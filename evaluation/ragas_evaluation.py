"""
Production-grade RAG evaluation through RAGAS.

Source/reference:
https://atalupadhyay.wordpress.com/2026/01/30/rag-evaluation-from-bleu-scores-to-production-ready-metrics/
"""
from langchain_openai import ChatOpenAI
from openai import OpenAI,AsyncOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
import sys
import json
import math
import asyncio
import time
import datetime
from pathlib import Path
from typing import List, Dict
import os   
from langchain_nvidia_ai_endpoints import ChatNVIDIA
import pandas as pd
from datasets import Dataset
from ragas.llms import llm_factory,LangchainLLMWrapper
from langchain_ollama import ChatOllama
from langchain_openai import OpenAIEmbeddings
from ragas.embeddings import embedding_factory,LangchainEmbeddingsWrapper
from ragas import evaluate
# from langchain_groq import ChatGroq
from groq import AsyncGroq
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    ContextEntityRecall,
    AnswerAccuracy,
    AnswerCorrectness,

)
from ragas import evaluate, EvaluationDataset
from dotenv import load_dotenv
load_dotenv()
# client=AsyncGroq()
client=AsyncOpenAI()
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from app.support_agent_workflow import call_workflow

api_key=os.getenv("GROQ_API_KEY")

# ---------------------------------------------------------
# Rate-limit logger
# ---------------------------------------------------------

class RateLogger:
    """
    Appends every rate-limit event (sleep, retry, recovery) to
    ``rate_limit_log.json`` so you can post-hoc inspect pacing.

    Log entry schema
    ----------------
    {
        "event"     : "sample_cooldown" | "experiment_cooldown" | "retry_429" | "recovery",
        "metric"    : "<metric name or 'N/A'>",
        "sample_idx": <int or null>,
        "sleep_s"   : <float>,
        "reason"    : "<human-readable string>",
        "ts"        : "<ISO-8601 UTC>"
    }
    """

    def __init__(self, log_path: Path):
        self.path = log_path
        # Start fresh each run (append mode would mix runs)
        self._entries: list = []
        self._flush()  # create / truncate the file

    def _flush(self) -> None:
        """Write all entries to disk atomically."""
        tmp = self.path.with_suffix(".tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(self._entries, fh, indent=2, ensure_ascii=False)
            tmp.replace(self.path)
        except OSError as exc:
            print(f"[RateLogger] ERROR writing log: {exc}")

    def log(self, event: str, metric: str, sample_idx, sleep_s: float, reason: str) -> None:
        entry = {
            "event"      : event,
            "metric"     : metric,
            "sample_idx" : sample_idx,
            "sleep_s"    : round(sleep_s, 3),
            "reason"     : reason,
            "ts"         : datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        self._entries.append(entry)
        self._flush()
        print(
            f"[RateLimit] {event:22s} | metric={metric:<22s} "
            f"| sample={str(sample_idx):<4} | sleep={sleep_s:.1f}s | {reason}"
        )


# ---------------------------------------------------------
# Rate-limited LLM wrapper
# ---------------------------------------------------------

class RateLimitedLLM:
    """
    Thin proxy that adds Groq-safe pacing around any RAGAS LLM.

    Pacing rules (matching the sequence diagram)
    -------------------------------------------
    sample_cooldown   : wait N seconds *after* each sample is scored
    experiment_cooldown: wait N seconds *after* all samples of one metric
    retry_wait        : wait N seconds then retry on HTTP 429
    max_retries       : give up after this many consecutive 429s
    """

    def __init__(
        self,
        llm,
        rate_logger: "RateLogger",
        sample_cooldown: float  = 25.0,
        experiment_cooldown: float = 35.0,
        retry_wait: float       = 65.0,
        max_retries: int        = 3,
    ):
        self._llm               = llm
        self._rl                = rate_logger
        self.sample_cooldown    = sample_cooldown
        self.experiment_cooldown = experiment_cooldown
        self.retry_wait         = retry_wait
        self.max_retries        = max_retries
        # State set by the evaluation loop
        self.current_metric: str = "N/A"
        self.current_sample: int = 0

    # Proxy every attribute access to the real LLM so RAGAS
    # can call .langchain_llm, .agenerate(), etc. transparently.
    def __getattr__(self, name):
        return getattr(self._llm, name)

    def _call_with_retry(self, fn, *args, **kwargs):
        """Call ``fn(*args, **kwargs)`` with 429 retry logic."""
        for attempt in range(self.max_retries + 1):
            try:
                result = fn(*args, **kwargs)
                # ── sample cooldown ──────────────────────────
                self._rl.log(
                    "sample_cooldown",
                    self.current_metric,
                    self.current_sample,
                    self.sample_cooldown,
                    "sleeping between samples",
                )
                time.sleep(self.sample_cooldown)
                self.current_sample += 1
                return result
            except Exception as exc:
                # Detect rate-limit by checking status code or message
                msg = str(exc).lower()
                is_429 = (
                    "429" in msg
                    or "rate limit" in msg
                    or "ratelimit" in msg
                    or "too many requests" in msg
                )
                if is_429 and attempt < self.max_retries:
                    self._rl.log(
                        "retry_429",
                        self.current_metric,
                        self.current_sample,
                        self.retry_wait,
                        f"attempt {attempt + 1}/{self.max_retries} — waiting for TPM window to clear",
                    )
                    time.sleep(self.retry_wait)
                    self._rl.log(
                        "recovery",
                        self.current_metric,
                        self.current_sample,
                        0.0,
                        "retrying after 429 cooldown",
                    )
                else:
                    raise

    # RAGAS calls generate / agenerate on the inner LLM.
    # We intercept both sync and async paths.
    def generate(self, *args, **kwargs):
        return self._call_with_retry(self._llm.generate, *args, **kwargs)

    async def agenerate(self, *args, **kwargs):
        # Run the sync wrapper in a thread to keep async compatibility
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._call_with_retry(self._llm.generate, *args, **kwargs),
        )


# ---------------------------------------------------------
# Checkpoint logger
# ---------------------------------------------------------

class CheckpointLogger:
    """
    Persists evaluation progress to a JSON file so that long
    runs can be inspected or resumed after interruption.

    Schema
    ------
    {
        "run_id"   : "<ISO-8601 timestamp of the run>",
        "phase1"   : [
            {
                "row_index"   : <int>,
                "question"    : "...",
                "answer"      : "...",
                "contexts"    : [...],
                "ground_truth": "...",
                "logged_at"   : "<ISO-8601>"
            },
            ...
        ],
        "phase2"   : [
            {
                "metric"      : "<metric_name>",
                "scores"      : [<float>, ...],   # one per sample
                "logged_at"   : "<ISO-8601>"
            },
            ...
        ]
    }
    """

    def __init__(self, checkpoint_path: Path):
        self.path = checkpoint_path
        self._data: dict = self._load()

    # --------------------------------------------------
    # Internal helpers
    # --------------------------------------------------

    def _load(self) -> dict:
        """Load an existing checkpoint file, or start fresh."""
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as fh:
                    return json.load(fh)
            except (json.JSONDecodeError, OSError):
                print(
                    f"[Checkpoint] WARNING: Could not read "
                    f"{self.path}. Starting fresh."
                )
        return {
            "run_id" : datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "phase1" : [],
            "phase2" : [],
        }

    def _save(self) -> None:
        """Atomically write the checkpoint to disk."""
        tmp = self.path.with_suffix(".tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(self._data, fh, indent=2, ensure_ascii=False)
            tmp.replace(self.path)
            print(f"[Checkpoint] Saved → {self.path}")
        except OSError as exc:
            print(f"[Checkpoint] ERROR saving checkpoint: {exc}")

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------

    def log_phase1_row(self, row_index: int, qa_row: dict) -> None:
        """
        Called after Phase 1 generates a RAG response for one golden row.

        Parameters
        ----------
        row_index : int
            Zero-based index of the golden sample.
        qa_row : dict
            Dict with keys: question, answer, contexts, ground_truth.
        """
        entry = {
            "row_index"   : row_index,
            "question"    : qa_row.get("question", ""),
            "answer"      : qa_row.get("answer", ""),
            "contexts"    : qa_row.get("contexts", []),
            "ground_truth": qa_row.get("ground_truth", ""),
            "logged_at"   : datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        self._data["phase1"].append(entry)
        self._save()
        print(
            f"[Checkpoint] Phase 1 — row {row_index} logged "
            f"(answer length: {len(entry['answer'])} chars, "
            f"{len(entry['contexts'])} context(s))"
        )

    def log_phase2_metric(self, metric: str, scores: List) -> None:
        """
        Called after Phase 2 calculates scores for one metric.

        Parameters
        ----------
        metric : str
            Name of the RAGAS metric (e.g. 'faithfulness').
        scores : list
            Per-sample scores as floats (NaN is serialised as null).
        """
        # Convert NaN/float to JSON-safe values
        safe_scores = [
            None if (isinstance(s, float) and math.isnan(s)) else s
            for s in scores
        ]
        entry = {
            "metric"    : metric,
            "scores"    : safe_scores,
            "logged_at" : datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        self._data["phase2"].append(entry)
        self._save()
        valid = [s for s in safe_scores if s is not None]
        avg   = sum(valid) / len(valid) if valid else float("nan")
        print(
            f"[Checkpoint] Phase 2 — metric '{metric}' logged "
            f"(avg={avg:.4f}, {len(valid)}/{len(safe_scores)} valid scores)"
        )

# ---------------------------------------------------------
# Project setup
# ---------------------------------------------------------



# ---------------------------------------------------------
# Prepare RAGAS dataset
# ---------------------------------------------------------

# ---------------------------------------------------------
# Token-budget constants (keep calls predictable for Groq)
# ---------------------------------------------------------
_MAX_CONTEXT_CHARS  = 400   # truncate each context string to this
_MAX_CONTEXT_CHUNKS = 2     # keep at most this many context chunks


def prepare_ragas_data(qa_pairs: List[Dict]) -> Dataset:
    """
    Convert QA results to RAGAS format.

    Context truncation
    ------------------
    Each context chunk is capped at ``_MAX_CONTEXT_CHARS`` characters
    and only the first ``_MAX_CONTEXT_CHUNKS`` chunks are kept, keeping
    per-call token usage predictable for rate-limited backends.
    """

    data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }

    for pair in qa_pairs:

        data["question"].append(pair["question"])
        data["answer"].append(pair["answer"])

        # RAGAS expects contexts as List[str]
        contexts = pair.get("contexts", [])

        if isinstance(contexts, str):
            contexts = [contexts] if contexts.strip() else []

        # ── Truncate for token budget ────────────────────────
        contexts = [
            c[:_MAX_CONTEXT_CHARS]
            for c in contexts[:_MAX_CONTEXT_CHUNKS]
        ]

        data["contexts"].append(contexts)

        data["ground_truth"].append(
            pair.get("ground_truth", "")
        )

    return Dataset.from_dict(data)


# ---------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------

def ragas_eval():

    # -----------------------------------------------------
    # Input / output files
    # -----------------------------------------------------

    evaluation_dir = Path(__file__).resolve().parent.parent

    input_file  = evaluation_dir / "golden_dataset.csv"
    

    # Checkpoint file lives next to this script
    checkpoint_file = Path(__file__).resolve().parent / "checkpoint.json"
    ckpt = CheckpointLogger(checkpoint_file)
    print(f"[Checkpoint] Run ID: {ckpt._data['run_id']}")

    df = pd.read_csv(input_file)
    df = df.head(1)
    print(f"Loaded {len(df)} golden examples")

    qa_data = []

    # -----------------------------------------------------
    # Run ResolveAI workflow
    # -----------------------------------------------------

    for index, row in df.iterrows():

        question = str(row["query"])

        print(
            f"\n[{index + 1}/{len(df)}] "
            f"Evaluating: {question[:100]}"
        )

        try:

            result = call_workflow(question)

            answer = result.get("draft", "")

            historical_evidence = result.get(
                "historical_evidence",
                []
            )

            # Normalize contexts
            if isinstance(historical_evidence, str):

                contexts = (
                    [historical_evidence]
                    if historical_evidence.strip()
                    else []
                )

            elif isinstance(historical_evidence, list):

                contexts = [
                    str(context)
                    for context in historical_evidence
                    if context
                ]

            else:

                contexts = []

            # -------------------------------------------------
            # Ground truth
            # -------------------------------------------------

            # Prefer ground_truth_answer if available.
            # Fall back to expected_action for now.
            if "ground_truth_answer" in df.columns:

                ground_truth = str(
                    row.get("ground_truth_answer", "")
                )

            else:

                ground_truth = str(
                    row.get("expected_action", "")
                )

            row_dict = {
                "question"    : question,
                "answer"      : answer,
                "contexts"    : contexts,
                "ground_truth": ground_truth,
            }
            qa_data.append(row_dict)

            # ── Phase 1 checkpoint ──────────────────────────
            ckpt.log_phase1_row(index, row_dict)

        except Exception as e:

            print(
                f"ERROR processing row {index}: {e}"
            )

            # Preserve row alignment
            row_dict = {
                "question"    : question,
                "answer"      : "",
                "contexts"    : [],
                "ground_truth": "",
            }
            qa_data.append(row_dict)

            # ── Phase 1 checkpoint (error row) ──────────────
            ckpt.log_phase1_row(index, row_dict)

    # ---------------------------------------------------------
    # Create RAGAS dataset
    # ---------------------------------------------------------

    eval_dataset = prepare_ragas_data(qa_data)

    # ---------------------------------------------------------
    # Evaluation models
    # ---------------------------------------------------------

    llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o", temperature=0.0))
    embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model="text-embedding-3-small")
    )
#     llm = LangchainLLMWrapper(AsyncOpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1"))
#     embeddings = LangchainEmbeddingsWrapper(
#     HuggingFaceEmbeddings(
#         model="sentence-transformers/all-MiniLM-L6-v2",
#     )
# )

    # ---------------------------------------------------------
    # Rate limiter + logger
    # ---------------------------------------------------------

    rate_log_file = Path(__file__).resolve().parent / "rate_limit_log.json"
    rate_logger   = RateLogger(rate_log_file)
    print(f"[RateLimit] Log → {rate_log_file}")

    # Wrap the LLM so every generate() call gets paced.
    # Switch the inner LLM here to ChatGroq / AsyncOpenAI as needed;
    # the wrapper is model-agnostic.
    paced_llm = RateLimitedLLM(
        llm,
        rate_logger,
        sample_cooldown    = 25.0,   # seconds between samples
        experiment_cooldown= 35.0,   # seconds between metrics
        retry_wait         = 65.0,   # seconds to wait after 429
        max_retries        = 3,
    )

    # ---------------------------------------------------------
    # Metrics  (one object per experiment)
    # ---------------------------------------------------------

    metric_definitions = [
        ("faithfulness",       Faithfulness(llm=llm)),
        ("answer_relevancy",   AnswerRelevancy(llm=llm, embeddings=embeddings)),
        ("context_precision",  ContextPrecision(llm=llm)),
        ("context_recall",     ContextRecall(llm=llm)),
        ("context_entity_recall", ContextEntityRecall(llm=llm)),
        ("nv_accuracy",        AnswerAccuracy(llm=llm)),
        ("answer_correctness", AnswerCorrectness(llm=llm, embeddings=embeddings)),
    ]

    metric_columns = [name for name, _ in metric_definitions]

    # ---------------------------------------------------------
    # Run RAGAS — one metric (experiment) at a time
    # ---------------------------------------------------------

    print("\nRunning RAGAS evaluation (rate-limited, one metric at a time)...\n")

    results_df = pd.DataFrame()   # will be built column by column

    for exp_idx, (metric_name, metric_obj) in enumerate(metric_definitions):

        print(f"\n[Experiment {exp_idx + 1}/{len(metric_definitions)}] Scoring: {metric_name}")

        # Tell the paced LLM which metric/sample we're on
        paced_llm.current_metric = metric_name
        paced_llm.current_sample = 0

        try:
            result = evaluate(
                eval_dataset,
                metrics=[metric_obj],
                llm=llm,
                embeddings=embeddings,
            )
            partial_df = result.to_pandas()

            if metric_name in partial_df.columns:
                results_df[metric_name] = pd.to_numeric(
                    partial_df[metric_name], errors="coerce"
                )
            else:
                results_df[metric_name] = float("nan")

        except Exception as exc:
            print(f"[Experiment] ERROR scoring '{metric_name}': {exc}")
            results_df[metric_name] = float("nan")

        # ── Phase 2 checkpoint ──────────────────────────────
        ckpt.log_phase2_metric(
            metric_name,
            results_df[metric_name].tolist(),
        )

        # ── Experiment cooldown (skip after the last metric) ─
        if exp_idx < len(metric_definitions) - 1:
            rate_logger.log(
                "experiment_cooldown",
                metric_name,
                None,
                paced_llm.experiment_cooldown,
                "fully reset TPM window before next metric",
            )
            time.sleep(paced_llm.experiment_cooldown)
            rate_logger.log(
                "recovery",
                metric_name,
                None,
                0.0,
                "experiment cooldown complete",
            )

    print("\n=== RAGAS Results ===")
    print(results_df.to_string())



# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------

if __name__ == "__main__":
    ragas_eval()


    """ Output: 
=== RAGAS Results ===
{'faithfulness': 1.0000, 'answer_relevancy': nan, 'context_precision': 1.0000, 'context_recall': 1.0000, 'context_entity_recall': 0.4000, 'nv_accuracy': 0.7500, 'answer_correctness': nan}

RAGAS result columns:
['user_input', 'retrieved_contexts', 'response', 'reference', 'faithfulness', 'answer_relevancy', 'context_precision', 'context_recall', 'context_entity_recall', 'nv_accuracy', 'answer_correctness']

Evaluation completed.
Output saved to:
D:\Projects\ResolveAI\golden_dataset_ragas_results.csv

=== Average RAGAS Scores ===
faithfulness             : 1.0000
answer_relevancy         : nan
context_precision        : 1.0000
context_recall           : 1.0000
context_entity_recall    : 0.4000
answer_accuracy          : nan
answer_correctness       : nan
    """