"""
Production-grade RAG evaluation through RAGAS.

Source/reference:
https://atalupadhyay.wordpress.com/2026/01/30/rag-evaluation-from-bleu-scores-to-production-ready-metrics/
"""
from langchain_openai import ChatOpenAI
from openai import OpenAI
from langchain_huggingface import HuggingFaceEmbeddings
import sys
import json
import time
from pathlib import Path
from typing import List, Dict
from langchain_nvidia_ai_endpoints import ChatNVIDIA
import pandas as pd
from datasets import Dataset
from ragas.llms import llm_factory,LangchainLLMWrapper
from langchain_ollama import ChatOllama
from langchain_openai import OpenAIEmbeddings
from ragas.embeddings import embedding_factory
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    ContextEntityRecall,
    AnswerAccuracy,
    AnswerCorrectness,
)
from dotenv import load_dotenv
load_dotenv()
client=OpenAI()
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from app.support_agent_workflow import call_workflow


# ---------------------------------------------------------
# Project setup
# ---------------------------------------------------------



# ---------------------------------------------------------
# Prepare RAGAS dataset
# ---------------------------------------------------------

def prepare_ragas_data(qa_pairs: List[Dict]) -> Dataset:
    """
    Convert QA results to RAGAS format.
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

    input_file = evaluation_dir / "golden_dataset.csv"
    output_file = evaluation_dir / "golden_dataset_ragas_results.csv"

    df = pd.read_csv(input_file)
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

            qa_data.append(
                {
                    "question": question,
                    "answer": answer,
                    "contexts": contexts,
                    "ground_truth": ground_truth,
                }
            )

        except Exception as e:

            print(
                f"ERROR processing row {index}: {e}"
            )

            # Preserve row alignment
            qa_data.append(
                {
                    "question": question,
                    "answer": "",
                    "contexts": [],
                    "ground_truth": "",
                }
            )

    # ---------------------------------------------------------
    # Create RAGAS dataset
    # ---------------------------------------------------------

    eval_dataset = prepare_ragas_data(qa_data)

    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------


    # ---------------------------------------------------------
    # Evaluation models
    # ---------------------------------------------------------

    # llm=llm_factory('gpt-4o-mini', client=client)
    llm=ChatOpenAI(model="gpt-4o",temperature=.2)
    llm=LangchainLLMWrapper(llm)
    # embeddings = embedding_factory("openai",model="text-embedding-ada-002",client=client,interface='modern')
    embeddings=OpenAIEmbeddings(model="text-embedding-ada-002",client=client)
   
    metrics = [
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm,embeddings=embeddings),
        ContextPrecision(llm=llm),
        ContextRecall(llm=llm),
        ContextEntityRecall(llm=llm),
        AnswerAccuracy(llm=llm),
        AnswerCorrectness(llm=llm,embeddings=embeddings),
    ]


    # ---------------------------------------------------------
    # Run RAGAS
    # ---------------------------------------------------------

    print("\nRunning RAGAS evaluation...\n")

    results = evaluate(
        eval_dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
    )

    print("\n=== RAGAS Results ===")
    print(results)

    # ---------------------------------------------------------
    # Convert RAGAS results to DataFrame
    # ---------------------------------------------------------

    results_df = results.to_pandas()

    print("\nRAGAS result columns:")
    print(results_df.columns.tolist())

    # ---------------------------------------------------------
    # Add metrics to original golden dataset
    # ---------------------------------------------------------

    metric_columns = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "context_entity_recall",
        "answer_accuracy",
        "answer_correctness",
    ]

    for metric in metric_columns:

        if metric in results_df.columns:

            df[metric] = pd.to_numeric(
                results_df[metric],
                errors="coerce"
            )

        else:

            # If RAGAS didn't calculate a metric,
            # preserve the column with NaN.
            df[metric] = float("nan")

    # ---------------------------------------------------------
    # Optional: add generated answer and retrieved context
    # ---------------------------------------------------------

    df["ragas_answer"] = [
        item["answer"]
        for item in qa_data
    ]

    df["ragas_context_count"] = [
        len(item["contexts"])
        for item in qa_data
    ]

    # ---------------------------------------------------------
    # Export
    # ---------------------------------------------------------

    df.to_csv(
        output_file,
        index=False,
        encoding="utf-8",
    )

    print(
        f"\nEvaluation completed."
        f"\nOutput saved to:"
        f"\n{output_file}"
    )

    # ---------------------------------------------------------
    # Dataset-level summary
    # ---------------------------------------------------------

    print("\n=== Average RAGAS Scores ===")

    for metric in metric_columns:

        if metric in df.columns:

            score = df[metric].mean()

            print(
                f"{metric:25s}: {score:.4f}"
            )

    return df


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