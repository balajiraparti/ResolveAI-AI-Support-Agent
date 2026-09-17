"""
Intent Classifier Pipeline for Customer Support Tweets
========================================================
BERTopic-style approach: embed -> reduce dims -> cluster -> LLM-label clusters -> assign intent to every tweet.

Pipeline stages:
  1. Load & preprocess tweets (clean text, filter inbound customer messages)
  2. Embed text with a sentence-transformer model
  3. Reduce dimensionality with UMAP
  4. Cluster with HDBSCAN
  5. Extract top keywords per cluster (class-based TF-IDF, c-TF-IDF)
  6. Label each cluster with an LLM (Anthropic Claude) using representative examples + keywords
  7. Assign intent label back to every tweet
  8. Save results (per-tweet CSV + cluster summary CSV)

Usage:
    python intent_pipeline.py --input tweets.csv --text-col text --output-dir ./output

Requirements:
    pip install pandas numpy sentence-transformers umap-learn hdbscan scikit-learn langchain-ollama tqdm --break-system-packages

    Also requires Ollama running locally with the model pulled:
        ollama pull llama3.2:1b
        ollama serve   (usually already running as a background service)
"""

import os
import re
import json
import time
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("intent_pipeline")


# ---------------------------------------------------------------------------
# 1. PREPROCESSING
# ---------------------------------------------------------------------------

URL_RE = re.compile(r"https?://\S+")
MENTION_RE = re.compile(r"@\w+")
WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Strip URLs and @mentions, collapse whitespace. Keeps punctuation/emoji (signal for intent)."""
    if not isinstance(text, str):
        return ""
    text = URL_RE.sub("", text)
    text = MENTION_RE.sub("", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def load_and_preprocess(
    input_path: str,
    text_col: str = "text",
    inbound_col: str = "inbound",
    min_chars: int = 8,
) -> pd.DataFrame:
    """Load CSV, keep only inbound (customer) messages, clean text, drop low-signal rows."""
    log.info(f"Loading {input_path}")
    df = pd.read_csv(input_path)

    if inbound_col in df.columns:
        # inbound may be read as bool or as string "True"/"False"
        mask = df[inbound_col].astype(str).str.upper() == "TRUE"
        df = df[mask].copy()
        log.info(f"Filtered to {len(df)} inbound (customer) messages")

    df["clean_text"] = df[text_col].apply(clean_text)
    before = len(df)
    df = df[df["clean_text"].str.len() >= min_chars].reset_index(drop=True)
    log.info(f"Dropped {before - len(df)} low-signal rows (< {min_chars} chars after cleaning)")

    df = df.drop_duplicates(subset="clean_text").reset_index(drop=True)
    log.info(f"{len(df)} unique rows remain after dedup")
    return df


# ---------------------------------------------------------------------------
# 2. EMBEDDING
# ---------------------------------------------------------------------------

def embed_texts(texts: list, model_name: str = "all-MiniLM-L6-v2", batch_size: int = 128) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    log.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    log.info(f"Embedding {len(texts)} texts...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return np.asarray(embeddings)


# ---------------------------------------------------------------------------
# 3. DIMENSIONALITY REDUCTION (UMAP)
# ---------------------------------------------------------------------------

def reduce_dimensions(
    embeddings: np.ndarray,
    n_components: int = 5,
    n_neighbors: int = 15,
    min_dist: float = 0.0,
    random_state: int = 42,
) -> np.ndarray:
    import umap

    log.info(f"Running UMAP: {embeddings.shape[1]}d -> {n_components}d")
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=random_state,
    )
    reduced = reducer.fit_transform(embeddings)
    return reduced


# ---------------------------------------------------------------------------
# 4. CLUSTERING (HDBSCAN)
# ---------------------------------------------------------------------------

def cluster_embeddings(
    reduced_embeddings: np.ndarray,
    min_cluster_size: int = 30,
    min_samples: int = None,
) -> np.ndarray:
    import hdbscan

    log.info(f"Running HDBSCAN (min_cluster_size={min_cluster_size})")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    labels = clusterer.fit_predict(reduced_embeddings)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    log.info(f"Found {n_clusters} clusters, {n_noise} noise points (label -1)")
    return labels


# ---------------------------------------------------------------------------
# 5. CLASS-BASED TF-IDF (c-TF-IDF) KEYWORDS PER CLUSTER
# ---------------------------------------------------------------------------

def extract_cluster_keywords(
    df: pd.DataFrame,
    text_col: str = "clean_text",
    cluster_col: str = "cluster",
    top_n: int = 10,
) -> dict:
    """BERTopic-style c-TF-IDF: treat each cluster as one 'document' and rank terms."""
    from sklearn.feature_extraction.text import CountVectorizer

    clusters = sorted(c for c in df[cluster_col].unique() if c != -1)
    docs = [" ".join(df.loc[df[cluster_col] == c, text_col]) for c in clusters]

    vectorizer = CountVectorizer(stop_words="english", ngram_range=(1, 2), max_features=5000)
    count_matrix = vectorizer.fit_transform(docs)  # (n_clusters, n_terms)
    words = vectorizer.get_feature_names_out()

    # c-TF-IDF: term freq within cluster, scaled by inverse doc freq across clusters
    tf = count_matrix.toarray()
    tf_norm = tf / (tf.sum(axis=1, keepdims=True) + 1e-9)
    df_count = (tf > 0).sum(axis=0)
    idf = np.log(1 + len(clusters) / (df_count + 1e-9))
    ctfidf = tf_norm * idf

    keywords = {}
    for i, c in enumerate(clusters):
        top_idx = np.argsort(ctfidf[i])[::-1][:top_n]
        keywords[c] = [words[j] for j in top_idx]
    return keywords


# ---------------------------------------------------------------------------
# 6. LLM CLUSTER LABELING (Anthropic Claude)
# ---------------------------------------------------------------------------

def extract_json_object(raw: str) -> dict:
    r"""Parse the FIRST complete JSON object out of a model response, ignoring anything after it.

    Small local models frequently repeat themselves or tack on extra chatter after a valid
    JSON object (e.g. `{"a": 1}{"a": 1}` or `{"a": 1}\n\nNote: ...`). A naive
    `re.search(r"\{.*\}")` greedily matches through to the LAST `}` in the string, which
    swallows the trailing junk too and causes `json.JSONDecodeError: Extra data`.
    `json.JSONDecoder.raw_decode` instead parses exactly one object starting at the first
    `{` and simply stops, so trailing text is safely ignored.
    """
    cleaned = re.sub(r"```(json)?", "", raw).strip()
    start = cleaned.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in model output: {cleaned[:200]}")
    decoder = json.JSONDecoder()
    obj, _end_index = decoder.raw_decode(cleaned, start)
    return obj


LABEL_PROMPT = """You are labeling a cluster of customer support tweets sent BY CUSTOMERS TO a brand.

Here are representative tweets from this cluster:
{examples}

Top keywords for this cluster (via c-TF-IDF): {keywords}

Respond with ONLY a single JSON object and nothing else. No explanation, no markdown fences, no extra words.
Output exactly this format:
{{"intent_label": "short_snake_case_label", "display_name": "Human Readable Label", "description": "one short sentence describing this intent"}}

The intent_label should be a concise category like "playback_bug_report", "subscription_cancellation",
"feature_request", "pricing_complaint", "account_access_issue", "general_praise", etc.

JSON:
"""


def get_representative_samples(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    cluster_col: str = "cluster",
    text_col: str = "clean_text",
    n_centroid: int = 12,
    n_random: int = 6,
    random_state: int = 42,
) -> dict:
    """For each cluster, pick texts closest to the centroid + a few random ones for diversity."""
    rng = np.random.default_rng(random_state)
    samples = {}
    for c in sorted(cid for cid in df[cluster_col].unique() if cid != -1):
        idx = df.index[df[cluster_col] == c].to_numpy()
        cluster_embs = embeddings[idx]
        centroid = cluster_embs.mean(axis=0)
        dists = np.linalg.norm(cluster_embs - centroid, axis=1)
        closest_order = idx[np.argsort(dists)]

        n_take_centroid = min(n_centroid, len(closest_order))
        centroid_idx = closest_order[:n_take_centroid]

        remaining = np.setdiff1d(idx, centroid_idx)
        n_take_random = min(n_random, len(remaining))
        random_idx = rng.choice(remaining, size=n_take_random, replace=False) if n_take_random > 0 else np.array([], dtype=int)

        chosen = np.concatenate([centroid_idx, random_idx])
        samples[c] = df.loc[chosen, text_col].tolist()
    return samples


def get_ollama_llm(model: str = "llama3.2:1b", base_url: str = "http://localhost:11434", temperature: float = 0.0):
    from langchain_ollama import OllamaLLM

    return OllamaLLM(model=model, base_url=base_url, temperature=temperature)


def label_clusters_with_llm(
    samples: dict,
    keywords: dict,
    model: str = "llama3.2:1b",
    base_url: str = "http://localhost:11434",
    max_examples_in_prompt: int = 15,
    max_retries: int = 2,
    retry_base_delay: float = 1.5,
) -> dict:
    """Call a local Ollama model (via LangChain) once per cluster to get an intent label.

    Retries with exponential backoff (retry_base_delay, then retry_base_delay*2, ...) on
    parse failures — this smooths over occasional garbage/truncated output from a small
    model without needing a human to re-run anything.
    """
    llm = get_ollama_llm(model=model, base_url=base_url)
    labels = {}

    for cluster_id, texts in tqdm(samples.items(), desc="Labeling clusters with LLM"):
        example_block = "\n".join(f"- {t}" for t in texts[:max_examples_in_prompt])
        kw = ", ".join(keywords.get(cluster_id, []))
        prompt = LABEL_PROMPT.format(examples=example_block, keywords=kw)

        parsed = None
        last_err = None
        for attempt in range(max_retries + 1):
            try:
                raw = llm.invoke(prompt).strip()
                candidate = extract_json_object(raw)
                # basic schema check - small models sometimes drop keys
                if not all(k in candidate and candidate[k] for k in ("intent_label", "display_name", "description")):
                    raise ValueError(f"Missing or empty expected keys in parsed JSON: {candidate}")
                # sanitize label into snake_case-ish token
                candidate["intent_label"] = re.sub(r"[^a-z0-9_]+", "_", candidate["intent_label"].lower()).strip("_")
                parsed = candidate
                break
            except Exception as e:
                last_err = e
                if attempt < max_retries:
                    delay = retry_base_delay * (2 ** attempt)
                    log.info(f"Cluster {cluster_id}: attempt {attempt + 1} failed ({e}); retrying in {delay:.1f}s")
                    time.sleep(delay)
                continue

        if parsed is None:
            log.warning(f"Cluster {cluster_id}: LLM labeling failed after retries ({last_err}); using fallback label")
            parsed = {
                "intent_label": f"cluster_{cluster_id}",
                "display_name": f"Cluster {cluster_id}",
                "description": "Auto-labeling failed; needs manual review.",
            }

        # Defensive: guarantee every entry has all three keys no matter what path got here
        parsed.setdefault("intent_label", f"cluster_{cluster_id}")
        parsed.setdefault("display_name", f"Cluster {cluster_id}")
        parsed.setdefault("description", "No description available.")

        labels[cluster_id] = parsed

    labels[-1] = {
        "intent_label": "uncategorized",
        "display_name": "Uncategorized / Noise",
        "description": "Did not fit clearly into any discovered cluster.",
    }
    return labels


# ---------------------------------------------------------------------------
# 7. OPTIONAL: MERGE NEAR-DUPLICATE CLUSTER LABELS VIA LLM REVIEW
# ---------------------------------------------------------------------------

MERGE_PROMPT = """Here is a list of intent labels discovered from clustering customer support tweets:

{label_list}

Some of these may be near-duplicates (e.g. "app_crashing" and "app_keeps_crashing" mean the same thing).
Respond with ONLY a single JSON object and nothing else. No explanation, no markdown fences.
Map every original label to a final, deduplicated label. Use the clearest / most general label as the
canonical name for each merged group. Every original label above MUST appear as a key.
Format: {{"original_label_1": "canonical_label", "original_label_2": "canonical_label", ...}}

JSON:
"""


def review_and_merge_labels(
    cluster_labels: dict,
    model: str = "llama3.2:1b",
    base_url: str = "http://localhost:11434",
) -> dict:
    unique_labels = sorted({v["intent_label"] for v in cluster_labels.values() if v["intent_label"] != "uncategorized"})
    if len(unique_labels) <= 1:
        return {lbl: lbl for lbl in unique_labels}

    llm = get_ollama_llm(model=model, base_url=base_url)
    prompt = MERGE_PROMPT.format(label_list="\n".join(f"- {l}" for l in unique_labels))
    try:
        raw = llm.invoke(prompt).strip()
        mapping = extract_json_object(raw)
    except Exception as e:
        # 1B models are unreliable on this "review the whole list" task - fail soft rather than crash the run
        log.warning(f"Label merge review failed ({e}); skipping merge step")
        mapping = {lbl: lbl for lbl in unique_labels}
    return mapping


# ---------------------------------------------------------------------------
# 8. MAIN PIPELINE
# ---------------------------------------------------------------------------

def run_pipeline(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load & preprocess
    df = load_and_preprocess(args.input, text_col=args.text_col, inbound_col=args.inbound_col)

    if args.sample_size and len(df) > args.sample_size:
        df = df.sample(n=args.sample_size, random_state=42).reset_index(drop=True)
        log.info(f"Subsampled to {len(df)} rows for this run")

    texts = df["clean_text"].tolist()

    # 2. Embed
    embeddings = embed_texts(texts, model_name=args.embedding_model)

    # 3. Reduce dims
    reduced = reduce_dimensions(embeddings, n_components=args.umap_components)

    # 4. Cluster
    labels = cluster_embeddings(reduced, min_cluster_size=args.min_cluster_size)
    df["cluster"] = labels

    # 5. Keywords per cluster
    keywords = extract_cluster_keywords(df)

    # 6. Representative samples + LLM labeling
    samples = get_representative_samples(df, embeddings)
    cluster_labels = label_clusters_with_llm(
        samples,
        keywords,
        model=args.llm_model,
        base_url=args.ollama_base_url,
        max_retries=args.max_retries,
        retry_base_delay=args.retry_base_delay,
    )

    # 7. Merge near-duplicate labels
    if args.merge_labels:
        mapping = review_and_merge_labels(cluster_labels, model=args.llm_model, base_url=args.ollama_base_url)
        for c, info in cluster_labels.items():
            if info["intent_label"] in mapping:
                info["intent_label"] = mapping[info["intent_label"]]

    # 8. Assign back to every tweet (use .get() with fallbacks — never let one malformed
    #    LLM response for a cluster crash the whole run at this stage)
    def safe_label_lookup(c, key, default):
        info = cluster_labels.get(c, {})
        value = info.get(key, default)
        return value if value else default

    df["intent_label"] = df["cluster"].map(lambda c: safe_label_lookup(c, "intent_label", f"cluster_{c}"))
    df["intent_display_name"] = df["cluster"].map(lambda c: safe_label_lookup(c, "display_name", f"Cluster {c}"))

    # Save outputs
    tweets_out = out_dir / "tweets_with_intent.csv"
    df.drop(columns=["clean_text"]).to_csv(tweets_out, index=False)
    log.info(f"Saved per-tweet results to {tweets_out}")

    summary_rows = []
    for c, info in sorted(cluster_labels.items()):
        summary_rows.append({
            "cluster_id": c,
            "intent_label": info["intent_label"],
            "display_name": info["display_name"],
            "description": info["description"],
            "num_tweets": int((df["cluster"] == c).sum()),
            "top_keywords": ", ".join(keywords.get(c, [])),
        })
    summary_df = pd.DataFrame(summary_rows).sort_values("num_tweets", ascending=False)
    summary_out = out_dir / "cluster_summary.csv"
    summary_df.to_csv(summary_out, index=False)
    log.info(f"Saved cluster summary to {summary_out}")

    log.info("Pipeline complete.")
    return df, summary_df


def parse_args():
    p = argparse.ArgumentParser(description="Intent classification pipeline for support tweets")
    p.add_argument("--input", required=True, help="Path to input CSV")
    p.add_argument("--text-col", default="text", help="Column containing tweet text")
    p.add_argument("--inbound-col", default="inbound", help="Column marking customer-authored rows")
    p.add_argument("--output-dir", default="./output", help="Directory to write output CSVs")
    p.add_argument("--embedding-model", default="all-MiniLM-L6-v2", help="sentence-transformers model name")
    p.add_argument("--llm-model", default="llama3.2:1b", help="Ollama model name for cluster labeling")
    p.add_argument("--ollama-base-url", default="http://localhost:11434", help="Ollama server URL")
    p.add_argument("--umap-components", type=int, default=5)
    p.add_argument("--min-cluster-size", type=int, default=30)
    p.add_argument("--sample-size", type=int, default=None, help="Optional cap on rows for a quick test run")
    p.add_argument("--merge-labels", action="store_true", help="Run an extra LLM pass to merge near-duplicate labels")
    p.add_argument("--max-retries", type=int, default=2, help="Retries per cluster if LLM output fails to parse")
    p.add_argument("--retry-base-delay", type=float, default=1.5, help="Base seconds for exponential backoff between retries")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)