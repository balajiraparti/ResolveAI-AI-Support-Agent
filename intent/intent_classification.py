import ast
import json
import os
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_nvidia_ai_endpoints import ChatNVIDIA

import pandas as pd
# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "intent"
    / "cluster_summary.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "intent"
    / "cluster_summary_with_embeddings.csv"
)


# ============================================================
# CONFIG
# ============================================================

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

TOP_K = 5


# ============================================================
# LOAD DATASET
# ============================================================



df = pd.read_csv(INPUT_FILE)

required_columns = {
    "intent_label",
    "display_name",
    "description",
    "num_tweets"
}

missing_columns = (
    required_columns - set(df.columns)
)

if missing_columns:
    raise ValueError(
        f"Missing columns: {sorted(missing_columns)}"
    )


# Make sure descriptions are strings
df["description"] = (
    df["description"]
    .fillna("")
    .astype(str)
    .str.strip()
)




# ============================================================
# HUGGING FACE EMBEDDING MODEL
# ============================================================



hf_embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,

    model_kwargs={
        "device": "cpu"
    },

    encode_kwargs={
        "normalize_embeddings": True
    }
)


# ============================================================
# GENERATE DESCRIPTION EMBEDDINGS
# ============================================================

def generate_embedding():

    print(
        "\nGenerating embeddings for descriptions..."
    )

    texts_to_embed = (
        df["description"]
        .tolist()
    )

    embeddings = (
        hf_embeddings.embed_documents(
            texts_to_embed
        )
    )

    # Add embedding column
    df["embedding"] = embeddings

    # Save
    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        "\nSuccess."
    )

    print(
        f"Embedding dimension: "
        f"{len(embeddings[0])}"
    )

    print(
        f"Saved to:\n{OUTPUT_FILE}"
    )


# ============================================================
# LOAD EMBEDDINGS FROM CSV
# ============================================================

def load_embeddings():

    if "embedding" not in df.columns:

        raise ValueError(
            "The CSV does not contain an "
            "'embedding' column.\n"
            "Run generate_embedding() first."
        )


 


    def parse_embedding(value):

        if pd.isna(value):

            raise ValueError(
                "Found empty embedding."
            )

        # CSV stores list as a string:
        #
        # "[0.123, -0.456, ...]"
        #
        return np.array(
            ast.literal_eval(str(value)),
            dtype=np.float32
        )


    embeddings = np.vstack(
        df["embedding"]
        .apply(parse_embedding)
        .values
    )


    # Normalize again for safety
    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True
    )

    embeddings = (
        embeddings /
        np.maximum(norms, 1e-12)
    )


   


    return embeddings


# ============================================================
# LOAD STORED EMBEDDINGS
# ============================================================

all_embeddings = None


def initialize_embeddings():

    global all_embeddings

    all_embeddings = load_embeddings()


# ============================================================
# CLASSIFY / RETRIEVE INTENT
# ============================================================

def validate_intent_with_llm(user_query, results):
    """Validate whether the extracted table intent matches the query.

    If not, ask the LLM to create a better intent label and description.
    Returns the same list shape with an updated top result and llm metadata.
    """
    if not results:
        return results

    try:
        llm = ChatNVIDIA(
            model="openai/gpt-oss-20b",
            temperature=0.1,
        )
    except Exception:
        for result in results:
            result["llm_validation"] = {
                "status": "skipped",
                "reason": "ChatNVIDIA unavailable",
                "is_correct": True,
            }
        return results

    candidate_summary = json.dumps(
        [
            {
                "intent_label": item.get("intent_label"),
                "display_name": item.get("display_name"),
                "description": item.get("description"),
            }
            for item in results[:5]
        ],
        ensure_ascii=False,
    )

    prompt = f"""
You are validating a retrieved intent label against a user's query.

User Query:
{user_query}

Candidate intent rows from the table:
{candidate_summary}

Task:
1. Decide if the top intent label is actually correct for the user's query.
2. If it is correct, keep it.
3. If it is not correct, create a better intent label, display name, and description.
4. Return JSON only with this exact structure:
5. If the user input is vague or any gibberish word/sentence then make the label as UNKNOWN
{{
  "is_correct": true,
  "corrected_label": "",
  "display_name": "",
  "description": "",
  "reason": "short explanation"
}}

Rules:
- Use clear, business-friendly intent names.
- Keep the label concise and machine-readable, e.g. "account_login_issue".
- The description should explain the customer concern in one sentence.
- If the table match is already right, set `is_correct` to true and leave corrected_label/display_name/description empty strings.
"""

    try:
        response = llm.invoke(prompt)
        raw_text = getattr(response, "content", str(response)).strip()

        if raw_text.startswith("```"):
            raw_text = raw_text.strip("`")
            if raw_text.lower().startswith("json"):
                raw_text = raw_text[4:].strip()

        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start != -1 and end != -1 and end > start:
            raw_text = raw_text[start:end + 1]

        parsed = json.loads(raw_text)
    except Exception:
        parsed = {
            "is_correct": True,
            "corrected_label": "",
            "display_name": "",
            "description": "",
            "reason": "LLM validation parse fallback",
        }

    validated_result = results[0].copy()
    validated_result["llm_validation"] = {
        "status": "validated",
        "is_correct": bool(parsed.get("is_correct", True)),
        "reason": str(parsed.get("reason", "Matched to query.")),
        "response": parsed,
    }

    if bool(parsed.get("is_correct", True)):
        validated_result["llm_validation"]["status"] = "matched"
        updated_results = results.copy()
        updated_results[0] = validated_result
        return updated_results

    corrected_label = str(parsed.get("corrected_label") or validated_result["intent_label"]).strip()
    corrected_display = str(parsed.get("display_name") or corrected_label.replace("_", " ").title()).strip()
    corrected_description = str(parsed.get("description") or f"Intent generated from query: {user_query}").strip()

    validated_result["intent_label"] = corrected_label
    validated_result["display_name"] = corrected_display
    validated_result["description"] = corrected_description
    validated_result["num_tweets"] = 0
    validated_result["similarity_score"] = round(float(validated_result.get("similarity_score", 0.0)), 4)
    validated_result["llm_validation"]["status"] = "replaced"
    validated_result["llm_validation"]["response"] = parsed

    updated_results = [validated_result]
    updated_results.extend(results[1:])
    return updated_results


def classify_intent(
    user_query,
    top_k=5
):

    global all_embeddings

    if all_embeddings is None:

        initialize_embeddings()


    # --------------------------------------------------------
    # Generate query embedding
    # --------------------------------------------------------

    query_embedding = (
        hf_embeddings.embed_query(
            user_query
        )
    )


    query_vector = np.array(
        query_embedding,
        dtype=np.float32
    ).reshape(1, -1)

    
    # --------------------------------------------------------
    # Normalize query
    # --------------------------------------------------------

    query_norm = np.linalg.norm(
        query_vector
    )

    query_vector = (
        query_vector /
        max(query_norm, 1e-12)
    )


    # --------------------------------------------------------
    # Cosine similarity
    # --------------------------------------------------------

    similarity_scores = (
        cosine_similarity(
            query_vector,
            all_embeddings
        )[0]
    )


    # --------------------------------------------------------
    # Get TOP-K matches
    # --------------------------------------------------------

    top_k = min(
        top_k,
        len(df)
    )

    top_indices = np.argsort(
        similarity_scores
    )[::-1][:top_k]


    # --------------------------------------------------------
    # Build results
    # --------------------------------------------------------

    results = []

    for rank, idx in enumerate(
        top_indices,
        start=1
    ):

        row = df.iloc[idx]

        results.append({

            "rank": rank,

            "intent_label": row[
                "intent_label"
            ],

            "display_name": row[
                "display_name"
            ],

            "description": row[
                "description"
            ],

            "num_tweets": int(
                row["num_tweets"]
            ),

            "similarity_score": round(
                float(
                    similarity_scores[idx]
                ),
                4
            )
        })

    results = validate_intent_with_llm(user_query, results)
    return results


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(
    user_query,
    results
):

    print("\n")
    print("=" * 100)

    print(
        f"Customer Query:\n{user_query}"
    )

    print("=" * 100)


    for result in results:

        print(
            f"\n#{result['rank']}"
        )

        print(
            f"Similarity: "
            f"{result['similarity_score']:.4f}"
        )

        print(
            f"Intent Label: "
            f"{result['intent_label']}"
        )

        print(
            f"Display Name: "
            f"{result['display_name']}"
        )

        print(
            f"Description: "
            f"{result['description']}"
        )

        print(
            f"Number of Tweets: "
            f"{result['num_tweets']:,}"
        )

        print("-" * 100)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "\nResolveAI Intent Retrieval"
    )

    print(
        "=" * 100
    )


    # --------------------------------------------------------
    # If embeddings don't exist, generate them.
    # --------------------------------------------------------

    if "embedding" not in df.columns:

        print(
            "\nNo embeddings found."
        )

        generate_embedding()

        # Reload from the newly generated data
        df = pd.read_csv(
            OUTPUT_FILE
        )

        df["description"] = (
            df["description"]
            .fillna("")
            .astype(str)
        )

    else:

        print(
            "\nEmbeddings already exist."
        )


    # --------------------------------------------------------
    # Load embedding matrix
    # --------------------------------------------------------

    initialize_embeddings()
    df = pd.read_csv("spotify_customer_queries_20.csv")
    predictions = []
    # --------------------------------------------------------
    # Test query
    # --------------------------------------------------------
    for _, row in df.iterrows():
        query = row["text"]
        expected_intent = row["intent_label"]

        print("\n" + "=" * 80)
        print(f"Query: {query}")
        print(f"Expected Intent: {expected_intent}")
        print("-" * 80)

        # Run intent classification
        results = classify_intent(
            query,
            top_k=5
        )
        predictions.append({
        "query": query,
        "expected_intent": expected_intent,
        "predicted_intent": predicted_intent,
        "correct": predicted_intent == expected_intent
    })
        predicted_intent = results[0]["intent_label"]
        # Print classifier results
        print_results(
            query,
            results
        )
    results_df = pd.DataFrame(predictions)

    accuracy = results_df["correct"].mean()

    print("\n" + "=" * 80)
    print("INTENT CLASSIFICATION RESULTS")
    print("=" * 80)

    print(results_df.to_string(index=False))

    print(f"\nAccuracy: {accuracy:.2%}")
    # query = (
    #     "who are you"
      
    # )


    # results = classify_intent(
    #     query,
    #     top_k=5
    # )


    # print_results(
    #     query,
    #     results
    # )