import ast
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from langchain_huggingface import HuggingFaceEmbeddings


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


    # --------------------------------------------------------
    # Test query
    # --------------------------------------------------------

    query = (
        "hey i my wifi and i have seperate account can we both merge to family plan"
      
    )


    results = classify_intent(
        query,
        top_k=5
    )


    print_results(
        query,
        results
    )