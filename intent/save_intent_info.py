import re
import pandas as pd

from pathlib import Path

from sentence_transformers import SentenceTransformer

from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired

from umap import UMAP
from hdbscan import HDBSCAN

from sklearn.feature_extraction.text import CountVectorizer


# ==================================================
# 1. Load dataset
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

file_path = (
    PROJECT_ROOT
    / "experiments"
    / "spotify_customer.csv"
)

df = pd.read_csv(file_path)

print(f"Total interactions: {len(df):,}")


# ==================================================
# 2. Prepare text
# ==================================================

df["text"] = (
    df["text"]
    .fillna("")
    .astype(str)
    .str.strip()
)


# ==================================================
# 3. Clean text specifically for intent discovery
# ==================================================

CUSTOM_STOPWORDS = {
    "spotify",
    "spotifycares",
    "please",
    "pls",
    "plz",
    "help",
    "hey",
    "hi",
    "hello",
    "thanks",
    "thank",
    "would",
    "could",
    "can",
    "just",
    "really",
    "also",
    "still",
    "get",
    "got",
    "want",
    "need",
}


def clean_for_intent(text):

    text = str(text)

    # Remove URLs
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    # Remove @mentions
    text = re.sub(
        r"@\w+",
        " ",
        text
    )

    # Remove standalone IDs/numbers
    text = re.sub(
        r"\b\d+\b",
        " ",
        text
    )

    # Remove HTML entities
    text = re.sub(
        r"&\w+;",
        " ",
        text
    )

    # Lowercase
    text = text.lower()

    # Keep alphabetic text
    text = re.sub(
        r"[^a-z\s]",
        " ",
        text
    )

    words = [
        word
        for word in text.split()
        if word not in CUSTOM_STOPWORDS
    ]

    return " ".join(words).strip()


df["intent_text"] = (
    df["text"]
    .apply(clean_for_intent)
)


# Remove very short documents
df = df[
    df["intent_text"].str.len() >= 10
].copy()


# Remove exact duplicates
df = df.drop_duplicates(
    subset=["intent_text"]
).reset_index(drop=True)


documents = df["intent_text"].tolist()

print(
    f"Documents for clustering: {len(documents):,}"
)


# ==================================================
# 4. Embedding model
# ==================================================

embedding_model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)


# ==================================================
# 5. UMAP
# ==================================================

umap_model = UMAP(
    n_neighbors=15,
    n_components=5,
    min_dist=0.0,
    metric="cosine",
    random_state=42
)


# ==================================================
# 6. HDBSCAN
# ==================================================

hdbscan_model = HDBSCAN(
    min_cluster_size=30,
    min_samples=10,
    metric="euclidean",
    cluster_selection_method="eom",
    prediction_data=True
)


# ==================================================
# 7. Vectorizer
# ==================================================

vectorizer_model = CountVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.95
)


# ==================================================
# 8. Topic representation
# ==================================================

representation_model = KeyBERTInspired()


# ==================================================
# 9. BERTopic
# ==================================================

topic_model = BERTopic(
    embedding_model=embedding_model,

    umap_model=umap_model,

    hdbscan_model=hdbscan_model,

    vectorizer_model=vectorizer_model,

    representation_model=representation_model,

    top_n_words=10,

    # Discover natural number of topics first
    nr_topics=20,

    calculate_probabilities=True,

    verbose=True
)


# ==================================================
# 10. Discover topics
# ==================================================

topics, probabilities = (
    topic_model.fit_transform(documents)
)


# ==================================================
# 11. Topic information
# ==================================================

topic_info = (
    topic_model
    .get_topic_info()
)


print("\n==============================")
print("DISCOVERED TOPICS")
print("==============================")

print(topic_info)


# ==================================================
# 12. Export
# ==================================================


topic_info.to_csv(
    "spotify_tweet_intent_30.csv",
    index=False
)

