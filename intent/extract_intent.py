import pandas as pd

from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from pathlib import Path 

# --------------------------------------------------
# 1. Load SpotifyCares dataset
# --------------------------------------------------
base_path=Path().resolve().parent
file_path=base_path/"experiments"/"spotify_customer.csv"
df = pd.read_csv(file_path)

print(f"Total interactions: {len(df)}")


# --------------------------------------------------
# 2. Prepare customer messages
# --------------------------------------------------

df["text"] = (
    df["text"]
    .fillna("")
    .astype(str)
    .str.strip()
)

# Remove empty messages
df = df[df["text"].str.len() > 5].copy()

documents = df["text"].tolist()

print(f"Documents for clustering: {len(documents)}")


# --------------------------------------------------
# 3. Load embedding model
# --------------------------------------------------

embedding_model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)


# --------------------------------------------------
# 4. UMAP
# --------------------------------------------------

umap_model = UMAP(
    n_neighbors=15,
    n_components=5,
    min_dist=0.0,
    metric="cosine",
    random_state=42
)


# --------------------------------------------------
# 5. HDBSCAN
# --------------------------------------------------

hdbscan_model = HDBSCAN(
    min_cluster_size=30,
    min_samples=10,
    metric="euclidean",
    cluster_selection_method="eom",
    prediction_data=True
)


# --------------------------------------------------
# 6. BERTopic
# --------------------------------------------------

topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,

    # Number of words representing each topic
    top_n_words=10,

    # Don't force a predefined number of topics
    nr_topics=None,

    calculate_probabilities=True,

    verbose=True
)


# --------------------------------------------------
# 7. Discover topics
# --------------------------------------------------

topics, probabilities = topic_model.fit_transform(
    documents
)


# --------------------------------------------------
# 8. Save topic assignments
# --------------------------------------------------

df["topic"] = topics

print("\nTopic discovery complete.")
print(df["topic"].value_counts().head(20))

topic_info = topic_model.get_topic_info()

print(topic_info)
# Add topic information to the dataframe
topic_info = topic_model.get_topic_info()

# Map topic ID -> topic name
topic_name_map = dict(
    zip(topic_info["Topic"], topic_info["Name"])
)

df["topic_name"] = df["topic"].map(topic_name_map)

# Export the complete dataframe
df.to_csv(
    "spotify_tweets_with_topics.csv",
    index=False
)

print("Exported:", len(df), "rows")
print("File: spotify_tweets_with_topics.csv")