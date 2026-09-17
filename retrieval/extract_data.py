# ============================================================
# retrieve.py  -- run this every time you want to query
# Connects to EXISTING Qdrant collection + EXISTING parent store.
# Does NOT touch CSVLoader, does NOT rebuild threads, does NOT re-embed anything.
# ============================================================

import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_classic.storage import LocalFileStore
from langchain_classic.storage._lc_store import create_kv_docstore
from qdrant_client import QdrantClient

load_dotenv()
base_path=Path().resolve().parent
parent_store_path=base_path/"ingestion"/"parent_store"

class TweetParentChildRetriever:
    def __init__(
        self,
        embedder_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        collection_name: str = "tweet_children",
        parent_store_dir: str = parent_store_path,
    ):
        self.embedder = HuggingFaceEmbeddings(model_name=embedder_model)

        client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
        )

        # Sanity check: fail loudly if the collection doesn't exist yet,
        # rather than silently trying to (re)create/ingest anything here.
        existing = [c.name for c in client.get_collections().collections]
        if collection_name not in existing:
            raise RuntimeError(
                f"Collection '{collection_name}' not found. Run ingest.py first."
            )

        self.vectorstore = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=self.embedder,
        )

        parent_dir = Path(parent_store_dir)
        if not parent_dir.exists():
            raise RuntimeError(
                f"Parent store directory '{parent_store_dir}' not found. Run ingest.py first."
            )
        fs_store = LocalFileStore(parent_store_dir)
        self.parent_store = create_kv_docstore(fs_store)

    def search(self, query: str, top_k_children: int = 10, top_n_parents: int = 5):
        child_results = self.vectorstore.similarity_search_with_score(query, k=top_k_children)

        parent_best = {}
        for doc, score in child_results:
            thread_id = doc.metadata["thread_id"]
            if thread_id not in parent_best or score > parent_best[thread_id]["score"]:
                parent_best[thread_id] = {"score": score, "matched_child": doc}

        ranked = sorted(parent_best.items(), key=lambda x: x[1]["score"], reverse=True)[:top_n_parents]

        thread_ids = [tid for tid, _ in ranked]
        parent_docs = self.parent_store.mget(thread_ids)

        results = []
        for (thread_id, info), parent_doc in zip(ranked, parent_docs):
            results.append({
                "thread_id": thread_id,
                "score": info["score"],
                "matched_tweet_text": info["matched_child"].page_content,
                "matched_tweet_id": info["matched_child"].metadata["tweet_id"],
                "parent_thread_text": parent_doc.page_content if parent_doc else None,
                "tweet_ids_in_thread": parent_doc.metadata["tweet_ids"] if parent_doc else [],
            })
        return results


if __name__ == "__main__":
    retriever = TweetParentChildRetriever()  # no CSV, no ingestion -- just connects

    results = retriever.search("ability stream from pc app")
    for r in results:
        print(f"[{r['score']:.3f}] tweet {r['matched_tweet_id']}: {r['matched_tweet_text']}")
        print(f"Thread:\n{r['parent_thread_text']}\n")