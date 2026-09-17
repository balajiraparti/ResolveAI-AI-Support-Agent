"""
retrieve.py
Run this every time you want to query.

Connects to the EXISTING Qdrant collection + EXISTING parent store built by ingest.py.
Does NOT touch CSVLoader, does NOT rebuild threads, does NOT re-embed the corpus --
only embeds the single query string.
"""

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
parent_store_path=base_path/"ResolveAI"/"ingestion"/"parent_store_turns"

class TurnChainRetriever:
    def __init__(
        self,
        embedder_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        collection_name: str = "tweet_turns",
        parent_store_dir: str = parent_store_path,
    ):
        self.embedder = HuggingFaceEmbeddings(model_name=embedder_model)
        client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))

        existing = [c.name for c in client.get_collections().collections]
        if collection_name not in existing:
            raise RuntimeError(f"Collection '{collection_name}' not found. Run ingest.py first.")

        self.vectorstore = QdrantVectorStore(
            client=client, collection_name=collection_name, embedding=self.embedder
        )

        parent_dir = Path(parent_store_dir)
        if not parent_dir.exists():
            raise RuntimeError(f"Parent store '{parent_store_dir}' not found. Run ingest.py first.")
        self.parent_store = create_kv_docstore(LocalFileStore(parent_store_dir))

    def search(self, query: str, top_k_children: int = 10, top_n_parents: int = 5):
        child_results = self.vectorstore.similarity_search_with_score(query, k=top_k_children)

        # Aggregate to parent thread level via max score across matched turns
        parent_best = {}
        for doc, score in child_results:
            thread_id = doc.metadata["thread_id"]
            if thread_id not in parent_best or score > parent_best[thread_id]["score"]:
                parent_best[thread_id] = {"score": score, "matched_turn": doc}

        ranked = sorted(parent_best.items(), key=lambda x: x[1]["score"], reverse=True)[:top_n_parents]

        thread_ids = [tid for tid, _ in ranked]
        parent_docs = self.parent_store.mget(thread_ids)

        results = []
        for (thread_id, info), parent_doc in zip(ranked, parent_docs):
            md = info["matched_turn"].metadata
            results.append({
                "thread_id": thread_id,
                "score": info["score"],
                "matched_turn_text": info["matched_turn"].page_content,
                "matched_turn_id": md["turn_id"],
                "prev_turn_id": md.get("prev_turn_id"),
                "next_turn_ids": md.get("next_turn_ids", []),  # list -- may contain multiple branches
                "parent_thread_text": parent_doc.page_content if parent_doc else None,
            })
        return results


if __name__ == "__main__":
    retriever = TurnChainRetriever()

    query = "hey my facebook account is locked and i am not able to acccess my spotify account because of it"
    results = retriever.search(query, top_k_children=10, top_n_parents=5)

    for r in results:
        print(f"[{r['score']:.3f}] turn_id={r['matched_turn_id']}")
        print(r["matched_turn_text"])
        if len(r["next_turn_ids"]) > 1:
            print(f"  (this turn branches into {len(r['next_turn_ids'])} follow-ups)")
        print(f"\nFull thread:\n{r['parent_thread_text']}\n")
        print("-" * 80)