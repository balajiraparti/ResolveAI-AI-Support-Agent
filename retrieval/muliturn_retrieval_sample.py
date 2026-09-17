"""
retrieve_chroma.py

Queries the local Chroma store built by ingest_chroma.py.
Opens both collections read-only -- no CSV loading, no re-embedding of the corpus,
only the query string is embedded.
"""

import json
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

base_path=Path().resolve().parent
parent_store_path=base_path/"ResolveAI"/"ingestion"/"chroma_store"
class TurnChainRetriever:
    def __init__(
        self,
        embedder_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        child_collection: str = "tweet_turns",
        parent_collection: str = "tweet_threads",
        persist_dir: str = parent_store_path,
    ):
        persist_path = Path(persist_dir).resolve()
        if not persist_path.exists():
            raise RuntimeError(f"Chroma store '{persist_path}' not found. Run ingest_chroma.py first.")

        self.embedder = HuggingFaceEmbeddings(model_name=embedder_model)

        self.children = Chroma(
            collection_name=child_collection,
            embedding_function=self.embedder,
            persist_directory=str(persist_path),
        )
        self.parents = Chroma(
            collection_name=parent_collection,
            embedding_function=self.embedder,
            persist_directory=str(persist_path),
        )

        n_child = self.children._collection.count()
        n_parent = self.parents._collection.count()
        if n_child == 0:
            raise RuntimeError(f"Child collection '{child_collection}' is empty. Run ingest_chroma.py first.")
        print(f"Connected: {n_child:,} children / {n_parent:,} parent threads")

    def _get_parent(self, thread_id: str):
        """Fetch a parent thread by its Chroma doc id (= thread_id). No embedding call."""
        got = self.parents.get(ids=[thread_id])
        if not got["ids"]:
            return None
        return {"text": got["documents"][0], "metadata": got["metadatas"][0]}

    def search(self, query: str, top_k_children: int = 10, top_n_parents: int = 5):
        # Chroma returns DISTANCE (lower = closer), not similarity
        hits = self.children.similarity_search_with_score(query, k=top_k_children)

        parent_best = {}
        for doc, distance in hits:
            thread_id = doc.metadata["thread_id"]
            if thread_id not in parent_best or distance < parent_best[thread_id]["distance"]:
                parent_best[thread_id] = {"distance": distance, "matched_turn": doc}

        ranked = sorted(parent_best.items(), key=lambda x: x[1]["distance"])[:top_n_parents]

        results = []
        for thread_id, info in ranked:
            md = info["matched_turn"].metadata
            parent = self._get_parent(thread_id)
            next_ids = json.loads(md.get("next_turn_ids", "[]"))
            results.append({
                "thread_id": thread_id,
                "distance": info["distance"],
                "similarity": 1.0 - info["distance"],   # rough, for readability only
                "matched_turn_id": md["turn_id"],
                "matched_turn_text": info["matched_turn"].page_content,
                "prev_turn_id": md.get("prev_turn_id") or None,
                "next_turn_ids": next_ids,
                "parent_thread_text": parent["text"] if parent else None,
                "n_turns_in_thread": parent["metadata"].get("n_turns") if parent else None,
            })
        return results


if __name__ == "__main__":
    retriever = TurnChainRetriever()

    query = "app keeps saying offline mode not available"
    for r in retriever.search(query, top_k_children=10, top_n_parents=5):
        print(f"\n[dist={r['distance']:.4f}] turn_id={r['matched_turn_id']} "
              f"(thread has {r['n_turns_in_thread']} turns)")
        print(r["matched_turn_text"])
        if len(r["next_turn_ids"]) > 1:
            print(f"  -> branches into {len(r['next_turn_ids'])} follow-ups")
        print(f"\nFull thread:\n{r['parent_thread_text']}")
        print("-" * 80)