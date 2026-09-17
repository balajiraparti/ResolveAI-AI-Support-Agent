"""
ingest_chroma.py

Parent-child ingestion over a 5,000-row sample of each CSV, using a LOCAL
persistent Chroma store with TWO collections:

  - "tweet_turns"   -> CHILDREN: one doc per Q->A turn (embedded, searched against)
  - "tweet_threads" -> PARENTS:  one doc per reconstructed conversation thread

Both are persisted to ./chroma_store so retrieval can run in a separate process
without re-ingesting.

Notes on Chroma:
  * Metadata values must be str / int / float / bool. Lists are rejected, so
    list fields (turn_ids, next_turn_ids, root_turns) are JSON-serialized.
  * None is also rejected -- omit the key or store a sentinel instead.
"""

import os
import ast
import json
from pathlib import Path
from collections import defaultdict

from dotenv import load_dotenv
from langchain_community.document_loaders import CSVLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

base_path = Path().resolve().parent
brand_file_path = base_path / "experiments" / "spotify_brand.csv"
customer_file_path = base_path / "experiments" / "spotify_customer.csv"


class TurnChainParentChildIngestor:
    def __init__(
        self,
        customer_csv_path: str,
        brand_csv_path: str,
        embedder_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        child_collection: str = "tweet_turns",
        parent_collection: str = "tweet_threads",
        persist_dir: str = "./chroma_store",
        sample_size: int = 5000,
    ):
        self.customer_csv_path = customer_csv_path
        self.brand_csv_path = brand_csv_path
        self.sample_size = sample_size

        self.embedder = HuggingFaceEmbeddings(model_name=embedder_model)

        self.persist_dir = str(Path(persist_dir).resolve())
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)

        self.child_collection = child_collection
        self.parent_collection = parent_collection

    # -----------------------------------------------------------
    # STEP 1: Load both CSVs (sampled), merge into one tweet lookup
    # -----------------------------------------------------------
    def _load_csv(self, path: str):
        loader = CSVLoader(
            file_path=str(path),
            encoding="utf-8",
            content_columns=["text"],
            metadata_columns=[
                "tweet_id", "author_id", "inbound", "created_at",
                "in_response_to_tweet_id", "response_ids", "missing_response_ids",
            ],
        )
        return loader.load()

    @staticmethod
    def _parse_response_ids(val):
        if not val or str(val) in ("", "[]", "nan", "None"):
            return []
        try:
            parsed = ast.literal_eval(str(val))
            return [int(x) for x in parsed] if isinstance(parsed, list) else [int(parsed)]
        except (ValueError, SyntaxError):
            return [int(x.strip()) for x in str(val).split(",") if x.strip().isdigit()]

    def build_tweet_lookup(self):
        customer_docs = self._load_csv(self.customer_csv_path)
        brand_docs = self._load_csv(self.brand_csv_path)

        # FIX: slice, not index. `docs[5000]` grabs ONE document; `docs[:5000]` takes 5,000.
        customer_sample = customer_docs[: self.sample_size]
        brand_sample = brand_docs[: self.sample_size]
        all_docs = customer_sample + brand_sample

        self.tweets = {}
        skipped = 0
        for doc in all_docs:
            md = doc.metadata
            raw_tid = str(md.get("tweet_id", "")).strip()
            if not raw_tid.isdigit():
                skipped += 1
                continue
            tid = int(raw_tid)

            raw_pred = str(md.get("in_response_to_tweet_id", "")).strip()
            pred = int(float(raw_pred)) if raw_pred not in ("", "nan", "None") else None

            self.tweets[tid] = {
                "tweet_id": tid,
                "text": doc.page_content,
                "author_id": md.get("author_id", ""),
                "inbound": str(md.get("inbound", "")).strip().lower() == "true",
                "created_at": md.get("created_at", ""),
                "in_response_to_tweet_id": pred,
                "response_ids": self._parse_response_ids(md.get("response_ids")),
            }

        print(f"Merged lookup: {len(self.tweets):,} tweets "
              f"({len(customer_sample):,} customer + {len(brand_sample):,} brand sampled"
              + (f", {skipped} skipped" if skipped else "") + ")")

    # -----------------------------------------------------------
    # STEP 2: Build TURNS = (customer_tweet, brand_tweet) pairs
    # -----------------------------------------------------------
    def build_turns(self):
        self.turns = {}
        self.turn_by_brand_tweet = {}

        for tid, t in self.tweets.items():
            if t["inbound"]:
                continue  # only brand tweets form the "answer" half
            pred = t["in_response_to_tweet_id"]
            if pred is None or pred not in self.tweets:
                continue
            customer_tweet = self.tweets[pred]
            if not customer_tweet["inbound"]:
                continue  # guard against brand-to-brand replies

            turn_id = f"{customer_tweet['tweet_id']}_{tid}"
            self.turns[turn_id] = {
                "turn_id": turn_id,
                "customer_tweet": customer_tweet,
                "brand_tweet": t,
                "created_at": customer_tweet["created_at"],
            }
            self.turn_by_brand_tweet[tid] = turn_id

        print(f"Built {len(self.turns):,} Q->A turns from {len(self.tweets):,} tweets")
        if not self.turns:
            print("  WARNING: zero turns. A head-sliced sample can split Q and A across "
                  "the cut. Consider sampling by thread instead (see note at bottom).")

    # -----------------------------------------------------------
    # STEP 3: Chain turns into THREADS (branch-aware)
    # -----------------------------------------------------------
    def chain_turns_into_threads(self):
        parent_ptr = {tid: tid for tid in self.turns}

        def find(x):
            while parent_ptr[x] != x:
                parent_ptr[x] = parent_ptr[parent_ptr[x]]
                x = parent_ptr[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent_ptr[ra] = rb

        self.next_turns = defaultdict(list)   # LIST -> branching-safe
        self.prev_turn = {}

        for turn_id, turn in self.turns.items():
            prior_pred = turn["customer_tweet"]["in_response_to_tweet_id"]
            if prior_pred in self.turn_by_brand_tweet:
                prior_turn_id = self.turn_by_brand_tweet[prior_pred]
                union(turn_id, prior_turn_id)
                self.prev_turn[turn_id] = prior_turn_id
                self.next_turns[prior_turn_id].append(turn_id)

        thread_groups = defaultdict(list)
        for turn_id in self.turns:
            thread_groups[find(turn_id)].append(turn_id)

        self.parent_docs = {}
        self.turn_thread_id = {}

        for root, turn_ids in thread_groups.items():
            root_turns = [tid for tid in turn_ids if tid not in self.prev_turn]
            root_turns.sort(key=lambda tid: self.turns[tid]["created_at"])

            lines = []
            visited = set()

            def render_branch(turn_id, depth=0, branch_label=""):
                if turn_id in visited:
                    return
                visited.add(turn_id)
                turn = self.turns[turn_id]
                indent = "  " * depth
                lines.append(f"{indent}{branch_label}Customer: {turn['customer_tweet']['text']}")
                lines.append(f"{indent}{branch_label}Agent: {turn['brand_tweet']['text']}")
                self.turn_thread_id[turn_id] = root

                children = sorted(self.next_turns.get(turn_id, []),
                                  key=lambda t: self.turns[t]["created_at"])
                multi = len(children) > 1
                for i, child_turn_id in enumerate(children):
                    label = f"[Branch {i + 1}] " if multi else ""
                    render_branch(child_turn_id, depth + 1, branch_label=label)

            for rt in root_turns:
                render_branch(rt)

            thread_id = str(root)
            self.parent_docs[thread_id] = Document(
                page_content="\n".join(lines),
                metadata={
                    "thread_id": thread_id,
                    # Chroma rejects lists -> JSON-serialize
                    "turn_ids": json.dumps(turn_ids),
                    "root_turns": json.dumps(root_turns),
                    "n_turns": len(turn_ids),
                },
            )

        branch_counts = [len(v) for v in self.next_turns.values() if len(v) > 1]
        print(f"Chained {len(self.turns):,} turns into {len(self.parent_docs):,} parent threads "
              f"(branch-aware)")
        print(f"  Turns with multiple successors (branches): {len(branch_counts)}")
        if branch_counts:
            print(f"  Max branches from a single turn: {max(branch_counts)}")

    # -----------------------------------------------------------
    # STEP 4: Persist PARENT threads into their own Chroma collection
    # -----------------------------------------------------------
    def persist_parents(self, batch_size: int = 256):
        parent_list = list(self.parent_docs.values())
        parent_ids = list(self.parent_docs.keys())

        self.parent_store = Chroma(
            collection_name=self.parent_collection,
            embedding_function=self.embedder,
            persist_directory=self.persist_dir,
        )

        existing = self.parent_store._collection.count()
        if existing > 0:
            print(f"Parent collection '{self.parent_collection}' already has "
                  f"{existing:,} docs. Skipping.")
            return

        print(f"Writing {len(parent_list):,} parent threads to Chroma "
              f"'{self.parent_collection}'...")
        for i in range(0, len(parent_list), batch_size):
            self.parent_store.add_documents(
                parent_list[i:i + batch_size],
                ids=parent_ids[i:i + batch_size],   # thread_id as the Chroma doc id
            )
            print(f"  {min(i + batch_size, len(parent_list)):,}/{len(parent_list):,}")
        print("Parent threads persisted.")

    # -----------------------------------------------------------
    # STEP 5: Build CHILD docs (one per turn) into the child collection
    # -----------------------------------------------------------
    def build_and_upsert_children(self, batch_size: int = 256):
        child_docs, child_ids = [], []
        for turn_id, turn in self.turns.items():
            combined_text = (
                f"Customer: {turn['customer_tweet']['text']}\n"
                f"Agent: {turn['brand_tweet']['text']}"
            )
            next_ids = self.next_turns.get(turn_id, [])
            child_docs.append(Document(
                page_content=combined_text,
                metadata={
                    "turn_id": turn_id,
                    "thread_id": str(self.turn_thread_id[turn_id]),
                    "customer_tweet_id": turn["customer_tweet"]["tweet_id"],
                    "brand_tweet_id": turn["brand_tweet"]["tweet_id"],
                    # None and lists are both rejected by Chroma -> sentinel + JSON
                    "prev_turn_id": self.prev_turn.get(turn_id) or "",
                    "next_turn_ids": json.dumps(next_ids),
                    "n_next": len(next_ids),
                },
            ))
            child_ids.append(turn_id)

        self.vectorstore = Chroma(
            collection_name=self.child_collection,
            embedding_function=self.embedder,
            persist_directory=self.persist_dir,
        )

        existing = self.vectorstore._collection.count()
        if existing > 0:
            print(f"Child collection '{self.child_collection}' already has "
                  f"{existing:,} docs. Skipping.")
            return

        print(f"Upserting {len(child_docs):,} turn-children to Chroma "
              f"'{self.child_collection}' in batches of {batch_size}...")
        for i in range(0, len(child_docs), batch_size):
            self.vectorstore.add_documents(
                child_docs[i:i + batch_size],
                ids=child_ids[i:i + batch_size],
            )
            print(f"  {min(i + batch_size, len(child_docs)):,}/{len(child_docs):,}")
        print("Turn ingestion complete.")

    def run(self):
        self.build_tweet_lookup()
        self.build_turns()
        self.chain_turns_into_threads()
        self.persist_parents()
        self.build_and_upsert_children()
        print(f"\nChroma store ready at: {self.persist_dir}")
        print(f"  children: '{self.child_collection}'   parents: '{self.parent_collection}'")


if __name__ == "__main__":
    ingestor = TurnChainParentChildIngestor(
        customer_csv_path=customer_file_path,
        brand_csv_path=brand_file_path,
        sample_size=5000,
    )
    ingestor.run()