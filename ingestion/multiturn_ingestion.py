import re
import ast
import pickle
from pathlib import Path
from collections import defaultdict

from langchain_community.document_loaders import CSVLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_classic.storage import LocalFileStore
from langchain_classic.storage._lc_store import create_kv_docstore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
import os
from dotenv import load_dotenv
base_path=Path().resolve().parent
brand_file_path=base_path/"experiments"/"spotify_brand.csv"
customer_file_path=base_path/"experiments"/"spotify_customer.csv"

load_dotenv()


class TurnChainParentChildIngestor:
    def __init__(
        self,
        customer_csv_path: str,
        brand_csv_path: str,
        embedder_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        collection_name: str = "tweet_turns",
        parent_store_dir: str = "./parent_store_turns",
    ):
        self.customer_csv_path = customer_csv_path
        self.brand_csv_path = brand_csv_path
        self.embedder = HuggingFaceEmbeddings(model_name=embedder_model)
        self.collection_name = collection_name

        Path(parent_store_dir).mkdir(exist_ok=True)
        fs_store = LocalFileStore(parent_store_dir)
        self.parent_store = create_kv_docstore(fs_store)

        self.qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
        )

    # -----------------------------------------------------------
    # STEP 1: Load both CSVs, tag inbound explicitly, merge into one lookup
    # -----------------------------------------------------------
    def _load_csv(self, path: str):
        loader = CSVLoader(
            file_path=path,
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
        if not val or val in ("", "[]", "nan"):
            return []
        try:
            parsed = ast.literal_eval(str(val))
            return [int(x) for x in parsed] if isinstance(parsed, list) else [int(parsed)]
        except (ValueError, SyntaxError):
            return [int(x.strip()) for x in str(val).split(",") if x.strip().isdigit()]

    def build_tweet_lookup(self):
        customer_docs = self._load_csv(self.customer_csv_path)
        brand_docs = self._load_csv(self.brand_csv_path)
        all_docs = customer_docs + brand_docs

        self.tweets = {}
        for doc in all_docs:
            md = doc.metadata
            tid = int(md["tweet_id"])
            self.tweets[tid] = {
                "tweet_id": tid,
                "text": doc.page_content,
                "author_id": md["author_id"],
                "inbound": str(md["inbound"]).strip().lower() == "true",
                "created_at": md["created_at"],
                "in_response_to_tweet_id": (
                    int(md["in_response_to_tweet_id"])
                    if md.get("in_response_to_tweet_id") not in (None, "", "nan") else None
                ),
                "response_ids": self._parse_response_ids(md.get("response_ids")),
            }
        print(f"Merged lookup: {len(self.tweets):,} tweets "
              f"({len(customer_docs):,} customer + {len(brand_docs):,} brand)")

    # -----------------------------------------------------------
    # STEP 2: Build TURNS = (customer_tweet, brand_tweet) pairs
    # A turn exists when a brand tweet's in_response_to_tweet_id
    # matches a customer tweet's tweet_id.
    # -----------------------------------------------------------
    def build_turns(self):
        self.turns = {}       # turn_id -> turn dict
        self.turn_by_brand_tweet = {}   # brand_tweet_id -> turn_id (for chaining lookups)

        for tid, t in self.tweets.items():
            if t["inbound"]:
                continue  # only iterate brand tweets to form turns
            pred = t["in_response_to_tweet_id"]
            if pred is None or pred not in self.tweets:
                continue
            customer_tweet = self.tweets[pred]
            if not customer_tweet["inbound"]:
                continue  # guard: predecessor should be a customer tweet, not brand-to-brand

            turn_id = f"{customer_tweet['tweet_id']}_{tid}"
            self.turns[turn_id] = {
                "turn_id": turn_id,
                "customer_tweet": customer_tweet,
                "brand_tweet": t,
                "created_at": customer_tweet["created_at"],  # order threads by when the question was asked
            }
            self.turn_by_brand_tweet[tid] = turn_id

        print(f"Built {len(self.turns):,} Q->A turns from {len(self.tweets):,} tweets")

    # -----------------------------------------------------------
    # STEP 3: Chain turns into THREADS.
    # Turn B follows Turn A if B's customer tweet replies to A's brand tweet
    # (i.e. the conversation continues: user Q -> brand A -> user Q -> brand A ...)
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

        # CHANGED: next_turn now holds a LIST of successor turn_ids (branching-safe)
        self.next_turns = defaultdict(list)
        self.prev_turn = {}   # a turn still has exactly ONE predecessor turn, so this stays 1:1

        for turn_id, turn in self.turns.items():
            customer_tweet = turn["customer_tweet"]
            prior_pred = customer_tweet["in_response_to_tweet_id"]
            if prior_pred in self.turn_by_brand_tweet:
                prior_turn_id = self.turn_by_brand_tweet[prior_pred]
                union(turn_id, prior_turn_id)
                self.prev_turn[turn_id] = prior_turn_id
                self.next_turns[prior_turn_id].append(turn_id)   # append, don't overwrite

        thread_groups = defaultdict(list)
        for turn_id in self.turns:
            thread_groups[find(turn_id)].append(turn_id)

        # -----------------------------------------------------------
        # Build parent text via TREE TRAVERSAL, not flat chronological sort,
        # so branches are preserved and clearly labeled instead of silently merged.
        # -----------------------------------------------------------
        self.parent_docs = {}
        self.turn_thread_id = {}

        for root, turn_ids in thread_groups.items():
            turn_id_set = set(turn_ids)
            # find root turns of this thread: turns with no prev_turn (start of a branch)
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
                prefix = f"{branch_label}" if branch_label else ""
                lines.append(f"{indent}{prefix}Customer: {turn['customer_tweet']['text']}")
                lines.append(f"{indent}{prefix}Agent: {turn['brand_tweet']['text']}")
                self.turn_thread_id[turn_id] = root

                children = sorted(self.next_turns.get(turn_id, []), key=lambda t: self.turns[t]["created_at"])
                for i, child_turn_id in enumerate(children):
                    if len(children) > 1:
                        render_branch(child_turn_id, depth + 1, branch_label=f"[Branch {i+1}] ")
                    else:
                        render_branch(child_turn_id, depth + 1)

            for rt in root_turns:
                render_branch(rt)

            thread_id = str(root)
            self.parent_docs[thread_id] = Document(
                page_content="\n".join(lines),
                metadata={"thread_id": thread_id, "turn_ids": turn_ids, "root_turns": root_turns},
            )

        print(f"Chained {len(self.turns):,} turns into {len(self.parent_docs):,} parent threads "
            f"(branch-aware)")
    # -----------------------------------------------------------
    # STEP 4: Persist parent thread documents
    # -----------------------------------------------------------
    def persist_parents(self):
        items = list(self.parent_docs.items())
        self.parent_store.mset(items)
        print(f"Persisted {len(items):,} parent threads to disk store")

    # -----------------------------------------------------------
    # STEP 5: Build CHILD documents = one per turn (Q+A combined),
    # embed, and upsert into Qdrant with chain metadata
    # -----------------------------------------------------------
    def build_and_upsert_children(self, batch_size: int = 256):
        child_docs = []
        for turn_id, turn in self.turns.items():
            combined_text = (
                f"Customer: {turn['customer_tweet']['text']}\n"
                f"Agent: {turn['brand_tweet']['text']}"
            )
            child_docs.append(Document(
                page_content=combined_text,
                metadata={
                    "turn_id": turn_id,
                    "thread_id": self.turn_thread_id[turn_id],
                    "customer_tweet_id": turn["customer_tweet"]["tweet_id"],
                    "brand_tweet_id": turn["brand_tweet"]["tweet_id"],
                    "prev_turn_id": self.prev_turn.get(turn_id),
                    "next_turn_id": self.next_turns.get(turn_id,[]),
                },
            ))

        sample_vec = self.embedder.embed_query("dimension probe")
        dim = len(sample_vec)

        existing = [c.name for c in self.qdrant_client.get_collections().collections]
        if self.collection_name not in existing:
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
            print(f"Created Qdrant collection '{self.collection_name}' (dim={dim})")
        else:
            count = self.qdrant_client.count(self.collection_name).count
            if count > 0:
                print(f"Collection already has {count:,} points. Skipping upsert.")
                self.vectorstore = QdrantVectorStore(
                    client=self.qdrant_client, collection_name=self.collection_name, embedding=self.embedder,
                )
                return

        self.vectorstore = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=self.collection_name,
            embedding=self.embedder,
        )

        print(f"Upserting {len(child_docs):,} turn-children in batches of {batch_size}...")
        for i in range(0, len(child_docs), batch_size):
            batch = child_docs[i:i + batch_size]
            self.vectorstore.add_documents(batch)
            print(f"  {i + len(batch):,}/{len(child_docs):,}")

        print("Turn ingestion complete.")

    def run(self):
        self.build_tweet_lookup()
        self.build_turns()
        self.chain_turns_into_threads()
        self.persist_parents()
        self.build_and_upsert_children()


if __name__ == "__main__":
    ingestor = TurnChainParentChildIngestor(
        customer_csv_path=customer_file_path,
        brand_csv_path=brand_file_path,
    )
    ingestor.run()