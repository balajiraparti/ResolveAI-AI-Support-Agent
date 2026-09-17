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

load_dotenv()
base_path=Path().resolve().parent
file_path=base_path/"experiments"/"spotify_brand.csv"

class TweetParentChildIngestor:
    def __init__(
        self,
        csv_path: str,
        embedder_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        collection_name: str = "tweet_children",
        parent_store_dir: str = "./parent_store",
    ):
        self.csv_path = csv_path
        self.embedder = HuggingFaceEmbeddings(model_name=embedder_model)
        self.collection_name = collection_name

        # Parent store: persisted key-value store on disk, keyed by thread_id
        Path(parent_store_dir).mkdir(exist_ok=True)
        fs_store = LocalFileStore(parent_store_dir)
        self.parent_store = create_kv_docstore(fs_store)

        self.qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
        )

    # -----------------------------------------------------------
    # STEP 1: Load raw CSV via CSVLoader
    # -----------------------------------------------------------
    def load_documents(self):
        loader = CSVLoader(
            file_path=self.csv_path,
            encoding="utf-8",
            csv_args={"delimiter": ","},
            content_columns=["text"],          # page_content = tweet text only
            metadata_columns=[                 # everything else stays as metadata
                "tweet_id", "author_id", "inbound", "created_at",
                "in_response_to_tweet_id", "response_ids", "missing_response_ids",
            ],
        )
        self.raw_docs = loader.load()
        print(f"Loaded {len(self.raw_docs):,} tweet documents via CSVLoader")

    # -----------------------------------------------------------
    # STEP 2: Build tweet_id -> Document lookup + parse response_ids
    # -----------------------------------------------------------
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
        self.tweets = {}
        for doc in self.raw_docs:
            md = doc.metadata
            tid = int(md["tweet_id"])
            self.tweets[tid] = {
                "tweet_id": tid,
                "text": doc.page_content,
                "author_id": md["author_id"],
                "inbound": str(md["inbound"]).lower() == "true",
                "created_at": md["created_at"],
                "in_response_to_tweet_id": (
                    int(md["in_response_to_tweet_id"])
                    if md.get("in_response_to_tweet_id") not in (None, "", "nan") else None
                ),
                "response_ids": self._parse_response_ids(md.get("response_ids")),
            }
        print(f"Tweet lookup built: {len(self.tweets):,} tweets")



    # -----------------------------------------------------------
    # STEP 4: Group tweets into PARENT threads via union-find over the graph
    # -----------------------------------------------------------
    def build_threads(self):
        parent_ptr = {tid: tid for tid in self.tweets}

        def find(x):
            while parent_ptr[x] != x:
                parent_ptr[x] = parent_ptr[parent_ptr[x]]
                x = parent_ptr[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent_ptr[ra] = rb

        for tid, t in self.tweets.items():
            if t["in_response_to_tweet_id"] in self.tweets:
                union(tid, t["in_response_to_tweet_id"])
            for sid in t["response_ids"]:
                if sid in self.tweets:
                    union(tid, sid)

        self.thread_of = {tid: find(tid) for tid in self.tweets}

        threads = defaultdict(list)
        for tid, root in self.thread_of.items():
            threads[root].append(tid)

        self.parent_docs = {}  # thread_id (str) -> Document
        for root, tids in threads.items():
            ordered = sorted(tids, key=lambda t: self.tweets[t]["created_at"])
            lines = [
                f"{'Customer' if self.tweets[t]['inbound'] else 'Agent'}: {self.tweets[t]['text']}"
                for t in ordered
            ]
            thread_id = str(root)
            self.parent_docs[thread_id] = Document(
                page_content="\n".join(lines),
                metadata={"thread_id": thread_id, "tweet_ids": ordered},
            )

        print(f"Grouped {len(self.tweets):,} tweets into {len(self.parent_docs):,} parent threads")

    # -----------------------------------------------------------
    # STEP 5: Persist parent docs to the docstore (keyed by thread_id)
    # -----------------------------------------------------------
    def persist_parents(self):
        items = list(self.parent_docs.items())
        self.parent_store.mset(items)
        print(f"Persisted {len(items):,} parent documents to disk store")

    # -----------------------------------------------------------
    # STEP 6: Build child documents (one per tweet), each tagged with its thread_id,
    # then embed + push into Qdrant
    # -----------------------------------------------------------
    def build_and_upsert_children(self, batch_size: int = 256):
        child_docs = []
        for tid, t in self.tweets.items():
            thread_id = str(self.thread_of[tid])
            child_docs.append(Document(
                page_content=t["text"],
                metadata={
                    "tweet_id": tid,
                    "thread_id": thread_id,                       # <-- links child to parent
                    "author_id": t["author_id"],
                    "inbound": t["inbound"],
                    "in_response_to_tweet_id": t["in_response_to_tweet_id"],
                    "response_ids": t["response_ids"],
                },
            ))

        # Ensure Qdrant collection exists with correct dimension for this embedder
        sample_vec = self.embedder.embed_query("dimension probe")
        dim = len(sample_vec)

        existing = [c.name for c in self.qdrant_client.get_collections().collections]
        if self.collection_name not in existing:
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
            print(f"Created Qdrant collection '{self.collection_name}' (dim={dim})")

        self.vectorstore = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=self.collection_name,
            embedding=self.embedder,
        )

        print(f"Upserting {len(child_docs):,} child documents in batches of {batch_size}...")
        for i in range(0, len(child_docs), batch_size):
            batch = child_docs[i:i + batch_size]
            self.vectorstore.add_documents(batch)
            print(f"  {i + len(batch):,}/{len(child_docs):,}")

        print("Child ingestion complete.")

    # -----------------------------------------------------------
    # Orchestration
    # -----------------------------------------------------------
    def run(self):
        self.load_documents()
        self.build_tweet_lookup()
        self.build_threads()
        self.persist_parents()
        self.build_and_upsert_children()


if __name__ == "__main__":
    ingestor = TweetParentChildIngestor(csv_path=file_path)
    ingestor.run()