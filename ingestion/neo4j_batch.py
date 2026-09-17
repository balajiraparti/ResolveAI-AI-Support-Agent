import asyncio
import time
from dotenv import load_dotenv
from langchain_experimental.graph_transformers import LLMGraphTransformer
import os
from langchain_community.document_loaders import CSVLoader
from pathlib import Path
from langchain_neo4j import Neo4jGraph
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

load_dotenv()

base_path = Path().resolve().parent
file_path = (base_path / "experiments" / "spotify_brand_customer_conversations.csv")


class IngestData:
    def __init__(self, batch_size: int = 10, concurrency: int = 4):
        """
        batch_size:   number of documents processed per batch (for status reporting
                      and per-batch Neo4j writes).
        concurrency:  number of LLM extraction calls fired concurrently WITHIN a batch.
                      Keep this aligned with your Ollama server's OLLAMA_NUM_PARALLEL
                      setting, otherwise requests just queue up locally with no real
                      speedup.
        """
        self.batch_size = batch_size
        self.concurrency = concurrency

        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.llm = ChatOllama(model="llama3.2:1b")
        self.llm_transformer = LLMGraphTransformer(llm=self.llm)

        self.graph = Neo4jGraph(
            url=os.getenv("NEO4J_CONNECTION"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD"),
            database=os.getenv("NEO4J_USERNAME"),  # fixed: was using username as db name
        )

        self.documents = []
        self.graph_documents = []

    # -----------------------------------------------------------
    # STEP 1: Load documents
    # -----------------------------------------------------------
    def load_doc(self):
        loader = CSVLoader(file_path=file_path, encoding="utf-8")
        self.documents = loader.load()
        print(f"Loaded {len(self.documents):,} documents")

    # -----------------------------------------------------------
    # STEP 2: Batched, concurrent graph extraction with per-batch status
    # -----------------------------------------------------------
    async def _process_one_batch(self, batch, batch_num, total_batches):
        start = time.time()
        try:
            # aconvert_to_graph_documents already processes the list concurrently
            # internally, but we cap batch_size so status prints stay frequent
            # and a single failed batch doesn't cost you the whole run.
            results = await self.llm_transformer.aconvert_to_graph_documents(batch)
            elapsed = time.time() - start
            node_count = sum(len(gd.nodes) for gd in results)
            rel_count = sum(len(gd.relationships) for gd in results)
            print(
                f"[Batch {batch_num}/{total_batches}] OK  "
                f"docs={len(batch)}  nodes={node_count}  rels={rel_count}  "
                f"time={elapsed:.1f}s"
            )
            return results
        except Exception as e:
            elapsed = time.time() - start
            print(
                f"[Batch {batch_num}/{total_batches}] FAILED  "
                f"docs={len(batch)}  time={elapsed:.1f}s  error={e}"
            )
            return []  # skip this batch, continue with the rest

    async def _convert_to_graph_async(self):
        total_batches = (len(self.documents) + self.batch_size - 1) // self.batch_size
        semaphore = asyncio.Semaphore(self.concurrency)

        async def bounded_batch(batch, batch_num):
            async with semaphore:
                return await self._process_one_batch(batch, batch_num, total_batches)

        batches = [
            self.documents[i:i + self.batch_size]
            for i in range(0, len(self.documents), self.batch_size)
        ]

        tasks = [bounded_batch(batch, i + 1) for i, batch in enumerate(batches)]

        overall_start = time.time()
        results = await asyncio.gather(*tasks)
        overall_elapsed = time.time() - overall_start

        self.graph_documents = [gd for batch_result in results for gd in batch_result]

        print(
            f"\nExtraction complete: {len(self.graph_documents):,} graph documents "
            f"from {len(self.documents):,} source docs in {overall_elapsed:.1f}s "
            f"({overall_elapsed / max(len(batches), 1):.1f}s/batch avg)"
        )

    def convert_to_graph(self):
        if not self.documents:
            print("No documents loaded -- call load_doc() first.")
            return
        asyncio.run(self._convert_to_graph_async())

        if self.graph_documents:
            print(f"\nSample -- Nodes: {self.graph_documents[0].nodes}")
            print(f"Sample -- Relationships: {self.graph_documents[0].relationships}")

    # -----------------------------------------------------------
    # STEP 3: Batched Neo4j writes with per-batch status
    # -----------------------------------------------------------
    def ingest_dataframe(self):
        if not self.graph_documents:
            print("No graph documents to ingest -- call convert_to_graph() first.")
            return

        total = len(self.graph_documents)
        total_batches = (total + self.batch_size - 1) // self.batch_size
        overall_start = time.time()

        for i in range(0, total, self.batch_size):
            batch_num = i // self.batch_size + 1
            batch = self.graph_documents[i:i + self.batch_size]
            start = time.time()
            try:
                self.graph.add_graph_documents(
                    batch,
                    baseEntityLabel=True,
                    include_source=True,
                )
                elapsed = time.time() - start
                print(
                    f"[Neo4j write {batch_num}/{total_batches}] OK  "
                    f"docs={len(batch)}  time={elapsed:.1f}s  "
                    f"progress={min(i + self.batch_size, total)}/{total}"
                )
            except Exception as e:
                elapsed = time.time() - start
                print(
                    f"[Neo4j write {batch_num}/{total_batches}] FAILED  "
                    f"docs={len(batch)}  time={elapsed:.1f}s  error={e}"
                )

        overall_elapsed = time.time() - overall_start
        print(f"\nNeo4j ingestion complete: {total:,} graph documents written "
              f"in {overall_elapsed:.1f}s")

        self.graph.close()


if __name__ == "__main__":
    # Tune batch_size / concurrency based on your Ollama throughput and
    # OLLAMA_NUM_PARALLEL setting -- start conservative, increase if stable.
    ingest = IngestData(batch_size=10, concurrency=2)

    ingest.load_doc()
    ingest.convert_to_graph()
    ingest.ingest_dataframe()