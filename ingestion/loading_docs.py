from langchain_neo4j import Neo4jGraph
from langchain_core.documents import Document
import os
from dotenv import load_dotenv

load_dotenv()


class Neo4jDocumentLoader:

    def __init__(self):
        self.graph = Neo4jGraph(
            url=os.getenv("NEO4J_CONNECTION"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD"),
            database=os.getenv("NEO4J_USERNAME")
            
        )

    def load_documents(self):

        query = """
        MATCH (d:Document)
        RETURN
            d.id AS id,
            d.row AS row,
            d.source AS source,
            d.text AS text
        ORDER BY d.row
        """

        results = self.graph.query(query)

        documents = []

        for row in results:

            if not row["text"]:
                continue

            documents.append(
                Document(
                    page_content=row["text"],
                    metadata={
                        "id": row["id"],
                        "row": row["row"],
                        "source": row["source"],
                    }
                )
            )

        print(f"Loaded {len(documents):,} documents from Neo4j")

        return documents
    def close(self):
        self.graph._driver.close()
loader = Neo4jDocumentLoader()

try:

        documents = loader.load_documents()

        for i, doc in enumerate(documents[:5], start=1):

            print(f"\n========== DOCUMENT {i} ==========")

            print("CONTENT:")
            print(doc.page_content)

            print("\nMETADATA:")
            print(doc.metadata)

finally:
        loader.close()