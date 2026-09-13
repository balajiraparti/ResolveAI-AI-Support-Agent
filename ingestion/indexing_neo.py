from dotenv import load_dotenv
from langchain_experimental.graph_transformers import LLMGraphTransformer
import os
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_community.document_loaders import CSVLoader
import pandas as pd
from pathlib import Path
# from langchain.docstore.document import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_neo4j import Neo4jGraph
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
load_dotenv()
from langchain_ollama import ChatOllama
base_path=Path().resolve().parent
file_path=(base_path/"experiments"/"spotify_brand_customer_conversations.csv")
class IngestData:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.text_splitter = SemanticChunker(self.embeddings)
        self.llm=ChatOllama(model="llama3.2:1b")
        self.llm_transformer = LLMGraphTransformer(llm=self.llm)
        self.graph=Neo4jGraph(url=os.getenv("NEO4J_CONNECTION"),username=os.getenv("NEO4J_USERNAME"),password=os.getenv("NEO4J_PASSWORD"),database=os.getenv("NEO4J_USERNAME"))
        self.db=None
        self.documents=[]
    def load_doc(self):
        loader = CSVLoader(
        file_path=file_path,
    encoding="utf-8"
)

        self.documents = loader.load()
        print(f"Loaded {len(self.documents):,} documents")  
    def convert_to_graph(self):
        if self.llm_transformer:
            if self.documents:
                self.graph_documents= self.llm_transformer.convert_to_graph_documents(self.documents)
        print(f'Nodes:{self.graph_documents[0].nodes}')
        print(f'Relationships:{self.graph_documents[0].relationships}')
     
    def ingest_dataframe(self):
        if self.graph_documents:
            self.db = self.graph.add_graph_documents(self.graph_documents
       ,baseEntityLabel=True,include_source=True
    )
        self.graph.close()
      

if __name__=="__main__":

     ingest=IngestData()

     ingest.load_doc()
     ingest.convert_to_graph()
     ingest.ingest_dataframe()