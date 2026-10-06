import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_classic.storage import LocalFileStore
from langchain_classic.storage._lc_store import create_kv_docstore
from qdrant_client import QdrantClient
from langchain_community.document_loaders import CSVLoader
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever 
import pandas as pd
from langchain_core.documents import Document

load_dotenv()
base_path=Path().resolve().parent
brand_customer_file=base_path/"ResolveAI"/"experiments"/"spotify_brand_customer_conversations.csv"
client=QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))

def load_doc():
    # loader=CSVLoader(brand_customer_file,source_column='text_question''text_answer')
    # docs=loader.load()
    # return docs
    df = pd.read_csv(brand_customer_file)

    documents = []

    for _, row in df.iterrows():

        text = (
            f"Customer: {row['text_question']}\n"
            f"SpotifyCares: {row['text_answer']}"
        )

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "tweet_id_question": str(row["tweet_id_question"]),
                    "author_id_question": str(row["author_id_question"]),
                }
            )
        )
    return documents

def sparse_retriever():
    docs=load_doc()
    retriever=BM25Retriever.from_documents(docs)
    retriever.k=3
    return retriever

def dense_retriever():
    embedder=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore=QdrantVectorStore(client=client, collection_name="tweet_turns", embedding=embedder)
    return vectorstore
    # return vectorstore.as_retriever(search_kwargs={"k":10})
    
    

def ensemble_retriever():
    sparse=sparse_retriever()
    dense=dense_retriever()
    ensemble_retriever=EnsembleRetriever(retrievers=[sparse,dense], weights=[0.4,0.5])
    return ensemble_retriever

if __name__ == "__main__":
    retriever=sparse_retriever()
    query="i have Lenovo Yoga, Windows 10, Spotify version 1.0.64.399.g4637b02a"
    results=retriever.invoke(query)
    print(results)
    for r in results:
        print(r.page_content)
    client.close()
