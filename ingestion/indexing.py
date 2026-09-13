from dotenv import load_dotenv
from mem0 import Memory
import os
load_dotenv()
key = os.getenv("OPENAI_API_KEY")
import pandas as pd
from pathlib import Path
base_path=Path().resolve().parent
file_path=(base_path/"experiments"/"spotify_brand_customer_conversations.csv")
class IngestData:
    def __init__(self):

        config = {
            "version": "v1.1",

            "embedder": {
                "provider": "huggingface",
                "config": {
            
                    "model": "sentence-transformers/all-MiniLM-L6-v2"
                }
            },

            "llm": {
                "provider": "ollama",
                "config": {
                   "ollama_base_url": "http://localhost:11434",
                    "model": "llama3.2:1b"
                }
            },

            "graph_store": {
                "provider": "neo4j",
                "config": {
                    "url": os.getenv("NEO4J_CONNECTION"),
                    "username": os.getenv("NEO4J_USERNAME"),
                    "password": os.getenv("NEO4J_PASSWORD"),
                    # "database":os.getenv("NEO4J_USERNAME")
                }
            },

            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "url": os.getenv("QDRANT_URL"),
                    "api_key": os.getenv("QDRANT_API_KEY"),
                    "collection_name":"customer_support",
                     "embedding_model_dims": 384 
                }
            }
        }
        self.mem_client = Memory.from_config(config)
    def ingest_conversations(self,row):
       
        messages = [
            {
                "role": "user",
                "content": str(row["text_question"])
            },
            {
                "role": "assistant",
                "content": str(row["text_answer"])
            }
        ]

        metadata = {
            "brand": "SpotifyCares",

            # -------------------------------
            # Conversation identifiers
            # -------------------------------

            "conversation_id": (
                f"{row['tweet_id_question']}_"
                f"{row['tweet_id_answer']}"
            ),

            # -------------------------------
            # Question metadata
            # -------------------------------

            "tweet_id_question": str(
                row["tweet_id_question"]
            ),

            "author_id_question": str(
                row["author_id_question"]
            ),

            "inbound_question": bool(
                row["inbound_question"]
            ),

            "created_at_question": str(
                row["created_at_question"]
            ),

            "in_response_to_tweet_id_question": (
                str(row["in_response_to_tweet_id_question"])
                if row["in_response_to_tweet_id_question"]
                is not None
                else None
            ),

            # -------------------------------
            # Answer metadata
            # -------------------------------

            "tweet_id_answer": str(
                row["tweet_id_answer"]
            ),

            "author_id_answer": str(
                row["author_id_answer"]
            ),

            "inbound_answer": bool(
                row["inbound_answer"]
            ),

            "created_at_answer": str(
                row["created_at_answer"]
            ),

            "in_response_to_tweet_id_answer": (
                str(row["in_response_to_tweet_id_answer"])
                if row["in_response_to_tweet_id_answer"]
                is not None
                else None
            ),

            # -------------------------------
            # Source
            # -------------------------------

            "source": "twitter_support_corpus",
            "memory_type": "historical_support_interaction"
        }

        result = self.mem_client.add(
            messages,
            user_id="spotify_support_knowledge",
            metadata=metadata
        )

        return result
    def ingest_dataframe(self, dataframe):

        results = []

        for index, row in dataframe.iterrows():

            try:

                result = self.ingest_conversations(row)

                results.append({
                    "index": index,
                    "tweet_id_question": row[
                        "tweet_id_question"
                    ],
                    "tweet_id_answer": row[
                        "tweet_id_answer"
                    ],
                    "status": "success"
                })

            except Exception as e:

                results.append({
                    "index": index,
                    "tweet_id_question": row[
                        "tweet_id_question"
                    ],
                    "tweet_id_answer": row[
                        "tweet_id_answer"
                    ],
                    "status": "failed",
                    "error": str(e)
                })

        return results

if __name__=="__main__":
    print(f"Loading dataset from:\n{file_path}")

    # Load CSV
    dataframe = pd.read_csv(file_path)

    print(f"Dataset loaded: {len(dataframe):,} records")

    # Initialize Mem0
    ingest = IngestData()
    for i in range(5):
        row = dataframe.iloc[i]

        print(f"\n--- Ingesting record {i + 1}/5 ---")
        print(f"Question: {row['text_question']}")
        print(f"Answer: {row['text_answer']}")

        result = ingest.ingest_conversations(row)

        print(f"Result: {result}")

        print("\nFirst 5 records ingested successfully.")
    # IMPORTANT:
    # Pass DataFrame, NOT str(file_path)
    # result = ingest.ingest_dataframe(dataframe)

    # print("\nIngestion completed.")

    # result_df = pd.DataFrame(result)

    # print(
    #     result_df["status"].value_counts()
   # )
