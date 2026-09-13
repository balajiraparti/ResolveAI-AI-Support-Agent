import os
import pandas as pd
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

# Neo4j connection
driver = GraphDatabase.driver(
    os.getenv("NEO4J_CONNECTION"),
    auth=(
        os.getenv("NEO4J_USERNAME"),
        os.getenv("NEO4J_PASSWORD")
    )
)

# Load dataset
file_path = r"D:\Projects\ResolveAI\experiments\spotify_brand_customer_conversations.csv"

print(f"Loading dataset from:\n{file_path}")

dataframe = pd.read_csv(file_path)

print(f"Dataset loaded: {len(dataframe):,} records")




# def delete_all(tx):
#     tx.run("""
#         MATCH (n)
#         DETACH DELETE n
#     """)


# with driver.session() as session:
#     session.execute_write(delete_all)

# print("All Neo4j nodes and relationships deleted.")
from neo4j import GraphDatabase
driver = GraphDatabase.driver(
    "neo4j://localhost:7687",
    auth=("neo4j","ResolveAI12345")
)
with driver.session(database="system") as session:
    result = session.run("SHOW DATABASES")
    for record in result:
        print(record)
