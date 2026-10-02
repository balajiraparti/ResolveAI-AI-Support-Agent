from neo4j_graphrag.retrievers import VectorCypherRetriever
import neo4j
from neo4j_graphrag.embeddings import OllamaEmbeddings
from neo4j_graphrag.llm import OllamaLLM
import os
from neo4j_graphrag.generation import GraphRAG, RagTemplate
from dotenv import load_dotenv
load_dotenv()

driver = neo4j.GraphDatabase.driver(
    os.getenv("NEO4J_CONNECTION"),
    auth=(
        os.getenv("NEO4J_USERNAME"),
        os.getenv("NEO4J_PASSWORD")
    )
)
# with driver.session() as session:
#     indexes = session.run("SHOW INDEXES").data()

# for index in indexes:
#     print(index)
embedder = OllamaEmbeddings(model="nomic-embed-text")
# graph_retriever = VectorCypherRetriever(
#     driver,
#     index_name="text_embeddings",
#     embedder=embedder,
#     retrieval_query="""
# //1) Go out 2-3 hops in the entity graph and get relationships
# WITH node AS chunk
# MATCH (chunk)<-[:FROM_CHUNK]-(entity)-[relList:!FROM_CHUNK]-{1,2}(nb)
# UNWIND relList AS rel

# //2) collect relationships and text chunks
# WITH collect(DISTINCT chunk) AS chunks, collect(DISTINCT rel) AS rels

# //3) format and return context
# RETURN apoc.text.join([c in chunks | c.text], '\n') + 
#   apoc.text.join([r in rels | 
#   startNode(r).name+' - '+type(r)+' '+r.details+' -> '+endNode(r).name],  
#   '\n') AS info
# """
# )
graph_retriever = VectorCypherRetriever(
    driver,
    index_name="text_embeddings",
    embedder=embedder,
    retrieval_query="""
    RETURN
        node.text AS info,
        node.index AS chunk_index
    """
)

llm=OllamaLLM(model_name="llama3.2:1b",  model_params={"temperature": 0.0})

rag_template = RagTemplate(template='''Answer the Question using the following Context. Only respond with information mentioned in the Context. Do not inject any speculative information not mentioned. 

# Question:
{query_text}
 
# Context:
{context}

# Answer:
''', expected_inputs=['query_text', 'context'])

# vector_rag  = GraphRAG(llm=llm, retriever=vector_retriever, prompt_template=rag_template)

graph_rag = GraphRAG(llm=llm, retriever=graph_retriever, prompt_template=rag_template)

q = "it says not available in offline mode."

# vector_rag.search(q, retriever_config={'top_k':5}).answer
try:
    result=graph_rag.search(q, retriever_config={'top_k':5}).answer
    print(result)
finally:
    driver.close()