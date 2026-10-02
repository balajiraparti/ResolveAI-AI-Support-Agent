import os
import asyncio
import pandas as pd
import neo4j

from dotenv import load_dotenv
from neo4j_graphrag.llm import OllamaLLM
from neo4j_graphrag.embeddings import OllamaEmbeddings
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.components.text_splitters.fixed_size_splitter import FixedSizeSplitter

load_dotenv()


# --------------------------------------------------
# 1. Neo4j connection
# --------------------------------------------------

driver = neo4j.GraphDatabase.driver(
    os.getenv("NEO4J_CONNECTION"),
    auth=(
        os.getenv("NEO4J_USERNAME"),
        os.getenv("NEO4J_PASSWORD")
    )
)
driver.verify_connectivity()
print("Connected to Neo4j")


# --------------------------------------------------
# 2. Local Ollama LLM
# --------------------------------------------------

ex_llm = OllamaLLM(
    model_name="llama3.2:1b",
    model_params={
        "response_format": {"type": "json_object"},
        "temperature": 0
    }
)


# --------------------------------------------------
# 3. Ollama embeddings
# --------------------------------------------------

embedder = OllamaEmbeddings(model="nomic-embed-text")


# --------------------------------------------------
# 4. Graph schema
# --------------------------------------------------

node_labels = [
    "Customer",
    "Question",
    "SupportResponse",
    "Issue",
    "Product",
    "Organization"
]

rel_types = [
    "ASKED",
    "ANSWERED_BY",
    "HAS_ISSUE",
    "ABOUT",
    "RESPONDED_TO",
    "MENTIONS"
]


# --------------------------------------------------
# 5. Extraction prompt
#    {text} and {schema} placeholders are filled by SimpleKGPipeline.
#    The JSON format block MUST be present — without it the LLM has no
#    idea what structure to return and produces free text, causing the
#    "LLM response is not valid JSON" error seen previously.
# --------------------------------------------------

prompt_template = """
You are a knowledge graph extractor for Spotify customer support conversations.
Extract ONLY entities and relationships that are explicitly stated in the text.
Do not infer or hallucinate anything not present in the text.

Available node labels: {schema}

Text:
{text}

Rules:
- Use ONLY the node labels listed above.
- Use ONLY these relationship types: ASKED, ANSWERED_BY, HAS_ISSUE, ABOUT, RESPONDED_TO, MENTIONS
- Each node must have a unique "id" (short snake_case string, e.g. "spotify_app", "customer_1").
- Customer nodes: id = "customer_<author_id>" if available, else "customer_unknown".
- If the text contains no extractable entities, return empty arrays — do not invent nodes.

Respond with ONLY a valid JSON object. No explanation, no markdown, no extra text.
Use exactly this structure (the double braces are literal — your output must use single braces):

{{
  "nodes": [
    {{"id": "<unique_id>", "label": "<NodeLabel>", "properties": {{"name": "<value>"}}}}
  ],
  "relationships": [
    {{"start_node_id": "<id>", "end_node_id": "<id>", "type": "<REL_TYPE>", "properties": {{}}}}
  ]
}}

JSON:
"""


# --------------------------------------------------
# 6. Build the KG pipeline
# --------------------------------------------------

kg_builder = SimpleKGPipeline(
    llm=ex_llm,
    driver=driver,
    text_splitter=FixedSizeSplitter(chunk_size=500, chunk_overlap=100),
    embedder=embedder,
    entities=node_labels,
    relations=rel_types,
    prompt_template=prompt_template,
    from_file=False,
    on_error="IGNORE"
)


# --------------------------------------------------
# 7. Load spotify_brand_customer_conversations.csv
#
#    The conversations CSV is already a pre-joined Q+A table:
#
#    Columns (customer / question side):
#      tweet_id_question, author_id_question, inbound_question,
#      created_at_question, text_question, response_tweet_id,
#      in_response_to_tweet_id_question, response_ids_question,
#      missing_response_ids_question
#
#    Columns (brand / answer side):
#      tweet_id_answer, author_id_answer, inbound_answer,
#      created_at_answer, text_answer, in_response_to_tweet_id_answer,
#      response_ids_answer, missing_response_ids_answer
#
#    No join needed — text_question + text_answer give the full
#    conversation context in a single row.
# --------------------------------------------------

CSV_PATH = "spotify_brand_customer_conversations.csv"

df = pd.read_csv(CSV_PATH)

print(f"Loaded {len(df)} conversation pairs")
print(f"Columns: {df.columns.tolist()}")

# Validate expected columns are present
required_cols = {"tweet_id_question", "text_question", "text_answer", "author_id_question"}
missing_cols = required_cols - set(df.columns)
if missing_cols:
    raise ValueError(
        f"CSV is missing expected columns: {missing_cols}\n"
        f"Found: {df.columns.tolist()}"
    )


def build_combined_text(row: pd.Series) -> str:
    """
    Combine the customer question and brand answer into one context block.
    The extra context (vs. a single tweet) is critical for a 1B model to
    extract meaningful entities — a bare '@SpotifyCares help' alone has
    almost nothing to extract.
    """
    q = str(row["text_question"]).strip() if pd.notna(row["text_question"]) else ""
    a = str(row["text_answer"]).strip()   if pd.notna(row["text_answer"])   else ""

    if q and a:
        return f"Customer: {q}\nSpotifyCares: {a}"
    elif q:
        return f"Customer: {q}"
    elif a:
        return f"SpotifyCares: {a}"
    return ""


df["combined_text"] = df.apply(build_combined_text, axis=1)

# Drop rows where both sides are empty after cleaning
df = df[df["combined_text"].str.strip().str.len() > 10].reset_index(drop=True)
print(f"Rows with usable text: {len(df)}")

has_answer = df["text_answer"].notna() & (df["text_answer"].str.strip() != "")
print(f"Rows with both Q+A:    {has_answer.sum()}")
print(f"Rows with Q only:      {(~has_answer).sum()}")


# --------------------------------------------------
# 8. Process rows — with retry + exponential backoff
#    for the "LLM response is not valid JSON" transient
#    failures that 1B models produce on complex prompts
# --------------------------------------------------

MAX_RETRIES    = 2
RETRY_DELAY_S  = 1.5


async def process_row(index: int, row: pd.Series) -> dict:
    text = row["combined_text"]

    print(f"\n{'=' * 70}")
    print(f"[{index}] tweet_id_question={row['tweet_id_question']}")
    print(f"Preview: {text[:140]}{'...' if len(text) > 140 else ''}")
    print("=" * 70)

    metadata = {
        "source":              CSV_PATH,
        "row_id":              str(index),
        "tweet_id_question":   str(row["tweet_id_question"]),
        "tweet_id_answer":     str(row.get("tweet_id_answer", "")),
        "author_id_question":  str(row.get("author_id_question", "")),
        "created_at_question": str(row.get("created_at_question", "")),
        "created_at_answer":   str(row.get("created_at_answer", "")),
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = await kg_builder.run_async(
                text=text,
                document_metadata=metadata
            )
            resolver = result.result.get("resolver", {})
            created  = resolver.get("number_of_created_nodes", 0) or 0
            resolved = resolver.get("number_of_nodes_to_resolve", 0) or 0
            print(f"[{index}] ✓ nodes created: {created}, resolved: {resolved}")
            return {"index": index, "status": "ok", "nodes_created": created}

        except Exception as e:
            print(f"[{index}] Attempt {attempt}/{MAX_RETRIES} — {type(e).__name__}: {e}")
            if attempt < MAX_RETRIES:
                await asyncio.sleep(RETRY_DELAY_S * attempt)

    print(f"[{index}] ✗ gave up after {MAX_RETRIES} attempts")
    return {"index": index, "status": "failed", "nodes_created": 0}


async def build_graph(n_rows: int = 5) -> None:
    """
    n_rows=5  → quick validation pass (run this first)
    n_rows=-1 → full dataset
    """
    sample = df if n_rows == -1 else df.head(n_rows)
    print(f"\nProcessing {len(sample)} rows...\n")

    results = []
    for index, row in sample.iterrows():
        r = await process_row(index, row)
        results.append(r)

    # Summary
    ok      = sum(1 for r in results if r["status"] == "ok")
    failed  = sum(1 for r in results if r["status"] == "failed")
    total_n = sum(r["nodes_created"] for r in results)

    print(f"\n{'=' * 70}")
    print(f"Done — {ok} succeeded, {failed} failed, {total_n} total nodes created")
    print("=" * 70)


# --------------------------------------------------
# 9. Run
# --------------------------------------------------

if __name__ == "__main__":
    try:
        # Change n_rows=-1 to process the full dataset once validated
        asyncio.run(build_graph(n_rows=-1))
    finally:
        driver.close()
        print("Neo4j connection closed")