# ResolveAI

**AI-powered intelligent customer support agent** using multi-turn RAG, historical conversation retrieval, evidence verification, human-in-the-loop escalation, and production-grade RAG evaluation.

ResolveAI is designed to automate customer support resolution for platforms like Spotify by leveraging past conversation history, intent classification, and AI-powered response generation with human oversight.

## 🎯 Key Features

- **Parent-Child RAG Architecture**: Efficient hierarchical retrieval for multi-turn conversations
- **Historical Evidence Retrieval**: Finds relevant past conversations to inform current responses
- **Intent Classification**: Automatically classifies customer queries into predefined intents
- **Evidence Verification**: Uses RAGAS metrics (Faithfulness, Answer Relevancy) to validate responses
- **Human-in-the-Loop (HITL)**: Escalates uncertain cases to human agents for review
- **Multi-Turn Conversation Support**: Maintains context across multiple customer-agent exchanges
- **Production-Ready Evaluation**: RAGAS-based RAG evaluation pipeline with multiple metrics
- **LangGraph Workflow Orchestration**: Complex agentic workflows with state management
- **FastAPI Backend**: RESTful API for integration with external systems
- **Streamlit Frontend**: Interactive UI for testing and demonstration

## 📁 Project Structure

```
ResolveAI/
├── app/
│   ├── main.py                           # FastAPI server with /ask endpoint
│   ├── support_agent_workflow.py          # LangGraph agent workflow (retrieval → evidence check → escalation)
│   └── __init__.py
│
├── frontend/
│   └── ui.py                             # Streamlit interactive UI
│
├── ingestion/
│   ├── indexing.py                       # Basic vector indexing
│   ├── indexing_neo.py                   # Neo4j graph-based indexing
│   ├── loading_docs.py                   # Document loading utilities
│   ├── multiturn_ingestion.py            # Full dataset ingestion (all conversations)
│   ├── multiturn_sample_indexing.py      # Sample dataset indexing (5,000 rows)
│   ├── neo4j_batch.py                    # Batch Neo4j operations
│   ├── parent_child_indexing.py          # Parent-child chunk hierarchy creation
│   ├── qdrant.py                         # Qdrant vector database integration
│   ├── chroma_store/                     # ChromaDB storage
│   └── parent_store/                     # Parent document storage
│
├── retrieval/
│   ├── extract_data.py                   # Data extraction utilities
│   ├── multiturn_retrieval.py            # Full dataset multi-turn retrieval
│   └── muliturn_retrieval_sample.py      # Sample dataset multi-turn retrieval (TurnChainRetriever)
│
├── intent/
│   ├── extract_intent.py                 # Intent extraction from conversations
│   ├── intent_classification.py          # Intent classification model
│   ├── intent_discovery_llm.py           # LLM-based intent discovery
│   ├── save_intent_info.py               # Intent metadata persistence
│   └── output/                           # Intent analysis outputs
│
├── evaluation/
│   ├── ragas_evaluation.py               # Production-grade RAG evaluation (Faithfulness, ContextRecall, etc.)
│   ├── custom_evaluation.py              # Custom evaluation metrics
│   └── __init__.py
│
├── neo4j/
│   ├── docker-compose.yml                # Neo4j Docker setup
│   └── data/                             # Neo4j data directory
│
├── banking77/
│   ├── banking77.py                      # Banking77 dataset utilities
│   ├── dataset_infos.json                # Dataset metadata
│   └── README.md
│
├── experiments/
│   ├── experiment.ipynb                  # Jupyter notebook for experiments
│   ├── spotify_*.csv                     # Spotify customer data (tweets, conversations)
│   └── tweets_with_intent.csv
│
├── twcs/
│   └── twcs.csv                          # Twitter Customer Service dataset
│
├── requirements.txt                      # Python dependencies
├── golden_dataset.csv                    # Ground truth dataset for evaluation
└── golden_dataset_ragas_results.csv      # RAGAS evaluation results
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- OpenAI API key (or alternative LLM)
- Mistral AI API key (optional)
- PostgreSQL/ChromaDB/Qdrant (for vector storage)
- Neo4j (optional, for graph-based retrieval)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd ResolveAI
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and database credentials
   ```
   Required variables:
   - `OPENAI_API_KEY` - OpenAI API key
   - `MISTRAL_API_KEY` - Mistral AI API key (optional)
   - `CHROMA_DB_PATH` - Path to ChromaDB storage
   - `NEO4J_URL` - Neo4j connection URL (optional)

### Data Ingestion Pipeline

The ingestion pipeline implements a **parent-child chunking strategy** optimized for multi-turn conversation retrieval:

#### Multi-Turn Context Chunking Strategy

Conversations are structured as sequential exchanges:
```
Customer Query 1
  └─→ Brand Response 1
       └─→ Customer Query 2
            └─→ Brand Response 2
                 └─→ Customer Query 3
                      └─→ Brand Response 3
```

**Chunking Process**:
1. **Parent Documents**: Full multi-turn conversation threads
2. **Child Documents**: Individual Q&A pairs with turn context
3. **Branch Creation**: Response IDs link related queries to historical solutions
4. **Context Window**: Each child chunk includes 1-2 previous exchanges for context

#### Data Cleaning (experiments/)

Raw data undergoes cleaning in Jupyter notebooks:
- Duplicate conversation removal
- Timestamp validation and ordering
- Text normalization (lowercase, whitespace cleanup)
- Invalid/incomplete exchanges filtering
- Multi-turn integrity checks

**Cleaned outputs**:
- `spotify_customer.csv` - Customer messages
- `spotify_brand.csv` - Brand responses  
- `spotify_tweet_cluster.csv` - Clustered conversations

#### Ingestion Execution

Choose one ingestion method based on your dataset size:

```bash
# Sample dataset (5,000 rows) - faster for testing
python ingestion/multiturn_sample_indexing.py

# Full dataset - all conversations  
python ingestion/multiturn_ingestion.py

# Graph-based indexing (Neo4j) - abandoned due to hardware constraints
python ingestion/indexing_neo.py
```

**Key Files**:
- `parent_child_indexing.py` - Implements parent-child chunking strategy
- `multiturn_ingestion.py` - Orchestrates full pipeline
- `chroma_store/` - Vector embeddings for child documents
- `parent_store/` - Raw parent documents for context retrieval

### Running the Application

**Option 1: API Server Only**
```bash
uvicorn app.main:app --reload --port 8000
```

**Option 2: With Streamlit UI**
```bash
# Terminal 1: Start API server
uvicorn app.main:app --reload --port 8000

# Terminal 2: Start Streamlit frontend
streamlit run frontend/ui.py
```

Access the UI at `http://localhost:8501`

### API Usage

**POST /ask** - Get support response for customer query

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I reset my password?"}'
```

**Response**:
```json
{
  "query": "How do I reset my password?",
  "response": "To reset your password, you can...",
  "human_required": false,
  "intent": "Account Management",
  "intent_description": "Password and account security issues",
  "historical_evidence": "Found 5 similar past conversations...",
  "is_approved": true,
  "hitl_reason": null
}
```

## 🔄 Workflow Architecture

The agent follows this LangGraph workflow:

```
START
  ↓
1. INTENT CLASSIFICATION
   └─→ Classify customer query intent
  ↓
2. MULTI-TURN RETRIEVAL
   └─→ Retrieve relevant historical conversations
  ↓
3. RESPONSE GENERATION
   └─→ Generate draft response using GPT-4o
  ↓
4. EVIDENCE VERIFICATION
   └─→ Validate response (Faithfulness, Relevancy)
  ↓
5. HUMAN-IN-THE-LOOP CHECK
   ├─→ If confident: Auto-approve and return response
   └─→ If uncertain: Escalate to human agent
  ↓
END
```

## 🧠 Intent Discovery & Classification

### Automated Intent Discovery Pipeline

The system performs unsupervised intent discovery followed by LLM-based intent naming:

#### 1. **Clustering Stage** (HDBSCAN)
```python
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer

# Embed customer messages
embeddings = SentenceTransformer("BAAI/bge-small-en-v1.5").encode(messages)

# Cluster with HDBSCAN (density-based, no K specification needed)
clustered = HDBSCAN(min_cluster_size=10).fit_predict(embeddings)
```

**Why HDBSCAN?**
- Density-based clustering (unlike K-means)
- Automatically determines optimal number of clusters
- Identifies noise/outlier messages
- No manual K selection required
- Hierarchical clustering structure

#### 2. **Intent Naming Stage** (LLM)

For each cluster, an LLM generates representative intent names:

```python
from langchain_openai import ChatOpenAI

# Get cluster samples
cluster_messages = [msg for msg, label in zip(messages, labels) if label == cluster_id]

# LLM generates intent name
intent_prompt = f"""
Given these customer support messages:
{cluster_messages[:5]}

Provide a concise intent name (2-4 words) that describes what these customers need.
Respond with ONLY the intent name.
"""

intent_name = llm.invoke(intent_prompt).content
```

#### 3. **Classification Stage** (TurnChainRetriever)

New queries are classified by finding closest embedding to intent cluster centroids.

### Intent Information Storage

Discovered intents are persisted in:
- `spotify_intent_info.csv` - Full intent inventory with descriptions
- `spotify_intent_info_50.csv` - Top 50 intents by frequency
- `output/` - Intent cluster details and sample conversations

### Supported Intent Sets

- **Banking77**: 77 predefined banking intents (for banking domain)
- **Custom Spotify Intents**: Auto-discovered from conversation clusters
- **Hierarchical Intent Models**: Parent-child intent relationships

### Files Involved

- `extract_intent.py` - Message extraction & preparation
- `intent_discovery_llm.py` - LLM-based intent naming
- `intent_classification.py` - Query-to-intent mapping
- `save_intent_info.py` - Intent persistence

## 📊 Evaluation Framework

Production-grade RAG evaluation using **RAGAS** metrics:

```bash
python evaluation/ragas_evaluation.py
```

**Metrics Calculated**:
- **Faithfulness**: Ensures answer is grounded in context
- **Answer Relevancy**: Measures response relevance to query
- **Context Precision**: Evaluates retrieval precision
- **Context Recall**: Measures retrieval comprehensiveness
- **Answer Accuracy**: Compares to ground truth
- **ContextEntityRecall**: Entity-level recall evaluation

**Output**: `golden_dataset_ragas_results.csv` with per-sample scores

## 📦 Key Dependencies

| Package | Purpose |
|---------|---------|
| `fastapi` | REST API framework |
| `streamlit` | Interactive UI |
| `langchain` | LLM framework & components |
| `langgraph` | Agentic workflow orchestration |
| `chromadb` | Vector database for child chunks |
| `qdrant-client` | Vector search engine |
| `langchain-neo4j` | Graph database integration |
| `langchain-openai` | OpenAI API integration |
| `langchain-mistralai` | Mistral AI integration |
| `ragas` | RAG evaluation framework |
| `deepeval` | AI evaluation platform |
| `sentence-transformers` | Embedding models (BAAI/bge-small-en-v1.5) |
| `hdbscan` | Density-based clustering for intent discovery |
| `umap` | Dimensionality reduction for clustering |
| `pandas` | Data manipulation & cleaning |
| `datasets` | Hugging Face datasets |

## 🔐 Environment Configuration

Create a `.env` file:

```
OPENAI_API_KEY=sk-...
MISTRAL_API_KEY=...
CHROMA_DB_PATH=./chroma_store
NEO4J_URL=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
```

## 🐳 Docker Setup (Neo4j)

For graph-based retrieval:

```bash
cd neo4j
docker-compose up -d

# Access Neo4j Browser at http://localhost:7474
```

## � Data Cleaning & Preparation (experiments/)

The `experiments/` directory contains Jupyter notebooks for data cleaning and preprocessing:

### Data Cleaning Process

**Raw Data Source**:
- Twitter Customer Service (TWCS) Dataset with ~100K records
- Original structure: `tweet_id`, `author_id`, `inbound`, `created_at`, `text`, `response_tweet_id`, `in_response_to_tweet_id`

### Data Quality Issues Identified & Fixed

| Issue | Detection | Solution |
|-------|-----------|----------|
| **Extra whitespace** | `author_id.str.strip()` | Strip leading/trailing spaces |
| **Date parsing** | Manual timezone parsing | `pd.to_datetime(..., format="%a %b %d %H:%M:%S %z %Y", errors="coerce")` |
| **Invalid dates** | Check `isna()` after parsing | Remove records with unparseable timestamps |
| **Multiple response IDs** | Detect commas in `response_tweet_id` | Parse comma-separated values → list |
| **Missing parent tweets** | Check if `in_response_to_tweet_id` in `tweet_id` set | Track missing refs (~3.86%) |
| **Missing response references** | Response IDs point to non-existent tweets | Reconstruct via `response_map` dictionary |
| **Orphaned tweets** | Spotify responses without customer parent | Use `response_ids` to find parent customers |

### Three Cleaned Datasets

**Dataset 1: spotify_brand.csv** - Spotify brand responses only
```python
spotify_tweets = df[df["author_id"] == "SpotifyCares"].copy()
# Reconstruct missing response_ids using response_map
# Output: ~5K-10K brand response records with reconstructed links
```

**Dataset 2: spotify_customer.csv** - Customer questions only
```python
# Find customer tweets that were responded to by Spotify
parent_ids = spotify_tweets['in_response_to_tweet_id'].dropna().astype('int64')
customer_questions = df[(df['tweet_id'].isin(parent_ids)) & (df['inbound'] == True)]
# Output: ~5K-10K customer question records
```

**Dataset 3: spotify_brand_customer_conversations.csv** - Q&A pairs
```python
# Merge customer questions with Spotify responses
qa_pairs = customer_questions.merge(
    spotify_tweets,
    left_on='tweet_id',
    right_on='in_response_to_tweet_id',
    suffixes=('_question', '_answer'),
    how='inner'
)
# Output: ~3K-8K complete Q&A pairs for parent-child ingestion
```

### Preprocessing Pipeline

1. **Load & Type Conversion**: Parse timestamps (timezone-aware), convert tweet_ids to int64, strip whitespace
2. **Reference Reconstruction**: Build response_map (parent → [children]), fill missing response_ids
3. **Quality Validation**: Check for nulls, verify parent-child links exist, sort by timestamp
4. **Dataset Split**: Extract Spotify tweets, customer questions, merge into Q&A pairs
5. **Intent Clustering**: Cluster customer messages (HDBSCAN), generate intent labels (LLM), tag tweets

### Preprocessing Statistics

```
Raw TWCS Dataset:                   ~100,000 records
├─ SpotifyCares brand:             ~10,000 records
├─ Referenced customers:            ~8,000 records
└─ Q&A pairs formed:                ~6,500 records

Data Loss:
├─ Missing parent refs:            ~3.86% (3,862 records)
├─ Unparseable dates:              ~0.5% (500 records)
└─ Other quality issues:           ~2%

Final Output:
├─ spotify_brand.csv:              ~10,000 records
├─ spotify_customer.csv:           ~8,000 records
└─ spotify_brand_customer_conversations.csv: ~6,500 Q&A pairs
```

### Key Utilities (experiment.ipynb)

- `parse_response_ids()` - Parse comma-separated response_ids string → list
- `find_parent_from_response_ids()` - Find parent tweet using response_ids
- `find_referencing_tweets()` - Reverse lookup: which tweets reference a given ID
- Response mapping merge - Link Spotify responses to customer questions

## �🧪 Testing & Evaluation

Run experiments and evaluation:

```bash
# Jupyter experiments
jupyter notebook experiments/experiment.ipynb

# Custom evaluation
python evaluation/custom_evaluation.py

# RAGAS evaluation pipeline
python evaluation/ragas_evaluation.py
```

## 📈 Performance Considerations

- **Retrieval**: Multi-turn context chunking with parent-child hierarchy
- **Vectorization**: Sentence-transformers with BAAI/bge-small-en-v1.5
- **Scaling**: Supports ChromaDB, Qdrant, and Neo4j backends
- **Latency**: Target <2s response time with sample dataset
- **Accuracy**: Evaluated with RAGAS metrics (target >0.7 for key metrics)

## 🤝 Human-in-the-Loop Features

- **Escalation Triggers**: Low confidence, complex queries, policy-sensitive issues
- **Human Review**: Draft response + context + evidence presented for approval
- **Feedback Loop**: Human corrections used to improve future responses
- **Audit Trail**: All decisions logged with reasoning

## 🔄 Multi-Turn Conversation Support

ResolveAI uses the **TurnChainRetriever** to intelligently retrieve conversation context:

### Retrieval Strategy

```
Query Embedding
      ↓
Vector Similarity Search (ChromaDB)
      ↓
Retrieve Child Documents (Q&A pairs)
      ↓
Fetch Parent Document (Full conversation thread)
      ↓
Branch Following (Linked response IDs)
      ↓
Context Assembly (3-5 previous turns)
      ↓
Return Historical Evidence
```

### How It Works

1. **Child Search**: Query embedded and compared against child chunk vectors
2. **Parent Retrieval**: Associated parent conversation thread loaded
3. **Context Expansion**: Previous Q&A exchanges added for turn context
4. **Branch Linking**: Related queries/responses followed via response IDs
5. **Deduplication**: Removes redundant turns from retrieved context

### Files & Components

- `muliturn_retrieval_sample.py` - TurnChainRetriever implementation
- `multiturn_retrieval.py` - Full dataset variant
- `parent_store/` - Parent conversation storage
- `chroma_store/` - Child chunk embeddings

### Configuration

- **Context Window**: Maintains 3-5 previous customer-agent exchanges
- **Turn Chaining**: Follows conversation branches via response IDs
- **Historical Matching**: Semantic similarity matching across turns
- **State Persistence**: Conversation state managed via LangGraph memory

## 📝 Datasets Included

- **Spotify Cares**: Real customer tweets and responses (experiments/)
- **Banking77**: 77 banking intent classes (banking77/)
- **TWCS**: Twitter Customer Service dataset (twcs/)
- **Golden Dataset**: Annotated ground truth (golden_dataset.csv)

## 🚦 Troubleshooting

### API Won't Start
- Check `.env` file and API key configuration
- Verify port 8000 is not in use
- Ensure vector database is running

### Slow Retrieval
- Check vector database indices
- Consider using sample dataset for testing
- Review logs for bottlenecks

### Low Evaluation Scores
- Verify ingestion completed successfully
- Check retrieved context relevance manually
- Adjust LLM temperature/prompts
- Review golden dataset quality

## 📚 References & Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [RAGAS Evaluation Framework](https://docs.ragas.io/)
- [FastAPI Guide](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [ChromaDB Vector Store](https://www.trychroma.com/)
- [BERTopic Clustering](https://maartengr.github.io/BERTopic/)

## 📄 License

[Add your license here]

## 👥 Contributing

[Add contribution guidelines here]

## 📞 Support

For issues and questions, please open a GitHub issue or contact the development team.

│

├── experiments/

│   ├── spotify\_brand\_customer.csv          # full customer/brand conversation dataset

│   └── spotify\_brand\_customer\_sample5000.csv  # 5,000-row sample for local/quick iteration

│

├── evaluation/

│   └── ...

│

├── .env

├── requirements.txt

└── README.md

```



This guide covers the \*\*Chroma (local, sample) + Qdrant (cloud, full)\*\* vector store setup and

the \*\*NVIDIA NIM / OpenAI / local Ollama\*\* LLM provider options.


\## 1. Setup



Clone the repository and move into the project directory:



```bash

git clone <YOUR\_REPOSITORY\_URL>

cd ResolveAI

```



Create and activate the virtual environment:



```bash

python -m venv .venv

```



Windows:



```powershell

.venv\\Scripts\\activate

```



Linux/macOS:



```bash

source .venv/bin/activate

```



Install dependencies:



```bash

pip install -r requirements.txt

```



\## 2. Environment Variables



Create a `.env` file in the project root.



```env

\# ---------------------------------------------------------------------------

\# LLM PROVIDER

\# I used : "nvidia\_nim" | "openai"(Inference) | "ollama"(ingestion)

\# ---------------------------------------------------------------------------

LLM\_PROVIDER=ollama



\# NVIDIA NIM (cloud-hosted inference microservices)

NVIDIA\_API\_KEY=your\_nvidia\_nim\_api\_key





\# OpenAI

OPENAI\_API\_KEY=your\_openai\_api\_key

OPENAI\_MODEL=gpt-4o-mini



\# Local LLM via Ollama

OLLAMA\_BASE\_URL=http://localhost:11434

OLLAMA\_MODEL=llama3.2:1b



\# ---------------------------------------------------------------------------

\# EMBEDDINGS

\# ---------------------------------------------------------------------------

HF\_EMBEDDING\_MODEL=sentence-transformers/all-MiniLM-L6-v2

HUGGINGFACEHUB\_API\_TOKEN=your\_hf\_token   # only needed for gated/private HF models



\# ---------------------------------------------------------------------------

\# VECTOR STORE

\# Choose one: "qdrant" | "chroma"

\# ---------------------------------------------------------------------------

VECTOR\_STORE=chroma



\# Qdrant (cloud) - used for the FULL dataset

QDRANT\_URL=https://your-cluster-url.qdrant.io

QDRANT\_API\_KEY=your\_qdrant\_api\_key

QDRANT\_COLLECTION=spotify\_support\_full



\# Chroma (local) - used for the 5,000-row SAMPLE dataset

CHROMA\_PERSIST\_DIR=./chroma\_db

CHROMA\_COLLECTION=spotify\_support\_sample



\# ---------------------------------------------------------------------------

\# DATASET

\# ---------------------------------------------------------------------------

DATASET\_MODE=sample   # "sample" (5,000 rows, Chroma) or "full" (Qdrant cloud)

SAMPLE\_SIZE=5000

```



Only fill in credentials for the provider(s) you're actually using — e.g. if `LLM\_PROVIDER=ollama`,

the `NVIDIA\_API\_KEY` / `OPENAI\_API\_KEY` rows can stay blank.



\## 3. Dataset








Use the 5,000-row sample for local development and quick iteration (Chroma needs no external

service). Switch to the full dataset + Qdrant cloud once the pipeline is validated locally.



The ingestion pipeline creates a parent-child representation of historical support conversations:



```text

Customer Conversation

&#x20;       |

&#x20;       +---- Parent Thread

&#x20;       |       |

&#x20;       |       +---- Child Chunk

&#x20;       |       +---- Child Chunk

&#x20;       |       +---- Child Chunk

&#x20;       |

&#x20;       +---- Parent Thread

&#x20;               |

&#x20;               +---- Child Chunk

&#x20;               +---- Child Chunk

```



Child chunks are used for semantic retrieval while parent threads preserve the complete

conversation context.



\## 4. Start the Vector Store



\*\*Local (Chroma, sample dataset)\*\* — no separate service needed; Chroma persists to

`CHROMA\_PERSIST\_DIR` on disk automatically when ingestion runs.



\*\*Cloud (Qdrant, full dataset)\*\* — either point `QDRANT\_URL` / `QDRANT\_API\_KEY` at a hosted

Qdrant Cloud cluster, or run Qdrant locally with Docker for testing against the same code path:



```bash

docker run -p 6333:6333 qdrant/qdrant

```



Verify:



```text

http://localhost:6333

```



\## 5. Start Ollama (if using a local LLM)



If `LLM\_PROVIDER=ollama`, pull the model and make sure the server is running:



```bash

ollama pull llama3.2:1b

ollama serve

```



Ollama listens at `OLLAMA\_BASE\_URL` (default `http://localhost:11434`).



\## 6. Run Sample Ingestion



Run the sample ingestion first to verify the ingestion and retrieval pipeline end-to-end on the

5,000-row dataset with Chroma:



```bash

python -m ingestion.multiturn\_sample\_indexing

```



This validates:



\* Dataset loading (`spotify\_brand\_customer\_sample5000.csv`)

\* Conversation processing

\* Parent-child creation

\* HuggingFace embedding generation

\* Chroma vector-store connection

\* Metadata creation



\## 7. Run Full Ingestion



After sample ingestion works, switch `.env` to `VECTOR\_STORE=qdrant` / `DATASET\_MODE=full` and

run the full ingestion against the complete dataset:



```bash

python -m ingestion.parent\_child\_indexing

```



or, for the multi-turn variant:



```bash

python -m ingestion.multiturn\_ingestion

```



Run only the ingestion pipeline required by your configured retrieval implementation.



\## 8. Test Retrieval



Test retrieval independently before starting the API:



```bash

python -m ingestion.retrieval

```



Example query:



```text

I cannot login to my Spotify account

```



The retrieval flow is:



```text

Query

&#x20; |

&#x20; v

Query Embedding (HuggingFace)

&#x20; |

&#x20; v

Child Vector Search (Chroma or Qdrant, per VECTOR\_STORE)

&#x20; |

&#x20; v

Top-K Child Chunks

&#x20; |

&#x20; v

Parent Thread Mapping

&#x20; |

&#x20; v

Top-N Parent Threads

&#x20; |

&#x20; v

Historical Evidence

```



\## 9. Start FastAPI



The backend is located at:



```text

app/main.py

```



Start the server from the project root:



```bash

uvicorn app.main:app --reload

```



API:



```text

http://127.0.0.1:8000

```



Swagger documentation:



```text

http://127.0.0.1:8000/docs

```



Test the API from Swagger before starting the frontend.



\## 10. Support Agent Workflow



The agent workflow is implemented in:



```text

app/support\_agent\_workflow.py

```



The workflow combines:



```text

Customer Query

&#x20;     |

&#x20;     v

Retrieval Decision

&#x20;     |

&#x20;     +---- No Retrieval ----> Writer (LLM\_PROVIDER: nvidia\_nim | openai | ollama)

&#x20;     |

&#x20;     +---- Retrieval -------> Historical Retrieval

&#x20;                                     |

&#x20;                                     v

&#x20;                               Evidence Check

&#x20;                                     |

&#x20;                        +------------+------------+

&#x20;                        |                         |

&#x20;                    Sufficient                Insufficient

&#x20;                        |                         |

&#x20;                        v                         v

&#x20;                      Writer                 Human Review

&#x20;                        |                         |

&#x20;                        +-----------+-------------+

&#x20;                                    |

&#x20;                                    v

&#x20;                                 Response

```



The workflow uses historical evidence to ground generated responses and routes to human review

when evidence is insufficient, regardless of which `LLM\_PROVIDER` is generating the final reply.



\## 11. Start Streamlit UI



The frontend is:



```text

frontend/ui.py

```



Start it from the project root:



```bash

streamlit run frontend/ui.py

```



Open:



```text

http://localhost:8501

```



The frontend communicates with FastAPI:



```text

Streamlit

&#x20;   |

&#x20;   | HTTP

&#x20;   v

FastAPI

&#x20;   |

&#x20;   v

Support Agent Workflow

&#x20;   |

&#x20;   v

RAG / Evidence / LLM (NVIDIA NIM / OpenAI / Ollama)

```



\## 12. Complete Run



Use separate terminals.



\### Terminal 1 - Vector store (only needed for local Qdrant; skip for Chroma)



```bash

docker run -p 6333:6333 qdrant/qdrant

```



\### Terminal 2 - Local LLM (only needed if LLM\_PROVIDER=ollama)



```bash

ollama serve

```



\### Terminal 3 - Ingestion



```bash

python -m ingestion.multiturn\_sample\_indexing

\# then, once validated:

python -m ingestion.parent\_child\_indexing

```



\### Terminal 4 - FastAPI



```bash

uvicorn app.main:app --reload

```



\### Terminal 5 - Streamlit



```bash

streamlit run frontend/ui.py

```



Open:



```text

http://localhost:8501

```



\## 13. End-to-End Flow



A customer query follows this path:



```text

&#x20;                   Customer

&#x20;                      |

&#x20;                      v

&#x20;                Streamlit UI

&#x20;                      |

&#x20;                      v

&#x20;                 FastAPI API

&#x20;                      |

&#x20;                      v

&#x20;             Support Agent Workflow

&#x20;                      |

&#x20;                      v

&#x20;               Retrieval Decision

&#x20;                      |

&#x20;                      v

&#x20;       Parent-Child Retrieval (Chroma / Qdrant)

&#x20;                      |

&#x20;                      v

&#x20;             Historical Evidence

&#x20;                      |

&#x20;                      v

&#x20;               Evidence Checking

&#x20;                      |

&#x20;               +------+------+

&#x20;               |             |

&#x20;            PASS          ESCALATE

&#x20;               |             |

&#x20;               v             v

&#x20;            LLM Writer    Human Review

&#x20;       (NIM/OpenAI/Ollama)  |

&#x20;               |             |

&#x20;               +------+------+

&#x20;                      |

&#x20;                      v

&#x20;                 Final Response

```



\## 14. Human-in-the-Loop



The agent supports three review actions:



```text

approve

```



Accept the generated response.



```text

feedback / reject

```



Provide feedback and regenerate the response.



```text

takeover

```



Stop automated handling and allow a human to handle the conversation.



High-risk or uncertain cases — low evidence confidence, billing/account-sensitive topics, or a

low-confidence intent classification upstream — should be routed to human review instead of

being automatically answered.



\## 15. Troubleshooting



\### Dataset Not Found



Run commands from the project root:



```bash

cd ResolveAI

python -m ingestion.multiturn\_sample\_indexing

```



Use project-root-relative paths in Python:



```python

from pathlib import Path



PROJECT\_ROOT = Path(\_\_file\_\_).resolve().parent.parent



DATASET\_PATH = (

&#x20;   PROJECT\_ROOT

&#x20;   / "experiments"

&#x20;   / ("spotify\_brand\_customer\_sample5000.csv" if DATASET\_MODE == "sample" else "spotify\_brand\_customer.csv")

)

```



\### FastAPI 422 Error



Make sure the Streamlit request format matches the FastAPI endpoint.



For a JSON request body:



```python

requests.post(

&#x20;   API\_URL,

&#x20;   json={"query": query},

&#x20;   timeout=120

)

```



For query parameters, the FastAPI endpoint must explicitly define the query parameter.



\### Qdrant Connection Error



Check that Qdrant is reachable and the API key is correct:



```text

https://your-cluster-url.qdrant.io   (cloud)

http://localhost:6333                (local Docker)

```



\### Chroma Persistence Issues



Confirm `CHROMA\_PERSIST\_DIR` points to a writable directory and that ingestion has actually run

at least once — Chroma creates the directory on first write, not on import.



\### Ollama Connection Error



Confirm the model is pulled and the server is running:



```bash

ollama list          # should show llama3.2:1b

ollama serve

```



Check `OLLAMA\_BASE\_URL` matches where Ollama is actually listening (default `http://localhost:11434`).



\### NVIDIA NIM / OpenAI Auth Errors



Confirm `NVIDIA\_API\_KEY` / `OPENAI\_API\_KEY` are set in `.env` and that `LLM\_PROVIDER` matches the

credentials you actually filled in — a mismatch (e.g. `LLM\_PROVIDER=openai` with only

`NVIDIA\_API\_KEY` set) will fail at the writer step, not at startup.



\### Import Errors



Run Python modules from the project root:



```bash

python -m ingestion.multiturn\_sample\_indexing

```



instead of running files from arbitrary working directories.



\## 16. Recommended Execution Order



```text

1\. Create virtual environment

2\. Install dependencies

3\. Configure .env (LLM\_PROVIDER, embeddings, VECTOR\_STORE, dataset)

4\. Place datasets in experiments/

5\. Start Chroma (automatic) or Qdrant (cloud or local Docker)

6\. Start Ollama, if LLM\_PROVIDER=ollama

7\. Run sample ingestion (5,000 rows, Chroma)

8\. Test retrieval

9\. Run full ingestion (Qdrant), once validated

10\. Start FastAPI

11\. Test /docs

12\. Start Streamlit

13\. Test end-to-end customer query

```



\## 17. Quick Start



```bash

\# Setup

python -m venv .venv

.venv\\Scripts\\activate

pip install -r requirements.txt



\# (if LLM\_PROVIDER=ollama)

ollama pull llama3.2:1b

ollama serve



\# Sample ingestion (5,000 rows, local Chroma)

python -m ingestion.multiturn\_sample\_indexing



\# Retrieval test

python -m ingestion.retrieval



\# Start API

uvicorn app.main:app --reload



\# Start UI in another terminal

streamlit run frontend/ui.py

```



Open:



```text

Frontend:

http://localhost:8501



API:

http://127.0.0.1:8000



Swagger:

http://127.0.0.1:8000/docs

```



\## License



This project is developed for educational, research, and demonstration purposes.

