# RAG vs GraphRAG Comparison Platform

This project provides a Streamlit-based UI to compare a **Standard RAG** pipeline (ChromaDB + embeddings) with a **GraphRAG** pipeline (Neo4j knowledge graph + vector store), including an LLM-as-a-Judge evaluation module for batch experiments using airline booking policy documents.

---

## 1. Project structure

Key Python modules:

- `app.py`: Streamlit application (UI, controls, dashboards, and evaluation workflows).
- `standard_rag.py`: Standard RAG implementation using ChromaDB and HuggingFace embeddings.
- `graph_rag.py`: GraphRAG implementation using Neo4j (knowledge graph + Neo4jVector). 
- `evaluator.py`: LLM-as-a-Judge evaluator with hybrid scoring (LLM + section matching). 
- `requirements.txt`: Python dependencies for all components.

The folder containing airline booking policy documents stored in `data/policies` to build the indices and graph.

---

## 2. Prerequisites

Before running the app locally, ensure the following are installed and running:

- **Python** (3.10+ recommended)  
- **Virtual environment** tool (e.g., `venv`)  
- **Ollama** (local LLM server) with a model exposed as `gemma3:27b`, used by:
  - Standard RAG answer generation.
  - GraphRAG answer generation.
  - LLM Judge evaluator.
- **Neo4j** (local instance) for GraphRAG:
  - Default URI expected: `neo4j://localhost:7687`.
  - Default credentials expected in code:
    - Username: `neo4j`  
    - Password: `password`

Set the following environment variables before starting the app (adjust to your local Neo4j settings):

```bash
# Linux / macOS
export NEO4J_URI="neo4j://localhost:7687"
export NEO4J_USERNAME="neo4j"
export NEO4J_PASSWORD="password"
```
---

## 3. Installation

1. **Clone or download the repository** to a folder, for example:

```bash
mkdir -p ~/projects
cd ~/projects
# Place this project (graph-rag-airline-policy-qa/) here
```

2. **Create and activate a virtual environment**:
```bash
cd graph-rag-airline-policy-qa
python -m venv .venv

# Activate on Linux/macOS
source .venv/bin/activate
```


3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

---

## 4. Running the Streamlit app

From the project root (where app.py is located), run:
```bash
streamlit run app.py --server.port=8501
```
This will start the Streamlit server and bind it to port 8501 on your local machine.

Then open a browser and go to:
```http://localhost:8501```

You should see the main page titled “RAG vs GraphRAG Comparison Platform”.

---
## 5. Using the application

### 5.1 Initialize systems

In the left sidebar:

1. Click “Initialize Both Systems”.

2. The app will:
   - Initialize Standard RAG (ChromaDB + embeddings + LLM).
   - Initialize GraphRAG (Neo4j graph + Neo4jVector + LLMs).
   - Instantiate the LLMJudgeEvaluator with both systems.

Status metrics for each system are displayed under “System Status” in the main view.

If initialization fails (e.g., missing ChromaDB index or Neo4j connection error), an error message will be shown in the UI.

### 5.2 Index and graph management

In the sidebar “Index Management” section:
- Rebuild RAG ChromaDB
    - Rebuilds the Standard RAG index from the policy folder using DirectoryLoader and Markdown splitting.
- Rebuild GraphRAG Neo4j Graph
  - Clears existing nodes and relationships, re-loads Markdown documents per airline, extracts structure, builds the Neo4j knowledge graph, and creates the Neo4jVector index.

Use these buttons whenever your policy documents change or when running the project for the first time.

### 5.3 Live query comparison

In the “Live Query Comparison” tab:
1. Type a booking policy question in the input field, e.g.:
What are the refund conditions for Economy Basic tickets?
2. Click “Search”.
3. The app:
   - Sends the query to Standard RAG and GraphRAG in parallel.
   - Displays:
     - Standard RAG answer + retrieved chunks (with airline, section, excerpt, and metrics).
     - GraphRAG answer + traversal path details (initial nodes, connected nodes, number of hops) and metrics.

This allows side-by-side qualitative comparison of both systems’ answers and retrieval behavior.


### 5.4 Batch evaluation (LLM-as-a-Judge)
In the “Batch Evaluation (LLM-as-a-Judge)” tab:

1. Upload the CSV containing the evaluation data. For testing, the file `airlines_multihop_eval_v1.csv` is used as an example in this repository.

2. The app previews the first few rows and reports how many questions will be evaluated.

3. Click “Run Evaluation”:
   - For each question, it queries Standard RAG and GraphRAG.
   - Uses the judge LLM to score each answer (0–100) based on completeness, accuracy, relevance, and clarity.
   - Computes a hybrid score: 50% LLM score + 50% section match (if ground-truth sections are provided).

4. The UI shows:
   - Overall performance metrics (average scores, success rate, median response time per system).
   - Score component breakdown (LLM vs section match), distribution tables, and win/loss/tie counts.
   - Per-question detailed analysis with:
       - Reference answer
       - Predicted answers (Standard RAG vs GraphRAG)
       - Retrieved sections and IDs
       - Judge explanations and comparative commentary.

5. Download buttons allow exporting:
   - Summary CSV of aggregated metrics.
   - Full results CSV with per-question scores and metadata.
   - Comparison CSV with side-by-side statistics.
