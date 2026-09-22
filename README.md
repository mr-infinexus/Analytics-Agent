# Intelligent Analytics Query Engine

A FastAPI service that translates natural language analytical questions into DuckDB SQL, executes them against an in-memory database, handles execution errors with self-healing reflection, and returns structured results with confidence scoring and explanations.

## Features

- **Text-to-SQL**: Converts natural language into DuckDB SQL queries using schema grounding and metric formula expansion.
- **Few-Shot RAG**: Retrieves relevant example queries from a local ChromaDB vector store.
- **Self-Healing Execution**: Catches DuckDB syntax or schema errors and asks the LLM to fix them (up to 3 retries).
- **Confidence Scoring**: Blends execution signals (empty results, syntax retries, dataset limits) with an LLM evaluation score.
- **Auditable Explanations**: Generates a breakdown of detected metrics, filters, groupings, and confidence factors.

## Tech Stack

- **Framework**: FastAPI + Uvicorn
- **Database**: DuckDB (in-memory)
- **Vector DB**: ChromaDB + Gemini text embeddings
- **LLM**: Groq (primary: `openai/gpt-oss-120b`, fallback: `qwen/qwen3.8-27b`) and Gemini (`gemini-3.8-flash`)

## Setup & Running

### 1. Install Dependencies
```bash
# Using uv:
uv sync
```

### 2. Environment Variables
Copy `.env.example` to `.env` and fill in your API keys:
```bash
cp .env.example .env
```
Ensure `GROQ_API_KEY` and `GEMINI_API_KEY` are set.

### 3. Run the Server
```bash
uv run main.py
```
Swagger API documentation is available at `http://localhost:7860/docs`.

### 4. Run Tests
```bash
uv run pytest tests/ -v
```

## Sample Request & Response

### Request
```bash
curl -X POST http://localhost:7860/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Total sales in India for March"}'
```

### Response
```json
{
  "query": "Total sales in India for March",
  "generated_logic": "SELECT SUM(quantity * unit_price * (1 - discount)) AS revenue FROM sales WHERE country = 'India' AND strftime(order_date, '%Y-%m') = '2024-03'",
  "result": 132.0,
  "confidence_score": 0.95,
  "explanation": "Interpreted metric as 'revenue'. Applied filters: [country = 'India' AND strftime(order_date, '%Y-%m') = '2024-03']. Confidence score: 0.95. Factors: LLM eval judge (openai/gpt-oss-20b) scored faithfulness at 0.95."
}
```
