from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.data.loader import get_data_layer
from app.explanation.generator import generate_explanation
from app.llm.client import get_llm_client
from app.llm.metric_resolver import resolve_metric_aggregate_sql
from app.llm.prompts import SYSTEM_PROMPT, build_sql_prompt
from app.schemas import AnalyticsResponse, HealthResponse, QueryRequest, SchemaInfoResponse
from app.scoring.confidence import get_confidence_scorer
from app.sql.executor import SQLExecutor
from app.vector_db.store import get_vector_store

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("analytics-agent")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize data layer and seed few-shot examples
    data_layer = get_data_layer()
    vector_store = get_vector_store()
    examples_file = data_layer.data_dir / "nl_queries.json"
    if not examples_file.exists():
        examples_file = data_layer.raw_data_dir / "nl_queries.json"
    vector_store.seed_from_file(examples_file)

    # Initialize LLM and scorer clients
    get_llm_client()
    get_confidence_scorer()
    yield


app = FastAPI(
    title="Analytics Query Agent",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health():
    data = get_data_layer()
    llm = get_llm_client()
    tables = [row[0] for row in data.con.execute("SHOW TABLES").fetchall()]

    return HealthResponse(
        status="healthy",
        providers={
            "groq": bool(llm.groq_api_key),
            "gemini": bool(llm.gemini_api_key),
        },
        duckdb_registered_tables=tables,
    )


@app.get("/schema", response_model=SchemaInfoResponse)
async def get_schema():
    data = get_data_layer()
    return SchemaInfoResponse(
        tables={
            "sales": list(data.sales.columns),
            "targets": list(data.targets.columns),
        },
        metrics=data.data_dictionary.get("metrics", {}),
        dimensions=data.data_dictionary.get("dimensions", []),
        synonyms=data.data_dictionary.get("synonyms", {}),
        known_dimension_values=data.known_values,
        max_date=data.max_date,
    )


@app.get("/examples")
async def get_examples():
    data = get_vector_store().collection.get()
    return {
        "total_examples": len(data.get("ids", [])),
        "examples": data.get("metadatas", []),
    }


@app.post("/query", response_model=AnalyticsResponse)
async def query(req: QueryRequest):
    user_query = req.query.strip()
    if not user_query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    data_layer = get_data_layer()
    vector_store = get_vector_store()
    llm = get_llm_client()
    scorer = get_confidence_scorer()
    executor = SQLExecutor(con=data_layer.con)

    # 1. Retrieve similar few-shot examples
    examples = vector_store.get_similar_examples(user_query, n=3)

    # 2. Pre-resolve metrics into SQL expressions
    metrics = {
        m: resolve_metric_aggregate_sql(m, data_layer.data_dictionary.get("metrics", {}))
        for m in data_layer.data_dictionary.get("metrics", {})
    }
    schema_summary = data_layer.get_schema_summary()

    # 3. Generate SQL using LLM
    prompt = build_sql_prompt(
        query=user_query,
        schema_summary=schema_summary,
        resolved_metrics=metrics,
        known_dimensions=data_layer.known_values,
        few_shot_examples=examples,
    )

    try:
        sql = await llm.generate_sql(user_prompt=prompt, system_prompt=SYSTEM_PROMPT)
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        raise HTTPException(status_code=502, detail=f"LLM generation failed: {e}")

    # 4. Execute SQL with self-healing retry on syntax error
    result, final_sql, retries, err, row_count = await executor.execute_with_self_healing(
        query=user_query,
        initial_sql=sql,
        llm_client=llm,
        schema_summary=schema_summary,
        max_retries=3,
        timeout=5.0,
    )

    # 5. Score confidence based on execution signals + LLM eval
    score, reasons = await scorer.calculate_confidence(
        query=user_query,
        sql=final_sql,
        result=result,
        row_count=row_count,
        retries_used=retries,
        execution_error=err,
        schema_summary=schema_summary,
    )

    # 6. Generate human-readable explanation
    explanation = generate_explanation(
        query=user_query,
        sql=final_sql,
        confidence_score=score,
        confidence_reasons=reasons,
        data_dictionary=data_layer.data_dictionary,
    )

    return AnalyticsResponse(
        query=user_query,
        generated_logic=final_sql,
        result=result,
        confidence_score=score,
        explanation=explanation,
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=7860)
