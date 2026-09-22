import pytest
import httpx


@pytest.mark.asyncio
async def test_health_endpoint(client: httpx.AsyncClient):
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "sales" in data["duckdb_registered_tables"]
    assert "targets" in data["duckdb_registered_tables"]


@pytest.mark.asyncio
async def test_schema_endpoint(client: httpx.AsyncClient):
    response = await client.get("/schema")
    assert response.status_code == 200
    data = response.json()
    assert "revenue" in data["metrics"]
    assert "avg_order_value" in data["metrics"]
    assert "country" in data["dimensions"]
    assert "India" in data["known_dimension_values"].get("country", [])


@pytest.mark.asyncio
async def test_examples_endpoint(client: httpx.AsyncClient):
    response = await client.get("/examples")
    assert response.status_code == 200
    data = response.json()
    assert "total_examples" in data


@pytest.mark.parametrize(
    "query_text",
    [
        "Total sales in India for March",
        "Top 2 cities by profit",
        "Average order value by region",
    ],
)
@pytest.mark.asyncio
async def test_benchmark_queries(client: httpx.AsyncClient, query_text: str):
    response = await client.post("/query", json={"query": query_text})
    assert response.status_code == 200
    data = response.json()

    # Validate 5 required fields per spec
    assert "query" in data
    assert "generated_logic" in data
    assert "result" in data
    assert "confidence_score" in data
    assert "explanation" in data

    assert data["query"] == query_text
    assert len(data["generated_logic"]) > 0
    assert data["result"] is not None
    assert 0.0 <= data["confidence_score"] <= 1.0
    assert len(data["explanation"]) > 0


@pytest.mark.parametrize(
    "unseen_query",
    [
        "Total revenue by product category",
        "Total orders by customer segment",
        "Profit in Germany for January",
    ],
)
@pytest.mark.asyncio
async def test_unseen_queries(client: httpx.AsyncClient, unseen_query: str):
    response = await client.post("/query", json={"query": unseen_query})
    assert response.status_code == 200
    data = response.json()

    assert data["query"] == unseen_query
    assert data["result"] is not None
    assert 0.0 <= data["confidence_score"] <= 1.0
