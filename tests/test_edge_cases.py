import pytest
import httpx
from app.llm.metric_resolver import (
    resolve_metric_expression,
    resolve_metric_aggregate_sql,
    MetricResolutionError,
)
from app.sql.executor import serialize_dataframe
import pandas as pd


@pytest.mark.asyncio
async def test_empty_query_rejected(client: httpx.AsyncClient):
    response = await client.post("/query", json={"query": "   "})
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_zero_results_penalized(client: httpx.AsyncClient):
    response = await client.post("/query", json={"query": "Total sales in Atlantis for March"})
    assert response.status_code == 200
    data = response.json()
    assert data["result"] == [] or data["result"] == 0 or data["result"] is None
    assert data["confidence_score"] <= 0.80
    assert "zero" in data["explanation"].lower() or "not match" in data["explanation"].lower() or data["confidence_score"] <= 0.80


@pytest.mark.asyncio
async def test_yoy_single_year_limitation(client: httpx.AsyncClient):
    response = await client.post("/query", json={"query": "YoY growth in revenue"})
    assert response.status_code == 200
    data = response.json()
    assert data["confidence_score"] <= 0.80
    assert "yoy" in data["explanation"].lower() or "2024" in data["explanation"].lower()


def test_metric_resolution_recursion():
    metric_defs = {
        "revenue": "quantity * unit_price * (1 - discount)",
        "orders": "count(order_id)",
        "avg_order_value": "revenue / orders",
    }
    resolved = resolve_metric_aggregate_sql("avg_order_value", metric_defs)
    assert "quantity" in resolved
    assert "unit_price" in resolved
    assert "COUNT(order_id)" in resolved


def test_metric_resolution_circular_dependency():
    circular_defs = {
        "metric_a": "metric_b * 2",
        "metric_b": "metric_a + 5",
    }
    with pytest.raises(MetricResolutionError):
        resolve_metric_expression("metric_a", circular_defs)


def test_serialize_dataframe():
    # 1x1 scalar
    df_scalar = pd.DataFrame({"revenue": [123.456]})
    assert serialize_dataframe(df_scalar) == 123.46

    # Empty
    df_empty = pd.DataFrame()
    assert serialize_dataframe(df_empty) == []

    # Records
    df_records = pd.DataFrame({"city": ["Berlin", "Paris"], "profit": [150.123, 120.456]})
    serialized = serialize_dataframe(df_records)
    assert len(serialized) == 2
    assert serialized[0]["city"] == "Berlin"
    assert serialized[0]["profit"] == 150.12
