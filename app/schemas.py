from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class QueryRequest(BaseModel):
    query: str


class AnalyticsResponse(BaseModel):
    query: str
    generated_logic: str
    result: Any
    confidence_score: float
    explanation: str


class HealthResponse(BaseModel):
    status: str
    providers: Dict[str, bool]
    duckdb_registered_tables: List[str]


class SchemaInfoResponse(BaseModel):
    tables: Dict[str, List[str]]
    metrics: Dict[str, str]
    dimensions: List[str]
    synonyms: Dict[str, str]
    known_dimension_values: Dict[str, List[str]]
    max_date: Optional[str]
