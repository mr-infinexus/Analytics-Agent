import asyncio
from typing import Any, Optional, Tuple
import duckdb
import pandas as pd

from app.llm.client import LLMClient
from app.llm.prompts import SYSTEM_PROMPT, build_repair_prompt


def serialize_dataframe(df: pd.DataFrame) -> Any:
    """Format DuckDB DataFrame as scalar, empty list, or list of dicts with 2-decimal floats."""
    if df.empty:
        return []

    # Single scalar value
    if df.shape == (1, 1):
        val = df.iloc[0, 0]
        if pd.isna(val):
            return None
        return round(float(val), 2) if isinstance(val, (float, int)) else str(val)

    # Multiple rows/columns
    records = df.to_dict(orient="records")
    for r in records:
        for k, v in r.items():
            if pd.isna(v):
                r[k] = None
            elif isinstance(v, float):
                r[k] = round(v, 2)
    return records


class SQLExecutor:
    def __init__(self, con: duckdb.DuckDBPyConnection):
        self.con = con

    async def execute_with_self_healing(
        self,
        query: str,
        initial_sql: str,
        llm_client: LLMClient,
        schema_summary: str,
        max_retries: int = 3,
        timeout: float = 5.0,
    ) -> Tuple[Any, str, int, Optional[str], int]:
        current_sql = initial_sql
        retries_used = 0
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                df = await asyncio.wait_for(
                    asyncio.to_thread(self.con.execute(current_sql).fetchdf),
                    timeout=timeout,
                )
                return serialize_dataframe(df), current_sql, retries_used, None, len(df)

            except Exception as e:
                retries_used = attempt + 1
                last_error = f"{type(e).__name__}: {e}"

                if attempt >= max_retries:
                    break

                # Ask LLM to fix the query error
                repair_prompt = build_repair_prompt(
                    original_query=query,
                    failed_sql=current_sql,
                    error_message=last_error,
                    schema_summary=schema_summary,
                )
                try:
                    repaired_sql, _, _ = await llm_client.generate_sql(
                        user_prompt=repair_prompt,
                        system_prompt=SYSTEM_PROMPT,
                    )
                    current_sql = repaired_sql
                except Exception:
                    break

        return None, current_sql, retries_used, last_error, 0
