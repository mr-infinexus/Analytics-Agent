import os
import re
from typing import Any, List, Optional, Tuple
from openai import AsyncOpenAI

from app.config import settings


class ConfidenceScorer:
    def __init__(self):
        self.api_key = settings.GROQ_API_KEY or os.environ.get("GROQ_API_KEY", "")
        self.model = settings.EVAL_MODEL
        self.client = (
            AsyncOpenAI(base_url=settings.GROQ_BASE_URL, api_key=self.api_key, timeout=20.0)
            if self.api_key
            else None
        )

    async def _eval_judge_score(self, query: str, sql: str, schema_summary: str) -> Optional[float]:
        """Ask Groq judge model to score how faithfully SQL satisfies the query."""
        if not self.client:
            return None

        prompt = f"""Rate from 0.0 (completely incorrect) to 1.0 (perfectly accurate)
how accurately this DuckDB SQL answers the natural language query given the schema.

SCHEMA:
{schema_summary}

QUERY:
"{query}"

SQL:
{sql}

Output only a single number between 0.0 and 1.0 on the first line (e.g. 0.95).
Score:"""

        try:
            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=50,
            )
            content = resp.choices[0].message.content or ""
            match = re.search(r"(\b[01](?:\.\d+)?\b)", content)
            if match:
                return max(0.0, min(1.0, float(match.group(1))))
        except Exception:
            pass
        return None

    async def calculate_confidence(
        self,
        query: str,
        sql: str,
        result: Any,
        row_count: int,
        retries_used: int,
        execution_error: Optional[str],
        schema_summary: str,
    ) -> Tuple[float, List[str]]:
        reasons = []

        # Query failed to execute
        if execution_error:
            reasons.append("SQL query execution failed against database")
            return 0.0, reasons

        penalty = 0.0

        # Penalty for empty or null result
        if row_count == 0 or result == [] or result is None:
            penalty += 0.25
            reasons.append("Query returned zero matching rows; filters may not match dataset")

        # Dataset limitation: only 2024 is available, so YoY queries cannot compare prior years
        q_lower = query.lower()
        if "yoy" in q_lower or "year over year" in q_lower or "prior year" in q_lower:
            penalty += 0.35
            reasons.append("Dataset only covers 2024; YoY calculation lacks prior year data")

        # Retries needed to fix syntax errors
        if retries_used > 0:
            penalty += 0.10 * retries_used
            reasons.append(f"SQL required {retries_used} self-healing correction retry(ies)")

        base_score = max(0.0, 1.0 - penalty)

        # Blend: 60% execution signals + 40% LLM judge score
        eval_score = await self._eval_judge_score(query, sql, schema_summary)
        if eval_score is not None:
            final_score = (0.60 * base_score) + (0.40 * eval_score)
            reasons.append(f"LLM eval judge ({self.model}) scored faithfulness at {eval_score:.2f}")
        else:
            final_score = base_score
            reasons.append("Execution verified cleanly against DuckDB in-memory schema")

        return max(0.0, min(1.0, round(final_score, 2))), reasons


_confidence_scorer: Optional[ConfidenceScorer] = None


def get_confidence_scorer() -> ConfidenceScorer:
    global _confidence_scorer
    if _confidence_scorer is None:
        _confidence_scorer = ConfidenceScorer()
    return _confidence_scorer
