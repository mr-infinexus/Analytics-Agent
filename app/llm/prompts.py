from typing import Any, Dict, List

SYSTEM_PROMPT = """You are an expert DuckDB SQL analytics engineer.
Your task is to convert natural language business questions into precise, executable DuckDB SQL queries.

CRITICAL RULES:
1. Output ONLY the raw executable SQL query. Do NOT include markdown code fences (```sql ... ```), explanations, notes, or preamble.
2. Tables available:
   - `sales`: contains historical transaction records.
   - `targets`: contains target revenue by region and month (format 'YYYY-MM').
3. Date format: `order_date` in `sales` is a TIMESTAMP/DATE. Use `strftime(order_date, '%Y-%m')` for monthly matching or `strftime(order_date, '%Y')` for yearly matching.
4. Relative dates: Anchor against the maximum date in the dataset provided in the context.
5. Analytical Scopes:
   - Standard Aggregation: Use correct metric calculations.
   - Ranking: Use `ORDER BY <metric> DESC/ASC LIMIT <n>`.
   - Contribution %: Use window functions like `ROUND(100.0 * <metric> / SUM(<metric>) OVER (), 2) AS contribution_pct`.
   - Target Comparison: Join `sales` with `targets` ON `sales.region = targets.region AND strftime(sales.order_date, '%Y-%m') = targets.month`.
   - Nested/Rank per group: Use window functions like `RANK() OVER (PARTITION BY <group> ORDER BY <metric> DESC)`.
6. Use column names strictly as specified in the schema.
"""


def build_sql_prompt(
    query: str,
    schema_summary: str,
    resolved_metrics: Dict[str, str],
    known_dimensions: Dict[str, List[str]],
    few_shot_examples: List[Dict[str, Any]],
) -> str:
    examples_str = ""
    if few_shot_examples:
        lines = ["RELEVANT FEW-SHOT EXAMPLES:"]
        for i, ex in enumerate(few_shot_examples, 1):
            lines.append(f"Example {i}:")
            lines.append(f"  Question: {ex.get('query', '')}")
            lines.append(f"  Expected Logic / SQL: {ex.get('expected_logic', '')}\n")
        examples_str = "\n".join(lines) + "\n"

    metrics_str = "\n".join(f"  - {m}: {sql}" for m, sql in resolved_metrics.items())

    dims_str = "\n".join(
        f"  - {col}: {', '.join(repr(v) for v in vals[:10])}{' ...' if len(vals) > 10 else ''}"
        for col, vals in known_dimensions.items()
    )

    return f"""DATABASE CONTEXT:
{schema_summary}

PRE-RESOLVED METRICS (DuckDB SQL aggregates):
{metrics_str}

KNOWN DIMENSION VALUES FOR FILTER MATCHING:
{dims_str}

{examples_str}USER QUESTION:
"{query}"

Generate the exact DuckDB SQL query to answer the question:"""


def build_repair_prompt(
    original_query: str,
    failed_sql: str,
    error_message: str,
    schema_summary: str,
) -> str:
    return f"""The previous SQL query failed during DuckDB execution. Please fix the error and output only the corrected DuckDB SQL.

DATABASE CONTEXT:
{schema_summary}

USER QUESTION:
"{original_query}"

FAILED SQL:
{failed_sql}

DUCKDB ERROR:
{error_message}

CRITICAL: Output ONLY the corrected raw DuckDB SQL. No markdown, no explanation."""
