import re
from typing import Any, Dict, List


def generate_explanation(
    query: str,
    sql: str,
    confidence_score: float,
    confidence_reasons: List[str],
    data_dictionary: Dict[str, Any],
) -> str:
    """Generate a readable summary of how the query was interpreted and executed."""
    sql_upper = sql.upper()
    parts = []

    # Detect metric
    metric = "metric"
    for m in ["avg_order_value", "revenue", "profit", "orders"]:
        if m in sql.lower() or m.replace("_", " ") in query.lower():
            metric = m
            break
    parts.append(f"Interpreted metric as '{metric}'.")

    # Detect groupings
    group_match = re.search(r"GROUP\s+BY\s+([^;\n]+)", sql, re.IGNORECASE)
    if group_match:
        dims = [d.strip() for d in group_match.group(1).split(",")]
        clean_dims = [d for d in dims if not d.upper().startswith("STRFTIME")]
        if clean_dims:
            parts.append(f"Grouped by: {', '.join(clean_dims)}.")

    # Detect filters
    where_match = re.search(r"WHERE\s+([^;\n]+?)(?:GROUP|ORDER|LIMIT|$)", sql, re.IGNORECASE | re.DOTALL)
    if where_match:
        where_clause = " ".join(where_match.group(1).strip().split())
        parts.append(f"Applied filters: [{where_clause}].")

    # Analytical features
    features = []
    if "LIMIT" in sql_upper:
        lim_match = re.search(r"LIMIT\s+(\d+)", sql, re.IGNORECASE)
        n = lim_match.group(1) if lim_match else "N"
        direction = "descending" if "DESC" in sql_upper else "ascending"
        features.append(f"ranking (top {n}, {direction})")

    if "OVER ()" in sql_upper or "OVER()" in sql_upper:
        features.append("contribution percentage (window calculation)")

    if "JOIN TARGETS" in sql_upper or "JOIN TARGET" in sql_upper:
        features.append("target comparison (joined with targets table)")

    if "PARTITION BY" in sql_upper:
        features.append("partitioned window ranking")

    if features:
        parts.append(f"Analytical features: {'; '.join(features)}.")

    # Confidence breakdown
    parts.append(f"Confidence score: {confidence_score:.2f}.")
    if confidence_reasons:
        parts.append(f"Factors: {'; '.join(confidence_reasons)}.")

    return " ".join(parts)
