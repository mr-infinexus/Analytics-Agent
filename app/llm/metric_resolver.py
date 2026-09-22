import re
from typing import Dict, Set, Optional


class MetricResolutionError(Exception):
    pass


def resolve_metric_expression(
    metric_name: str,
    metric_defs: Dict[str, str],
    visited: Optional[Set[str]] = None,
) -> str:
    """Recursively expand a metric formula into base column expressions."""
    if visited is None:
        visited = set()

    if metric_name in visited:
        raise MetricResolutionError(f"Circular dependency detected in metric: {metric_name}")

    if metric_name not in metric_defs:
        return metric_name

    visited.add(metric_name)
    expr = metric_defs[metric_name]

    for other_name in metric_defs:
        if other_name != metric_name:
            pattern = rf"\b{re.escape(other_name)}\b"
            if re.search(pattern, expr):
                child = resolve_metric_expression(other_name, metric_defs, set(visited))
                expr = re.sub(pattern, f"({child})", expr)

    return expr


def resolve_metric_aggregate_sql(metric_name: str, metric_defs: Dict[str, str]) -> str:
    """Convert metric to DuckDB SQL aggregate expression."""
    if metric_name not in metric_defs:
        return f"SUM({metric_name})"

    expr = resolve_metric_expression(metric_name, metric_defs)

    if metric_name == "orders":
        return expr.replace("count(", "COUNT(")

    if metric_name == "avg_order_value":
        rev = resolve_metric_aggregate_sql("revenue", metric_defs)
        orders = resolve_metric_aggregate_sql("orders", metric_defs)
        return f"({rev}) / NULLIF({orders}, 0)"

    if re.search(r"\b(sum|count|avg|min|max)\s*\(", expr, flags=re.IGNORECASE):
        return expr

    return f"SUM({expr})"
