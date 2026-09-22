import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import duckdb
import pandas as pd

from app.config import settings


def clean_csv(src: Path, dst: Path):
    """Strip extra quotes and BOM from CSV files."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(src, "r", encoding="utf-8-sig") as f_in, open(dst, "w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.rstrip("\r\n")
            if line.startswith('"') and line.endswith('"'):
                line = line[1:-1].replace('""', '"')
            f_out.write(line + "\n")


class DataLayer:
    def __init__(self, data_dir: Optional[str] = None, raw_data_dir: Optional[str] = None):
        self.data_dir = settings.resolve_path(data_dir or settings.DATA_DIR)
        self.raw_data_dir = settings.resolve_path(raw_data_dir or settings.RAW_DATA_DIR)

        sales_path = self.data_dir / "sales_data.csv"
        targets_path = self.data_dir / "targets.csv"
        dict_path = self.data_dir / "data_dictionary.json"

        # Clean raw files if processed ones are missing
        if not (sales_path.exists() and targets_path.exists() and dict_path.exists()):
            if self.raw_data_dir.exists():
                for f in os.listdir(self.raw_data_dir):
                    in_f = self.raw_data_dir / f
                    if in_f.is_file():
                        clean_csv(in_f, self.data_dir / f)

        self.sales = pd.read_csv(sales_path, parse_dates=["order_date"])
        self.targets = pd.read_csv(targets_path)
        with open(dict_path, "r", encoding="utf-8") as f:
            self.data_dictionary = json.load(f)

        # In-memory DuckDB setup
        self.con = duckdb.connect(database=":memory:")
        self.con.register("sales", self.sales)
        self.con.register("targets", self.targets)

        self.max_date = str(self.sales["order_date"].max().date())
        self.known_values = self._get_dimension_values()

    def _get_dimension_values(self) -> Dict[str, List[str]]:
        dims = self.data_dictionary.get("dimensions", [])
        values = {}
        for d in dims:
            if d in self.sales.columns and d != "order_date":
                values[d] = sorted(self.sales[d].dropna().unique().astype(str).tolist())
        return values

    def get_schema_summary(self) -> str:
        sales_cols = ", ".join(f"{col} ({dtype})" for col, dtype in zip(self.sales.columns, self.sales.dtypes))
        targets_cols = ", ".join(f"{col} ({dtype})" for col, dtype in zip(self.targets.columns, self.targets.dtypes))

        metrics_desc = "\n".join(
            f"  - {m}: {formula}"
            for m, formula in self.data_dictionary.get("metrics", {}).items()
        )
        synonyms_desc = "\n".join(
            f"  - '{s}' -> {target}"
            for s, target in self.data_dictionary.get("synonyms", {}).items()
        )

        return f"""TABLE sales:
  Columns: {sales_cols}

TABLE targets:
  Columns: {targets_cols}

METRIC FORMULAS (from data dictionary):
{metrics_desc}

SYNONYMS:
{synonyms_desc}

DATASET TEMPORAL CONTEXT:
  Max date in dataset: {self.max_date} (use this as the anchor for relative dates like 'last month', 'this quarter')
"""


_data_layer: Optional[DataLayer] = None


def get_data_layer() -> DataLayer:
    global _data_layer
    if _data_layer is None:
        _data_layer = DataLayer()
    return _data_layer
