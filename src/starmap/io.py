from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_table(path: str | Path, **kwargs) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    if p.suffix.lower() in {'.parquet', '.pq'}:
        return pd.read_parquet(p, **kwargs)
    return pd.read_csv(p, **kwargs)


def write_table(df: pd.DataFrame, path: str | Path, index: bool = False) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() in {'.parquet', '.pq'}:
        df.to_parquet(p, index=index)
    else:
        df.to_csv(p, index=index)
    return p


def sanitize_name(value: object) -> str:
    return re.sub(r"[^\w\-\.]+", "_", str(value)).strip("_")


def first_existing_column(df: pd.DataFrame, names: Iterable[str]) -> str | None:
    lower_to_actual = {c.lower(): c for c in df.columns}
    for name in names:
        if name in df.columns:
            return name
        if name.lower() in lower_to_actual:
            return lower_to_actual[name.lower()]
    return None
