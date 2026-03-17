"""I/O helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_parent_dir(path: Path) -> None:
    """Create the parent directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    """Save a dataframe as parquet."""
    ensure_parent_dir(path)
    df.to_parquet(path, index=False)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    """Save a dataframe as CSV."""
    ensure_parent_dir(path)
    df.to_csv(path, index=False)


def save_json(payload: dict[str, Any], path: Path) -> None:
    """Save a dictionary as JSON."""
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
