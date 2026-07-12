"""Small file I/O helpers used across training, evaluation, and scripts.

This module keeps the most common save helpers in one place so the rest of the
codebase does not repeat the same directory-creation and serialization logic.

Main helpers
------------
ensure_parent_dir
    Create the parent folder for an output path when needed.
save_parquet / save_csv / save_json
    Save common artifact formats with consistent parent-directory handling.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_parent_dir(path: Path) -> None:
    """Create the parent directory for ``path`` if it does not already exist."""
    path.parent.mkdir(parents=True, exist_ok=True)


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    """Save a dataframe to parquet after making sure the parent folder exists."""
    ensure_parent_dir(path)
    df.to_parquet(path, index=False)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    """Save a dataframe to CSV after making sure the parent folder exists."""
    ensure_parent_dir(path)
    df.to_csv(path, index=False)


def save_json(payload: dict[str, Any], path: Path) -> None:
    """Save a dictionary as nicely formatted JSON after creating the parent folder."""
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
