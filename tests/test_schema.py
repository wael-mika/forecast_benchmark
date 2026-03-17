from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.schema import validate_canonical_dataframe, validate_required_columns


def test_validate_required_columns_accepts_canonical_schema() -> None:
    df = pd.DataFrame(
        {
            "unique_id": ["station_a"],
            "ds": pd.to_datetime(["2020-01-01"]),
            "y": [1.23],
        }
    )
    validate_required_columns(df)


def test_validate_canonical_dataframe_rejects_non_datetime_ds() -> None:
    df = pd.DataFrame(
        {
            "unique_id": ["station_a"],
            "ds": ["2020-01-01"],
            "y": [1.23],
        }
    )
    with pytest.raises(TypeError):
        validate_canonical_dataframe(df)
