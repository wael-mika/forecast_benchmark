from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.clean import remove_exact_duplicates
from src.data.load_grdc import discover_raw_files, ingest_grdc_directory, parse_grdc_file


def test_remove_exact_duplicates_keeps_one_copy() -> None:
    df = pd.DataFrame(
        {
            "unique_id": ["station_a", "station_a"],
            "ds": pd.to_datetime(["2020-01-01", "2020-01-01"]),
            "y": [10.0, 10.0],
        }
    )

    cleaned = remove_exact_duplicates(df)

    assert len(cleaned) == 1


def test_ingest_grdc_directory_writes_canonical_parquet(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    processed_path = tmp_path / "processed" / "discharge_daily.parquet"
    raw_dir.mkdir(parents=True)

    (raw_dir / "station_1001.csv").write_text(
        "station_id,date,discharge\n"
        "1001,2020-01-01,5.0\n"
        "1001,2020-01-01,5.0\n"
        "1001,2020-01-02,\n",
        encoding="utf-8",
    )

    df = ingest_grdc_directory(
        raw_data_dir=raw_dir,
        processed_data_path=processed_path,
        date_column_candidates=["date", "ds"],
        value_column_candidates=["discharge", "y"],
        station_id_column_candidates=["station_id", "unique_id"],
    )

    assert list(df.columns) == ["unique_id", "ds", "y"]
    assert pd.api.types.is_datetime64_any_dtype(df["ds"])
    assert not df.duplicated(["unique_id", "ds"]).any()
    assert processed_path.exists()

    stored = pd.read_parquet(processed_path)
    assert list(stored.columns) == ["unique_id", "ds", "y"]
    assert len(stored) == 2


def test_discover_raw_files_recurses_and_filters_by_pattern(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    nested_dir = raw_dir / "Data_slovakia"
    nested_dir.mkdir(parents=True)

    (nested_dir / "6142150_Q_Day.Cmd.txt").write_text("YYYY-MM-DD;Value\n2020-01-01;1.0\n", encoding="utf-8")
    (nested_dir / "6142150_Q_Month.txt").write_text("YYYY-MM-DD;Value\n2020-01-01;2.0\n", encoding="utf-8")

    discovered = discover_raw_files(raw_dir, glob_patterns=["**/*_Q_Day*.txt"])

    assert [path.name for path in discovered] == ["6142150_Q_Day.Cmd.txt"]


def test_parse_grdc_file_extracts_station_id_from_header_and_preserves_missing_values(tmp_path: Path) -> None:
    file_path = tmp_path / "6142150_Q_Day.Cmd.txt"
    file_path.write_text(
        "# GRDC-No.:              6142150\n"
        "# DATA\n"
        "YYYY-MM-DD;hh:mm; Value\n"
        "2020-01-01;--:--;1.000\n"
        "2020-01-02;--:--;-999.000\n",
        encoding="utf-8",
    )

    parsed = parse_grdc_file(
        file_path=file_path,
        date_column_candidates=["YYYY-MM-DD", "date"],
        value_column_candidates=["Value", "y"],
        station_id_column_candidates=["station_id"],
    )

    assert parsed["unique_id"].tolist() == ["6142150", "6142150"]
    assert parsed.loc[0, "y"] == 1.0
    assert pd.isna(parsed.loc[1, "y"])
