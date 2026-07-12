"""Build the canonical discharge dataset from raw GRDC files.

This script is the first data-preparation step in the repository. It loads
configs/data.yaml, reads the raw station files, normalizes the station/date/
value columns, and writes the processed canonical parquet used by the later
feature-engineering scripts.

Use this script after placing the raw GRDC exports in the directory referenced
by configs/data.yaml.

Usage
-----
    .venv/Scripts/python scripts/prepare_data.py

Inputs
------
    raw_data_dir from configs/data.yaml
    date/value/station column candidates from the same config

Outputs
-------
    processed_data_path from configs/data.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_grdc import ingest_grdc_directory
from src.utils.config import load_yaml_config
from src.utils.logging import get_logger


def main() -> None:
    """Load the data config and write the canonical processed dataset."""
    logger = get_logger("prepare_data")
    config = load_yaml_config(PROJECT_ROOT / "configs" / "data.yaml")

    raw_data_dir = PROJECT_ROOT / config["raw_data_dir"]
    processed_data_path = PROJECT_ROOT / config["processed_data_path"]

    df = ingest_grdc_directory(
        raw_data_dir=raw_data_dir,
        processed_data_path=processed_data_path,
        date_column_candidates=config["date_column_candidates"],
        value_column_candidates=config["value_column_candidates"],
        station_id_column_candidates=config["station_id_column_candidates"],
        raw_file_patterns=config.get("raw_file_patterns"),
        logger=logger,
    )
    logger.info("Wrote %s rows from %s stations to %s", len(df), df["unique_id"].nunique(), processed_data_path)


if __name__ == "__main__":
    main()
