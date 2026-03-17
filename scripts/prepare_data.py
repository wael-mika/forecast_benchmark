"""Run the Milestone 1 GRDC ingestion pipeline."""

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
    """Load config and prepare the processed canonical dataset."""
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
