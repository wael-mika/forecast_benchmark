"""Download public ERA5-style daily reanalysis features for Slovakia stations."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.reanalysis import build_station_request_specs, download_open_meteo_reanalysis
from src.utils.config import load_yaml_config
from src.utils.logging import get_logger


def main(argv: list[str] | None = None) -> None:
    """Download public reanalysis data and save one processed parquet dataset."""
    active_argv = argv or sys.argv
    logger = get_logger("download_reanalysis_data")

    config_path = PROJECT_ROOT / "configs" / "reanalysis.yaml"
    if len(active_argv) > 1:
        config_path = (PROJECT_ROOT / active_argv[1]).resolve()
    config = load_yaml_config(config_path)

    canonical_df = pd.read_parquet(PROJECT_ROOT / config["canonical_data_path"])
    station_specs = build_station_request_specs(
        canonical_df,
        PROJECT_ROOT / config["station_metadata_path"],
        min_start_date=str(config.get("min_start_date", "1940-01-01")),
    )
    reanalysis_df = download_open_meteo_reanalysis(
        station_specs,
        raw_output_dir=PROJECT_ROOT / config["raw_output_dir"],
        processed_output_path=PROJECT_ROOT / config["processed_output_path"],
        metadata_output_path=PROJECT_ROOT / config["metadata_output_path"],
    )

    logger.info(
        "Downloaded reanalysis features for %s stations and wrote %s rows to %s",
        len(station_specs),
        len(reanalysis_df),
        PROJECT_ROOT / config["processed_output_path"],
    )


if __name__ == "__main__":
    main()
