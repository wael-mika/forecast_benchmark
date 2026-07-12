from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.reanalysis import StationRequestSpec, build_reanalysis_frame_from_cache


def test_build_reanalysis_frame_from_cache_uses_cached_seamless_payloads(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    processed_path = tmp_path / "processed" / "reanalysis.parquet"
    metadata_path = tmp_path / "processed" / "metadata.csv"
    seamless_dir = raw_dir / "era5_seamless"
    seamless_dir.mkdir(parents=True)

    payload = {
        "latitude": 48.1,
        "longitude": 17.1,
        "elevation": 120.0,
        "daily": {
            "time": ["2020-01-01", "2020-01-02"],
            "temperature_2m_mean": [1.0, 2.0],
            "precipitation_sum": [0.5, 1.5],
        },
    }
    (seamless_dir / "station_a.json").write_text(json.dumps(payload), encoding="utf-8")

    station_specs = [
        StationRequestSpec(
            unique_id="station_a",
            latitude=48.1,
            longitude=17.1,
            start_date="2020-01-01",
            end_date="2020-01-02",
        )
    ]

    reanalysis_df = build_reanalysis_frame_from_cache(
        station_specs,
        raw_output_dir=raw_dir,
        processed_output_path=processed_path,
        metadata_output_path=metadata_path,
    )

    assert processed_path.exists()
    assert metadata_path.exists()
    assert reanalysis_df.columns.tolist() == ["ds", "unique_id", "era5_temperature_2m_mean", "era5_precipitation_sum"]
    assert reanalysis_df.loc[0, "unique_id"] == "station_a"
    assert reanalysis_df.loc[1, "era5_precipitation_sum"] == 1.5

    stored = pd.read_parquet(processed_path)
    assert len(stored) == 2
