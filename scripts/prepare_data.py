"""Build the canonical discharge parquet from raw GRDC station files.

This is the first data-preparation step. It reads the daily GRDC exports listed
by ``configs/data.yaml`` (``raw_file_patterns`` under ``raw_data_dir``), parses
each station's mean-daily-discharge series, reindexes it onto a gap-free daily
calendar (so later ``shift(k)`` operations mean exactly k calendar days), drops
stations shorter than ``min_station_length`` valid observations, and writes the
canonical ``data/processed/discharge_daily.parquet`` consumed by
``scripts/prepare_features.py``.

The GRDC daily file format is a ``;``-delimited table preceded by ``#`` comment
lines; the station number is taken from the ``# GRDC-No.:`` header (falling back
to the file name) and missing values (-999.0) are converted to NaN.

Usage
-----
    .venv/bin/python scripts/prepare_data.py

Note
----
    The processed parquet is distributed with the repository as the authoritative
    artifact. Re-running this script requires the raw GRDC exports, which are not
    redistributed (see the README data-availability section).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_yaml_config
from src.utils.logging import get_logger

MISSING_VALUE = -999.0
_GRDC_NO_RE = re.compile(r"#\s*GRDC-No\.?:\s*(\d+)", re.IGNORECASE)


def _station_id_from_file(path: Path, lines: list[str]) -> str:
    """Extract the station identifier from the GRDC header or the file name."""
    for line in lines:
        if not line.startswith("#"):
            break
        match = _GRDC_NO_RE.search(line)
        if match:
            return match.group(1)
    return path.name.split("_")[0]


def _parse_grdc_daily(path: Path) -> pd.DataFrame:
    """Parse one GRDC daily-discharge file into a (unique_id, ds, y) frame."""
    lines = path.read_text(encoding="latin-1").splitlines()
    station_id = _station_id_from_file(path, lines)

    records: list[tuple[pd.Timestamp, float]] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.lower().startswith("yyyy"):  # in-table header row
            continue
        parts = stripped.split(";")
        if len(parts) < 3:
            continue
        date_token = parts[0].strip()
        value_token = parts[-1].strip()
        try:
            ds = pd.Timestamp(date_token)
            value = float(value_token)
        except (ValueError, TypeError):
            continue
        if np.isclose(value, MISSING_VALUE):
            value = np.nan
        records.append((ds, value))

    if not records:
        return pd.DataFrame(columns=["unique_id", "ds", "y"])

    frame = pd.DataFrame(records, columns=["ds", "y"]).drop_duplicates("ds")
    frame = frame.sort_values("ds")
    # Reindex onto a continuous daily calendar so shifts are calendar-correct.
    full_index = pd.date_range(frame["ds"].min(), frame["ds"].max(), freq="D")
    frame = frame.set_index("ds").reindex(full_index).rename_axis("ds").reset_index()
    frame.insert(0, "unique_id", str(station_id))
    return frame


def main() -> None:
    """Load the data config and write the canonical processed discharge parquet."""
    logger = get_logger("prepare_data")
    config = load_yaml_config(PROJECT_ROOT / "configs" / "data.yaml")

    raw_data_dir = PROJECT_ROOT / config["raw_data_dir"]
    processed_data_path = PROJECT_ROOT / config["processed_data_path"]
    patterns = config.get("raw_file_patterns", ["**/*_Q_Day*.txt"])
    min_station_length = int(config.get("min_station_length", 365))

    if not raw_data_dir.exists():
        raise FileNotFoundError(
            f"Raw data directory {raw_data_dir} not found. Place the GRDC exports there "
            "(see the README data-availability section) or use the distributed "
            "data/processed/discharge_daily.parquet directly."
        )

    files: list[Path] = []
    for pattern in patterns:
        files.extend(sorted(raw_data_dir.glob(pattern)))
    files = sorted(set(files))
    logger.info("Found %d raw daily files under %s", len(files), raw_data_dir)

    frames: list[pd.DataFrame] = []
    for path in files:
        station_frame = _parse_grdc_daily(path)
        if station_frame.empty:
            continue
        valid_count = int(station_frame["y"].notna().sum())
        if valid_count < min_station_length:
            logger.info("Skipping %s (%d valid obs < %d)", path.name, valid_count, min_station_length)
            continue
        frames.append(station_frame)

    if not frames:
        raise RuntimeError("No station files produced usable data; check raw_data_dir and patterns.")

    canonical = pd.concat(frames, ignore_index=True)
    canonical["unique_id"] = canonical["unique_id"].astype("string")
    canonical["ds"] = pd.to_datetime(canonical["ds"])
    canonical = canonical.sort_values(["unique_id", "ds"]).reset_index(drop=True)

    processed_data_path.parent.mkdir(parents=True, exist_ok=True)
    canonical.to_parquet(processed_data_path, index=False)
    logger.info(
        "Wrote %d rows from %d stations to %s",
        len(canonical), canonical["unique_id"].nunique(), processed_data_path,
    )


if __name__ == "__main__":
    main()
