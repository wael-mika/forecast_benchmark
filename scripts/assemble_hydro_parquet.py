"""Assemble cached hydro reanalysis JSON files into one daily parquet.

This script turns the raw JSON responses downloaded under
data/raw/reanalysis_open_meteo_hydro/ into one long-format parquet keyed by
[unique_id, ds]. It is the final assembly step after the hydro download
scripts finish.

Use this script when the raw cache folders already contain the per-station
Open-Meteo responses and you want the processed file used by feature
engineering.

Inputs
------
    data/raw/reanalysis_open_meteo_hydro/era5_seamless/*.json
    data/raw/reanalysis_open_meteo_hydro/era5_land/*.json

Outputs
-------
    data/processed/reanalysis_hydro_daily.parquet

Usage
-----
    .venv/Scripts/python scripts/assemble_hydro_parquet.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "reanalysis_open_meteo_hydro"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "reanalysis_hydro_daily.parquet"


def _daily_json_to_frame(path: Path, prefix: str) -> pd.DataFrame:
    with path.open(encoding="utf-8") as fh:
        p = json.load(fh)
    daily = p.get("daily", {})
    frame = pd.DataFrame({
        "unique_id": path.stem,
        "ds": pd.to_datetime(daily["time"]),
    })
    for k, v in daily.items():
        if k != "time":
            frame[f"{prefix}{k}"] = v
    return frame


def main() -> None:
    """Combine cached hydro JSON payloads into one processed parquet file."""
    # ------------------------------------------------------------------
    # 1. ERA5-seamless daily (all 21 stations)
    # ------------------------------------------------------------------
    seamless_frames = []
    for path in sorted((RAW_DIR / "era5_seamless").glob("*.json")):
        seamless_frames.append(_daily_json_to_frame(path, prefix="era5_"))
    seamless_df = pd.concat(seamless_frames, ignore_index=True)
    print(f"era5_seamless: {seamless_df['unique_id'].nunique()} stations, {len(seamless_df):,} rows")
    print(f"  columns: {[c for c in seamless_df.columns if c not in ('unique_id','ds')]}")

    # ------------------------------------------------------------------
    # 2. ERA5-Land daily (soil temp + soil moisture, up to 20 stations)
    # ------------------------------------------------------------------
    era5l_frames = []
    for path in sorted((RAW_DIR / "era5_land").glob("*.json")):
        era5l_frames.append(_daily_json_to_frame(path, prefix="era5l_"))

    if era5l_frames:
        era5l_df = pd.concat(era5l_frames, ignore_index=True)
        print(f"era5_land:     {era5l_df['unique_id'].nunique()} stations, {len(era5l_df):,} rows")
        print(f"  columns: {[c for c in era5l_df.columns if c not in ('unique_id','ds')]}")
    else:
        era5l_df = None
        print("era5_land: no data found — skipping.")

    # ------------------------------------------------------------------
    # 3. Merge
    # ------------------------------------------------------------------
    if era5l_df is not None:
        result = seamless_df.merge(era5l_df, on=["unique_id", "ds"], how="left")
    else:
        result = seamless_df

    result = result.sort_values(["unique_id", "ds"], kind="stable").reset_index(drop=True)

    # ------------------------------------------------------------------
    # 4. Save
    # ------------------------------------------------------------------
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(OUT_PATH, index=False)
    print(f"\nSaved {len(result):,} rows × {result.shape[1]} columns → {OUT_PATH}")
    hydro_cols = [c for c in result.columns if c not in ("unique_id", "ds")]
    print(f"Hydro columns ({len(hydro_cols)}): {hydro_cols}")

    # Coverage report
    missing_era5l = set(seamless_df["unique_id"].unique()) - set((era5l_df["unique_id"].unique() if era5l_df is not None else []))
    if missing_era5l:
        print(f"\nNote: era5_land data missing for {len(missing_era5l)} station(s): {sorted(missing_era5l)}")
        print("  Those stations will have NaN for soil temp/moisture columns.")


if __name__ == "__main__":
    main()
