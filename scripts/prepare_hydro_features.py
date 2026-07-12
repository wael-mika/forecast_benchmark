"""Augment the weather feature frame with hydro reanalysis variables.

This script merges the daily hydro reanalysis parquet into an existing weather
feature frame using [unique_id, forecast_origin_ds] and writes a new
hydro-weather feature parquet with lagged and rolling hydro columns.

Use this script after the base weather feature parquet and
data/processed/reanalysis_hydro_daily.parquet both exist. The exact input and
output paths are controlled by BASE_PARQUET, HYDRO_PARQUET, and OUT_PARQUET
near the top of the file.

Inputs
------
    BASE_PARQUET
        Existing weather feature frame.
    HYDRO_PARQUET
        Daily hydro reanalysis data assembled from the Open-Meteo caches.

Outputs
-------
    OUT_PARQUET
        Weather feature frame plus hydro lag and rolling-window columns.

Usage
-----
    .venv/Scripts/python scripts/prepare_hydro_features.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

BASE_PARQUET = PROJECT_ROOT / "data/processed/xgboost/features_weather_plus_w14_h3.parquet"
HYDRO_PARQUET = PROJECT_ROOT / "data/processed/reanalysis_hydro_daily.parquet"
OUT_PARQUET = PROJECT_ROOT / "data/processed/xgboost/features_hydro_weather_w30_h3.parquet"

LAGS = list(range(31))   # 0 … 30
WINDOWS = [3, 7, 14, 21]

# Rolling aggregation per variable (sum for flux/accumulation, mean for states)
_VAR_AGG: dict[str, str] = {
    "era5_shortwave_radiation_sum": "sum",
    "era5_wind_speed_10m_mean": "mean",
    "era5_et0_fao_evapotranspiration": "sum",
    "era5l_soil_temperature_0_to_7cm_mean": "mean",
    "era5l_soil_temperature_7_to_28cm_mean": "mean",
    "era5l_soil_temperature_28_to_100cm_mean": "mean",
    "era5l_soil_temperature_100_to_255cm_mean": "mean",
    "era5l_soil_moisture_0_to_7cm_mean": "mean",
    "era5l_soil_moisture_7_to_28cm_mean": "mean",
    "era5l_soil_moisture_28_to_100cm_mean": "mean",
    "era5l_soil_moisture_100_to_255cm_mean": "mean",
}


def _build_hydro_lag_frame(hydro_df: pd.DataFrame) -> pd.DataFrame:
    """Create lagged and rolling hydro features keyed by station and date."""
    hydro_vars = [c for c in hydro_df.columns if c not in ("unique_id", "ds")]
    hydro_df = hydro_df.sort_values(["unique_id", "ds"]).copy()

    new_cols: dict[str, pd.Series] = {}

    for var in hydro_vars:
        agg = _VAR_AGG.get(var, "mean")
        grp = hydro_df.groupby("unique_id")[var]

        # Raw value (alias for lag_0 – matches weather feature frame convention)
        new_cols[var] = hydro_df[var]

        for lag in LAGS:
            new_cols[f"{var}_lag_{lag}"] = grp.shift(lag)

        for w in WINDOWS:
            if agg == "sum":
                new_cols[f"{var}_sum_{w}"] = grp.transform(
                    lambda x, _w=w: x.rolling(_w, min_periods=1).sum()
                )
            else:
                new_cols[f"{var}_mean_{w}"] = grp.transform(
                    lambda x, _w=w: x.rolling(_w, min_periods=1).mean()
                )

    lag_frame = pd.DataFrame(new_cols, index=hydro_df.index)
    lag_frame.insert(0, "unique_id", hydro_df["unique_id"].values)
    lag_frame.insert(1, "ds", hydro_df["ds"].values)
    return lag_frame


def main() -> None:
    """Merge hydro reanalysis columns into the configured weather feature frame."""
    print(f"Loading base feature frame from:\n  {BASE_PARQUET}")
    base_df = pd.read_parquet(BASE_PARQUET)
    print(f"  {base_df.shape[0]:,} rows × {base_df.shape[1]} columns")

    print(f"\nLoading hydro reanalysis from:\n  {HYDRO_PARQUET}")
    hydro_df = pd.read_parquet(HYDRO_PARQUET)
    hydro_vars = [c for c in hydro_df.columns if c not in ("unique_id", "ds")]
    print(f"  {hydro_df.shape[0]:,} rows, {len(hydro_vars)} hydro variables: {hydro_vars}")

    # Align types
    hydro_df["unique_id"] = hydro_df["unique_id"].astype(str)
    hydro_df["ds"] = pd.to_datetime(hydro_df["ds"])
    base_df["unique_id"] = base_df["unique_id"].astype(str)
    base_df["forecast_origin_ds"] = pd.to_datetime(base_df["forecast_origin_ds"])

    print("\nBuilding lag / rolling-window features …")
    hydro_lag_frame = _build_hydro_lag_frame(hydro_df)
    n_hydro_cols = hydro_lag_frame.shape[1] - 2  # minus unique_id and ds
    print(f"  {n_hydro_cols} new feature columns per row")

    print("\nMerging with base feature frame on [unique_id, forecast_origin_ds] …")
    merged = base_df.merge(
        hydro_lag_frame,
        left_on=["unique_id", "forecast_origin_ds"],
        right_on=["unique_id", "ds"],
        how="left",
        suffixes=("", "_hydro"),
    )
    # Drop the duplicate 'ds' column that came from the right side
    if "ds_hydro" in merged.columns:
        merged = merged.drop(columns=["ds_hydro"])

    added = merged.shape[1] - base_df.shape[1]
    print(f"  Added {added} columns. Total: {merged.shape[1]}")

    # Restrict to 1984+ so every hydro feature has real observed values.
    # Pre-1984 rows have no ERA5-hydro coverage; keeping them would impute NaN
    # as the training-set mean, which is physically meaningless.
    hydro_start = pd.Timestamp("1984-01-01")
    before = len(merged)
    merged = merged[merged["forecast_origin_ds"] >= hydro_start].copy()
    after = len(merged)
    print(f"\nFiltered to forecast_origin_ds >= {hydro_start.date()}: "
          f"{before:,} -> {after:,} rows ({before - after:,} pre-1984 rows dropped)")
    print("Split distribution after filter:")
    print(f"  {merged['split'].value_counts().to_dict()}")

    # Impute any remaining NaN with the per-column training-set mean.
    # The 6% residual NaN comes from station 6158100 which has no ERA5-Land
    # coverage; using the mean of the other stations is physically reasonable.
    hydro_feature_cols = [
        c for c in merged.columns
        if c not in base_df.columns or c == "ds"  # new hydro columns only
    ]
    hydro_feature_cols = [c for c in merged.columns if c not in set(base_df.columns)]
    train_mask = merged["split"] == "train"
    n_imputed = 0
    for col in hydro_feature_cols:
        if merged[col].isna().any():
            col_mean = merged.loc[train_mask, col].mean()
            n_filled = merged[col].isna().sum()
            merged[col] = merged[col].fillna(col_mean)
            n_imputed += n_filled
    if n_imputed:
        print(f"\nImputed {n_imputed:,} NaN values with per-column training-set mean.")
    else:
        print("\nNo NaN values remain after 1984 filter.")

    # Final NaN check
    print("Residual NaN check (should all be 0.0%):")
    for var in hydro_vars:
        col = f"{var}_lag_0"
        if col in merged.columns:
            pct_nan = merged[col].isna().mean() * 100
            print(f"  {var}: {pct_nan:.1f}% NaN")

    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(OUT_PARQUET, index=False)
    print(f"\nSaved: {OUT_PARQUET}")
    print(f"Final shape: {merged.shape[0]:,} rows x {merged.shape[1]} columns")


if __name__ == "__main__":
    main()
