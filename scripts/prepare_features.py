"""Build the unified feature frames for the streamflow benchmark.

This script produces SIX feature parquets plus a split-boundary table from the
canonical discharge parquet and the two reanalysis parquets:

    data/processed/features/context_w14_h3.parquet
    data/processed/features/weather_w14_h3.parquet
    data/processed/features/hydro_w14_h3.parquet
    data/processed/features/context_w30_h3.parquet
    data/processed/features/weather_w30_h3.parquet
    data/processed/features/hydro_w30_h3.parquet
    data/processed/features/split_boundaries.csv

Design
------
All six frames share ONE master ``(unique_id, forecast_origin_ds)`` index and
ONE per-station chronological 70/15/15 split. The master index is defined by the
strictest stage: 20 stations (station 6144500 is dropped), forecast origins on or
after 1984-01-01, and rows for which all 30 discharge lags and the h1-h3 targets
are present. Every frame is built by attaching stage/window specific columns to
those exact rows, so the frames differ only in their columns, never in their rows
or split assignment.

Usage
-----
    .venv/bin/python scripts/prepare_features.py
"""

from __future__ import annotations

import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Constants — shared across all frames
# ---------------------------------------------------------------------------

HORIZONS = [1, 2, 3]
DROP_STATION = "6144500"          # 3-year record — excluded from the benchmark
START_DATE = pd.Timestamp("1984-01-01")
MASTER_LAGS = 30                  # strictest history requirement (defines the row set)
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15

# Lookback windows and their rolling-window sizes.
WINDOW_ROLLINGS: dict[int, list[int]] = {
    14: [3, 7, 14],
    30: [3, 7, 14, 21],
}

# 17 flow-context station IDs (other-station same-day and previous-day discharge).
FLOW_CONTEXT_IDS = [
    6142150, 6142200, 6142520, 6142551, 6142601,
    6142620, 6142640, 6142650, 6142660, 6142680,
    6144100, 6144150, 6144200, 6144300, 6144350,
    6144400, 6158100,
]
FLOW_CONTEXT_LAGS = [0, 1]

# 7 ERA5 weather variables (from data/processed/reanalysis_daily.parquet).
WEATHER_VARS = [
    "era5_precipitation_sum",
    "era5_rain_sum",
    "era5_snowfall_sum",
    "era5_precipitation_hours",
    "era5_temperature_2m_mean",
    "era5_temperature_2m_max",
    "era5_temperature_2m_min",
]

# 11 hydro variables (from data/processed/reanalysis_hydro_daily.parquet).
HYDRO_VARS = [
    "era5_shortwave_radiation_sum",
    "era5_wind_speed_10m_mean",
    "era5_et0_fao_evapotranspiration",
    "era5l_soil_temperature_0_to_7cm_mean",
    "era5l_soil_temperature_7_to_28cm_mean",
    "era5l_soil_temperature_28_to_100cm_mean",
    "era5l_soil_temperature_100_to_255cm_mean",
    "era5l_soil_moisture_0_to_7cm_mean",
    "era5l_soil_moisture_7_to_28cm_mean",
    "era5l_soil_moisture_28_to_100cm_mean",
    "era5l_soil_moisture_100_to_255cm_mean",
]

# Rolling aggregation per variable: 'sum' for fluxes/accumulations, 'mean' for states.
_FLUX_VARS = {
    "era5_precipitation_sum",
    "era5_rain_sum",
    "era5_snowfall_sum",
    "era5_precipitation_hours",
    "era5_shortwave_radiation_sum",
    "era5_et0_fao_evapotranspiration",
}


def _rolling_agg(var: str) -> str:
    return "sum" if var in _FLUX_VARS else "mean"


# ---------------------------------------------------------------------------
# Split assignment
# ---------------------------------------------------------------------------

def assign_split(master_df: pd.DataFrame) -> pd.Series:
    """Assign 'train'/'validation'/'test' per station (70/15/15 time-ordered)."""
    result = pd.Series("", index=master_df.index, dtype=object)
    for _uid, grp in master_df.groupby("unique_id", sort=False):
        idx = grp.sort_values("forecast_origin_ds").index
        n = len(idx)
        t_end = int(np.floor(n * TRAIN_FRAC))
        v_end = int(np.floor(n * (TRAIN_FRAC + VAL_FRAC)))
        result.loc[idx[:t_end]] = "train"
        result.loc[idx[t_end:v_end]] = "validation"
        result.loc[idx[v_end:]] = "test"
    return result


def build_split_boundaries(master_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize per-station split counts and boundary dates for the paper."""
    rows: list[dict] = []
    for uid, grp in master_df.groupby("unique_id", sort=True):
        grp = grp.sort_values("forecast_origin_ds")
        rec: dict = {"unique_id": uid, "n_total": len(grp)}
        for split in ("train", "validation", "test"):
            sub = grp.loc[grp["split"] == split, "forecast_origin_ds"]
            rec[f"n_{split}"] = int(len(sub))
            rec[f"{split}_start_ds"] = sub.min() if len(sub) else pd.NaT
            rec[f"{split}_end_ds"] = sub.max() if len(sub) else pd.NaT
        rows.append(rec)
    return pd.DataFrame(rows).sort_values("unique_id").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Discharge feature construction (defines the master index)
# ---------------------------------------------------------------------------

def build_discharge_master(discharge_df: pd.DataFrame) -> pd.DataFrame:
    """Build discharge lags/targets and restrict to the master row set + split.

    The returned frame holds the identifying columns, lags 1..30, current_y, the
    h1-h3 targets and their dates, and the shared split label. Window-specific
    columns are derived from this frame later.
    """
    df = discharge_df.copy()
    df["unique_id"] = df["unique_id"].astype(str)
    df = df[df["unique_id"] != DROP_STATION]
    df = df.sort_values(["unique_id", "ds"]).reset_index(drop=True)

    g = df.groupby("unique_id")["y"]
    for k in range(1, MASTER_LAGS + 1):
        df[f"lag_{k}"] = g.shift(k)
    df["current_y"] = df["y"]

    g_all = df.groupby("unique_id")
    for h in HORIZONS:
        df[f"target_h{h}"] = g_all["y"].shift(-h).values
        df[f"target_h{h}_ds"] = g_all["ds"].shift(-h).values

    df["forecast_origin_ds"] = df["ds"]

    lag_cols = [f"lag_{k}" for k in range(1, MASTER_LAGS + 1)]
    tgt_cols = [f"target_h{h}" for h in HORIZONS]
    n_before = len(df)
    df = df.dropna(subset=lag_cols + tgt_cols).copy()
    df = df[df["forecast_origin_ds"] >= START_DATE].copy()
    df = df.reset_index(drop=True)
    print(f"  master rows: {n_before:,} -> {len(df):,} after 30-lag/target dropna + {START_DATE.date()} filter")

    df["split"] = assign_split(df)
    counts = {s: int((df["split"] == s).sum()) for s in ("train", "validation", "test")}
    print(f"  master split: {counts}  stations={df['unique_id'].nunique()}")
    return df


def build_context_frame(master_df: pd.DataFrame, window: int, flow_context: pd.DataFrame) -> pd.DataFrame:
    """Assemble the context frame for one window from the master discharge frame."""
    keep_lags = [f"lag_{k}" for k in range(1, window + 1)]
    id_cols = ["unique_id", "ds", "y", "current_y", "forecast_origin_ds", "split"]
    tgt_cols = [f"target_h{h}" for h in HORIZONS] + [f"target_h{h}_ds" for h in HORIZONS]
    df = master_df.loc[:, id_cols + keep_lags + tgt_cols].copy()

    # Window stats over lags 1..window.
    lag_matrix = np.column_stack([df[c].to_numpy() for c in keep_lags])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        df["lag_mean"] = np.nanmean(lag_matrix, axis=1)
        df["lag_std"] = np.nanstd(lag_matrix, axis=1)
        df["lag_min"] = np.nanmin(lag_matrix, axis=1)
        df["lag_max"] = np.nanmax(lag_matrix, axis=1)

    # Deltas between successive lags.
    for k in range(1, window):
        df[f"delta_{k}"] = df[f"lag_{k}"] - df[f"lag_{k + 1}"]

    # Flow-context is keyed by origin date only (other stations' discharge is
    # shared across all target stations at a given ds).
    df = df.merge(flow_context, on="ds", how="left")
    return df


def build_flow_context(discharge_df: pd.DataFrame) -> pd.DataFrame:
    """Build other-station same-day/previous-day discharge, keyed by [unique_id, ds]."""
    disch = discharge_df.copy()
    disch["unique_id"] = disch["unique_id"].astype(str)
    ctx_wide = disch.pivot_table(index="ds", columns="unique_id", values="y", aggfunc="first")
    ctx_wide.columns = [str(c) for c in ctx_wide.columns]

    fc_cols: dict[str, pd.Series] = {}
    for sid in FLOW_CONTEXT_IDS:
        col = str(sid)
        for lag in FLOW_CONTEXT_LAGS:
            if col in ctx_wide.columns:
                fc_cols[f"flow_context_{sid}_lag_{lag}"] = ctx_wide[col].shift(lag)
            else:
                fc_cols[f"flow_context_{sid}_lag_{lag}"] = pd.Series(np.nan, index=ctx_wide.index)
    fc_df = pd.DataFrame(fc_cols).reset_index()  # ds + flow_context columns
    # The flow-context columns carry the ds of the *forecast origin* row they attach to.
    return fc_df


# ---------------------------------------------------------------------------
# ERA5 / hydro reanalysis lag + rolling construction
# ---------------------------------------------------------------------------

def build_reanalysis_features(
    reanalysis_df: pd.DataFrame,
    variables: list[str],
    *,
    lags: list[int],
    rolling_windows: list[int],
) -> pd.DataFrame:
    """Build raw/lag/rolling columns for the given variables, keyed by [unique_id, ds].

    Rolling features use ``shift(1)`` before aggregating, so window w covers days
    t-1 ... t-w and never includes the origin day t (which is already available as
    the lag-0 alias). Fluxes are summed; state variables are averaged.
    """
    df = reanalysis_df.copy()
    df["unique_id"] = df["unique_id"].astype(str)
    df = df.sort_values(["unique_id", "ds"]).reset_index(drop=True)

    parts: list[pd.DataFrame] = []
    for _uid, grp in df.groupby("unique_id", sort=False):
        cols: dict[str, np.ndarray] = {
            "unique_id": grp["unique_id"].to_numpy(),
            "ds": grp["ds"].to_numpy(),
        }
        for var in variables:
            if var not in grp.columns:
                continue
            s = grp[var].reset_index(drop=True)
            cols[var] = s.to_numpy()  # raw value (lag-0 alias)
            for lag in lags:
                cols[f"{var}_lag_{lag}"] = s.shift(lag).to_numpy()
            agg = _rolling_agg(var)
            shifted = s.shift(1)
            for w in rolling_windows:
                roller = shifted.rolling(w, min_periods=1)
                rolled = roller.sum() if agg == "sum" else roller.mean()
                cols[f"{var}_{agg}_{w}"] = rolled.to_numpy()
        parts.append(pd.DataFrame(cols))
    return pd.concat(parts, ignore_index=True)


def _impute_hydro_columns(frame: pd.DataFrame, new_cols: list[str]) -> pd.DataFrame:
    """Fill NaNs in the newly added hydro columns with the per-column TRAIN mean.

    Station 6158100 has no ERA5-Land coverage, so its soil columns are entirely
    missing; filling them with the cross-station training-split mean keeps the
    frame dense. This substitution is physically approximate and is disclosed.
    """
    train_mask = frame["split"] == "train"
    n_imputed = 0
    for col in new_cols:
        if frame[col].isna().any():
            col_mean = frame.loc[train_mask, col].mean()
            n_filled = int(frame[col].isna().sum())
            frame[col] = frame[col].fillna(col_mean)
            n_imputed += n_filled
    if n_imputed:
        print(f"    imputed {n_imputed:,} NaN hydro values with per-column train-split mean")
    return frame


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------

def assert_no_future_columns(frame: pd.DataFrame, name: str) -> None:
    leaky = [c for c in frame.columns if re.search(r"_future_h\d+", c)]
    if leaky:
        raise AssertionError(f"Frame '{name}' contains forbidden future columns: {leaky}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    data_dir = PROJECT_ROOT / "data" / "processed"
    out_dir = data_dir / "features"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading source parquets …")
    discharge = pd.read_parquet(data_dir / "discharge_daily.parquet")
    discharge["unique_id"] = discharge["unique_id"].astype(str)
    discharge_20 = discharge[discharge["unique_id"] != DROP_STATION].copy()
    weather_src = pd.read_parquet(data_dir / "reanalysis_daily.parquet")
    hydro_src = pd.read_parquet(data_dir / "reanalysis_hydro_daily.parquet")

    print("\nBuilding master discharge frame …")
    master = build_discharge_master(discharge)

    print("\nWriting split boundaries …")
    boundaries = build_split_boundaries(master)
    boundaries_path = out_dir / "split_boundaries.csv"
    boundaries.to_csv(boundaries_path, index=False)
    print(f"  saved {boundaries_path.name}  ({len(boundaries)} stations)")

    print("\nBuilding flow-context …")
    flow_context = build_flow_context(discharge_20)

    level_name = {"context": "context", "weather": "weather", "hydro": "hydro"}

    for window, rolling_windows in WINDOW_ROLLINGS.items():
        print(f"\n=== Window w{window} (rollings {rolling_windows}) ===")

        # --- context -----------------------------------------------------
        context = build_context_frame(master, window, flow_context)
        assert_no_future_columns(context, f"context_w{window}")
        ctx_path = out_dir / f"{level_name['context']}_w{window}_h3.parquet"
        context.to_parquet(ctx_path, index=False)
        print(f"  context: {len(context):,} rows x {len(context.columns)} cols -> {ctx_path.name}")

        # --- weather -----------------------------------------------------
        era_wide = build_reanalysis_features(
            weather_src, WEATHER_VARS, lags=list(range(window + 1)), rolling_windows=rolling_windows
        )
        weather = context.merge(era_wide, on=["unique_id", "ds"], how="left")
        assert_no_future_columns(weather, f"weather_w{window}")
        wth_path = out_dir / f"{level_name['weather']}_w{window}_h3.parquet"
        weather.to_parquet(wth_path, index=False)
        print(f"  weather: {len(weather):,} rows x {len(weather.columns)} cols -> {wth_path.name}")

        # --- hydro (built on the same-window weather frame) --------------
        hydro_wide = build_reanalysis_features(
            hydro_src, HYDRO_VARS, lags=list(range(window + 1)), rolling_windows=rolling_windows
        )
        new_hydro_cols = [c for c in hydro_wide.columns if c not in ("unique_id", "ds")]
        hydro = weather.merge(hydro_wide, on=["unique_id", "ds"], how="left")
        hydro = _impute_hydro_columns(hydro, new_hydro_cols)
        assert_no_future_columns(hydro, f"hydro_w{window}")
        # No hydro column may remain NaN after imputation.
        residual = hydro[new_hydro_cols].isna().any().any()
        if residual:
            raise AssertionError(f"hydro_w{window} still has NaN hydro columns after imputation")
        hyd_path = out_dir / f"{level_name['hydro']}_w{window}_h3.parquet"
        hydro.to_parquet(hyd_path, index=False)
        print(f"  hydro:   {len(hydro):,} rows x {len(hydro.columns)} cols -> {hyd_path.name}")

    print("\nDone. All six frames share the master index and split.")


if __name__ == "__main__":
    main()
