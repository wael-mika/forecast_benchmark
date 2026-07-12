"""Build 30-day feature frames for the context and weather benchmark levels.

This script creates the w30/h3 tabular datasets used by the advanced models
and XGBoost runs. It starts from the canonical discharge parquet and, for the
weather variant, joins daily reanalysis features before generating lagged
columns, forecast targets, and split labels.

Use this script after both of these files exist:
- the canonical discharge parquet, and
- the base reanalysis parquet.

Outputs
-------
    data/processed/xgboost/features_context_w30_h3.parquet
    data/processed/xgboost/features_weather_plus_w30_h3.parquet

Usage
-----
    .venv/Scripts/python scripts/prepare_features_w30.py

Notes
-----
    The generated columns are designed to stay compatible with the training
    code used for the advanced neural and XGBoost experiments.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Constants — must match the neural model configs
# ---------------------------------------------------------------------------

WINDOW      = 30
HORIZONS    = [1, 2, 3]
TRAIN_FRAC  = 0.70
VAL_FRAC    = 0.15
START_DATE  = "1984-01-01"

# 17 flow-context station IDs (from configs/advanced_data_context.yaml)
FLOW_CONTEXT_IDS = [
    6142150, 6142200, 6142520, 6142551, 6142601,
    6142620, 6142640, 6142650, 6142660, 6142680,
    6144100, 6144150, 6144200, 6144300, 6144350,
    6144400, 6158100,
]
FLOW_CONTEXT_LAGS = [0, 1]

# ERA5 variables (from configs/advanced_data_weather.yaml)
ERA5_VARS = [
    "era5_precipitation_sum",
    "era5_rain_sum",
    "era5_snowfall_sum",
    "era5_temperature_2m_mean",
    "era5_temperature_2m_max",
    "era5_temperature_2m_min",
    "era5_precipitation_hours",
]
ERA5_LAGS    = list(range(31))   # 0 … 30
ERA5_WINDOWS = [3, 7, 14, 21]    # rolling-mean windows


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assign_split(df: pd.DataFrame) -> pd.Series:
    """Assign 'train'/'validation'/'test' per station (70/15/15 time-sorted)."""
    result = pd.Series("", index=df.index, dtype=str)
    for uid, grp in df.groupby("unique_id", sort=False):
        idx = grp.sort_values("split_reference_ds").index
        n = len(idx)
        t_end = int(np.floor(n * TRAIN_FRAC))
        v_end = int(np.floor(n * (TRAIN_FRAC + VAL_FRAC)))
        result.loc[idx[:t_end]]       = "train"
        result.loc[idx[t_end:v_end]]  = "validation"
        result.loc[idx[v_end:]]       = "test"
    return result


# ---------------------------------------------------------------------------
# Build functions
# ---------------------------------------------------------------------------

def build_context_frame(canonical_df: pd.DataFrame) -> pd.DataFrame:
    """Build the context feature frame (discharge lags + flow context)."""
    print("Building context w30 features …")
    df = canonical_df.sort_values(["unique_id", "ds"]).reset_index(drop=True).copy()

    # ── Discharge lags ───────────────────────────────────────────────────────
    g = df.groupby("unique_id")["y"]
    for k in range(1, WINDOW + 1):
        df[f"lag_{k}"] = g.shift(k)
    df["current_y"] = df["y"]

    # ── Window stats (over lags 1..WINDOW) ──────────────────────────────────
    # Early rows (< WINDOW days history) will be all-NaN and dropped by dropna below
    lag_matrix = np.column_stack([df[f"lag_{k}"].values for k in range(1, WINDOW + 1)])
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        df["lag_mean"] = np.nanmean(lag_matrix, axis=1)
        df["lag_std"]  = np.nanstd(lag_matrix, axis=1)
        df["lag_min"]  = np.nanmin(lag_matrix, axis=1)
        df["lag_max"]  = np.nanmax(lag_matrix, axis=1)

    # ── Deltas ───────────────────────────────────────────────────────────────
    for k in range(1, WINDOW):
        df[f"delta_{k}"] = df[f"lag_{k}"] - df[f"lag_{k + 1}"]

    # ── Targets ──────────────────────────────────────────────────────────────
    g_all = df.groupby("unique_id")
    for h in HORIZONS:
        df[f"target_h{h}"]    = g_all["y"].shift(-h).values
        df[f"target_h{h}_ds"] = g_all["ds"].shift(-h).values

    df["forecast_origin_ds"] = df["ds"]
    df["split_reference_ds"] = df["ds"]

    # ── Flow context ─────────────────────────────────────────────────────────
    print(f"  Adding flow-context for {len(FLOW_CONTEXT_IDS)} stations …")
    # Pivot discharge to wide format (date × station)
    ctx_wide = canonical_df.pivot_table(
        index="ds", columns="unique_id", values="y", aggfunc="first"
    )
    ctx_wide.columns = [str(c) for c in ctx_wide.columns]

    fc_cols: dict[str, pd.Series] = {}
    for sid in FLOW_CONTEXT_IDS:
        col = str(sid)
        if col in ctx_wide.columns:
            for lag in FLOW_CONTEXT_LAGS:
                fc_cols[f"flow_context_{sid}_lag_{lag}"] = ctx_wide[col].shift(lag)
        else:
            for lag in FLOW_CONTEXT_LAGS:
                fc_cols[f"flow_context_{sid}_lag_{lag}"] = pd.Series(
                    np.nan, index=ctx_wide.index
                )

    fc_df = pd.DataFrame(fc_cols).reset_index()   # ds + flow_context columns
    df = df.merge(fc_df, on="ds", how="left")

    # ── Filter ───────────────────────────────────────────────────────────────
    lag_cols = [f"lag_{k}" for k in range(1, WINDOW + 1)]
    tgt_cols = [f"target_h{h}" for h in HORIZONS]
    n_before = len(df)
    df = df.dropna(subset=lag_cols + tgt_cols).copy()
    df = df[df["forecast_origin_ds"] >= pd.Timestamp(START_DATE)].copy()
    print(f"  Rows: {n_before:,} -> {len(df):,} after dropna + date filter")

    # ── Split ────────────────────────────────────────────────────────────────
    df["split"] = _assign_split(df)
    print(f"  Split: { {s: int((df['split']==s).sum()) for s in ['train','validation','test']} }")
    print(f"  Columns: {len(df.columns)}")
    return df.reset_index(drop=True)


def build_weather_frame(context_df: pd.DataFrame, reanalysis_df: pd.DataFrame) -> pd.DataFrame:
    """Extend context frame with ERA5 lags and rolling windows."""
    print("\nBuilding weather w30 features …")
    era = reanalysis_df.sort_values(["unique_id", "ds"]).reset_index(drop=True).copy()

    # Build ERA5 lag + rolling columns per station
    era_parts = []
    for uid, grp in era.groupby("unique_id", sort=False):
        row = {"unique_id": grp["unique_id"].values, "ds": grp["ds"].values}
        for var in ERA5_VARS:
            if var not in grp.columns:
                continue
            s = grp[var].reset_index(drop=True)
            row[var] = s.values                          # current (lag-0 alias, mirrors w14)
            for lag in ERA5_LAGS:
                row[f"{var}_lag_{lag}"] = s.shift(lag).values
            for win in ERA5_WINDOWS:
                row[f"{var}_roll_{win}"] = s.shift(1).rolling(win, min_periods=1).mean().values
        era_parts.append(pd.DataFrame(row))

    era_wide = pd.concat(era_parts, ignore_index=True)

    df = context_df.merge(era_wide, on=["unique_id", "ds"], how="left")
    print(f"  Rows: {len(df):,}  Columns: {len(df.columns)}")
    for sp in ["train", "validation", "test"]:
        print(f"  {sp}: {(df['split']==sp).sum():,}")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    canonical_path   = PROJECT_ROOT / "data/processed/discharge_daily.parquet"
    reanalysis_path  = PROJECT_ROOT / "data/processed/reanalysis_daily.parquet"
    out_ctx_path     = PROJECT_ROOT / "data/processed/xgboost/features_context_w30_h3.parquet"
    out_wth_path     = PROJECT_ROOT / "data/processed/xgboost/features_weather_plus_w30_h3.parquet"
    out_ctx_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading canonical data  …  {canonical_path}")
    canon = pd.read_parquet(canonical_path)
    print(f"  {len(canon):,} rows, {canon['unique_id'].nunique()} stations\n")

    ctx = build_context_frame(canon)
    ctx.to_parquet(out_ctx_path, index=False)
    print(f"\nSaved: {out_ctx_path.name}  ({len(ctx):,} rows × {len(ctx.columns)} cols)")

    print(f"\nLoading reanalysis data …  {reanalysis_path}")
    era = pd.read_parquet(reanalysis_path)
    print(f"  {len(era):,} rows")

    wth = build_weather_frame(ctx, era)
    wth.to_parquet(out_wth_path, index=False)
    print(f"\nSaved: {out_wth_path.name}  ({len(wth):,} rows × {len(wth.columns)} cols)")

    print("\nDone.")


if __name__ == "__main__":
    main()
