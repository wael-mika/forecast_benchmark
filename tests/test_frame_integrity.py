"""Acceptance suite for the unified feature frames and the fixed models.

The frame checks mirror an empirical audit and must pass with zero mismatches:
exact lag/target alignment against the canonical discharge parquet, strict
rolling causality (t-1..t-w only), one shared row set / split / boundary across
all six frames, no leaky future columns, and ERA5 lag alignment. Model checks
cover the Mamba parallel-vs-sequential scan and the PatchTST / (bi)LSTM forward
shapes. Frame tests skip when the built frames are absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRAMES_DIR = PROJECT_ROOT / "data" / "processed" / "features"
DATA_DIR = PROJECT_ROOT / "data" / "processed"

FRAME_STEMS = ["context", "weather", "hydro"]
WINDOWS = [14, 30]
# Rolling-window sizes per lookback window (must match scripts/prepare_features.py).
ROLLING_WINDOWS = {14: [3, 7, 14], 30: [3, 7, 14, 21]}
ALL_FRAMES = [f"{stem}_w{w}_h3" for w in WINDOWS for stem in FRAME_STEMS]
HORIZONS = [1, 2, 3]
DROPPED_STATION = "6144500"


def _frame_path(name: str) -> Path:
    return FRAMES_DIR / f"{name}.parquet"


def _require_frames() -> None:
    missing = [name for name in ALL_FRAMES if not _frame_path(name).exists()]
    if missing:
        pytest.skip(f"feature frames not built: {missing}")


def _load(name: str) -> pd.DataFrame:
    return pd.read_parquet(_frame_path(name))


def _window_of(name: str) -> int:
    return int(name.split("_w")[1].split("_")[0])


def _discharge_lookup() -> dict[tuple[str, pd.Timestamp], float]:
    if not (DATA_DIR / "discharge_daily.parquet").exists():
        pytest.skip("discharge_daily.parquet not present")
    d = pd.read_parquet(DATA_DIR / "discharge_daily.parquet")
    d["unique_id"] = d["unique_id"].astype(str)
    return {(u, pd.Timestamp(t)): v for u, t, v in zip(d["unique_id"], d["ds"], d["y"])}


# ---------------------------------------------------------------------------
# Group 1: lag / target alignment vs discharge_daily.parquet
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ALL_FRAMES)
def test_lag_and_target_alignment(name: str) -> None:
    _require_frames()
    lut = _discharge_lookup()
    df = _load(name)
    window = _window_of(name)
    sample = df.sample(min(2000, len(df)), random_state=7)
    lags_to_check = sorted({1, window // 2, window})
    mism = 0
    for _, row in sample.iterrows():
        uid = str(row["unique_id"])
        t = pd.Timestamp(row["forecast_origin_ds"])
        for k in lags_to_check:
            if not np.isclose(row[f"lag_{k}"], lut[(uid, t - pd.Timedelta(days=k))]):
                mism += 1
        assert np.isclose(row["current_y"], lut[(uid, t)])
        for h in HORIZONS:
            if not np.isclose(row[f"target_h{h}"], lut[(uid, t + pd.Timedelta(days=h))]):
                mism += 1
            assert pd.Timestamp(row[f"target_h{h}_ds"]) == t + pd.Timedelta(days=h)
    assert mism == 0, f"{name}: {mism} lag/target mismatches"


# ---------------------------------------------------------------------------
# Group 2: rolling causality (t-1..t-w only)
# ---------------------------------------------------------------------------

def _era_pivot(var: str) -> pd.DataFrame:
    era = pd.read_parquet(DATA_DIR / "reanalysis_daily.parquet")
    era["unique_id"] = era["unique_id"].astype(str)
    return era.pivot_table(index="ds", columns="unique_id", values=var, aggfunc="first")


@pytest.mark.parametrize("name", ["weather_w14_h3", "weather_w30_h3"])
def test_rolling_causality(name: str) -> None:
    _require_frames()
    if not (DATA_DIR / "reanalysis_daily.parquet").exists():
        pytest.skip("reanalysis_daily.parquet not present")
    df = _load(name)
    window = _window_of(name)
    roll = max(ROLLING_WINDOWS[window])  # largest rolling window for this frame
    sample = df.sample(min(1500, len(df)), random_state=11)

    # A flux var (sum) and a state var (mean), with the largest rolling window.
    checks = [("era5_precipitation_sum", "sum"), ("era5_temperature_2m_mean", "mean")]
    for var, agg in checks:
        pv = _era_pivot(var)
        col = f"{var}_{agg}_{roll}"
        causal_mismatch = 0
        alt_future_mismatch = 0
        alt_inclusive_mismatch = 0
        for _, row in sample.iterrows():
            uid = str(row["unique_id"])
            t = pd.Timestamp(row["forecast_origin_ds"])
            series = pv[uid]
            causal = series.loc[t - pd.Timedelta(days=roll): t - pd.Timedelta(days=1)]
            inclusive = series.loc[t - pd.Timedelta(days=roll - 1): t]
            future = series.loc[t + pd.Timedelta(days=1): t + pd.Timedelta(days=roll)]
            causal_val = causal.sum() if agg == "sum" else causal.mean()
            incl_val = inclusive.sum() if agg == "sum" else inclusive.mean()
            fut_val = future.sum() if agg == "sum" else future.mean()
            if not np.isclose(row[col], causal_val, equal_nan=True):
                causal_mismatch += 1
            if not np.isclose(row[col], incl_val, equal_nan=True):
                alt_inclusive_mismatch += 1
            if not np.isclose(row[col], fut_val, equal_nan=True):
                alt_future_mismatch += 1
        assert causal_mismatch == 0, f"{name} {col}: {causal_mismatch} causal mismatches"
        # The wrong hypotheses must be clearly rejected on the majority of rows.
        assert alt_inclusive_mismatch > 0.5 * len(sample), f"{col} inclusive hypothesis not rejected"
        assert alt_future_mismatch > 0.5 * len(sample), f"{col} future hypothesis not rejected"


# ---------------------------------------------------------------------------
# Group 3: shared row set and split across all six frames
# ---------------------------------------------------------------------------

def test_frames_share_rows_and_split() -> None:
    _require_frames()
    base = None
    for name in ALL_FRAMES:
        keys = (
            _load(name)[["unique_id", "forecast_origin_ds", "split"]]
            .assign(unique_id=lambda d: d["unique_id"].astype(str))
            .sort_values(["unique_id", "forecast_origin_ds"])
            .reset_index(drop=True)
        )
        if base is None:
            base = keys
        else:
            assert base.equals(keys), f"{name} differs in (unique_id, origin, split)"


# ---------------------------------------------------------------------------
# Group 4: per-station boundaries equal across frames and match split_boundaries.csv
# ---------------------------------------------------------------------------

def _station_boundaries(df: pd.DataFrame) -> pd.DataFrame:
    df = df.assign(unique_id=lambda d: d["unique_id"].astype(str))
    rows = []
    for uid, grp in df.groupby("unique_id"):
        rec = {"unique_id": uid}
        for split in ("train", "validation", "test"):
            sub = grp.loc[grp["split"] == split, "forecast_origin_ds"]
            rec[f"{split}_start"] = pd.Timestamp(sub.min()) if len(sub) else pd.NaT
            rec[f"{split}_end"] = pd.Timestamp(sub.max()) if len(sub) else pd.NaT
        rows.append(rec)
    return pd.DataFrame(rows).sort_values("unique_id").reset_index(drop=True)


def test_boundaries_match_across_frames_and_csv() -> None:
    _require_frames()
    reference = _station_boundaries(_load(ALL_FRAMES[0]))
    for name in ALL_FRAMES[1:]:
        assert reference.equals(_station_boundaries(_load(name))), f"{name} boundaries differ"

    csv_path = FRAMES_DIR / "split_boundaries.csv"
    assert csv_path.exists(), "split_boundaries.csv missing"
    csv = pd.read_csv(csv_path, parse_dates=[
        "train_start_ds", "train_end_ds", "validation_start_ds",
        "validation_end_ds", "test_start_ds", "test_end_ds",
    ])
    csv["unique_id"] = csv["unique_id"].astype(str)
    csv = csv.sort_values("unique_id").reset_index(drop=True)
    for split in ("train", "validation", "test"):
        assert (csv[f"{split}_start_ds"].values == reference[f"{split}_start"].values).all()
        assert (csv[f"{split}_end_ds"].values == reference[f"{split}_end"].values).all()


# ---------------------------------------------------------------------------
# Group 5: station set, duplicates, no future columns, ERA5 lag alignment
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ALL_FRAMES)
def test_station_set_and_no_duplicates_and_no_future(name: str) -> None:
    _require_frames()
    import re
    df = _load(name)
    df["unique_id"] = df["unique_id"].astype(str)
    assert df["unique_id"].nunique() == 20
    assert DROPPED_STATION not in set(df["unique_id"])
    assert not df.duplicated(["unique_id", "forecast_origin_ds"]).any()
    leaky = [c for c in df.columns if re.search(r"_future_h\d+", c)]
    assert leaky == [], f"{name} has future columns: {leaky}"


@pytest.mark.parametrize("name", ["weather_w14_h3", "weather_w30_h3"])
def test_era5_lag_alignment(name: str) -> None:
    _require_frames()
    if not (DATA_DIR / "reanalysis_daily.parquet").exists():
        pytest.skip("reanalysis_daily.parquet not present")
    df = _load(name)
    window = _window_of(name)
    var = "era5_precipitation_sum"
    pv = _era_pivot(var)
    sample = df.sample(min(1500, len(df)), random_state=13)
    lags = sorted({0, window // 2, window})
    mism = 0
    for _, row in sample.iterrows():
        uid = str(row["unique_id"])
        t = pd.Timestamp(row["forecast_origin_ds"])
        for k in lags:
            expected = pv[uid].get(t - pd.Timedelta(days=k), np.nan)
            if not np.isclose(row[f"{var}_lag_{k}"], expected, equal_nan=True):
                mism += 1
    assert mism == 0, f"{name}: {mism} ERA5 lag mismatches"


# ---------------------------------------------------------------------------
# Model checks: Mamba scan equivalence and forward shapes
# ---------------------------------------------------------------------------

def test_mamba_parallel_matches_sequential() -> None:
    import torch

    from src.models.advanced_neural import _MambaBlock

    torch.manual_seed(0)
    block = _MambaBlock(128, state_dim=64, expand=2).eval()
    u = torch.randn(4, 31, block.d_inner)
    with torch.no_grad():
        parallel = block._ssm_parallel(u)
        sequential = block._ssm_sequential(u)
    assert torch.max(torch.abs(parallel - sequential)).item() < 1e-4


def _toy_forward_inputs(seq_dim: int, static_dim: int, future_dim: int, horizon: int, stations: int):
    import torch

    batch, length = 6, 15
    return (
        torch.randn(batch, length, seq_dim),
        torch.randn(batch, static_dim),
        torch.randn(batch, horizon, future_dim),
        torch.randint(0, stations, (batch,)),
        torch.randn(batch, horizon),
    )


def test_patchtst_and_lstm_forward_shapes() -> None:
    import torch

    from src.models.advanced_neural import (
        ResidualAdvancedLSTMForecaster,
        ResidualAdvancedPatchTSTForecaster,
    )

    seq_dim, static_dim, future_dim, horizon, stations = 3, 8, 4, 3, 20
    inputs = _toy_forward_inputs(seq_dim, static_dim, future_dim, horizon, stations)

    patchtst = ResidualAdvancedPatchTSTForecaster(
        sequence_input_dim=seq_dim, sequence_length=15, static_input_dim=static_dim,
        future_input_dim=future_dim, horizon_count=horizon, station_count=stations,
    ).eval()
    with torch.no_grad():
        assert patchtst(*inputs).shape == (6, horizon)

    for bidirectional in (False, True):
        lstm = ResidualAdvancedLSTMForecaster(
            sequence_input_dim=seq_dim, static_input_dim=static_dim, future_input_dim=future_dim,
            horizon_count=horizon, station_count=stations, bidirectional=bidirectional,
        ).eval()
        with torch.no_grad():
            assert lstm(*inputs).shape == (6, horizon)
