"""Tests for the scaled-target (log1p_station_z) direct XGBoost variant."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.train import predict_direct_xgboost, train_direct_xgboost_experiment

pytest.importorskip("xgboost")


def _build_two_station_frame(rows_per_station: int = 160) -> pd.DataFrame:
    """Two-station direct feature frame whose stations differ ~100x in scale."""
    rng = np.random.default_rng(0)
    frames = []
    for station, scale in [("small_creek", 1.0), ("big_river", 100.0)]:
        t = np.arange(rows_per_station, dtype=float)
        base = scale * (1.0 + 0.4 * np.sin(2.0 * np.pi * t / 30.0)) + rng.normal(0.0, 0.02 * scale, rows_per_station)
        base = np.clip(base, 0.05 * scale, None)
        origins = pd.date_range("2020-01-01", periods=rows_per_station, freq="D")
        n_train = int(rows_per_station * 0.7)
        n_val = int(rows_per_station * 0.15)
        split = np.array(
            ["train"] * n_train + ["validation"] * n_val + ["test"] * (rows_per_station - n_train - n_val)
        )
        frames.append(
            pd.DataFrame(
                {
                    "unique_id": station,
                    "forecast_origin_ds": origins,
                    "split": split,
                    "lag_3": np.roll(base, 3),
                    "lag_2": np.roll(base, 2),
                    "lag_1": np.roll(base, 1),
                    "current_y": base,
                    "target_h1": np.roll(base, -1),
                    "target_h2": np.roll(base, -2),
                    "target_h1_ds": origins + pd.Timedelta(days=1),
                    "target_h2_ds": origins + pd.Timedelta(days=2),
                }
            ).iloc[3:-2]
        )
    return pd.concat(frames, ignore_index=True)


def _base_config(artifact_dir: Path) -> dict:
    return {
        "model_name": "xgboost",
        "split_column": "split",
        "use_station_id_as_feature": True,
        "num_boost_round": 40,
        "early_stopping_rounds": 10,
        "checkpoint_interval": 0,
        "verbose_eval": 0,
        "seed": 42,
        "verbosity": 0,
        "artifact_dir": str(artifact_dir),
    }


def test_scaled_target_trains_and_returns_physical_unit_predictions(tmp_path: Path) -> None:
    feature_df = _build_two_station_frame()
    config = {**_base_config(tmp_path / "scaled"), "target_transform": "log1p_station_z"}

    experiment = train_direct_xgboost_experiment(feature_df, config)

    assert experiment.normalizer is not None
    assert (tmp_path / "scaled" / "scaler_by_station.csv").exists()
    assert experiment.training_summary["target_transform"] == "log1p_station_z"
    # The returned feature frame keeps raw physical targets for evaluation.
    assert experiment.feature_frame["target_h1"].max() > 50.0

    predictions = predict_direct_xgboost(
        experiment.boosters,
        experiment.feature_frame,
        feature_columns=experiment.feature_columns,
        enable_categorical="station_id_feature" in experiment.feature_columns,
        normalizer=experiment.normalizer,
    )

    small_mask = (experiment.feature_frame["unique_id"] == "small_creek").to_numpy()
    big_mask = ~small_mask
    for column in ["prediction_h1", "prediction_h2"]:
        values = predictions[column].to_numpy()
        assert np.all(values >= 0.0)
        # Predictions land inside each station's physical range despite the
        # 100x scale gap: the small station is not dragged to the big scale.
        assert values[small_mask].max() < 10.0
        assert values[small_mask].min() > 0.05
        assert values[big_mask].max() < 1000.0
        assert values[big_mask].min() > 10.0


def test_default_raw_target_path_is_unchanged(tmp_path: Path) -> None:
    feature_df = _build_two_station_frame()
    config = _base_config(tmp_path / "raw")

    experiment = train_direct_xgboost_experiment(feature_df, config)

    assert experiment.normalizer is None
    assert "target_transform" not in experiment.training_summary
    assert not (tmp_path / "raw" / "scaler_by_station.csv").exists()

    predictions = predict_direct_xgboost(
        experiment.boosters,
        experiment.feature_frame,
        feature_columns=experiment.feature_columns,
        enable_categorical="station_id_feature" in experiment.feature_columns,
    )
    assert list(predictions.columns) == ["prediction_h1", "prediction_h2"]
    assert len(predictions) == len(feature_df)


def test_unknown_target_transform_is_rejected(tmp_path: Path) -> None:
    feature_df = _build_two_station_frame(rows_per_station=60)
    config = {**_base_config(tmp_path / "bad"), "target_transform": "zscore"}
    with pytest.raises(ValueError, match="target_transform"):
        train_direct_xgboost_experiment(feature_df, config)
