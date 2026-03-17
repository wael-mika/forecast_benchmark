from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.advanced_neural import _build_advanced_model, prepare_advanced_neural_window_bundle


def _toy_feature_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unique_id": ["station_a", "station_a", "station_b", "station_b"],
            "split": ["train", "validation", "train", "test"],
            "forecast_origin_ds": pd.date_range("2020-01-01", periods=4, freq="D"),
            "split_reference_ds": pd.date_range("2020-01-04", periods=4, freq="D"),
            "current_y": [5.0, 6.0, 50.0, 55.0],
            "lag_1": [4.0, 5.0, 45.0, 50.0],
            "lag_2": [3.0, 4.0, 40.0, 45.0],
            "lag_3": [2.0, 3.0, 35.0, 40.0],
            "lag_4": [1.0, 2.0, 30.0, 35.0],
            "lag_mean": [2.5, 3.5, 37.5, 42.5],
            "delta_1": [1.0, 1.0, 5.0, 5.0],
            "delta_2": [1.0, 1.0, 5.0, 5.0],
            "target_h1": [6.0, 7.0, 55.0, 60.0],
            "target_h2": [7.0, 8.0, 60.0, 65.0],
            "target_h3": [8.0, 9.0, 65.0, 70.0],
            "target_h1_ds": pd.date_range("2020-01-02", periods=4, freq="D"),
            "target_h2_ds": pd.date_range("2020-01-03", periods=4, freq="D"),
            "target_h3_ds": pd.date_range("2020-01-04", periods=4, freq="D"),
            "era5_precipitation_sum_lag_0": [0.1, 0.2, 0.3, 0.4],
            "era5_precipitation_sum_lag_1": [0.0, 0.1, 0.2, 0.3],
            "era5_precipitation_sum_lag_2": [0.0, 0.0, 0.1, 0.2],
            "era5_precipitation_sum_lag_3": [0.2, 0.1, 0.2, 0.3],
            "era5_precipitation_sum_lag_4": [0.3, 0.2, 0.2, 0.3],
            "era5_precipitation_sum_future_h1": [0.1, 0.1, 0.2, 0.2],
            "era5_precipitation_sum_future_h2": [0.2, 0.2, 0.3, 0.3],
            "era5_precipitation_sum_future_h3": [0.3, 0.3, 0.4, 0.4],
        }
    )


def test_prepare_advanced_neural_window_bundle_builds_history_static_and_future_tensors() -> None:
    bundle = prepare_advanced_neural_window_bundle(_toy_feature_frame(), min_sequence_coverage=0.5)

    assert bundle.sequence_features.shape == (4, 5, 2)
    assert bundle.context_features.shape[0] == 4
    assert bundle.future_features.shape == (4, 3, 5)
    assert bundle.prediction_columns == ["prediction_h1", "prediction_h2", "prediction_h3"]
    assert "target_history" in bundle.sequence_channel_names
    assert "era5_precipitation_sum" in bundle.sequence_channel_names


def test_advanced_model_variants_return_horizon_predictions() -> None:
    bundle = prepare_advanced_neural_window_bundle(_toy_feature_frame(), min_sequence_coverage=0.5)
    sequence_features = torch.tensor(bundle.sequence_features[:2], dtype=torch.float32)
    flat_features = torch.tensor(bundle.flat_features[:2], dtype=torch.float32)
    static_features = torch.tensor(bundle.context_features[:2], dtype=torch.float32)
    future_features = torch.tensor(bundle.future_features[:2], dtype=torch.float32)
    station_indices = torch.tensor(bundle.station_indices[:2], dtype=torch.long)
    baseline = torch.tensor(bundle.baseline[:2], dtype=torch.float32)

    for model_name in ["ann", "lstm", "nhits", "patchtst", "tft", "xlstm", "mamba", "hybrid"]:
        model = _build_advanced_model(model_name, bundle, {"model_name": model_name})
        if model_name == "ann":
            output = model(sequence_features, flat_features, future_features, station_indices, baseline)
        else:
            output = model(sequence_features, static_features, future_features, station_indices, baseline)
        assert output.shape == (2, 3)
