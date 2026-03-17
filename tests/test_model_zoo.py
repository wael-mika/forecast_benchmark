from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.neural import (
    ResidualMambaForecaster,
    ResidualNHiTSForecaster,
    ResidualPatchTSTForecaster,
    ResidualTemporalFusionTransformerForecaster,
    ResidualXLSTMForecaster,
)
from src.training.train import run_direct_seasonal_naive_experiment


def _toy_batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    sequence_features = torch.randn(4, 14, 1)
    static_features = torch.randn(4, 12)
    station_index = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    baseline = torch.randn(4, 3)
    return sequence_features, static_features, station_index, baseline


def test_sequence_model_variants_return_horizon_predictions() -> None:
    sequence_features, static_features, station_index, baseline = _toy_batch()
    models = [
        ResidualNHiTSForecaster(
            sequence_length=14,
            static_input_dim=12,
            horizon_count=3,
            station_count=2,
        ),
        ResidualPatchTSTForecaster(
            sequence_input_dim=1,
            sequence_length=14,
            static_input_dim=12,
            horizon_count=3,
            station_count=2,
        ),
        ResidualTemporalFusionTransformerForecaster(
            sequence_input_dim=1,
            static_input_dim=12,
            horizon_count=3,
            station_count=2,
        ),
        ResidualXLSTMForecaster(
            sequence_input_dim=1,
            static_input_dim=12,
            horizon_count=3,
            station_count=2,
        ),
        ResidualMambaForecaster(
            sequence_input_dim=1,
            static_input_dim=12,
            horizon_count=3,
            station_count=2,
        ),
    ]

    for model in models:
        output = model(sequence_features, static_features, station_index, baseline)
        assert output.shape == (4, 3)


def test_run_direct_seasonal_naive_experiment_uses_persistence(tmp_path: Path) -> None:
    feature_df = pd.DataFrame(
        {
            "unique_id": ["station_a", "station_a"],
            "split": ["train", "test"],
            "forecast_origin_ds": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "current_y": [10.0, 11.0],
            "lag_1": [9.0, 10.0],
            "target_h1": [11.0, 12.0],
            "target_h2": [12.0, 13.0],
            "target_h1_ds": pd.to_datetime(["2020-01-02", "2020-01-03"]),
            "target_h2_ds": pd.to_datetime(["2020-01-03", "2020-01-04"]),
            "split_reference_ds": pd.to_datetime(["2020-01-03", "2020-01-04"]),
        }
    )

    experiment = run_direct_seasonal_naive_experiment(
        feature_df,
        {
            "model_name": "seasonal_naive",
            "artifact_dir": str(tmp_path / "artifacts" / "test_seasonal_naive"),
            "baseline_type": "persistence",
        },
    )

    assert experiment.prediction_columns_df["prediction_h1"].tolist() == [10.0, 11.0]
    assert experiment.prediction_columns_df["prediction_h2"].tolist() == [10.0, 11.0]
