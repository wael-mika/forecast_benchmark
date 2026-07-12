from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.neural import prepare_neural_window_bundle


def test_prepare_neural_window_bundle_normalizes_and_restores_per_station() -> None:
    feature_df = pd.DataFrame(
        {
            "unique_id": ["station_a", "station_a", "station_b", "station_b"],
            "split": ["train", "validation", "train", "test"],
            "forecast_origin_ds": pd.date_range("2020-01-01", periods=4, freq="D"),
            "lag_1": [5.0, 6.0, 50.0, 55.0],
            "lag_2": [4.0, 5.0, 45.0, 50.0],
            "lag_3": [3.0, 4.0, 40.0, 45.0],
            "lag_4": [2.0, 3.0, 35.0, 40.0],
            "lag_5": [1.0, 2.0, 30.0, 35.0],
            "target_h1": [6.0, 7.0, 55.0, 60.0],
            "target_h2": [7.0, 8.0, 60.0, 65.0],
            "target_h3": [8.0, 9.0, 65.0, 70.0],
            "target_h1_ds": pd.date_range("2020-01-02", periods=4, freq="D"),
            "target_h2_ds": pd.date_range("2020-01-03", periods=4, freq="D"),
            "target_h3_ds": pd.date_range("2020-01-04", periods=4, freq="D"),
        }
    )

    bundle = prepare_neural_window_bundle(feature_df)

    assert bundle.lag_columns == ["lag_5", "lag_4", "lag_3", "lag_2", "lag_1"]
    assert bundle.sequence_features.shape == (4, 5, 1)
    assert bundle.flat_features.shape[1] == 13
    assert bundle.context_features.shape[1] == 8
    assert bundle.prediction_columns == ["prediction_h1", "prediction_h2", "prediction_h3"]

    restored_targets = bundle.normalizer.inverse_transform(bundle.targets, bundle.station_indices)
    np.testing.assert_allclose(
        restored_targets,
        feature_df.loc[:, ["target_h1", "target_h2", "target_h3"]].to_numpy(dtype=float),
        rtol=1e-5,
        atol=1e-5,
    )
