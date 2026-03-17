from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import build_scale_reference
from src.evaluation.pipeline import build_direct_prediction_frame, evaluate_direct_prediction_frame


def test_direct_prediction_frame_and_metrics_include_horizons() -> None:
    feature_df = pd.DataFrame(
        {
            "unique_id": ["station_a", "station_a", "station_a", "station_a"],
            "forecast_origin_ds": pd.date_range("2020-01-01", periods=4, freq="D"),
            "split": ["train", "train", "validation", "test"],
            "target_h1_ds": pd.date_range("2020-01-02", periods=4, freq="D"),
            "target_h2_ds": pd.date_range("2020-01-03", periods=4, freq="D"),
            "target_h1": [1.0, 2.0, 3.0, 4.0],
            "target_h2": [2.0, 3.0, 4.0, 5.0],
        }
    )
    prediction_columns_df = pd.DataFrame(
        {
            "prediction_h1": [1.0, 2.0, 3.5, 4.5],
            "prediction_h2": [2.0, 3.0, 4.5, 5.5],
        }
    )

    prediction_df = build_direct_prediction_frame(feature_df, prediction_columns_df)
    overall_df, per_station_df = evaluate_direct_prediction_frame(prediction_df)

    assert sorted(prediction_df["horizon"].unique().tolist()) == [1, 2]
    assert set(overall_df["aggregation"]) == {"micro", "macro"}
    assert sorted(overall_df["horizon"].unique().tolist()) == [1, 2]
    assert sorted(per_station_df["horizon"].unique().tolist()) == [1, 2]


def test_direct_evaluation_accepts_external_scale_reference() -> None:
    prediction_df = pd.DataFrame(
        {
            "unique_id": ["station_a", "station_a"],
            "forecast_origin_ds": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "target_ds": pd.to_datetime(["2020-01-02", "2020-01-03"]),
            "split": ["validation", "test"],
            "horizon": [1, 1],
            "y_true": [2.0, 3.0],
            "y_pred": [2.5, 2.8],
            "residual": [0.5, -0.2],
        }
    )
    scale_reference_df = build_scale_reference(
        pd.DataFrame(
            {
                "unique_id": ["station_a", "station_a", "station_a"],
                "target_ds": pd.to_datetime(["2019-12-30", "2019-12-31", "2020-01-01"]),
                "target": [1.0, 2.0, 3.0],
            }
        ),
        group_column="unique_id",
        time_column="target_ds",
        target_column="target",
    )

    overall_df, per_station_df = evaluate_direct_prediction_frame(prediction_df, scale_reference_df=scale_reference_df)

    assert not overall_df.empty
    assert not per_station_df.empty
    assert overall_df["mase"].notna().any()
