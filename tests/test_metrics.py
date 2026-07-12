from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import build_scale_reference, compute_metric_bundle, summarize_prediction_metrics


def test_compute_metric_bundle_returns_expected_core_metrics() -> None:
    df = pd.DataFrame(
        {
            "y_true": [1.0, 2.0, 3.0],
            "y_pred": [1.0, 2.0, 4.0],
            "mase_denominator": [1.0, 1.0, 1.0],
            "rmsse_denominator": [1.0, 1.0, 1.0],
        }
    )

    metrics = compute_metric_bundle(df)

    assert metrics["n_obs"] == 3
    assert metrics["mae"] == 1.0 / 3.0
    assert metrics["mse"] == 1.0 / 3.0
    assert round(metrics["rmse"], 6) == round((1.0 / 3.0) ** 0.5, 6)
    assert round(metrics["mase"], 6) == round(1.0 / 3.0, 6)
    assert round(metrics["rmsse"], 6) == round((1.0 / 3.0) ** 0.5, 6)


def test_build_scale_reference_uses_groupwise_differences() -> None:
    df = pd.DataFrame(
        {
            "unique_id": ["station_a"] * 4,
            "target_ds": pd.date_range("2020-01-01", periods=4, freq="D"),
            "target": [1.0, 2.0, 4.0, 7.0],
        }
    )

    scale_reference = build_scale_reference(df)

    assert scale_reference.loc[0, "mase_denominator"] == 2.0
    assert scale_reference.loc[0, "rmsse_denominator"] == (1.0**2 + 2.0**2 + 3.0**2) / 3.0


def test_summarize_prediction_metrics_returns_micro_and_macro_rows() -> None:
    prediction_df = pd.DataFrame(
        {
            "unique_id": ["station_a", "station_a", "station_b", "station_b"],
            "split": ["test", "test", "test", "test"],
            "y_true": [1.0, 2.0, 2.0, 4.0],
            "y_pred": [1.0, 1.0, 2.0, 5.0],
            "mase_denominator": [1.0, 1.0, 2.0, 2.0],
            "rmsse_denominator": [1.0, 1.0, 4.0, 4.0],
        }
    )

    overall_df, per_station_df = summarize_prediction_metrics(prediction_df)

    assert set(overall_df["aggregation"]) == {"micro", "macro"}
    assert set(per_station_df["unique_id"]) == {"station_a", "station_b"}
