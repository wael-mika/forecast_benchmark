from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import plots as evaluation_plots
from src.evaluation.plots import discover_round_model_paths
from src.training.advanced_neural import _compute_epoch_metric_bundle
from src.training.train import prepare_inference_frame


def test_prepare_inference_frame_rebuilds_station_id_feature() -> None:
    feature_df = pd.DataFrame(
        {
            "unique_id": ["station_a", "station_b"],
            "target": [1.0, 2.0],
        }
    )

    model_frame = prepare_inference_frame(feature_df, ["target", "station_id_feature"])

    assert "station_id_feature" in model_frame.columns
    assert str(model_frame["station_id_feature"].dtype) == "category"
    assert model_frame["station_id_feature"].astype(str).tolist() == ["station_a", "station_b"]


def test_discover_round_model_paths_uses_checkpoints_and_final_model(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts"
    checkpoint_dir = artifact_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)

    (checkpoint_dir / "xgboost_1.ubj").write_text("checkpoint", encoding="utf-8")
    (checkpoint_dir / "xgboost_2.ubj").write_text("checkpoint", encoding="utf-8")
    (artifact_dir / "model_rounds_0003.json").write_text("final", encoding="utf-8")

    round_paths = discover_round_model_paths(artifact_dir, final_round=3)

    assert [round_number for round_number, _ in round_paths] == [1, 2, 3]
    assert round_paths[-1][1].name == "model_rounds_0003.json"


def test_compute_epoch_metric_bundle_includes_mase_and_rmsse() -> None:
    y_true = np.array([10.0, 20.0, 30.0], dtype=float)
    y_pred = np.array([12.0, 17.0, 33.0], dtype=float)

    metrics = _compute_epoch_metric_bundle(
        y_true,
        y_pred,
        mase_denominator=np.array([2.0, 2.0, 2.0], dtype=float),
        rmsse_denominator=np.array([4.0, 4.0, 4.0], dtype=float),
    )

    assert metrics["mase"] == pytest.approx((1.0 + 1.5 + 1.5) / 3.0)
    assert metrics["rmsse"] == pytest.approx(np.sqrt((1.0 + 2.25 + 2.25) / 3.0))


def test_plot_epoch_metric_curves_skips_metrics_with_only_nan_values(tmp_path: Path) -> None:
    epoch_metrics_df = pd.DataFrame(
        {
            "epoch": [1, 1, 2, 2],
            "split": ["validation", "test", "validation", "test"],
            "horizon": [1, 1, 1, 1],
            "aggregation": ["micro", "micro", "micro", "micro"],
            "rmse": [2.0, 2.5, 1.8, 2.1],
            "mase": [np.nan, np.nan, np.nan, np.nan],
            "rmsse": [np.nan, np.nan, np.nan, np.nan],
        }
    )

    output_path = tmp_path / "primary_metric_curves.png"
    evaluation_plots.plot_epoch_metric_curves(
        epoch_metrics_df,
        output_path=output_path,
        metrics=("rmse", "mase", "rmsse"),
    )

    assert output_path.exists()


def test_generate_neural_plot_bundle_manifest_omits_retired_station_rmse_plot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir(parents=True)

    pd.DataFrame({"epoch": [1], "split": ["train"], "loss": [0.5]}).to_csv(artifact_dir / "loss_history.csv", index=False)
    pd.DataFrame(
        {
            "epoch": [1, 1],
            "split": ["validation", "test"],
            "horizon": [1, 1],
            "aggregation": ["micro", "micro"],
            "rmse": [1.2, 1.3],
            "mae": [0.8, 0.9],
            "mape": [0.1, 0.1],
            "smape": [0.1, 0.1],
            "mase": [0.7, 0.8],
            "rmsse": [0.9, 1.0],
        }
    ).to_csv(artifact_dir / "epoch_metrics.csv", index=False)
    pd.DataFrame(
        {
            "split": ["test"],
            "horizon": [1],
            "aggregation": ["micro"],
            "rmse": [1.3],
            "mae": [0.9],
            "mape": [0.1],
            "smape": [0.1],
            "mase": [0.8],
            "rmsse": [1.0],
        }
    ).to_csv(artifact_dir / "metrics_summary.csv", index=False)
    pd.DataFrame({"split": ["test"], "horizon": [1], "unique_id": ["station_a"], "rmse": [1.3]}).to_csv(
        artifact_dir / "metrics_by_station.csv",
        index=False,
    )
    pd.DataFrame(
        {
            "split": ["test"],
            "horizon": [1],
            "unique_id": ["station_a"],
            "forecast_origin_ds": pd.to_datetime(["2024-01-01"]),
            "target_ds": pd.to_datetime(["2024-01-02"]),
            "y_true": [1.0],
            "y_pred": [1.1],
            "residual": [0.1],
        }
    ).to_parquet(artifact_dir / "predictions.parquet", index=False)

    called: list[str] = []

    def _record(name: str):
        return lambda *args, **kwargs: called.append(name)

    monkeypatch.setattr(evaluation_plots, "plot_loss_curves", _record("plot_loss_curves"))
    monkeypatch.setattr(evaluation_plots, "plot_epoch_metric_curves", _record("plot_epoch_metric_curves"))
    monkeypatch.setattr(evaluation_plots, "plot_direct_metrics_heatmap", _record("plot_direct_metrics_heatmap"))
    monkeypatch.setattr(evaluation_plots, "plot_direct_final_metric_bars", _record("plot_direct_final_metric_bars"))
    monkeypatch.setattr(evaluation_plots, "plot_direct_actual_vs_predicted", _record("plot_direct_actual_vs_predicted"))
    monkeypatch.setattr(evaluation_plots, "plot_direct_residual_distribution", _record("plot_direct_residual_distribution"))
    monkeypatch.setattr(evaluation_plots, "plot_direct_station_examples", _record("plot_direct_station_examples"))
    monkeypatch.setattr(evaluation_plots, "plot_direct_forecast_window_examples", _record("plot_direct_forecast_window_examples"))

    manifest = evaluation_plots.generate_neural_plot_bundle(artifact_dir=artifact_dir, model_label="ANN")

    assert "test_worst_station_rmse" not in manifest
    assert "test_residual_distribution" in manifest
    assert (artifact_dir / "plots" / "plot_manifest.json").exists()
    assert "plot_direct_residual_distribution" in called
