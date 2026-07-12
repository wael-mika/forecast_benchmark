"""Run one configured experiment from training through metric export.

This script reads a single YAML config, loads the referenced feature parquet,
trains the requested model, builds a prediction frame, evaluates it, and writes
predictions plus metrics into the configured artifact directory.

Supported config families
-------------------------
- XGBoost single-output and direct multi-horizon runs.
- Neural runs, including baseline and advanced variants.
- Direct seasonal naive baselines.

Use this script when you want to execute one experiment directly from a config
file instead of using the higher-level suite runner.

Usage
-----
    .venv/Scripts/python scripts/run_experiment.py
    .venv/Scripts/python scripts/run_experiment.py configs/xgboost.yaml
    .venv/Scripts/python scripts/run_experiment.py configs/ann_advanced_weather.yaml

Inputs
------
    YAML config
        Selects the model, feature parquet, split columns, and artifact dir.
    Feature parquet
        Loaded from config["feature_frame_path"].

Outputs
-------
    The configured artifact directory receives model files, predictions, and
    evaluation tables saved through src.evaluation.pipeline.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.pipeline import (
    build_direct_prediction_frame,
    build_prediction_frame,
    evaluate_direct_prediction_frame,
    evaluate_prediction_frame,
    save_evaluation_outputs,
)
from src.training.train import (
    predict_direct_xgboost,
    predict_with_xgboost,
    run_direct_seasonal_naive_experiment,
    train_direct_xgboost_experiment,
    train_xgboost_experiment,
)
from src.utils.config import load_yaml_config
from src.utils.logging import get_logger


def _resolve_config_path(argv: list[str]) -> Path:
    if len(argv) > 1:
        return (PROJECT_ROOT / argv[1]).resolve()
    return PROJECT_ROOT / "configs" / "xgboost.yaml"


def main(argv: list[str] | None = None) -> None:
    """Run one experiment end to end using the selected config file."""
    active_argv = argv or sys.argv
    logger = get_logger("run_experiment")
    config_path = _resolve_config_path(active_argv)
    config = load_yaml_config(config_path)

    model_name = str(config.get("model_name", "")).lower()

    feature_frame_path = PROJECT_ROOT / config["feature_frame_path"]
    if not feature_frame_path.exists():
        raise FileNotFoundError(
            f"Feature frame not found at {feature_frame_path}. Run scripts/prepare_xgboost_data.py first."
        )

    feature_df = pd.read_parquet(feature_frame_path)

    # Reject any frame carrying known-future weather columns (leakage guard).
    leaky_columns = [c for c in feature_df.columns if re.search(r"_future_h\d+", c)]
    if leaky_columns:
        raise ValueError(
            f"Feature frame {feature_frame_path} contains forbidden future columns: {leaky_columns}"
        )

    training_config = {**config, "artifact_dir": str(PROJECT_ROOT / config["artifact_dir"])}
    if model_name == "xgboost" and "horizons" in config:
        experiment = train_direct_xgboost_experiment(feature_df, training_config)
        enable_categorical = "station_id_feature" in experiment.feature_columns
        prediction_columns_df = predict_direct_xgboost(
            experiment.boosters,
            experiment.feature_frame,
            feature_columns=experiment.feature_columns,
            enable_categorical=enable_categorical,
        )
        prediction_df = build_direct_prediction_frame(
            experiment.feature_frame,
            prediction_columns_df,
            split_column=config.get("split_column", "split"),
        )
        overall_metrics_df, per_station_metrics_df = evaluate_direct_prediction_frame(
            prediction_df,
            split_column=config.get("split_column", "split"),
            group_column="unique_id",
        )
        artifact_dir = experiment.artifact_dir
    elif model_name == "xgboost":
        experiment = train_xgboost_experiment(feature_df, training_config)
        enable_categorical = "station_id_feature" in experiment.feature_columns
        predictions = predict_with_xgboost(
            experiment.booster,
            experiment.feature_frame,
            feature_columns=experiment.feature_columns,
            target_column=config.get("target_column", "target"),
            enable_categorical=enable_categorical,
        )

        prediction_df = build_prediction_frame(
            experiment.feature_frame,
            predictions.to_numpy(),
            actual_column=config.get("target_column", "target"),
            split_column=config.get("split_column", "split"),
        )
        overall_metrics_df, per_station_metrics_df = evaluate_prediction_frame(
            prediction_df,
            split_column=config.get("split_column", "split"),
            group_column="unique_id",
        )
        artifact_dir = experiment.artifact_dir
    elif model_name in {"ann", "lstm", "bilstm", "nhits", "patchtst", "tft", "xlstm", "mamba", "hybrid", "flownet"}:
        model_variant = str(config.get("model_variant", "baseline")).lower()
        if model_variant == "advanced" or model_name in {"hybrid", "flownet"}:
            from src.training.advanced_neural import train_advanced_neural_experiment

            experiment = train_advanced_neural_experiment(feature_df, training_config)
        else:
            from src.training.neural import train_neural_experiment

            experiment = train_neural_experiment(feature_df, training_config)
        prediction_df = experiment.prediction_df
        overall_metrics_df = experiment.overall_metrics_df
        per_station_metrics_df = experiment.per_station_metrics_df
        artifact_dir = experiment.artifact_dir
    elif model_name == "seasonal_naive":
        experiment = run_direct_seasonal_naive_experiment(feature_df, training_config)
        prediction_df = build_direct_prediction_frame(
            experiment.feature_frame,
            experiment.prediction_columns_df,
            split_column=config.get("split_column", "split"),
        )
        overall_metrics_df, per_station_metrics_df = evaluate_direct_prediction_frame(
            prediction_df,
            split_column=config.get("split_column", "split"),
            group_column="unique_id",
        )
        artifact_dir = experiment.artifact_dir
    else:
        raise ValueError(f"Unsupported model_name in {config_path}: {config.get('model_name')!r}")

    save_evaluation_outputs(
        prediction_df,
        overall_metrics_df,
        per_station_metrics_df,
        artifact_dir=artifact_dir,
    )

    logger.info("Saved model, checkpoints, predictions, and metrics under %s", artifact_dir)
    if not overall_metrics_df.empty:
        logger.info("Validation/test summary:\n%s", overall_metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
