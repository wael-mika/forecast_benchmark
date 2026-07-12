"""Generate cross-model comparison plots for the advanced benchmark suite.

This script collects metrics from completed advanced-model artifact directories,
combines them by regime, and calls the shared comparison plot bundle builder.
It produces one comparison folder for context runs and one for weather runs.

Use this script after several advanced model runs have completed and you want a
side-by-side comparison across architectures.

Inputs
------
    metrics_summary.csv from the configured artifact directories in CONFIG_MAP

Outputs
-------
    artifacts/advanced_seq/model_comparison_context/
    artifacts/advanced_seq/model_comparison_weather/

Usage
-----
    .venv/Scripts/python scripts/compare_advanced_results.py
    .venv/Scripts/python scripts/compare_advanced_results.py ann patchtst tft
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.plots import generate_model_comparison_plot_bundle
from src.utils.config import load_yaml_config
from src.utils.io import save_csv
from src.utils.logging import get_logger


CONFIG_MAP = {
    "context": {
        "xgboost": PROJECT_ROOT / "configs" / "xgboost_advanced_context.yaml",
        "ann": PROJECT_ROOT / "configs" / "ann_advanced_context.yaml",
        "lstm": PROJECT_ROOT / "configs" / "lstm_advanced_context.yaml",
        "nhits": PROJECT_ROOT / "configs" / "nhits_advanced_context.yaml",
        "patchtst": PROJECT_ROOT / "configs" / "patchtst_advanced_context.yaml",
        "tft": PROJECT_ROOT / "configs" / "tft_advanced_context.yaml",
        "xlstm": PROJECT_ROOT / "configs" / "xlstm_advanced_context.yaml",
        "mamba": PROJECT_ROOT / "configs" / "mamba_advanced_context.yaml",
        "hybrid": PROJECT_ROOT / "configs" / "hybrid_context.yaml",
        "flownet": PROJECT_ROOT / "configs" / "flownet_context.yaml",
    },
    "weather": {
        "xgboost": PROJECT_ROOT / "configs" / "xgboost_advanced_weather.yaml",
        "ann": PROJECT_ROOT / "configs" / "ann_advanced_weather.yaml",
        "lstm": PROJECT_ROOT / "configs" / "lstm_advanced_weather.yaml",
        "nhits": PROJECT_ROOT / "configs" / "nhits_advanced_weather.yaml",
        "patchtst": PROJECT_ROOT / "configs" / "patchtst_advanced_weather.yaml",
        "tft": PROJECT_ROOT / "configs" / "tft_advanced_weather.yaml",
        "xlstm": PROJECT_ROOT / "configs" / "xlstm_advanced_weather.yaml",
        "mamba": PROJECT_ROOT / "configs" / "mamba_advanced_weather.yaml",
        "hybrid": PROJECT_ROOT / "configs" / "hybrid_weather.yaml",
        "flownet": PROJECT_ROOT / "configs" / "flownet_weather.yaml",
    },
}


def _resolve_artifact_map(config_map: dict[str, Path]) -> dict[str, Path]:
    return {
        model_name: (PROJECT_ROOT / load_yaml_config(config_path)["artifact_dir"]).resolve()
        for model_name, config_path in config_map.items()
    }


def _load_metrics(artifact_map: dict[str, Path]) -> tuple[pd.DataFrame, list[str]]:
    frames: list[pd.DataFrame] = []
    missing_models: list[str] = []
    for model_name, artifact_dir in artifact_map.items():
        metrics_path = artifact_dir / "metrics_summary.csv"
        if not metrics_path.exists():
            missing_models.append(model_name)
            continue
        metrics_df = pd.read_csv(metrics_path)
        metrics_df["model_name"] = model_name
        frames.append(metrics_df)
    if not frames:
        raise FileNotFoundError("No completed model metrics were found for the requested comparison.")
    return (
        pd.concat(frames, ignore_index=True).sort_values(
            ["model_name", "horizon", "split", "aggregation"],
            kind="stable",
        ),
        missing_models,
    )


def main(argv: list[str] | None = None) -> None:
    """Build comparison tables and plots for the requested advanced models."""
    active_argv = argv or sys.argv
    logger = get_logger("compare_advanced_results")
    requested_models = active_argv[1:]

    for regime_name, config_map in CONFIG_MAP.items():
        resolved_artifact_map = _resolve_artifact_map(config_map)
        if requested_models:
            resolved_artifact_map = {
                model_name: artifact_dir
                for model_name, artifact_dir in resolved_artifact_map.items()
                if model_name in requested_models
            }
        if not resolved_artifact_map:
            continue

        comparison_metrics_df, missing_models = _load_metrics(resolved_artifact_map)
        if missing_models:
            logger.warning("Skipping incomplete comparison artifacts for: %s", ", ".join(sorted(missing_models)))
        output_dir = PROJECT_ROOT / "artifacts" / "advanced_seq" / f"model_comparison_{regime_name}"
        save_csv(comparison_metrics_df, output_dir / "comparison_metrics.csv")
        manifest = generate_model_comparison_plot_bundle(
            output_dir=output_dir,
            comparison_metrics_df=comparison_metrics_df,
        )
        logger.info(
            "Generated %s comparison plots under %s",
            len(manifest),
            output_dir,
        )


if __name__ == "__main__":
    main()
