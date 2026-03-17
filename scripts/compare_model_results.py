"""Create comparison plots across benchmark model artifact directories."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.plots import generate_model_comparison_plot_bundle
from src.utils.io import save_csv
from src.utils.logging import get_logger


DEFAULT_ARTIFACTS = {
    "seasonal_naive": PROJECT_ROOT / "artifacts" / "seasonal_naive_weather_context_w14_h3",
    "xgboost": PROJECT_ROOT / "artifacts" / "xgboost_weather_context_w14_h3",
    "ann": PROJECT_ROOT / "artifacts" / "ann_weather_context_w14_h3",
    "lstm": PROJECT_ROOT / "artifacts" / "lstm_weather_context_w14_h3",
    "nhits": PROJECT_ROOT / "artifacts" / "nhits_weather_context_w14_h3",
    "patchtst": PROJECT_ROOT / "artifacts" / "patchtst_weather_context_w14_h3",
    "tft": PROJECT_ROOT / "artifacts" / "tft_weather_context_w14_h3",
    "xlstm": PROJECT_ROOT / "artifacts" / "xlstm_weather_context_w14_h3",
    "mamba": PROJECT_ROOT / "artifacts" / "mamba_weather_context_w14_h3_smallbatch",
}


def main(argv: list[str] | None = None) -> None:
    """Generate comparison tables and plots for the configured benchmark models."""
    active_argv = argv or sys.argv
    logger = get_logger("compare_model_results")

    output_dir = (PROJECT_ROOT / "artifacts" / "model_comparison").resolve()
    metrics_frames: list[pd.DataFrame] = []
    artifact_map = DEFAULT_ARTIFACTS
    if len(active_argv) > 1:
        requested_model_names = active_argv[1:]
        artifact_map = {
            model_name: artifact_dir
            for model_name, artifact_dir in DEFAULT_ARTIFACTS.items()
            if model_name in requested_model_names
        }
        if not artifact_map:
            raise ValueError(f"No matching model names found in comparison defaults: {requested_model_names}")

    for model_name, artifact_dir in artifact_map.items():
        metrics_path = artifact_dir / "metrics_summary.csv"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing metrics summary for {model_name}: {metrics_path}")
        metrics_df = pd.read_csv(metrics_path)
        metrics_df["model_name"] = model_name
        metrics_frames.append(metrics_df)

    comparison_metrics_df = pd.concat(metrics_frames, ignore_index=True).sort_values(
        ["model_name", "horizon", "split", "aggregation"],
        kind="stable",
    )
    save_csv(comparison_metrics_df, output_dir / "comparison_metrics.csv")
    manifest = generate_model_comparison_plot_bundle(
        output_dir=output_dir,
        comparison_metrics_df=comparison_metrics_df,
    )
    logger.info("Generated %s comparison plots under %s", len(manifest), output_dir)


if __name__ == "__main__":
    main()
