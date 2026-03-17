"""Quantify the effect of adding weather variables across advanced models."""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_yaml_config
from src.utils.io import ensure_parent_dir, save_csv, save_json
from src.utils.logging import get_logger


CONTEXT_CONFIGS = {
    "xgboost": PROJECT_ROOT / "configs" / "xgboost_advanced_context.yaml",
    "ann": PROJECT_ROOT / "configs" / "ann_advanced_context.yaml",
    "lstm": PROJECT_ROOT / "configs" / "lstm_advanced_context.yaml",
    "nhits": PROJECT_ROOT / "configs" / "nhits_advanced_context.yaml",
    "patchtst": PROJECT_ROOT / "configs" / "patchtst_advanced_context.yaml",
    "tft": PROJECT_ROOT / "configs" / "tft_advanced_context.yaml",
    "xlstm": PROJECT_ROOT / "configs" / "xlstm_advanced_context.yaml",
    "mamba": PROJECT_ROOT / "configs" / "mamba_advanced_context.yaml",
    "hybrid": PROJECT_ROOT / "configs" / "hybrid_context.yaml",
}

WEATHER_CONFIGS = {
    "xgboost": PROJECT_ROOT / "configs" / "xgboost_advanced_weather.yaml",
    "ann": PROJECT_ROOT / "configs" / "ann_advanced_weather.yaml",
    "lstm": PROJECT_ROOT / "configs" / "lstm_advanced_weather.yaml",
    "nhits": PROJECT_ROOT / "configs" / "nhits_advanced_weather.yaml",
    "patchtst": PROJECT_ROOT / "configs" / "patchtst_advanced_weather.yaml",
    "tft": PROJECT_ROOT / "configs" / "tft_advanced_weather.yaml",
    "xlstm": PROJECT_ROOT / "configs" / "xlstm_advanced_weather.yaml",
    "mamba": PROJECT_ROOT / "configs" / "mamba_advanced_weather.yaml",
    "hybrid": PROJECT_ROOT / "configs" / "hybrid_weather.yaml",
}


def _resolve_artifact_path(config_path: Path) -> Path:
    return (PROJECT_ROOT / load_yaml_config(config_path)["artifact_dir"]).resolve()


def _load_test_micro_metrics(path: Path) -> pd.DataFrame:
    metrics_df = pd.read_csv(path)
    return metrics_df.loc[
        (metrics_df["split"] == "test") & (metrics_df["aggregation"] == "micro")
    ].copy()


def _plot_gain_bars(avg_gain_df: pd.DataFrame, *, metric: str, output_path: Path, title: str) -> None:
    ordered = avg_gain_df.sort_values(metric, ascending=False, kind="stable")
    figure, axis = plt.subplots(figsize=(9.0, max(4.5, len(ordered) * 0.4)), constrained_layout=True)
    colors = np.where(ordered[metric] >= 0.0, "#2ca02c", "#d62728")
    axis.barh(ordered["model_name"], ordered[metric], color=colors)
    axis.axvline(0.0, color="#111111", linestyle="--", linewidth=1.2)
    axis.set_xlabel(metric.upper())
    axis.set_title(title)
    axis.grid(True, axis="x", alpha=0.25)
    ensure_parent_dir(output_path)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main(argv: list[str] | None = None) -> None:
    """Generate tables and plots of the weather-variable gains for advanced models."""
    active_argv = argv or sys.argv
    logger = get_logger("compare_weather_ablations")
    requested_models = active_argv[1:] or list(CONTEXT_CONFIGS)

    rows: list[pd.DataFrame] = []
    skipped_models: list[str] = []
    for model_name in requested_models:
        context_path = _resolve_artifact_path(CONTEXT_CONFIGS[model_name]) / "metrics_summary.csv"
        weather_path = _resolve_artifact_path(WEATHER_CONFIGS[model_name]) / "metrics_summary.csv"
        if not context_path.exists() or not weather_path.exists():
            skipped_models.append(model_name)
            continue

        context_df = _load_test_micro_metrics(context_path).add_suffix("_context")
        weather_df = _load_test_micro_metrics(weather_path).add_suffix("_weather")
        merged = context_df.merge(
            weather_df,
            left_on="horizon_context",
            right_on="horizon_weather",
            how="inner",
        )
        merged["model_name"] = model_name
        merged["horizon"] = merged["horizon_context"]
        merged["rmse_gain"] = merged["rmse_context"] - merged["rmse_weather"]
        merged["mae_gain"] = merged["mae_context"] - merged["mae_weather"]
        merged["r2_gain"] = merged["r2_weather"] - merged["r2_context"]
        merged["nse_gain"] = merged["nse_weather"] - merged["nse_context"]
        rows.append(
            merged.loc[
                :,
                [
                    "model_name",
                    "horizon",
                    "rmse_context",
                    "rmse_weather",
                    "rmse_gain",
                    "mae_context",
                    "mae_weather",
                    "mae_gain",
                    "r2_context",
                    "r2_weather",
                    "r2_gain",
                    "nse_context",
                    "nse_weather",
                    "nse_gain",
                ],
            ]
        )

    if not rows:
        raise FileNotFoundError("No completed context/weather model pairs were found for the requested comparison.")
    if skipped_models:
        logger.warning("Skipping incomplete weather-ablation artifacts for: %s", ", ".join(sorted(skipped_models)))

    weather_effect_df = pd.concat(rows, ignore_index=True).sort_values(["model_name", "horizon"], kind="stable")
    average_gain_df = (
        weather_effect_df.groupby("model_name", dropna=False)[["rmse_gain", "mae_gain", "r2_gain", "nse_gain"]]
        .mean()
        .reset_index()
        .sort_values("rmse_gain", ascending=False, kind="stable")
    )

    output_dir = PROJECT_ROOT / "artifacts" / "advanced_seq" / "weather_ablation"
    save_csv(weather_effect_df, output_dir / "weather_effect_by_horizon.csv")
    save_csv(average_gain_df, output_dir / "weather_effect_average.csv")

    _plot_gain_bars(
        average_gain_df,
        metric="rmse_gain",
        output_path=output_dir / "avg_rmse_gain.png",
        title="Average RMSE Gain From Weather Variables",
    )
    _plot_gain_bars(
        average_gain_df,
        metric="mae_gain",
        output_path=output_dir / "avg_mae_gain.png",
        title="Average MAE Gain From Weather Variables",
    )
    _plot_gain_bars(
        average_gain_df,
        metric="r2_gain",
        output_path=output_dir / "avg_r2_gain.png",
        title="Average R2 Gain From Weather Variables",
    )

    save_json(
        {
            "weather_effect_by_horizon": str(output_dir / "weather_effect_by_horizon.csv"),
            "weather_effect_average": str(output_dir / "weather_effect_average.csv"),
            "avg_rmse_gain": str(output_dir / "avg_rmse_gain.png"),
            "avg_mae_gain": str(output_dir / "avg_mae_gain.png"),
            "avg_r2_gain": str(output_dir / "avg_r2_gain.png"),
        },
        output_dir / "plot_manifest.json",
    )
    logger.info("Saved weather-ablation outputs under %s", output_dir)


if __name__ == "__main__":
    main()
