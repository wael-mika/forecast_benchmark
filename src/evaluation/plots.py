"""Plotting helpers for benchmark experiment artifacts."""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation.metrics import DEFAULT_METRIC_COLUMNS
from src.evaluation.pipeline import build_prediction_frame, evaluate_prediction_frame
from src.training.train import infer_direct_target_columns, predict_with_xgboost
from src.utils.io import ensure_parent_dir, save_csv, save_json


DEFAULT_SPLITS = ("train", "validation", "test")
PRIMARY_METRICS = ("rmse", "mae", "mape", "smape", "mase", "rmsse")


def load_json(path: Path) -> dict:
    """Load a JSON object from disk."""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object at {path}.")
    return payload


def _require_xgboost():
    try:
        import xgboost as xgb  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise ImportError("xgboost is required to rebuild round-by-round metric curves.") from exc
    return xgb


def discover_round_model_paths(artifact_dir: Path, final_round: int) -> list[tuple[int, Path]]:
    """Discover saved checkpoint models for each boosting round."""
    round_paths: dict[int, Path] = {}
    checkpoint_dir = artifact_dir / "checkpoints"
    if checkpoint_dir.exists():
        for checkpoint_path in checkpoint_dir.glob("xgboost_*.ubj"):
            match = re.search(r"_(\d+)$", checkpoint_path.stem)
            if match is None:
                continue
            round_paths[int(match.group(1))] = checkpoint_path

    final_model_path = artifact_dir / f"model_rounds_{final_round:04d}.json"
    if final_model_path.exists():
        round_paths[final_round] = final_model_path
    elif (artifact_dir / "model.json").exists():
        round_paths[final_round] = artifact_dir / "model.json"

    if not round_paths:
        raise FileNotFoundError(f"No checkpoint or final model artifacts found under {artifact_dir}.")

    return sorted(round_paths.items())


def load_booster(model_path: Path):
    """Load a saved XGBoost booster from disk."""
    xgb = _require_xgboost()
    booster = xgb.Booster()
    booster.load_model(str(model_path))
    return booster


def build_round_metric_history(
    feature_df: pd.DataFrame,
    *,
    artifact_dir: Path,
    feature_columns: Iterable[str],
    target_column: str = "target",
    split_column: str = "split",
) -> pd.DataFrame:
    """Recompute split-level metrics for each saved boosting round."""
    training_summary = load_json(artifact_dir / "training_summary.json")
    final_round = int(training_summary["trained_num_boost_round"])
    enable_categorical = "station_id_feature" in set(feature_columns)

    history_frames: list[pd.DataFrame] = []
    for round_number, model_path in discover_round_model_paths(artifact_dir, final_round):
        booster = load_booster(model_path)
        predictions = predict_with_xgboost(
            booster,
            feature_df,
            feature_columns=list(feature_columns),
            target_column=target_column,
            enable_categorical=enable_categorical,
        )
        prediction_df = build_prediction_frame(
            feature_df,
            predictions.to_numpy(),
            actual_column=target_column,
            split_column=split_column,
        )
        overall_df, _ = evaluate_prediction_frame(
            prediction_df,
            split_column=split_column,
            group_column="unique_id",
        )
        overall_df["round"] = round_number
        overall_df["model_path"] = str(model_path)
        history_frames.append(overall_df)

    return (
        pd.concat(history_frames, ignore_index=True)
        .sort_values(["aggregation", "split", "round"], kind="stable")
        .reset_index(drop=True)
    )


def build_direct_round_metric_history(
    feature_df: pd.DataFrame,
    *,
    artifact_dir: Path,
    feature_columns: Iterable[str],
    split_column: str = "split",
) -> pd.DataFrame:
    """Recompute split-level metrics by round for each direct forecast horizon."""
    training_summary = load_json(artifact_dir / "training_summary.json")
    target_columns = training_summary.get("target_columns") or infer_direct_target_columns(feature_df)
    enable_categorical = "station_id_feature" in set(feature_columns)
    history_frames: list[pd.DataFrame] = []

    for target_column in target_columns:
        horizon = int(target_column.removeprefix("target_h"))
        horizon_dir = artifact_dir / f"h{horizon}"
        horizon_summary = load_json(horizon_dir / "training_summary.json")
        final_round = int(horizon_summary["trained_num_boost_round"])

        horizon_feature_df = feature_df.loc[
            :,
            ["unique_id", "forecast_origin_ds", split_column, f"target_h{horizon}_ds", target_column],
        ].rename(columns={f"target_h{horizon}_ds": "target_ds"})

        for round_number, model_path in discover_round_model_paths(horizon_dir, final_round):
            booster = load_booster(model_path)
            predictions = predict_with_xgboost(
                booster,
                feature_df,
                feature_columns=list(feature_columns),
                target_column=target_column,
                enable_categorical=enable_categorical,
            )
            prediction_df = build_prediction_frame(
                horizon_feature_df,
                predictions.to_numpy(),
                actual_column=target_column,
                split_column=split_column,
            )
            overall_df, _ = evaluate_prediction_frame(
                prediction_df,
                split_column=split_column,
                group_column="unique_id",
            )
            overall_df["round"] = round_number
            overall_df["horizon"] = horizon
            overall_df["model_path"] = str(model_path)
            history_frames.append(overall_df)

    return (
        pd.concat(history_frames, ignore_index=True)
        .sort_values(["horizon", "aggregation", "split", "round"], kind="stable")
        .reset_index(drop=True)
    )


def _prepare_curve_frame(round_metrics_df: pd.DataFrame, aggregation: str = "micro") -> pd.DataFrame:
    curve_df = round_metrics_df.loc[round_metrics_df["aggregation"] == aggregation].copy()
    if curve_df.empty:
        raise ValueError(f"No rows found for aggregation={aggregation!r}.")
    return curve_df.sort_values(["split", "round"], kind="stable")


def _new_figure_grid(metric_count: int, columns: int = 3) -> tuple[plt.Figure, np.ndarray]:
    rows = math.ceil(metric_count / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(columns * 5.2, rows * 3.8), constrained_layout=True)
    axes_array = np.atleast_1d(axes).reshape(rows, columns)
    return figure, axes_array


def plot_metric_curves(
    round_metrics_df: pd.DataFrame,
    *,
    output_path: Path,
    metrics: Iterable[str] = DEFAULT_METRIC_COLUMNS,
    splits: Iterable[str] = DEFAULT_SPLITS,
    aggregation: str = "micro",
    title: str = "XGBoost Metric Curves By Boosting Round",
) -> None:
    """Plot one subplot per metric with train/validation/test curves."""
    curve_df = _prepare_curve_frame(round_metrics_df, aggregation=aggregation)
    metric_names = list(metrics)
    figure, axes = _new_figure_grid(len(metric_names))
    color_map = {"train": "#1f77b4", "validation": "#ff7f0e", "test": "#2ca02c"}

    for axis, metric_name in zip(axes.flatten(), metric_names):
        for split_name in splits:
            split_df = curve_df.loc[curve_df["split"] == split_name]
            if split_df.empty or metric_name not in split_df.columns:
                continue
            axis.plot(
                split_df["round"],
                split_df[metric_name],
                marker="o",
                linewidth=2.0,
                label=split_name.title(),
                color=color_map.get(split_name),
            )
        axis.set_title(metric_name.upper())
        axis.set_xlabel("Boosting Round")
        axis.set_ylabel(metric_name.upper())
        axis.grid(True, alpha=0.25)

    for axis in axes.flatten()[len(metric_names):]:
        axis.axis("off")

    handles, labels = axes.flatten()[0].get_legend_handles_labels()
    if handles:
        axes.flatten()[0].legend(handles, labels, loc="best", frameon=False)
    figure.suptitle(title, fontsize=16)
    ensure_parent_dir(output_path)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_metrics_heatmap(
    metrics_summary_df: pd.DataFrame,
    *,
    output_path: Path,
    metrics: Iterable[str] = PRIMARY_METRICS,
    aggregation: str = "micro",
    title: str = "Final Split Metrics",
) -> None:
    """Plot a split-by-metric heatmap for the final evaluation summary."""
    filtered = metrics_summary_df.loc[metrics_summary_df["aggregation"] == aggregation].copy()
    if filtered.empty:
        raise ValueError(f"No rows found for aggregation={aggregation!r}.")

    metric_names = [metric for metric in metrics if metric in filtered.columns]
    heatmap_df = filtered.set_index("split").loc[:, metric_names]
    figure, axis = plt.subplots(figsize=(max(6, len(metric_names) * 1.2), 3.8), constrained_layout=True)
    image = axis.imshow(heatmap_df.to_numpy(dtype=float), aspect="auto", cmap="YlGnBu")

    axis.set_xticks(np.arange(len(metric_names)))
    axis.set_xticklabels([metric.upper() for metric in metric_names], rotation=30, ha="right")
    axis.set_yticks(np.arange(len(heatmap_df.index)))
    axis.set_yticklabels([str(label).title() for label in heatmap_df.index])
    axis.set_title(title)

    for row_index in range(heatmap_df.shape[0]):
        for column_index in range(heatmap_df.shape[1]):
            axis.text(
                column_index,
                row_index,
                f"{heatmap_df.iat[row_index, column_index]:.3f}",
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )

    figure.colorbar(image, ax=axis, shrink=0.85)
    ensure_parent_dir(output_path)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_feature_importance(
    feature_importance_df: pd.DataFrame,
    *,
    output_path: Path,
    top_n: int = 15,
    title: str = "Top Feature Importance By Gain",
) -> None:
    """Plot the highest-gain features from the trained XGBoost model."""
    top_features = feature_importance_df.sort_values("gain", ascending=False, kind="stable").head(top_n).copy()
    top_features = top_features.sort_values("gain", ascending=True, kind="stable")

    figure, axis = plt.subplots(figsize=(8.5, max(4.5, top_n * 0.35)), constrained_layout=True)
    axis.barh(top_features["feature"], top_features["gain"], color="#1f77b4")
    axis.set_xlabel("Gain")
    axis.set_title(title)
    axis.grid(True, axis="x", alpha=0.25)

    ensure_parent_dir(output_path)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_actual_vs_predicted(
    prediction_df: pd.DataFrame,
    *,
    output_path: Path,
    split: str = "test",
    max_points: int = 5000,
    title: str = "Actual Vs Predicted",
) -> None:
    """Plot a sampled actual-vs-predicted scatter for a chosen split."""
    split_df = prediction_df.loc[prediction_df["split"] == split].copy()
    if split_df.empty:
        raise ValueError(f"No prediction rows found for split={split!r}.")
    if len(split_df) > max_points:
        split_df = split_df.sample(max_points, random_state=42)

    min_value = min(split_df["y_true"].min(), split_df["y_pred"].min())
    max_value = max(split_df["y_true"].max(), split_df["y_pred"].max())

    figure, axis = plt.subplots(figsize=(6.5, 6.0), constrained_layout=True)
    axis.scatter(split_df["y_true"], split_df["y_pred"], alpha=0.25, s=10, color="#1f77b4")
    axis.plot([min_value, max_value], [min_value, max_value], linestyle="--", color="#d62728", linewidth=1.5)
    axis.set_xlabel("Actual")
    axis.set_ylabel("Predicted")
    axis.set_title(f"{title} ({split.title()})")
    axis.grid(True, alpha=0.25)

    ensure_parent_dir(output_path)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_residual_distribution(
    prediction_df: pd.DataFrame,
    *,
    output_path: Path,
    split: str = "test",
    bins: int = 50,
    title: str = "Residual Distribution",
) -> None:
    """Plot the residual histogram for one split."""
    split_df = prediction_df.loc[prediction_df["split"] == split].copy()
    if split_df.empty:
        raise ValueError(f"No prediction rows found for split={split!r}.")

    figure, axis = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
    axis.hist(split_df["residual"], bins=bins, color="#ff7f0e", alpha=0.85)
    axis.axvline(0.0, color="#111111", linestyle="--", linewidth=1.5)
    axis.set_xlabel("Residual (Prediction - Actual)")
    axis.set_ylabel("Count")
    axis.set_title(f"{title} ({split.title()})")
    axis.grid(True, axis="y", alpha=0.25)

    ensure_parent_dir(output_path)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_station_metric_bars(
    per_station_metrics_df: pd.DataFrame,
    *,
    output_path: Path,
    split: str = "test",
    metric: str = "rmse",
    top_n: int = 10,
    title: str = "Worst Stations By RMSE",
) -> None:
    """Plot the highest-error stations for one metric and split."""
    split_df = per_station_metrics_df.loc[per_station_metrics_df["split"] == split].copy()
    if split_df.empty or metric not in split_df.columns:
        raise ValueError(f"No per-station rows found for split={split!r} and metric={metric!r}.")

    ranked = split_df.sort_values(metric, ascending=False, kind="stable").head(top_n).copy()
    ranked = ranked.sort_values(metric, ascending=True, kind="stable")

    figure, axis = plt.subplots(figsize=(8.5, max(4.5, top_n * 0.35)), constrained_layout=True)
    axis.barh(ranked["unique_id"], ranked[metric], color="#d62728")
    axis.set_xlabel(metric.upper())
    axis.set_title(f"{title} ({split.title()})")
    axis.grid(True, axis="x", alpha=0.25)

    ensure_parent_dir(output_path)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_station_examples(
    prediction_df: pd.DataFrame,
    *,
    output_path: Path,
    split: str = "test",
    station_ids: Iterable[str] | None = None,
    max_stations: int = 3,
    max_points_per_station: int = 120,
    title: str = "Example Station Forecasts",
) -> None:
    """Plot actual and predicted series for a few representative stations."""
    split_df = prediction_df.loc[prediction_df["split"] == split].copy()
    if split_df.empty:
        raise ValueError(f"No prediction rows found for split={split!r}.")

    if station_ids is None:
        station_ids = (
            split_df.groupby("unique_id")["target_ds"]
            .size()
            .sort_values(ascending=False)
            .head(max_stations)
            .index
            .tolist()
        )
    else:
        station_ids = list(station_ids)[:max_stations]

    figure, axes = plt.subplots(len(list(station_ids)), 1, figsize=(11, max(3.2, 3.0 * len(list(station_ids)))), constrained_layout=True)
    axes_array = np.atleast_1d(axes)

    for axis, station_id in zip(axes_array, station_ids):
        station_df = (
            split_df.loc[split_df["unique_id"] == station_id]
            .sort_values("target_ds", kind="stable")
            .tail(max_points_per_station)
        )
        axis.plot(station_df["target_ds"], station_df["y_true"], label="Actual", linewidth=2.0, color="#1f77b4")
        axis.plot(station_df["target_ds"], station_df["y_pred"], label="Predicted", linewidth=1.8, color="#ff7f0e")
        axis.set_title(f"Station {station_id}")
        axis.grid(True, alpha=0.25)
        axis.tick_params(axis="x", rotation=30)

    axes_array[0].legend(loc="upper right", frameon=False)
    figure.suptitle(f"{title} ({split.title()})", fontsize=16)

    ensure_parent_dir(output_path)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _metric_label(metric_name: str) -> str:
    return metric_name.upper()


def plot_direct_metric_curves(
    round_metrics_df: pd.DataFrame,
    *,
    output_path: Path,
    metrics: Iterable[str] = DEFAULT_METRIC_COLUMNS,
    splits: Iterable[str] = DEFAULT_SPLITS,
    aggregation: str = "micro",
    title: str = "Direct Forecast Metric Curves By Boosting Round",
) -> None:
    """Plot average metric curves across horizons for direct multi-horizon experiments."""
    curve_df = _prepare_curve_frame(round_metrics_df, aggregation=aggregation)
    metric_names = [metric for metric in metrics if metric in curve_df.columns]
    figure, axes = _new_figure_grid(len(metric_names))
    color_map = {"train": "#1f77b4", "validation": "#ff7f0e", "test": "#2ca02c"}

    for axis, metric_name in zip(axes.flatten(), metric_names):
        for split_name in splits:
            split_df = curve_df.loc[curve_df["split"] == split_name]
            if split_df.empty:
                continue
            averaged = (
                split_df.groupby("round", dropna=False)[metric_name]
                .mean()
                .reset_index()
                .sort_values("round", kind="stable")
            )
            axis.plot(
                averaged["round"],
                averaged[metric_name],
                marker="o",
                linewidth=2.0,
                label=split_name.title(),
                color=color_map.get(split_name),
            )
        axis.set_title(_metric_label(metric_name))
        axis.set_xlabel("Round")
        axis.set_ylabel(_metric_label(metric_name))
        axis.grid(True, alpha=0.25)

    for axis in axes.flatten()[len(metric_names):]:
        axis.axis("off")

    handles, labels = axes.flatten()[0].get_legend_handles_labels()
    if handles:
        axes.flatten()[0].legend(handles, labels, loc="best", frameon=False)
    figure.suptitle(title, fontsize=16)
    ensure_parent_dir(output_path)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_direct_horizon_metric_curves(
    round_metrics_df: pd.DataFrame,
    *,
    output_path: Path,
    metric: str = "rmse",
    split: str = "test",
    aggregation: str = "micro",
    title: str = "Test RMSE By Horizon",
) -> None:
    """Plot one line per horizon for a chosen split and metric."""
    curve_df = _prepare_curve_frame(round_metrics_df, aggregation=aggregation)
    curve_df = curve_df.loc[curve_df["split"] == split].copy()
    if curve_df.empty or "horizon" not in curve_df.columns or metric not in curve_df.columns:
        raise ValueError(f"Cannot plot direct horizon metric curves for split={split!r} and metric={metric!r}.")

    figure, axis = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    for horizon, horizon_df in curve_df.groupby("horizon", dropna=False):
        axis.plot(
            horizon_df["round"],
            horizon_df[metric],
            marker="o",
            linewidth=2.0,
            label=f"H{int(horizon)}",
        )
    axis.set_title(title)
    axis.set_xlabel("Round")
    axis.set_ylabel(_metric_label(metric))
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best", frameon=False)
    ensure_parent_dir(output_path)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_direct_metrics_heatmap(
    metrics_summary_df: pd.DataFrame,
    *,
    output_path: Path,
    metrics: Iterable[str] = PRIMARY_METRICS,
    aggregation: str = "micro",
    title: str = "Final Direct Forecast Metrics",
) -> None:
    """Plot a split/horizon-by-metric heatmap for multi-horizon runs."""
    filtered = metrics_summary_df.loc[metrics_summary_df["aggregation"] == aggregation].copy()
    if filtered.empty or "horizon" not in filtered.columns:
        raise ValueError("Direct metric heatmaps require horizon-aware summary rows.")

    metric_names = [metric for metric in metrics if metric in filtered.columns]
    filtered["row_label"] = filtered["split"].astype(str).str.title() + " H" + filtered["horizon"].astype(int).astype(str)
    heatmap_df = filtered.set_index("row_label").loc[:, metric_names]

    figure, axis = plt.subplots(figsize=(max(6.5, len(metric_names) * 1.2), max(4.0, len(heatmap_df) * 0.45)), constrained_layout=True)
    image = axis.imshow(heatmap_df.to_numpy(dtype=float), aspect="auto", cmap="YlGnBu")

    axis.set_xticks(np.arange(len(metric_names)))
    axis.set_xticklabels([_metric_label(metric) for metric in metric_names], rotation=30, ha="right")
    axis.set_yticks(np.arange(len(heatmap_df.index)))
    axis.set_yticklabels(heatmap_df.index.tolist())
    axis.set_title(title)

    for row_index in range(heatmap_df.shape[0]):
        for column_index in range(heatmap_df.shape[1]):
            axis.text(
                column_index,
                row_index,
                f"{heatmap_df.iat[row_index, column_index]:.3f}",
                ha="center",
                va="center",
                fontsize=7,
                color="black",
            )

    figure.colorbar(image, ax=axis, shrink=0.85)
    ensure_parent_dir(output_path)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_direct_final_metric_bars(
    metrics_summary_df: pd.DataFrame,
    *,
    output_path: Path,
    metrics: Iterable[str] = ("rmse", "mae", "r2"),
    split: str = "test",
    aggregation: str = "micro",
    title: str = "Test Metrics By Horizon",
) -> None:
    """Plot grouped bars of final metrics by forecast horizon."""
    filtered = metrics_summary_df.loc[
        (metrics_summary_df["split"] == split) & (metrics_summary_df["aggregation"] == aggregation)
    ].copy()
    if filtered.empty or "horizon" not in filtered.columns:
        raise ValueError(f"No direct summary rows found for split={split!r} and aggregation={aggregation!r}.")

    metric_names = [metric for metric in metrics if metric in filtered.columns]
    figure, axes = plt.subplots(1, len(metric_names), figsize=(max(6.5, len(metric_names) * 4.2), 4.3), constrained_layout=True)
    axes_array = np.atleast_1d(axes)

    horizon_labels = [f"H{int(horizon)}" for horizon in filtered["horizon"]]
    x_positions = np.arange(len(filtered))
    for axis, metric_name in zip(axes_array, metric_names):
        axis.bar(x_positions, filtered[metric_name], color="#1f77b4")
        axis.set_xticks(x_positions)
        axis.set_xticklabels(horizon_labels)
        axis.set_title(_metric_label(metric_name))
        axis.grid(True, axis="y", alpha=0.25)

    figure.suptitle(title, fontsize=16)
    ensure_parent_dir(output_path)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_direct_actual_vs_predicted(
    prediction_df: pd.DataFrame,
    *,
    output_path: Path,
    split: str = "test",
    max_points_per_horizon: int = 3000,
    title: str = "Actual Vs Predicted",
) -> None:
    """Plot actual-vs-predicted scatter panels by forecast horizon."""
    split_df = prediction_df.loc[prediction_df["split"] == split].copy()
    horizons = sorted(split_df["horizon"].dropna().unique().tolist())
    if split_df.empty or not horizons:
        raise ValueError(f"No direct prediction rows found for split={split!r}.")

    figure, axes = plt.subplots(1, len(horizons), figsize=(max(6.5, len(horizons) * 4.6), 4.8), constrained_layout=True)
    axes_array = np.atleast_1d(axes)

    for axis, horizon in zip(axes_array, horizons):
        horizon_df = split_df.loc[split_df["horizon"] == horizon].copy()
        if len(horizon_df) > max_points_per_horizon:
            horizon_df = horizon_df.sample(max_points_per_horizon, random_state=42)
        min_value = min(horizon_df["y_true"].min(), horizon_df["y_pred"].min())
        max_value = max(horizon_df["y_true"].max(), horizon_df["y_pred"].max())
        axis.scatter(horizon_df["y_true"], horizon_df["y_pred"], alpha=0.22, s=8, color="#1f77b4")
        axis.plot([min_value, max_value], [min_value, max_value], linestyle="--", color="#d62728", linewidth=1.3)
        axis.set_title(f"H{int(horizon)}")
        axis.set_xlabel("Actual")
        axis.set_ylabel("Predicted")
        axis.grid(True, alpha=0.25)

    figure.suptitle(f"{title} ({split.title()})", fontsize=16)
    ensure_parent_dir(output_path)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_direct_residual_distribution(
    prediction_df: pd.DataFrame,
    *,
    output_path: Path,
    split: str = "test",
    bins: int = 50,
    title: str = "Residual Distribution",
) -> None:
    """Plot residual histograms by horizon."""
    split_df = prediction_df.loc[prediction_df["split"] == split].copy()
    horizons = sorted(split_df["horizon"].dropna().unique().tolist())
    if split_df.empty or not horizons:
        raise ValueError(f"No direct prediction rows found for split={split!r}.")

    figure, axes = plt.subplots(1, len(horizons), figsize=(max(7.0, len(horizons) * 4.2), 4.4), constrained_layout=True)
    axes_array = np.atleast_1d(axes)

    for axis, horizon in zip(axes_array, horizons):
        horizon_df = split_df.loc[split_df["horizon"] == horizon]
        axis.hist(horizon_df["residual"], bins=bins, color="#ff7f0e", alpha=0.85)
        axis.axvline(0.0, color="#111111", linestyle="--", linewidth=1.3)
        axis.set_title(f"H{int(horizon)}")
        axis.set_xlabel("Residual")
        axis.set_ylabel("Count")
        axis.grid(True, axis="y", alpha=0.25)

    figure.suptitle(f"{title} ({split.title()})", fontsize=16)
    ensure_parent_dir(output_path)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_direct_station_metric_bars(
    per_station_metrics_df: pd.DataFrame,
    *,
    output_path: Path,
    split: str = "test",
    metric: str = "rmse",
    top_n: int = 8,
    title: str = "Worst Stations By RMSE",
) -> None:
    """Plot worst-performing stations for each horizon."""
    split_df = per_station_metrics_df.loc[per_station_metrics_df["split"] == split].copy()
    horizons = sorted(split_df["horizon"].dropna().unique().tolist())
    if split_df.empty or metric not in split_df.columns or not horizons:
        raise ValueError(f"No direct per-station rows found for split={split!r} and metric={metric!r}.")

    figure, axes = plt.subplots(1, len(horizons), figsize=(max(8.5, len(horizons) * 4.4), max(4.8, top_n * 0.32)), constrained_layout=True)
    axes_array = np.atleast_1d(axes)

    for axis, horizon in zip(axes_array, horizons):
        horizon_df = split_df.loc[split_df["horizon"] == horizon].copy()
        ranked = horizon_df.sort_values(metric, ascending=False, kind="stable").head(top_n)
        ranked = ranked.sort_values(metric, ascending=True, kind="stable")
        axis.barh(ranked["unique_id"], ranked[metric], color="#d62728")
        axis.set_title(f"H{int(horizon)}")
        axis.set_xlabel(_metric_label(metric))
        axis.grid(True, axis="x", alpha=0.25)

    figure.suptitle(f"{title} ({split.title()})", fontsize=16)
    ensure_parent_dir(output_path)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_direct_station_examples(
    prediction_df: pd.DataFrame,
    *,
    output_path: Path,
    split: str = "test",
    max_stations: int = 2,
    max_points_per_station: int = 90,
    title: str = "Example Station Forecasts",
) -> None:
    """Plot actual and predicted series for a few stations across horizons."""
    split_df = prediction_df.loc[prediction_df["split"] == split].copy()
    horizons = sorted(split_df["horizon"].dropna().unique().tolist())
    if split_df.empty or not horizons:
        raise ValueError(f"No direct prediction rows found for split={split!r}.")

    station_ids = (
        split_df.loc[split_df["horizon"] == horizons[0]]
        .groupby("unique_id")["target_ds"]
        .size()
        .sort_values(ascending=False)
        .head(max_stations)
        .index
        .tolist()
    )
    figure, axes = plt.subplots(
        len(station_ids),
        len(horizons),
        figsize=(max(10.0, len(horizons) * 4.2), max(4.8, len(station_ids) * 3.0)),
        constrained_layout=True,
    )
    axes_array = np.atleast_2d(axes)

    for row_index, station_id in enumerate(station_ids):
        for column_index, horizon in enumerate(horizons):
            axis = axes_array[row_index, column_index]
            station_df = (
                split_df.loc[(split_df["unique_id"] == station_id) & (split_df["horizon"] == horizon)]
                .sort_values("target_ds", kind="stable")
                .tail(max_points_per_station)
            )
            axis.plot(station_df["target_ds"], station_df["y_true"], label="Actual", linewidth=2.0, color="#1f77b4")
            axis.plot(station_df["target_ds"], station_df["y_pred"], label="Predicted", linewidth=1.6, color="#ff7f0e")
            axis.set_title(f"{station_id} | H{int(horizon)}")
            axis.grid(True, alpha=0.25)
            axis.tick_params(axis="x", rotation=30)

    axes_array[0, 0].legend(loc="upper right", frameon=False)
    figure.suptitle(f"{title} ({split.title()})", fontsize=16)
    ensure_parent_dir(output_path)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_direct_forecast_window_examples(
    prediction_df: pd.DataFrame,
    *,
    output_path: Path,
    split: str = "test",
    max_examples: int = 6,
    title: str = "Predicted Vs Target Forecast Windows",
) -> None:
    """Plot a few complete multi-horizon forecast windows for direct models."""
    split_df = prediction_df.loc[prediction_df["split"] == split].copy()
    horizons = sorted(split_df["horizon"].dropna().unique().tolist())
    if split_df.empty or not horizons:
        raise ValueError(f"No direct prediction rows found for split={split!r}.")

    window_df = (
        split_df.pivot_table(
            index=["unique_id", "forecast_origin_ds"],
            columns="horizon",
            values=["y_true", "y_pred"],
            aggfunc="first",
        )
        .sort_index()
        .dropna()
    )
    if window_df.empty:
        raise ValueError(f"No complete forecast windows found for split={split!r}.")

    available_windows = window_df.reset_index()
    top_station_ids = (
        available_windows.groupby("unique_id")["forecast_origin_ds"]
        .size()
        .sort_values(ascending=False)
        .head(min(3, available_windows["unique_id"].nunique()))
        .index
        .tolist()
    )

    selected_frames: list[pd.DataFrame] = []
    per_station_examples = max(1, math.ceil(max_examples / max(1, len(top_station_ids))))
    for station_id in top_station_ids:
        selected_frames.append(
            available_windows.loc[available_windows["unique_id"] == station_id]
            .sort_values("forecast_origin_ds", kind="stable")
            .tail(per_station_examples)
        )

    selected_windows = (
        pd.concat(selected_frames, ignore_index=True)
        .sort_values(["unique_id", "forecast_origin_ds"], kind="stable")
        .head(max_examples)
    )
    x_positions = np.arange(1, len(horizons) + 1)

    figure, axes = plt.subplots(
        math.ceil(len(selected_windows) / 2),
        2,
        figsize=(12.0, max(4.6, math.ceil(len(selected_windows) / 2) * 3.6)),
        constrained_layout=True,
    )
    axes_array = np.atleast_1d(axes).reshape(-1)

    for axis, (_, window_row) in zip(axes_array, selected_windows.iterrows()):
        actual_values = [float(window_row[("y_true", horizon)]) for horizon in horizons]
        predicted_values = [float(window_row[("y_pred", horizon)]) for horizon in horizons]
        if isinstance(window_row.index, pd.MultiIndex):
            unique_id = window_row[("unique_id", "")]
            origin_raw = window_row[("forecast_origin_ds", "")]
        else:
            unique_id = window_row["unique_id"]
            origin_raw = window_row["forecast_origin_ds"]
        axis.plot(x_positions, actual_values, marker="o", linewidth=2.0, color="#1f77b4", label="Target")
        axis.plot(x_positions, predicted_values, marker="o", linewidth=2.0, color="#ff7f0e", label="Prediction")
        axis.set_xticks(x_positions)
        axis.set_xticklabels([f"H{int(horizon)}" for horizon in horizons])
        origin_label = pd.to_datetime(origin_raw).strftime("%Y-%m-%d")
        axis.set_title(f"{unique_id} | origin {origin_label}")
        axis.grid(True, alpha=0.25)

    for axis in axes_array[len(selected_windows):]:
        axis.axis("off")

    axes_array[0].legend(loc="best", frameon=False)
    figure.suptitle(f"{title} ({split.title()})", fontsize=16)
    ensure_parent_dir(output_path)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_direct_feature_importance(
    feature_importance_by_horizon: dict[int, pd.DataFrame],
    *,
    output_path: Path,
    top_n: int = 10,
    title: str = "Top Feature Importance By Horizon",
) -> None:
    """Plot top feature importance values for each direct horizon model."""
    horizons = sorted(feature_importance_by_horizon)
    figure, axes = plt.subplots(1, len(horizons), figsize=(max(8.5, len(horizons) * 4.2), max(4.8, top_n * 0.32)), constrained_layout=True)
    axes_array = np.atleast_1d(axes)

    for axis, horizon in zip(axes_array, horizons):
        top_features = feature_importance_by_horizon[horizon].sort_values("gain", ascending=False, kind="stable").head(top_n)
        top_features = top_features.sort_values("gain", ascending=True, kind="stable")
        axis.barh(top_features["feature"], top_features["gain"], color="#1f77b4")
        axis.set_title(f"H{int(horizon)}")
        axis.set_xlabel("Gain")
        axis.grid(True, axis="x", alpha=0.25)

    figure.suptitle(title, fontsize=16)
    ensure_parent_dir(output_path)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_loss_curves(
    loss_history_df: pd.DataFrame,
    *,
    output_path: Path,
    title: str = "Loss Curves",
) -> None:
    """Plot train/validation/test loss by epoch."""
    if loss_history_df.empty:
        raise ValueError("loss_history_df is empty.")

    figure, axis = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    color_map = {"train": "#1f77b4", "validation": "#ff7f0e", "test": "#2ca02c"}
    for split_name, split_df in loss_history_df.groupby("split", dropna=False):
        axis.plot(
            split_df["epoch"],
            split_df["loss"],
            marker="o",
            linewidth=2.0,
            label=str(split_name).title(),
            color=color_map.get(str(split_name)),
        )
    axis.set_title(title)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best", frameon=False)
    ensure_parent_dir(output_path)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_epoch_metric_curves(
    epoch_metrics_df: pd.DataFrame,
    *,
    output_path: Path,
    metrics: Iterable[str] = PRIMARY_METRICS,
    splits: Iterable[str] = ("validation", "test"),
    aggregation: str = "micro",
    title: str = "Epoch Metric Curves",
) -> None:
    """Plot epoch-by-epoch metrics averaged across horizons."""
    filtered = epoch_metrics_df.loc[epoch_metrics_df["aggregation"] == aggregation].copy()
    if filtered.empty:
        raise ValueError(f"No rows found for aggregation={aggregation!r}.")

    metric_names = [metric for metric in metrics if metric in filtered.columns]
    figure, axes = _new_figure_grid(len(metric_names))
    color_map = {"validation": "#ff7f0e", "test": "#2ca02c"}

    for axis, metric_name in zip(axes.flatten(), metric_names):
        for split_name in splits:
            split_df = filtered.loc[filtered["split"] == split_name]
            if split_df.empty:
                continue
            averaged = split_df.groupby("epoch", dropna=False)[metric_name].mean().reset_index()
            axis.plot(
                averaged["epoch"],
                averaged[metric_name],
                marker="o",
                linewidth=2.0,
                label=str(split_name).title(),
                color=color_map.get(str(split_name)),
            )
        axis.set_title(_metric_label(metric_name))
        axis.set_xlabel("Epoch")
        axis.set_ylabel(_metric_label(metric_name))
        axis.grid(True, alpha=0.25)

    for axis in axes.flatten()[len(metric_names):]:
        axis.axis("off")

    handles, labels = axes.flatten()[0].get_legend_handles_labels()
    if handles:
        axes.flatten()[0].legend(handles, labels, loc="best", frameon=False)
    figure.suptitle(title, fontsize=16)
    ensure_parent_dir(output_path)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_model_comparison_bars(
    metrics_df: pd.DataFrame,
    *,
    output_path: Path,
    metric: str = "rmse",
    split: str = "test",
    aggregation: str = "micro",
    title: str = "Model Comparison",
) -> None:
    """Plot grouped bars comparing models by horizon for one metric."""
    filtered = metrics_df.loc[
        (metrics_df["split"] == split) & (metrics_df["aggregation"] == aggregation)
    ].copy()
    if filtered.empty or metric not in filtered.columns:
        raise ValueError(f"No comparison rows found for split={split!r} and metric={metric!r}.")

    model_names = sorted(filtered["model_name"].unique().tolist())
    horizons = sorted(filtered["horizon"].unique().tolist())
    x_positions = np.arange(len(horizons))
    width = 0.8 / max(1, len(model_names))

    figure, axis = plt.subplots(figsize=(max(7.0, len(horizons) * 2.5), 4.8), constrained_layout=True)
    for index, model_name in enumerate(model_names):
        model_df = filtered.loc[filtered["model_name"] == model_name].sort_values("horizon", kind="stable")
        axis.bar(
            x_positions + ((index - (len(model_names) - 1) / 2) * width),
            model_df[metric],
            width=width,
            label=str(model_name).upper(),
        )

    axis.set_xticks(x_positions)
    axis.set_xticklabels([f"H{int(horizon)}" for horizon in horizons])
    axis.set_xlabel("Forecast Horizon")
    axis.set_ylabel(_metric_label(metric))
    axis.set_title(title)
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend(loc="best", frameon=False)
    ensure_parent_dir(output_path)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_model_comparison_heatmap(
    metrics_df: pd.DataFrame,
    *,
    output_path: Path,
    metrics: Iterable[str] = ("rmse", "mae", "r2"),
    split: str = "test",
    aggregation: str = "micro",
    title: str = "Average Test Metrics By Model",
) -> None:
    """Plot a model-by-metric heatmap averaged across horizons."""
    filtered = metrics_df.loc[
        (metrics_df["split"] == split) & (metrics_df["aggregation"] == aggregation)
    ].copy()
    if filtered.empty:
        raise ValueError(f"No comparison rows found for split={split!r}.")

    metric_names = [metric for metric in metrics if metric in filtered.columns]
    heatmap_df = (
        filtered.groupby("model_name", dropna=False)[metric_names]
        .mean()
        .sort_index(kind="stable")
    )

    figure, axis = plt.subplots(figsize=(max(6.5, len(metric_names) * 1.6), max(3.8, len(heatmap_df) * 0.8)), constrained_layout=True)
    image = axis.imshow(heatmap_df.to_numpy(dtype=float), aspect="auto", cmap="YlGnBu")
    axis.set_xticks(np.arange(len(metric_names)))
    axis.set_xticklabels([_metric_label(metric) for metric in metric_names], rotation=30, ha="right")
    axis.set_yticks(np.arange(len(heatmap_df.index)))
    axis.set_yticklabels([str(label).upper() for label in heatmap_df.index])
    axis.set_title(title)

    for row_index in range(heatmap_df.shape[0]):
        for column_index in range(heatmap_df.shape[1]):
            axis.text(
                column_index,
                row_index,
                f"{heatmap_df.iat[row_index, column_index]:.3f}",
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )

    figure.colorbar(image, ax=axis, shrink=0.85)
    ensure_parent_dir(output_path)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def generate_xgboost_plot_bundle(
    *,
    artifact_dir: Path,
    feature_df: pd.DataFrame,
    feature_columns: Iterable[str],
    target_column: str = "target",
    split_column: str = "split",
) -> dict[str, str]:
    """Generate round-by-round and summary plots for one XGBoost artifact directory."""
    training_summary = load_json(artifact_dir / "training_summary.json")
    if "per_horizon" in training_summary:
        plot_dir = artifact_dir / "plots"
        ensure_parent_dir(plot_dir / ".keep")

        round_metrics_df = build_direct_round_metric_history(
            feature_df,
            artifact_dir=artifact_dir,
            feature_columns=feature_columns,
            split_column=split_column,
        )
        save_csv(round_metrics_df, plot_dir / "round_metrics.csv")

        metrics_summary_df = pd.read_csv(artifact_dir / "metrics_summary.csv")
        per_station_metrics_df = pd.read_csv(artifact_dir / "metrics_by_station.csv")
        prediction_df = pd.read_parquet(artifact_dir / "predictions.parquet")

        feature_importance_by_horizon = {
            int(horizon_dir.name.removeprefix("h")): pd.read_csv(horizon_dir / "feature_importance.csv")
            for horizon_dir in sorted(artifact_dir.glob("h[0-9]*"))
            if (horizon_dir / "feature_importance.csv").exists()
        }

        plot_direct_metric_curves(
            round_metrics_df,
            output_path=plot_dir / "all_metric_curves.png",
            metrics=DEFAULT_METRIC_COLUMNS,
            title="Direct XGBoost Metric Curves By Boosting Round",
        )
        plot_direct_metric_curves(
            round_metrics_df,
            output_path=plot_dir / "primary_metric_curves.png",
            metrics=PRIMARY_METRICS,
            title="Primary Direct XGBoost Metric Curves By Boosting Round",
        )
        plot_direct_horizon_metric_curves(
            round_metrics_df,
            output_path=plot_dir / "test_rmse_by_horizon_curve.png",
            metric="rmse",
            split="test",
            title="Test RMSE By Horizon Across Boosting Rounds",
        )
        plot_direct_metrics_heatmap(
            metrics_summary_df,
            output_path=plot_dir / "final_metric_heatmap.png",
            title="Final Direct XGBoost Metrics",
        )
        plot_direct_final_metric_bars(
            metrics_summary_df,
            output_path=plot_dir / "test_metric_bars.png",
            title="Direct XGBoost Test Metrics By Horizon",
        )
        if feature_importance_by_horizon:
            plot_direct_feature_importance(
                feature_importance_by_horizon,
                output_path=plot_dir / "feature_importance_top10.png",
            )
        plot_direct_actual_vs_predicted(
            prediction_df,
            output_path=plot_dir / "test_actual_vs_predicted.png",
        )
        plot_direct_residual_distribution(
            prediction_df,
            output_path=plot_dir / "test_residual_distribution.png",
        )
        plot_direct_station_metric_bars(
            per_station_metrics_df,
            output_path=plot_dir / "test_worst_station_rmse.png",
        )
        plot_direct_station_examples(
            prediction_df,
            output_path=plot_dir / "test_station_examples.png",
        )
        plot_direct_forecast_window_examples(
            prediction_df,
            output_path=plot_dir / "test_forecast_windows.png",
        )

        manifest = {
            "round_metrics_csv": str(plot_dir / "round_metrics.csv"),
            "all_metric_curves": str(plot_dir / "all_metric_curves.png"),
            "primary_metric_curves": str(plot_dir / "primary_metric_curves.png"),
            "test_rmse_by_horizon_curve": str(plot_dir / "test_rmse_by_horizon_curve.png"),
            "final_metric_heatmap": str(plot_dir / "final_metric_heatmap.png"),
            "test_metric_bars": str(plot_dir / "test_metric_bars.png"),
            "test_actual_vs_predicted": str(plot_dir / "test_actual_vs_predicted.png"),
            "test_residual_distribution": str(plot_dir / "test_residual_distribution.png"),
            "test_worst_station_rmse": str(plot_dir / "test_worst_station_rmse.png"),
            "test_station_examples": str(plot_dir / "test_station_examples.png"),
            "test_forecast_windows": str(plot_dir / "test_forecast_windows.png"),
        }
        if feature_importance_by_horizon:
            manifest["feature_importance_top10"] = str(plot_dir / "feature_importance_top10.png")
        save_json(manifest, plot_dir / "plot_manifest.json")
        return manifest

    plot_dir = artifact_dir / "plots"
    ensure_parent_dir(plot_dir / ".keep")

    round_metrics_df = build_round_metric_history(
        feature_df,
        artifact_dir=artifact_dir,
        feature_columns=feature_columns,
        target_column=target_column,
        split_column=split_column,
    )
    save_csv(round_metrics_df, plot_dir / "round_metrics.csv")

    metrics_summary_df = pd.read_csv(artifact_dir / "metrics_summary.csv")
    per_station_metrics_df = pd.read_csv(artifact_dir / "metrics_by_station.csv")
    prediction_df = pd.read_parquet(artifact_dir / "predictions.parquet")
    feature_importance_df = pd.read_csv(artifact_dir / "feature_importance.csv")

    plot_metric_curves(
        round_metrics_df,
        output_path=plot_dir / "all_metric_curves.png",
        metrics=DEFAULT_METRIC_COLUMNS,
    )
    plot_metric_curves(
        round_metrics_df,
        output_path=plot_dir / "primary_metric_curves.png",
        metrics=PRIMARY_METRICS,
        title="Primary XGBoost Metric Curves By Boosting Round",
    )
    plot_metrics_heatmap(
        metrics_summary_df,
        output_path=plot_dir / "final_metric_heatmap.png",
    )
    plot_feature_importance(
        feature_importance_df,
        output_path=plot_dir / "feature_importance_top15.png",
    )
    plot_actual_vs_predicted(
        prediction_df,
        output_path=plot_dir / "test_actual_vs_predicted.png",
    )
    plot_residual_distribution(
        prediction_df,
        output_path=plot_dir / "test_residual_distribution.png",
    )
    plot_station_metric_bars(
        per_station_metrics_df,
        output_path=plot_dir / "test_worst_station_rmse.png",
    )
    plot_station_examples(
        prediction_df,
        output_path=plot_dir / "test_station_examples.png",
    )

    manifest = {
        "round_metrics_csv": str(plot_dir / "round_metrics.csv"),
        "all_metric_curves": str(plot_dir / "all_metric_curves.png"),
        "primary_metric_curves": str(plot_dir / "primary_metric_curves.png"),
        "final_metric_heatmap": str(plot_dir / "final_metric_heatmap.png"),
        "feature_importance_top15": str(plot_dir / "feature_importance_top15.png"),
        "test_actual_vs_predicted": str(plot_dir / "test_actual_vs_predicted.png"),
        "test_residual_distribution": str(plot_dir / "test_residual_distribution.png"),
        "test_worst_station_rmse": str(plot_dir / "test_worst_station_rmse.png"),
        "test_station_examples": str(plot_dir / "test_station_examples.png"),
    }
    save_json(manifest, plot_dir / "plot_manifest.json")
    return manifest


def generate_neural_plot_bundle(
    *,
    artifact_dir: Path,
    model_label: str,
) -> dict[str, str]:
    """Generate loss, metric, and prediction plots for ANN/LSTM artifact directories."""
    plot_dir = artifact_dir / "plots"
    ensure_parent_dir(plot_dir / ".keep")

    loss_history_df = pd.read_csv(artifact_dir / "loss_history.csv")
    epoch_metrics_df = pd.read_csv(artifact_dir / "epoch_metrics.csv")
    metrics_summary_df = pd.read_csv(artifact_dir / "metrics_summary.csv")
    per_station_metrics_df = pd.read_csv(artifact_dir / "metrics_by_station.csv")
    prediction_df = pd.read_parquet(artifact_dir / "predictions.parquet")

    plot_loss_curves(
        loss_history_df,
        output_path=plot_dir / "loss_curves.png",
        title=f"{model_label} Loss Curves",
    )
    plot_epoch_metric_curves(
        epoch_metrics_df,
        output_path=plot_dir / "primary_metric_curves.png",
        metrics=PRIMARY_METRICS,
        title=f"{model_label} Validation/Test Metric Curves",
    )
    plot_direct_metrics_heatmap(
        metrics_summary_df,
        output_path=plot_dir / "final_metric_heatmap.png",
        title=f"Final {model_label} Metrics",
    )
    plot_direct_final_metric_bars(
        metrics_summary_df,
        output_path=plot_dir / "test_metric_bars.png",
        title=f"{model_label} Test Metrics By Horizon",
    )
    plot_direct_actual_vs_predicted(
        prediction_df,
        output_path=plot_dir / "test_actual_vs_predicted.png",
    )
    plot_direct_residual_distribution(
        prediction_df,
        output_path=plot_dir / "test_residual_distribution.png",
    )
    plot_direct_station_metric_bars(
        per_station_metrics_df,
        output_path=plot_dir / "test_worst_station_rmse.png",
    )
    plot_direct_station_examples(
        prediction_df,
        output_path=plot_dir / "test_station_examples.png",
    )
    plot_direct_forecast_window_examples(
        prediction_df,
        output_path=plot_dir / "test_forecast_windows.png",
    )

    manifest = {
        "loss_curves": str(plot_dir / "loss_curves.png"),
        "primary_metric_curves": str(plot_dir / "primary_metric_curves.png"),
        "final_metric_heatmap": str(plot_dir / "final_metric_heatmap.png"),
        "test_metric_bars": str(plot_dir / "test_metric_bars.png"),
        "test_actual_vs_predicted": str(plot_dir / "test_actual_vs_predicted.png"),
        "test_residual_distribution": str(plot_dir / "test_residual_distribution.png"),
        "test_worst_station_rmse": str(plot_dir / "test_worst_station_rmse.png"),
        "test_station_examples": str(plot_dir / "test_station_examples.png"),
        "test_forecast_windows": str(plot_dir / "test_forecast_windows.png"),
    }
    save_json(manifest, plot_dir / "plot_manifest.json")
    return manifest


def generate_model_comparison_plot_bundle(
    *,
    output_dir: Path,
    comparison_metrics_df: pd.DataFrame,
) -> dict[str, str]:
    """Generate grouped comparison plots across multiple models."""
    ensure_parent_dir(output_dir / ".keep")

    plot_model_comparison_bars(
        comparison_metrics_df,
        output_path=output_dir / "test_rmse_by_horizon.png",
        metric="rmse",
        title="Test RMSE Comparison By Horizon",
    )
    plot_model_comparison_bars(
        comparison_metrics_df,
        output_path=output_dir / "test_mae_by_horizon.png",
        metric="mae",
        title="Test MAE Comparison By Horizon",
    )
    plot_model_comparison_bars(
        comparison_metrics_df,
        output_path=output_dir / "test_r2_by_horizon.png",
        metric="r2",
        title="Test R2 Comparison By Horizon",
    )
    plot_model_comparison_heatmap(
        comparison_metrics_df,
        output_path=output_dir / "test_metric_heatmap.png",
    )

    manifest = {
        "test_rmse_by_horizon": str(output_dir / "test_rmse_by_horizon.png"),
        "test_mae_by_horizon": str(output_dir / "test_mae_by_horizon.png"),
        "test_r2_by_horizon": str(output_dir / "test_r2_by_horizon.png"),
        "test_metric_heatmap": str(output_dir / "test_metric_heatmap.png"),
    }
    save_json(manifest, output_dir / "plot_manifest.json")
    return manifest
