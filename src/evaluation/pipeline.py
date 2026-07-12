"""Evaluation pipeline helpers that turn raw model outputs into saved artifacts.

This module sits between model inference and the final evaluation files. It
builds long-form prediction tables, attaches train-based scaling references
for MASE and RMSSE, computes split-level and station-level summaries, and
writes the results to each experiment directory.

Main helpers
------------
build_prediction_frame
    Attach single-output predictions to the base feature frame.
build_direct_prediction_frame
    Convert direct multi-horizon predictions from wide format to long format.
evaluate_prediction_frame
    Score a standard prediction frame and return overall plus per-station metrics.
evaluate_direct_prediction_frame
    Score a direct multi-horizon prediction frame by split and horizon.
save_evaluation_outputs
    Save predictions, metric tables, and a small manifest to disk.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.metrics import DEFAULT_METRIC_COLUMNS, build_scale_reference, compute_metric_bundle, summarize_prediction_metrics
from src.utils.io import save_csv, save_json, save_parquet


def build_prediction_frame(
    feature_df: pd.DataFrame,
    predictions: np.ndarray,
    *,
    actual_column: str = "target",
    split_column: str = "split",
) -> pd.DataFrame:
    """Attach one predicted value per row to the supervised feature frame."""
    prediction_df = feature_df.loc[:, ["unique_id", "forecast_origin_ds", "target_ds", split_column, actual_column]].copy()
    prediction_df = prediction_df.rename(columns={actual_column: "y_true"})
    prediction_df["y_pred"] = predictions
    prediction_df["residual"] = prediction_df["y_pred"] - prediction_df["y_true"]
    return prediction_df.sort_values(["unique_id", "target_ds"], kind="stable").reset_index(drop=True)


def evaluate_prediction_frame(
    prediction_df: pd.DataFrame,
    *,
    split_column: str = "split",
    group_column: str = "unique_id",
    scale_reference_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score a standard prediction frame and return overall plus per-station summaries."""
    if scale_reference_df is None:
        train_reference = prediction_df.loc[prediction_df[split_column] == "train", [group_column, "target_ds", "y_true"]]
        scale_reference = build_scale_reference(
            train_reference.rename(columns={"y_true": "target"}),
            group_column=group_column,
            time_column="target_ds",
            target_column="target",
        )
    else:
        scale_reference = scale_reference_df.copy()

    enriched_predictions = prediction_df.merge(scale_reference, on=group_column, how="left", validate="m:1")
    return summarize_prediction_metrics(
        enriched_predictions,
        split_column=split_column,
        group_column=group_column,
        actual_column="y_true",
        prediction_column="y_pred",
    )


def save_evaluation_outputs(
    prediction_df: pd.DataFrame,
    overall_metrics_df: pd.DataFrame,
    per_station_metrics_df: pd.DataFrame,
    *,
    artifact_dir: Path,
) -> None:
    """Save predictions, summary tables, and a manifest into one artifact directory."""
    save_parquet(prediction_df, artifact_dir / "predictions.parquet")
    save_csv(overall_metrics_df, artifact_dir / "metrics_summary.csv")
    save_csv(per_station_metrics_df, artifact_dir / "metrics_by_station.csv")

    summary_payload = {
        "metrics_summary_path": str(artifact_dir / "metrics_summary.csv"),
        "metrics_by_station_path": str(artifact_dir / "metrics_by_station.csv"),
        "predictions_path": str(artifact_dir / "predictions.parquet"),
    }
    save_json(summary_payload, artifact_dir / "evaluation_manifest.json")


def build_direct_prediction_frame(
    feature_df: pd.DataFrame,
    prediction_columns_df: pd.DataFrame,
    *,
    split_column: str = "split",
) -> pd.DataFrame:
    """Convert wide direct predictions into one long table with one row per horizon."""
    prediction_frames: list[pd.DataFrame] = []
    prediction_columns = sorted(
        [column for column in prediction_columns_df.columns if column.startswith("prediction_h")],
        key=lambda column: int(column.removeprefix("prediction_h")),
    )
    for prediction_column in prediction_columns:
        horizon = int(prediction_column.removeprefix("prediction_h"))
        target_column = f"target_h{horizon}"
        target_ds_column = f"target_h{horizon}_ds"
        prediction_frame = feature_df.loc[
            :,
            ["unique_id", "forecast_origin_ds", split_column, target_ds_column, target_column],
        ].copy()
        prediction_frame = prediction_frame.rename(columns={target_ds_column: "target_ds", target_column: "y_true"})
        prediction_frame["horizon"] = horizon
        prediction_frame["y_pred"] = prediction_columns_df[prediction_column].to_numpy()
        prediction_frame["residual"] = prediction_frame["y_pred"] - prediction_frame["y_true"]
        prediction_frames.append(prediction_frame)

    return (
        pd.concat(prediction_frames, ignore_index=True)
        .sort_values(["horizon", "unique_id", "target_ds"], kind="stable")
        .reset_index(drop=True)
    )


def evaluate_direct_prediction_frame(
    prediction_df: pd.DataFrame,
    *,
    split_column: str = "split",
    group_column: str = "unique_id",
    scale_reference_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score direct multi-horizon predictions separately for each split and horizon."""
    if scale_reference_df is None:
        train_reference = (
            prediction_df.loc[prediction_df[split_column] == "train", [group_column, "target_ds", "y_true"]]
            .drop_duplicates(subset=[group_column, "target_ds"])
            .rename(columns={"y_true": "target"})
        )
        scale_reference = build_scale_reference(
            train_reference,
            group_column=group_column,
            time_column="target_ds",
            target_column="target",
        )
    else:
        scale_reference = scale_reference_df.copy()
    enriched_predictions = prediction_df.merge(scale_reference, on=group_column, how="left", validate="m:1")

    per_station_rows: list[dict[str, object]] = []
    for (split_name, horizon, group_name), group_df in enriched_predictions.groupby(
        [split_column, "horizon", group_column],
        dropna=False,
    ):
        metrics = compute_metric_bundle(group_df, actual_column="y_true", prediction_column="y_pred")
        per_station_rows.append({split_column: split_name, "horizon": horizon, group_column: group_name, **metrics})

    per_station_df = pd.DataFrame(per_station_rows)

    overall_rows: list[dict[str, object]] = []
    for (split_name, horizon), group_df in enriched_predictions.groupby([split_column, "horizon"], dropna=False):
        micro_metrics = compute_metric_bundle(group_df, actual_column="y_true", prediction_column="y_pred")
        overall_rows.append({split_column: split_name, "horizon": horizon, "aggregation": "micro", **micro_metrics})

        split_station_df = per_station_df.loc[
            (per_station_df[split_column] == split_name) & (per_station_df["horizon"] == horizon)
        ]
        if not split_station_df.empty:
            macro_metrics = {
                metric_name: float(split_station_df[metric_name].mean())
                for metric_name in DEFAULT_METRIC_COLUMNS
                if metric_name in split_station_df.columns
            }
            overall_rows.append(
                {
                    split_column: split_name,
                    "horizon": horizon,
                    "aggregation": "macro",
                    "n_obs": int(split_station_df["n_obs"].sum()),
                    "n_groups": int(split_station_df[group_column].nunique()),
                    **macro_metrics,
                }
            )

    overall_metrics_df = pd.DataFrame(overall_rows)
    if not overall_metrics_df.empty:
        overall_metrics_df = overall_metrics_df.loc[
            :,
            [
                column
                for column in ["split", "horizon", "aggregation", "n_obs", "n_groups", *DEFAULT_METRIC_COLUMNS]
                if column in overall_metrics_df.columns
            ],
        ].sort_values(["horizon", "split", "aggregation"], kind="stable").reset_index(drop=True)

    if not per_station_df.empty:
        per_station_df = per_station_df.loc[
            :,
            [
                column
                for column in ["split", "horizon", "unique_id", "n_obs", *DEFAULT_METRIC_COLUMNS]
                if column in per_station_df.columns
            ],
        ].sort_values(["horizon", "split", "unique_id"], kind="stable").reset_index(drop=True)

    return overall_metrics_df, per_station_df
