"""Forecast evaluation metrics for benchmark experiments."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


EPSILON = 1e-8
DEFAULT_METRIC_COLUMNS = (
    "bias",
    "mae",
    "mse",
    "rmse",
    "r2",
    "nse",
    "mape",
    "smape",
    "wape",
    "mase",
    "rmsse",
)


def _clean_prediction_frame(
    df: pd.DataFrame,
    *,
    actual_column: str,
    prediction_column: str,
) -> pd.DataFrame:
    required_columns = [actual_column, prediction_column]
    return df.dropna(subset=required_columns).reset_index(drop=True)


def _safe_mean(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.mean(values))


def _safe_ratio(numerator: float, denominator: float) -> float:
    if abs(denominator) <= EPSILON:
        return float("nan")
    return float(numerator / denominator)


def build_scale_reference(
    df: pd.DataFrame,
    *,
    group_column: str = "unique_id",
    time_column: str = "target_ds",
    target_column: str = "target",
) -> pd.DataFrame:
    """Build MASE/RMSSE denominators from in-sample target history."""
    ordered = df.sort_values([group_column, time_column], kind="stable").copy()
    diffs = ordered.groupby(group_column)[target_column].diff()

    ordered["abs_diff"] = diffs.abs()
    ordered["squared_diff"] = diffs.pow(2)

    scale_reference = (
        ordered.groupby(group_column, dropna=False)
        .agg(
            mase_denominator=("abs_diff", "mean"),
            rmsse_denominator=("squared_diff", "mean"),
            reference_observations=(target_column, "size"),
        )
        .reset_index()
    )
    return scale_reference


def compute_metric_bundle(
    df: pd.DataFrame,
    *,
    actual_column: str = "y_true",
    prediction_column: str = "y_pred",
    mase_denominator_column: str = "mase_denominator",
    rmsse_denominator_column: str = "rmsse_denominator",
) -> dict[str, float]:
    """Compute a broad metric bundle on a prediction frame."""
    cleaned = _clean_prediction_frame(df, actual_column=actual_column, prediction_column=prediction_column)
    if cleaned.empty:
        return {metric_name: float("nan") for metric_name in DEFAULT_METRIC_COLUMNS} | {"n_obs": 0}

    y_true = cleaned[actual_column].to_numpy(dtype=float)
    y_pred = cleaned[prediction_column].to_numpy(dtype=float)
    errors = y_pred - y_true
    absolute_errors = np.abs(errors)
    squared_errors = np.square(errors)

    mean_true = np.mean(y_true)
    total_variance = np.sum(np.square(y_true - mean_true))
    sum_squared_errors = np.sum(squared_errors)

    nonzero_true_mask = np.abs(y_true) > EPSILON
    smape_denominator = np.abs(y_true) + np.abs(y_pred)
    valid_smape_mask = smape_denominator > EPSILON

    metrics = {
        "n_obs": int(cleaned.shape[0]),
        "bias": _safe_mean(errors),
        "mae": _safe_mean(absolute_errors),
        "mse": _safe_mean(squared_errors),
        "rmse": float(np.sqrt(_safe_mean(squared_errors))),
        "r2": float("nan") if total_variance <= EPSILON else float(1.0 - (sum_squared_errors / total_variance)),
        "nse": float("nan") if total_variance <= EPSILON else float(1.0 - (sum_squared_errors / total_variance)),
        "mape": float("nan")
        if not np.any(nonzero_true_mask)
        else float(np.mean(absolute_errors[nonzero_true_mask] / np.abs(y_true[nonzero_true_mask]))),
        "smape": float("nan")
        if not np.any(valid_smape_mask)
        else float(np.mean((2.0 * absolute_errors[valid_smape_mask]) / smape_denominator[valid_smape_mask])),
        "wape": _safe_ratio(float(np.sum(absolute_errors)), float(np.sum(np.abs(y_true)))),
        "mase": float("nan"),
        "rmsse": float("nan"),
    }

    if mase_denominator_column in cleaned.columns:
        valid_mase_mask = cleaned[mase_denominator_column].to_numpy(dtype=float) > EPSILON
        if np.any(valid_mase_mask):
            mase_denominator = cleaned.loc[valid_mase_mask, mase_denominator_column].to_numpy(dtype=float)
            metrics["mase"] = float(np.mean(absolute_errors[valid_mase_mask] / mase_denominator))

    if rmsse_denominator_column in cleaned.columns:
        valid_rmsse_mask = cleaned[rmsse_denominator_column].to_numpy(dtype=float) > EPSILON
        if np.any(valid_rmsse_mask):
            rmsse_denominator = cleaned.loc[valid_rmsse_mask, rmsse_denominator_column].to_numpy(dtype=float)
            metrics["rmsse"] = float(np.sqrt(np.mean(squared_errors[valid_rmsse_mask] / rmsse_denominator)))

    return metrics


def summarize_prediction_metrics(
    predictions_df: pd.DataFrame,
    *,
    split_column: str = "split",
    group_column: str = "unique_id",
    actual_column: str = "y_true",
    prediction_column: str = "y_pred",
    metric_columns: Iterable[str] = DEFAULT_METRIC_COLUMNS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize metrics both overall by split and by split/station."""
    metric_names = list(metric_columns)

    per_station_rows: list[dict[str, object]] = []
    for (split_name, group_name), group_df in predictions_df.groupby([split_column, group_column], dropna=False):
        metrics = compute_metric_bundle(
            group_df,
            actual_column=actual_column,
            prediction_column=prediction_column,
        )
        per_station_rows.append(
            {
                split_column: split_name,
                group_column: group_name,
                **metrics,
            }
        )

    per_station_df = pd.DataFrame(per_station_rows)

    overall_rows: list[dict[str, object]] = []
    for split_name, group_df in predictions_df.groupby(split_column, dropna=False):
        micro_metrics = compute_metric_bundle(
            group_df,
            actual_column=actual_column,
            prediction_column=prediction_column,
        )
        overall_rows.append({split_column: split_name, "aggregation": "micro", **micro_metrics})

        split_station_metrics = per_station_df.loc[per_station_df[split_column] == split_name]
        if not split_station_metrics.empty:
            macro_metrics = {
                metric_name: float(split_station_metrics[metric_name].mean())
                for metric_name in metric_names
                if metric_name in split_station_metrics.columns
            }
            overall_rows.append(
                {
                    split_column: split_name,
                    "aggregation": "macro",
                    "n_obs": int(split_station_metrics["n_obs"].sum()),
                    "n_groups": int(split_station_metrics[group_column].nunique()),
                    **macro_metrics,
                }
            )

    overall_df = pd.DataFrame(overall_rows)
    if not overall_df.empty:
        ordered_columns = [split_column, "aggregation", "n_obs", "n_groups", *metric_names]
        overall_df = overall_df.loc[:, [column for column in ordered_columns if column in overall_df.columns]]

    if not per_station_df.empty:
        ordered_columns = [split_column, group_column, "n_obs", *metric_names]
        per_station_df = per_station_df.loc[:, [column for column in ordered_columns if column in per_station_df.columns]]

    return overall_df, per_station_df
