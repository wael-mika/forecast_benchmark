"""Training helpers for scaled neural benchmark models and the hybrid architecture."""

from __future__ import annotations

from dataclasses import dataclass
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.metrics import DEFAULT_METRIC_COLUMNS, EPSILON
from src.evaluation.pipeline import build_direct_prediction_frame, evaluate_direct_prediction_frame
from src.models.advanced_neural import (
    ResidualAdvancedANNForecaster,
    ResidualAdvancedLSTMForecaster,
    ResidualAdvancedMambaForecaster,
    ResidualAdvancedNHiTSForecaster,
    ResidualAdvancedPatchTSTForecaster,
    ResidualAdvancedTemporalFusionTransformerForecaster,
    ResidualAdvancedXLSTMForecaster,
    ResidualHydroHybridForecaster,
)
from src.training.neural import StationNormalizer, TrainedNeuralExperiment
from src.training.train import infer_direct_target_columns
from src.utils.io import ensure_parent_dir, save_csv, save_json
from src.utils.seed import set_global_seed


NON_FEATURE_COLUMNS = {
    "unique_id",
    "ds",
    "y",
    "forecast_origin_ds",
    "split",
    "split_reference_ds",
}


@dataclass
class AdvancedNeuralWindowBundle:
    """Prepared tensors and metadata for advanced neural training."""

    feature_df: pd.DataFrame
    lag_columns: list[str]
    target_columns: list[str]
    prediction_columns: list[str]
    station_indices: np.ndarray
    sequence_features: np.ndarray
    flat_features: np.ndarray
    context_features: np.ndarray
    future_features: np.ndarray
    targets: np.ndarray
    baseline: np.ndarray
    split_labels: np.ndarray
    normalizer: StationNormalizer
    sequence_channel_names: list[str]
    static_feature_names: list[str]
    future_feature_names: list[str]


def _require_torch():
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise ImportError(
            "PyTorch is required for advanced neural training. Install the project dependencies before running this model."
        ) from exc
    return torch, nn, DataLoader, TensorDataset


def _resolve_device(torch: Any, requested_device: str = "auto") -> Any:
    normalized = str(requested_device).lower()
    if normalized != "auto":
        return torch.device(normalized)
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _infer_target_lag_columns(feature_df: pd.DataFrame) -> list[str]:
    lag_columns = sorted(
        [column for column in feature_df.columns if re.fullmatch(r"lag_\d+", column)],
        key=lambda column: int(column.removeprefix("lag_")),
        reverse=True,
    )
    if not lag_columns:
        raise ValueError("No lag columns were found in the prepared feature frame.")
    return lag_columns


def _fit_station_normalizer(
    log_history_values: np.ndarray,
    log_target_values: np.ndarray,
    station_indices: np.ndarray,
    split_labels: np.ndarray,
    station_ids: list[str],
) -> StationNormalizer:
    mean_by_station = np.zeros(len(station_ids), dtype=np.float32)
    std_by_station = np.ones(len(station_ids), dtype=np.float32)
    train_mask = split_labels == "train"

    for station_index, _station_id in enumerate(station_ids):
        station_mask = train_mask & (station_indices == station_index)
        if not np.any(station_mask):
            continue

        observed_values = np.concatenate(
            [log_history_values[station_mask].reshape(-1), log_target_values[station_mask].reshape(-1)]
        )
        mean_value = float(observed_values.mean())
        std_value = float(observed_values.std())
        mean_by_station[station_index] = mean_value
        std_by_station[station_index] = std_value if std_value > 1e-6 else 1.0

    station_to_index = {station_id: index for index, station_id in enumerate(station_ids)}
    return StationNormalizer(
        station_ids=station_ids,
        station_to_index=station_to_index,
        mean_by_station=mean_by_station,
        std_by_station=std_by_station,
    )


def _standardize_array(values: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.float32)

    if values.ndim == 2:
        train_values = values[train_mask]
        means = np.nanmean(train_values, axis=0)
        stds = np.nanstd(train_values, axis=0)
        stds = np.where((stds > 1e-6) & np.isfinite(stds), stds, 1.0)
        standardized = (values - means) / stds
    elif values.ndim == 3:
        train_values = values[train_mask].reshape(-1, values.shape[-1])
        means = np.nanmean(train_values, axis=0)
        stds = np.nanstd(train_values, axis=0)
        stds = np.where((stds > 1e-6) & np.isfinite(stds), stds, 1.0)
        standardized = (values - means.reshape(1, 1, -1)) / stds.reshape(1, 1, -1)
    else:  # pragma: no cover - defensive branch
        raise ValueError(f"Unsupported array rank for standardization: {values.ndim}")

    return np.nan_to_num(standardized, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _group_historic_exogenous_columns(feature_df: pd.DataFrame) -> dict[str, dict[int, str]]:
    grouped: dict[str, dict[int, str]] = {}
    for column in feature_df.columns:
        if re.fullmatch(r"lag_\d+", column):
            continue
        match = re.fullmatch(r"(.+)_lag_(\d+)", column)
        if match is None:
            continue
        base_name = str(match.group(1))
        lag_value = int(match.group(2))
        grouped.setdefault(base_name, {})[lag_value] = column
    return grouped


def _group_future_columns(feature_df: pd.DataFrame) -> dict[str, dict[int, str]]:
    grouped: dict[str, dict[int, str]] = {}
    for column in feature_df.columns:
        match = re.fullmatch(r"(.+)_future_h(\d+)", column)
        if match is None:
            continue
        base_name = str(match.group(1))
        horizon = int(match.group(2))
        grouped.setdefault(base_name, {})[horizon] = column
    return grouped


def _build_future_calendar_features(
    feature_df: pd.DataFrame,
    *,
    target_columns: list[str],
) -> tuple[np.ndarray, list[str]]:
    if not target_columns:
        return np.zeros((len(feature_df), 0, 0), dtype=np.float32), []

    feature_names = [
        "future_month",
        "future_dayofweek",
        "future_dayofyear_sin",
        "future_dayofyear_cos",
    ]
    calendar_features = np.zeros((len(feature_df), len(target_columns), len(feature_names)), dtype=np.float32)
    for horizon_index, target_column in enumerate(target_columns):
        horizon = int(target_column.removeprefix("target_h"))
        target_ds = pd.to_datetime(feature_df[f"target_h{horizon}_ds"])
        day_of_year = target_ds.dt.dayofyear.to_numpy(dtype=np.float32)
        calendar_features[:, horizon_index, 0] = target_ds.dt.month.to_numpy(dtype=np.float32)
        calendar_features[:, horizon_index, 1] = target_ds.dt.dayofweek.to_numpy(dtype=np.float32)
        calendar_features[:, horizon_index, 2] = np.sin((2.0 * np.pi * day_of_year) / 365.25).astype(np.float32)
        calendar_features[:, horizon_index, 3] = np.cos((2.0 * np.pi * day_of_year) / 365.25).astype(np.float32)
    return calendar_features, feature_names


def prepare_advanced_neural_window_bundle(
    feature_df: pd.DataFrame,
    *,
    min_sequence_coverage: float = 0.6,
) -> AdvancedNeuralWindowBundle:
    """Build multivariate history, static covariates, and future-known covariates."""
    lag_columns = _infer_target_lag_columns(feature_df)
    target_columns = infer_direct_target_columns(feature_df)
    if not target_columns:
        raise ValueError("No direct target columns found in the prepared feature frame.")

    station_ids = sorted(feature_df["unique_id"].astype(str).unique().tolist())
    station_to_index = {station_id: index for index, station_id in enumerate(station_ids)}
    station_indices = feature_df["unique_id"].astype(str).map(station_to_index).to_numpy(dtype=np.int64)
    split_labels = feature_df["split"].astype(str).to_numpy()
    train_mask = split_labels == "train"

    history_columns = list(lag_columns)
    history_lags = [int(column.removeprefix("lag_")) for column in lag_columns]
    if "current_y" in feature_df.columns:
        history_columns.append("current_y")
        history_lags.append(0)

    history_values = feature_df.loc[:, history_columns].to_numpy(dtype=np.float32)
    target_values = feature_df.loc[:, target_columns].to_numpy(dtype=np.float32)
    log_history_values = np.log1p(history_values)
    log_target_values = np.log1p(target_values)

    normalizer = _fit_station_normalizer(
        log_history_values,
        log_target_values,
        station_indices,
        split_labels,
        station_ids,
    )
    means = normalizer.mean_by_station[station_indices][:, None]
    stds = normalizer.std_by_station[station_indices][:, None]
    normalized_target_history = ((log_history_values - means) / stds).astype(np.float32)
    normalized_targets = ((log_target_values - means) / stds).astype(np.float32)
    baseline = np.repeat(normalized_target_history[:, -1:], normalized_targets.shape[1], axis=1).astype(np.float32)

    grouped_historic_columns = _group_historic_exogenous_columns(feature_df)
    dense_sequence_channels: list[str] = []
    dense_sequence_column_names: set[str] = set()
    sparse_lag_column_names: list[str] = []
    sequence_step_count = len(history_lags)
    for base_name, lag_map in sorted(grouped_historic_columns.items()):
        coverage = sum(1 for lag in history_lags if lag in lag_map) / max(1, sequence_step_count)
        if coverage >= min_sequence_coverage:
            dense_sequence_channels.append(base_name)
            dense_sequence_column_names.update(lag_map.values())
        else:
            dropped = sorted(lag_map.values())
            sparse_lag_column_names.extend(dropped)
            import warnings
            warnings.warn(
                f"Feature '{base_name}' has only {coverage:.1%} sequence coverage "
                f"(threshold={min_sequence_coverage:.1%}). Dropping columns: {dropped}. "
                "Consider lowering min_sequence_coverage or removing this feature.",
                UserWarning,
                stacklevel=3,
            )

    sequence_feature_arrays: list[np.ndarray] = [normalized_target_history[:, :, None]]
    sequence_channel_names = ["target_history"]
    for base_name in dense_sequence_channels:
        lag_map = grouped_historic_columns[base_name]
        values = np.full((len(feature_df), sequence_step_count), np.nan, dtype=np.float32)
        for step_index, lag_value in enumerate(history_lags):
            column_name = lag_map.get(lag_value)
            if column_name is None:
                continue
            values[:, step_index] = feature_df[column_name].to_numpy(dtype=np.float32)
        standardized_values = _standardize_array(values[:, :, None], train_mask)
        sequence_feature_arrays.append(standardized_values)
        sequence_channel_names.append(base_name)

    sequence_features = np.concatenate(sequence_feature_arrays, axis=2).astype(np.float32)

    grouped_future_columns = _group_future_columns(feature_df)
    horizon_numbers = [int(target_column.removeprefix("target_h")) for target_column in target_columns]
    future_feature_arrays: list[np.ndarray] = []
    future_feature_names: list[str] = []
    dense_future_columns: set[str] = set()
    for base_name, horizon_map in sorted(grouped_future_columns.items()):
        values = np.full((len(feature_df), len(target_columns), 1), np.nan, dtype=np.float32)
        for horizon_index, horizon in enumerate(horizon_numbers):
            column_name = horizon_map.get(horizon)
            if column_name is None:
                continue
            values[:, horizon_index, 0] = feature_df[column_name].to_numpy(dtype=np.float32)
            dense_future_columns.add(column_name)
        future_feature_arrays.append(_standardize_array(values, train_mask))
        future_feature_names.append(base_name)

    future_calendar_features, future_calendar_names = _build_future_calendar_features(
        feature_df,
        target_columns=target_columns,
    )
    if future_calendar_features.size > 0:
        future_feature_arrays.append(_standardize_array(future_calendar_features, train_mask))
        future_feature_names.extend(future_calendar_names)

    future_features = (
        np.concatenate(future_feature_arrays, axis=2).astype(np.float32)
        if future_feature_arrays
        else np.zeros((len(feature_df), len(target_columns), 0), dtype=np.float32)
    )

    excluded_static_columns = {
        *NON_FEATURE_COLUMNS,
        *lag_columns,
        *target_columns,
        *dense_sequence_column_names,
        *dense_future_columns,
        "current_y",
        *(f"{target_column}_ds" for target_column in target_columns),
    }
    static_feature_names = [
        column
        for column in feature_df.columns
        if column not in excluded_static_columns and pd.api.types.is_numeric_dtype(feature_df[column])
    ]
    if static_feature_names:
        static_values = feature_df.loc[:, static_feature_names].to_numpy(dtype=np.float32)
        context_features = _standardize_array(static_values, train_mask)
    else:
        context_features = np.zeros((len(feature_df), 0), dtype=np.float32)

    flat_features = np.concatenate(
        [
            sequence_features.reshape(len(feature_df), -1),
            context_features,
            future_features.reshape(len(feature_df), -1),
        ],
        axis=1,
    ).astype(np.float32)

    return AdvancedNeuralWindowBundle(
        feature_df=feature_df.copy(),
        lag_columns=lag_columns,
        target_columns=target_columns,
        prediction_columns=[f"prediction_h{index}" for index in range(1, len(target_columns) + 1)],
        station_indices=station_indices,
        sequence_features=sequence_features,
        flat_features=flat_features,
        context_features=context_features,
        future_features=future_features,
        targets=normalized_targets,
        baseline=baseline,
        split_labels=split_labels,
        normalizer=normalizer,
        sequence_channel_names=sequence_channel_names,
        static_feature_names=static_feature_names,
        future_feature_names=future_feature_names,
    )


def _slice_loader_dataset(torch: Any, TensorDataset: Any, bundle: AdvancedNeuralWindowBundle, index_array: np.ndarray) -> Any:
    return TensorDataset(
        torch.tensor(bundle.sequence_features[index_array], dtype=torch.float32),
        torch.tensor(bundle.flat_features[index_array], dtype=torch.float32),
        torch.tensor(bundle.context_features[index_array], dtype=torch.float32),
        torch.tensor(bundle.future_features[index_array], dtype=torch.float32),
        torch.tensor(bundle.station_indices[index_array], dtype=torch.long),
        torch.tensor(bundle.targets[index_array], dtype=torch.float32),
        torch.tensor(bundle.baseline[index_array], dtype=torch.float32),
    )


def _build_advanced_model(
    model_name: str,
    bundle: AdvancedNeuralWindowBundle,
    config: dict[str, Any],
) -> Any:
    horizon_count = len(bundle.target_columns)
    station_count = len(bundle.normalizer.station_ids)
    sequence_input_dim = int(bundle.sequence_features.shape[2])
    sequence_length = int(bundle.sequence_features.shape[1])
    static_input_dim = int(bundle.context_features.shape[1])
    future_input_dim = int(bundle.future_features.shape[2])
    embedding_dim = int(config.get("embedding_dim", 16))
    dropout = float(config.get("dropout", 0.1))

    if model_name == "ann":
        return ResidualAdvancedANNForecaster(
            input_dim=int(bundle.flat_features.shape[1]),
            horizon_count=horizon_count,
            station_count=station_count,
            embedding_dim=embedding_dim,
            hidden_dim=int(config.get("hidden_dim", 512)),
            num_blocks=int(config.get("num_blocks", 4)),
            dropout=dropout,
        )
    if model_name == "lstm":
        return ResidualAdvancedLSTMForecaster(
            sequence_input_dim=sequence_input_dim,
            static_input_dim=static_input_dim,
            future_input_dim=future_input_dim,
            horizon_count=horizon_count,
            station_count=station_count,
            embedding_dim=embedding_dim,
            model_dim=int(config.get("model_dim", 128)),
            hidden_size=int(config.get("hidden_size", 128)),
            num_layers=int(config.get("num_layers", 2)),
            kernel_size=int(config.get("kernel_size", 5)),
            dropout=dropout,
            head_hidden_dim=int(config.get("head_hidden_dim", 256)),
        )
    if model_name == "nhits":
        return ResidualAdvancedNHiTSForecaster(
            sequence_length=sequence_length,
            sequence_input_dim=sequence_input_dim,
            static_input_dim=static_input_dim,
            future_input_dim=future_input_dim,
            horizon_count=horizon_count,
            station_count=station_count,
            embedding_dim=embedding_dim,
            hidden_dims=config.get("hidden_dims", [512, 512]),
            pool_kernels=config.get("pool_kernels", [1, 2, 4, 8]),
            dropout=dropout,
            condition_dim=int(config.get("condition_dim", 256)),
        )
    if model_name == "patchtst":
        return ResidualAdvancedPatchTSTForecaster(
            sequence_input_dim=sequence_input_dim,
            sequence_length=sequence_length,
            static_input_dim=static_input_dim,
            future_input_dim=future_input_dim,
            horizon_count=horizon_count,
            station_count=station_count,
            embedding_dim=embedding_dim,
            patch_len=int(config.get("patch_len", 4)),
            patch_stride=int(config.get("patch_stride", 2)),
            model_dim=int(config.get("model_dim", 128)),
            num_heads=int(config.get("num_heads", 8)),
            num_layers=int(config.get("num_layers", 3)),
            ff_multiplier=int(config.get("ff_multiplier", 4)),
            dropout=dropout,
            head_hidden_dim=int(config.get("head_hidden_dim", 256)),
        )
    if model_name == "tft":
        return ResidualAdvancedTemporalFusionTransformerForecaster(
            sequence_input_dim=sequence_input_dim,
            static_input_dim=static_input_dim,
            future_input_dim=future_input_dim,
            horizon_count=horizon_count,
            station_count=station_count,
            embedding_dim=embedding_dim,
            hidden_size=int(config.get("hidden_size", 128)),
            lstm_layers=int(config.get("lstm_layers", 2)),
            attention_heads=int(config.get("attention_heads", 8)),
            dropout=dropout,
            head_hidden_dim=int(config.get("head_hidden_dim", 256)),
        )
    if model_name == "xlstm":
        return ResidualAdvancedXLSTMForecaster(
            sequence_input_dim=sequence_input_dim,
            static_input_dim=static_input_dim,
            future_input_dim=future_input_dim,
            horizon_count=horizon_count,
            station_count=station_count,
            embedding_dim=embedding_dim,
            model_dim=int(config.get("model_dim", 128)),
            num_blocks=int(config.get("num_blocks", 4)),
            kernel_size=int(config.get("kernel_size", 4)),
            dropout=dropout,
            head_hidden_dim=int(config.get("head_hidden_dim", 256)),
        )
    if model_name == "mamba":
        return ResidualAdvancedMambaForecaster(
            sequence_input_dim=sequence_input_dim,
            static_input_dim=static_input_dim,
            future_input_dim=future_input_dim,
            horizon_count=horizon_count,
            station_count=station_count,
            embedding_dim=embedding_dim,
            model_dim=int(config.get("model_dim", 128)),
            state_dim=int(config.get("state_dim", 64)),
            num_blocks=int(config.get("num_blocks", 4)),
            kernel_size=int(config.get("kernel_size", 5)),
            dropout=dropout,
            head_hidden_dim=int(config.get("head_hidden_dim", 256)),
        )
    if model_name == "hybrid":
        return ResidualHydroHybridForecaster(
            sequence_input_dim=sequence_input_dim,
            static_input_dim=static_input_dim,
            future_input_dim=future_input_dim,
            horizon_count=horizon_count,
            station_count=station_count,
            embedding_dim=embedding_dim,
            model_dim=int(config.get("model_dim", 128)),
            conv_blocks=int(config.get("conv_blocks", 3)),
            conv_kernel_size=int(config.get("conv_kernel_size", 5)),
            recurrent_hidden_size=int(config.get("recurrent_hidden_size", 128)),
            recurrent_layers=int(config.get("recurrent_layers", 2)),
            attention_heads=int(config.get("attention_heads", 8)),
            dropout=dropout,
            head_hidden_dim=int(config.get("head_hidden_dim", 256)),
        )
    raise ValueError(f"Unsupported advanced neural model_name: {model_name!r}")


def _move_batch_to_device(batch: tuple[Any, ...], device: Any) -> tuple[Any, ...]:
    return tuple(tensor.to(device) for tensor in batch)


def _forward_batch(model_name: str, model: Any, batch: tuple[Any, ...]) -> tuple[Any, Any]:
    sequence_features, flat_features, context_features, future_features, station_indices, targets, baseline = batch
    if model_name == "ann":
        predictions = model(sequence_features, flat_features, future_features, station_indices, baseline)
    else:
        predictions = model(sequence_features, context_features, future_features, station_indices, baseline)
    return predictions, targets


def _create_loss_function(nn: Any, config: dict[str, Any], *, horizon_count: int) -> Any:
    import torch
    import torch.nn.functional as F

    loss_name = str(config.get("loss_name", "smooth_l1")).lower()
    if loss_name == "mse":
        return nn.MSELoss()
    if loss_name == "smooth_l1":
        beta = float(config.get("smooth_l1_beta", 1.0))
        return nn.SmoothL1Loss(beta=beta)
    if loss_name == "trajectory":
        point_loss_name = str(config.get("trajectory_point_loss", "smooth_l1")).lower()
        smooth_l1_beta = float(config.get("smooth_l1_beta", 1.0))
        horizon_weights_config = config.get("loss_horizon_weights")
        if horizon_weights_config is None:
            horizon_weights = torch.linspace(1.0, 1.35, steps=max(1, horizon_count), dtype=torch.float32)
        else:
            horizon_weights = torch.tensor(horizon_weights_config, dtype=torch.float32)
            if int(horizon_weights.numel()) != int(horizon_count):
                raise ValueError(
                    f"loss_horizon_weights must have exactly {horizon_count} entries for trajectory loss."
                )
        diff_weight = float(config.get("loss_diff_weight", 0.35))
        curvature_weight = float(config.get("loss_curvature_weight", 0.10))

        class _TrajectoryLoss(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("horizon_weights", horizon_weights)
                self.diff_weight = diff_weight
                self.curvature_weight = curvature_weight
                self.point_loss_name = point_loss_name
                self.smooth_l1_beta = smooth_l1_beta

            def _weighted_point_loss(self, predictions: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
                if self.point_loss_name == "mse":
                    losses = torch.square(predictions - targets)
                else:
                    losses = F.smooth_l1_loss(predictions, targets, reduction="none", beta=self.smooth_l1_beta)
                normalized_weights = weights / weights.mean().clamp_min(1e-6)
                return torch.mean(losses * normalized_weights.view(1, -1))

            def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
                total_loss = self._weighted_point_loss(predictions, targets, self.horizon_weights)
                if predictions.size(1) > 1 and self.diff_weight > 0.0:
                    difference_weights = self.horizon_weights[1:]
                    prediction_deltas = predictions[:, 1:] - predictions[:, :-1]
                    target_deltas = targets[:, 1:] - targets[:, :-1]
                    total_loss = total_loss + (self.diff_weight * self._weighted_point_loss(
                        prediction_deltas,
                        target_deltas,
                        difference_weights,
                    ))
                if predictions.size(1) > 2 and self.curvature_weight > 0.0:
                    curvature_weights = self.horizon_weights[2:]
                    prediction_curvature = predictions[:, 2:] - (2.0 * predictions[:, 1:-1]) + predictions[:, :-2]
                    target_curvature = targets[:, 2:] - (2.0 * targets[:, 1:-1]) + targets[:, :-2]
                    total_loss = total_loss + (self.curvature_weight * self._weighted_point_loss(
                        prediction_curvature,
                        target_curvature,
                        curvature_weights,
                    ))
                return total_loss

        return _TrajectoryLoss()
    raise ValueError(f"Unsupported loss_name: {loss_name!r}")


def _predict_dataset(
    model_name: str,
    model: Any,
    loader: Any,
    *,
    device: Any,
    loss_function: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    station_indices: list[np.ndarray] = []
    total_loss = 0.0
    total_rows = 0

    model.eval()
    import torch

    with torch.no_grad():
        for raw_batch in loader:
            batch = _move_batch_to_device(raw_batch, device)
            batch_predictions, batch_targets = _forward_batch(model_name, model, batch)
            batch_loss = loss_function(batch_predictions, batch_targets)
            total_loss += float(batch_loss.item()) * int(batch_targets.shape[0])
            total_rows += int(batch_targets.shape[0])

            predictions.append(batch_predictions.detach().cpu().numpy())
            targets.append(batch_targets.detach().cpu().numpy())
            station_indices.append(batch[4].detach().cpu().numpy())

    average_loss = total_loss / total_rows if total_rows else float("nan")
    return (
        np.concatenate(predictions, axis=0),
        np.concatenate(targets, axis=0),
        np.concatenate(station_indices, axis=0),
        average_loss,
    )


def _compute_epoch_metric_bundle(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute lightweight micro metrics directly from arrays."""
    valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(valid_mask):
        return {metric_name: float("nan") for metric_name in DEFAULT_METRIC_COLUMNS} | {"n_obs": 0}

    cleaned_true = y_true[valid_mask].astype(np.float64, copy=False)
    cleaned_pred = y_pred[valid_mask].astype(np.float64, copy=False)
    errors = cleaned_pred - cleaned_true
    absolute_errors = np.abs(errors)
    squared_errors = np.square(errors)

    mean_true = float(np.mean(cleaned_true))
    total_variance = float(np.sum(np.square(cleaned_true - mean_true)))
    sum_squared_errors = float(np.sum(squared_errors))

    nonzero_true_mask = np.abs(cleaned_true) > EPSILON
    smape_denominator = np.abs(cleaned_true) + np.abs(cleaned_pred)
    valid_smape_mask = smape_denominator > EPSILON

    mse = float(np.mean(squared_errors))
    metrics = {
        "n_obs": int(cleaned_true.size),
        "bias": float(np.mean(errors)),
        "mae": float(np.mean(absolute_errors)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": float("nan") if total_variance <= EPSILON else float(1.0 - (sum_squared_errors / total_variance)),
        "nse": float("nan") if total_variance <= EPSILON else float(1.0 - (sum_squared_errors / total_variance)),
        "mape": float("nan")
        if not np.any(nonzero_true_mask)
        else float(np.mean(absolute_errors[nonzero_true_mask] / np.abs(cleaned_true[nonzero_true_mask]))),
        "smape": float("nan")
        if not np.any(valid_smape_mask)
        else float(np.mean((2.0 * absolute_errors[valid_smape_mask]) / smape_denominator[valid_smape_mask])),
        "wape": float("nan")
        if float(np.sum(np.abs(cleaned_true))) <= EPSILON
        else float(np.sum(absolute_errors) / np.sum(np.abs(cleaned_true))),
        "mase": float("nan"),
        "rmsse": float("nan"),
    }
    return metrics


def _build_epoch_metric_frame(
    *,
    split_name: str,
    target_columns: list[str],
    actual_targets: np.ndarray,
    predicted_targets: np.ndarray,
    station_indices: np.ndarray,
) -> pd.DataFrame:
    """Create per-horizon micro AND macro metrics for epoch tracking.

    Macro aggregation (mean across stations) is the primary early-stopping signal
    because micro-averaging is dominated by high-flow large stations and hides
    poor performance on small tributaries.
    """
    rows: list[dict[str, Any]] = []
    unique_stations = np.unique(station_indices) if station_indices.size else np.array([], dtype=np.int64)
    group_count = int(unique_stations.size)

    for horizon_index, target_column in enumerate(target_columns):
        horizon = int(target_column.removeprefix("target_h"))
        y_true_h = actual_targets[:, horizon_index]
        y_pred_h = predicted_targets[:, horizon_index]

        # Micro: all stations pooled
        micro_bundle = _compute_epoch_metric_bundle(y_true_h, y_pred_h)
        rows.append(
            {
                "split": split_name,
                "horizon": horizon,
                "aggregation": "micro",
                "n_groups": group_count,
                **micro_bundle,
            }
        )

        # Macro: compute per-station metrics then average across stations
        if group_count > 0:
            station_nse_values: list[float] = []
            station_rmse_values: list[float] = []
            for station_idx in unique_stations:
                mask = station_indices == station_idx
                if np.sum(mask) < 2:
                    continue
                station_bundle = _compute_epoch_metric_bundle(y_true_h[mask], y_pred_h[mask])
                if np.isfinite(station_bundle["nse"]):
                    station_nse_values.append(station_bundle["nse"])
                if np.isfinite(station_bundle["rmse"]):
                    station_rmse_values.append(station_bundle["rmse"])

            macro_nse = float(np.mean(station_nse_values)) if station_nse_values else float("nan")
            macro_rmse = float(np.mean(station_rmse_values)) if station_rmse_values else float("nan")
            rows.append(
                {
                    "split": split_name,
                    "horizon": horizon,
                    "aggregation": "macro",
                    "n_groups": group_count,
                    "n_obs": micro_bundle["n_obs"],
                    "bias": float("nan"),
                    "mae": float("nan"),
                    "mse": float("nan"),
                    "rmse": macro_rmse,
                    "r2": macro_nse,
                    "nse": macro_nse,
                    "mape": float("nan"),
                    "smape": float("nan"),
                    "wape": float("nan"),
                    "mase": float("nan"),
                    "rmsse": float("nan"),
                }
            )

    return pd.DataFrame(rows)


def _sample_epoch_eval_index(
    index_array: np.ndarray,
    *,
    fraction: float,
    max_rows: int,
    seed: int,
) -> np.ndarray:
    if index_array.size == 0:
        return index_array
    bounded_fraction = min(max(float(fraction), 0.0), 1.0)
    if bounded_fraction <= 0.0:
        return np.array([], dtype=index_array.dtype)
    target_size = int(np.ceil(index_array.size * bounded_fraction))
    if max_rows > 0:
        target_size = min(target_size, int(max_rows))
    target_size = max(1, min(target_size, int(index_array.size)))
    if target_size >= int(index_array.size):
        return index_array
    generator = np.random.default_rng(int(seed))
    selection = generator.choice(index_array, size=target_size, replace=False)
    return np.sort(selection.astype(index_array.dtype, copy=False))


def _save_torch_checkpoint(torch: Any, payload: dict[str, Any], path: Path) -> None:
    ensure_parent_dir(path)
    torch.save(payload, path)


def train_advanced_neural_experiment(feature_df: pd.DataFrame, config: dict[str, Any]) -> TrainedNeuralExperiment:
    """Train an advanced neural benchmark model on the prepared direct feature frame."""
    torch, nn, DataLoader, TensorDataset = _require_torch()

    model_name = str(config.get("model_name", "")).lower()
    supported_model_names = {"ann", "lstm", "nhits", "patchtst", "tft", "xlstm", "mamba", "hybrid"}
    if model_name not in supported_model_names:
        raise ValueError(f"train_advanced_neural_experiment supports only {sorted(supported_model_names)}.")

    artifact_dir = Path(config.get("artifact_dir", f"artifacts/{model_name}_advanced"))
    ensure_parent_dir(artifact_dir / "training_summary.json")
    set_global_seed(int(config.get("seed", 42)))

    bundle = prepare_advanced_neural_window_bundle(
        feature_df,
        min_sequence_coverage=float(config.get("min_sequence_coverage", 0.6)),
    )
    baseline_strategy = str(config.get("baseline_strategy", "persistence")).lower()
    if baseline_strategy == "zero":
        bundle.baseline = np.zeros_like(bundle.baseline, dtype=np.float32)
    elif baseline_strategy != "persistence":
        raise ValueError(f"Unsupported baseline_strategy: {baseline_strategy!r}")
    device = _resolve_device(torch, str(config.get("device", "auto")))
    loss_function = _create_loss_function(nn, config, horizon_count=len(bundle.target_columns)).to(device)
    model = _build_advanced_model(model_name, bundle, config).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 1e-3)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
        betas=tuple(config.get("adam_betas", [0.9, 0.999])),
    )
    scheduler_name = str(config.get("scheduler_name", "cosine")).lower()
    if scheduler_name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(config.get("lr_decay_factor", 0.5)),
            patience=int(config.get("lr_patience", 3)),
        )
    elif scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(config.get("max_epochs", 30)),
            eta_min=float(config.get("min_learning_rate", 1e-5)),
        )
    else:
        scheduler = None

    max_epochs = int(config.get("max_epochs", 30))
    patience = int(config.get("early_stopping_patience", 8))
    checkpoint_interval = int(config.get("checkpoint_interval", 0))
    gradient_clip_norm = float(config.get("gradient_clip_norm", 1.0))
    train_fraction = float(config.get("train_fraction", 1.0))
    train_max_rows = int(config.get("train_max_rows", 0))
    warmup_epochs = int(config.get("warmup_epochs", 5))
    base_lr = float(config.get("learning_rate", 1e-3))

    train_index = np.where(bundle.split_labels == "train")[0]
    validation_index = np.where(bundle.split_labels == "validation")[0]
    test_index = np.where(bundle.split_labels == "test")[0]
    if train_index.size == 0 or validation_index.size == 0 or test_index.size == 0:
        raise ValueError("Advanced neural training requires non-empty train, validation, and test splits.")

    batch_size = int(config.get("batch_size", 2048))
    eval_batch_size = int(config.get("eval_batch_size", batch_size * 2))

    train_fit_index = _sample_epoch_eval_index(
        train_index,
        fraction=train_fraction,
        max_rows=train_max_rows,
        seed=int(config.get("seed", 42)) + 11,
    )

    train_dataset = _slice_loader_dataset(torch, TensorDataset, bundle, train_fit_index)
    train_full_dataset = _slice_loader_dataset(torch, TensorDataset, bundle, train_index)
    validation_dataset = _slice_loader_dataset(torch, TensorDataset, bundle, validation_index)
    test_dataset = _slice_loader_dataset(torch, TensorDataset, bundle, test_index)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    validation_loader = DataLoader(validation_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=0)

    validation_eval_fraction = float(config.get("validation_eval_fraction", 1.0))
    validation_eval_max_rows = int(config.get("validation_eval_max_rows", 0))
    test_eval_interval = int(config.get("test_eval_interval", 0))
    test_eval_fraction = float(config.get("test_eval_fraction", validation_eval_fraction))
    test_eval_max_rows = int(config.get("test_eval_max_rows", validation_eval_max_rows))

    validation_eval_index = _sample_epoch_eval_index(
        validation_index,
        fraction=validation_eval_fraction,
        max_rows=validation_eval_max_rows,
        seed=int(config.get("seed", 42)) + 101,
    )
    test_eval_index = _sample_epoch_eval_index(
        test_index,
        fraction=test_eval_fraction,
        max_rows=test_eval_max_rows,
        seed=int(config.get("seed", 42)) + 202,
    )
    validation_epoch_loader = DataLoader(
        _slice_loader_dataset(torch, TensorDataset, bundle, validation_eval_index),
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_epoch_loader = None
    if test_eval_interval > 0 and test_eval_index.size > 0:
        test_epoch_loader = DataLoader(
            _slice_loader_dataset(torch, TensorDataset, bundle, test_eval_index),
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=0,
        )

    epoch_split_loaders = {"validation": validation_epoch_loader}
    epoch_split_feature_frames = {
        "validation": bundle.feature_df.iloc[validation_eval_index].reset_index(drop=True),
    }
    if test_epoch_loader is not None:
        epoch_split_loaders["test"] = test_epoch_loader
        epoch_split_feature_frames["test"] = bundle.feature_df.iloc[test_eval_index].reset_index(drop=True)

    full_split_feature_frames = {
        "validation": bundle.feature_df.iloc[validation_index].reset_index(drop=True),
        "test": bundle.feature_df.iloc[test_index].reset_index(drop=True),
    }
    # Track macro-averaged NSE (mean across stations, then across horizons) as the primary
    # early-stopping criterion. Macro-NSE treats all stations equally regardless of flow
    # magnitude, which is critical for Slovak river data with highly heterogeneous stations.
    best_validation_macro_nse = float("-inf")
    best_epoch = 0
    epochs_without_improvement = 0
    best_model_path = artifact_dir / "model.pt"
    best_model_epoch_path = artifact_dir / "model_epoch_0000.pt"

    loss_history_rows: list[dict[str, Any]] = []
    epoch_metric_frames: list[pd.DataFrame] = []

    save_json(config, artifact_dir / "config_snapshot.json")
    save_csv(bundle.normalizer.to_frame(), artifact_dir / "scaler_by_station.csv")

    for epoch in range(1, max_epochs + 1):
        # Linear warmup: ramp LR from base_lr/warmup_epochs to base_lr over warmup_epochs.
        # After warmup the main scheduler takes over. This prevents large early gradient steps
        # that destabilize training when models are randomly initialized.
        if warmup_epochs > 0 and epoch <= warmup_epochs:
            warmup_scale = epoch / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group["lr"] = base_lr * warmup_scale

        model.train()
        total_train_loss = 0.0
        total_train_rows = 0
        for raw_batch in train_loader:
            batch = _move_batch_to_device(raw_batch, device)
            predictions, targets = _forward_batch(model_name, model, batch)
            loss = loss_function(predictions, targets)
            optimizer.zero_grad()
            loss.backward()
            if gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()
            total_train_loss += float(loss.item()) * int(targets.shape[0])
            total_train_rows += int(targets.shape[0])

        train_loss = total_train_loss / total_train_rows if total_train_rows else float("nan")
        loss_history_rows.append({"epoch": epoch, "split": "train", "loss": train_loss})

        epoch_metric_rows: list[pd.DataFrame] = []
        splits_to_evaluate = ["validation"]
        if test_epoch_loader is not None and epoch % test_eval_interval == 0:
            splits_to_evaluate.append("test")

        for split_name in splits_to_evaluate:
            loader = epoch_split_loaders[split_name]
            normalized_predictions, _normalized_targets, station_indices, split_loss = _predict_dataset(
                model_name,
                model,
                loader,
                device=device,
                loss_function=loss_function,
            )
            loss_history_rows.append({"epoch": epoch, "split": split_name, "loss": split_loss})
            restored_predictions = bundle.normalizer.inverse_transform(normalized_predictions, station_indices)
            actual_targets = epoch_split_feature_frames[split_name].loc[:, bundle.target_columns].to_numpy(dtype=np.float32)
            epoch_metric_rows.append(
                _build_epoch_metric_frame(
                    split_name=split_name,
                    target_columns=bundle.target_columns,
                    actual_targets=actual_targets,
                    predicted_targets=restored_predictions,
                    station_indices=station_indices,
                )
            )

        overall_metrics_df = pd.concat(epoch_metric_rows, ignore_index=True)
        overall_metrics_df["epoch"] = epoch
        epoch_metric_frames.append(overall_metrics_df)

        # Primary early-stopping signal: macro-averaged NSE across stations and horizons.
        # Falls back to micro-RMSE if no macro rows exist (e.g., single-station datasets).
        macro_val_rows = overall_metrics_df.loc[
            (overall_metrics_df["split"] == "validation") & (overall_metrics_df["aggregation"] == "macro")
        ]
        micro_val_rows = overall_metrics_df.loc[
            (overall_metrics_df["split"] == "validation") & (overall_metrics_df["aggregation"] == "micro")
        ]
        if not macro_val_rows.empty and macro_val_rows["nse"].notna().any():
            validation_macro_nse = float(macro_val_rows["nse"].mean())
        else:
            # Fallback: convert micro-RMSE to a pseudo-NSE-like score (negated, so higher is better)
            validation_macro_nse = -float(micro_val_rows["rmse"].mean()) if not micro_val_rows.empty else float("-inf")
        validation_micro_rmse = float(micro_val_rows["rmse"].mean()) if not micro_val_rows.empty else float("nan")

        validation_loss_rows = [row for row in loss_history_rows if row["epoch"] == epoch and row["split"] == "validation"]
        validation_loss = float(validation_loss_rows[-1]["loss"]) if validation_loss_rows else float("nan")
        test_loss_rows = [row for row in loss_history_rows if row["epoch"] == epoch and row["split"] == "test"]
        test_loss = float(test_loss_rows[-1]["loss"]) if test_loss_rows else float("nan")
        print(
            f"[advanced:{model_name}] epoch={epoch:03d} train_loss={train_loss:.5f} "
            f"val_loss={validation_loss:.5f} test_loss={test_loss:.5f} "
            f"val_macro_nse={validation_macro_nse:.5f} val_micro_rmse={validation_micro_rmse:.3f}",
            flush=True,
        )
        if scheduler_name == "plateau" and scheduler is not None:
            # ReduceLROnPlateau: pass negative NSE so "min" mode corresponds to max NSE
            scheduler.step(-validation_macro_nse if np.isfinite(validation_macro_nse) else validation_micro_rmse)
        elif scheduler_name == "cosine" and scheduler is not None:
            scheduler.step()

        checkpoint_payload = {
            "epoch": epoch,
            "model_name": model_name,
            "model_variant": "advanced",
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "validation_macro_nse": validation_macro_nse,
            "validation_micro_rmse": validation_micro_rmse,
        }

        if validation_macro_nse > best_validation_macro_nse:
            best_validation_macro_nse = validation_macro_nse
            best_epoch = epoch
            epochs_without_improvement = 0
            best_model_epoch_path = artifact_dir / f"model_epoch_{epoch:04d}.pt"
            _save_torch_checkpoint(torch, checkpoint_payload, best_model_path)
            _save_torch_checkpoint(torch, checkpoint_payload, best_model_epoch_path)
        else:
            epochs_without_improvement += 1

        if checkpoint_interval > 0 and epoch % checkpoint_interval == 0:
            _save_torch_checkpoint(
                torch,
                checkpoint_payload,
                artifact_dir / "checkpoints" / f"{model_name}_epoch_{epoch:04d}.pt",
            )

        if epochs_without_improvement >= patience:
            break

    best_checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])

    final_loaders = {
        "train": DataLoader(train_full_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=0),
        "validation": validation_loader,
        "test": test_loader,
    }
    final_feature_frames = {
        "train": bundle.feature_df.iloc[train_index].reset_index(drop=True),
        "validation": full_split_feature_frames["validation"],
        "test": full_split_feature_frames["test"],
    }

    final_prediction_frames: list[pd.DataFrame] = []
    for split_name, loader in final_loaders.items():
        normalized_predictions, _normalized_targets, station_indices, split_loss = _predict_dataset(
            model_name,
            model,
            loader,
            device=device,
            loss_function=loss_function,
        )
        restored_predictions = bundle.normalizer.inverse_transform(normalized_predictions, station_indices)
        prediction_columns_df = pd.DataFrame(restored_predictions, columns=bundle.prediction_columns)
        final_prediction_frames.append(
            build_direct_prediction_frame(
                final_feature_frames[split_name],
                prediction_columns_df,
                split_column=str(config.get("split_column", "split")),
            )
        )
        if not any(
            row["epoch"] == best_epoch and row["split"] == split_name and abs(float(row["loss"]) - split_loss) < 1e-12
            for row in loss_history_rows
        ):
            loss_history_rows.append({"epoch": best_epoch, "split": split_name, "loss": split_loss})

    final_prediction_df = pd.concat(final_prediction_frames, ignore_index=True)
    overall_metrics_df, per_station_metrics_df = evaluate_direct_prediction_frame(
        final_prediction_df,
        split_column=str(config.get("split_column", "split")),
        group_column="unique_id",
    )

    loss_history_df = (
        pd.DataFrame(loss_history_rows)
        .drop_duplicates(subset=["epoch", "split"], keep="last")
        .sort_values(["epoch", "split"], kind="stable")
        .reset_index(drop=True)
    )
    epoch_metrics_df = (
        pd.concat(epoch_metric_frames, ignore_index=True)
        .sort_values(["epoch", "horizon", "split", "aggregation"], kind="stable")
        .reset_index(drop=True)
    )

    save_csv(loss_history_df, artifact_dir / "loss_history.csv")
    save_csv(epoch_metrics_df, artifact_dir / "epoch_metrics.csv")

    training_summary = {
        "model_name": model_name,
        "model_variant": "advanced",
        "loss_name": str(config.get("loss_name", "smooth_l1")),
        "baseline_strategy": baseline_strategy,
        "scheduler_name": scheduler_name,
        "requested_max_epochs": max_epochs,
        "trained_num_epochs": int(loss_history_df["epoch"].max()),
        "best_epoch": best_epoch,
        "best_validation_macro_nse": best_validation_macro_nse,
        "device": str(device),
        "batch_size": batch_size,
        "eval_batch_size": eval_batch_size,
        "gradient_clip_norm": gradient_clip_norm,
        "window_size": int(bundle.sequence_features.shape[1]),
        "sequence_channel_count": int(bundle.sequence_features.shape[2]),
        "sequence_channel_names": bundle.sequence_channel_names,
        "static_feature_count": int(bundle.context_features.shape[1]),
        "future_feature_count": int(bundle.future_features.shape[2]),
        "future_feature_names": bundle.future_feature_names,
        "loss_horizon_weights": config.get("loss_horizon_weights"),
        "loss_diff_weight": float(config.get("loss_diff_weight", 0.35)),
        "loss_curvature_weight": float(config.get("loss_curvature_weight", 0.10)),
        "train_fraction": train_fraction,
        "train_max_rows": train_max_rows,
        "validation_eval_fraction": validation_eval_fraction,
        "validation_eval_max_rows": validation_eval_max_rows,
        "test_eval_interval": test_eval_interval,
        "test_eval_fraction": test_eval_fraction,
        "test_eval_max_rows": test_eval_max_rows,
        "train_rows": int(train_index.size),
        "train_fit_rows": int(train_fit_index.size),
        "validation_rows": int(validation_index.size),
        "test_rows": int(test_index.size),
        "prediction_columns": bundle.prediction_columns,
        "target_columns": bundle.target_columns,
        "best_model_path": str(best_model_path),
        "best_model_epoch_path": str(best_model_epoch_path),
    }
    save_json(training_summary, artifact_dir / "training_summary.json")

    return TrainedNeuralExperiment(
        model=model,
        model_name=model_name,
        feature_frame=bundle.feature_df,
        artifact_dir=artifact_dir,
        training_summary=training_summary,
        prediction_df=final_prediction_df,
        overall_metrics_df=overall_metrics_df,
        per_station_metrics_df=per_station_metrics_df,
        loss_history_df=loss_history_df,
        epoch_metrics_df=epoch_metrics_df,
    )
