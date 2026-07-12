"""Training helpers for the compact neural baseline models.

This module prepares the normalized input tensors used by the lightweight
neural baselines in ``src.models.neural`` and runs their training loop. The
focus here is on simple, fair baseline models: one shared data pipeline, one
shared training loop, and one consistent evaluation path across ANN, LSTM,
N-HiTS, PatchTST, TFT, xLSTM, and Mamba variants.

Main helpers
------------
prepare_neural_window_bundle
    Turn the direct feature frame into normalized tensors for baseline models.
train_neural_experiment
    Train one compact neural model and save checkpoints plus evaluation tables.
StationNormalizer
    Store per-station log-scale statistics and undo normalization at inference.
TrainedNeuralExperiment
    Return the trained model together with saved metrics and prediction tables.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.metrics import build_scale_reference
from src.evaluation.pipeline import build_direct_prediction_frame, evaluate_direct_prediction_frame
from src.models.neural import (
    ResidualANNForecaster,
    ResidualBidirectionalLSTMForecaster,
    ResidualMambaForecaster,
    ResidualNHiTSForecaster,
    ResidualPatchTSTForecaster,
    ResidualTemporalFusionTransformerForecaster,
    ResidualXLSTMForecaster,
)
from src.training.train import infer_direct_target_columns
from src.utils.io import ensure_parent_dir, save_csv, save_json
from src.utils.seed import set_global_seed


@dataclass
class StationNormalizer:
    """Store per-station log-space normalization statistics for the baseline models."""

    station_ids: list[str]
    station_to_index: dict[str, int]
    mean_by_station: np.ndarray
    std_by_station: np.ndarray

    def inverse_transform(self, values: np.ndarray, station_indices: np.ndarray) -> np.ndarray:
        means = self.mean_by_station[station_indices][:, None]
        stds = self.std_by_station[station_indices][:, None]
        restored = np.expm1((values * stds) + means)
        return np.clip(restored, a_min=0.0, a_max=None)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "unique_id": self.station_ids,
                "station_index": list(range(len(self.station_ids))),
                "log1p_mean": self.mean_by_station,
                "log1p_std": self.std_by_station,
            }
        )


@dataclass
class NeuralWindowBundle:
    """All tensors and metadata needed to train one compact neural baseline."""

    feature_df: pd.DataFrame
    lag_columns: list[str]
    target_columns: list[str]
    prediction_columns: list[str]
    station_indices: np.ndarray
    sequence_features: np.ndarray
    flat_features: np.ndarray
    context_features: np.ndarray
    targets: np.ndarray
    baseline: np.ndarray
    split_labels: np.ndarray
    normalizer: StationNormalizer


@dataclass
class TrainedNeuralExperiment:
    """Return bundle for one compact neural training run."""

    model: Any
    model_name: str
    feature_frame: pd.DataFrame
    artifact_dir: Path
    training_summary: dict[str, Any]
    prediction_df: pd.DataFrame
    overall_metrics_df: pd.DataFrame
    per_station_metrics_df: pd.DataFrame
    loss_history_df: pd.DataFrame
    epoch_metrics_df: pd.DataFrame


def _require_torch():
    """Import PyTorch lazily and apply a few environment safeguards first."""
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise ImportError(
            "PyTorch is required for ANN/LSTM training. Install the neural dependencies before running this model."
        ) from exc
    return torch, nn, DataLoader, TensorDataset


def _resolve_device(torch: Any, requested_device: str = "auto") -> Any:
    """Pick the requested device or choose the best available accelerator automatically."""
    normalized = str(requested_device).lower()
    if normalized != "auto":
        return torch.device(normalized)
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _infer_lag_columns(feature_df: pd.DataFrame) -> list[str]:
    """Return lag columns ordered from oldest lag to most recent lag."""
    lag_columns = sorted(
        [column for column in feature_df.columns if column.startswith("lag_") and column[4:].isdigit()],
        key=lambda column: int(column.removeprefix("lag_")),
        reverse=True,
    )
    if not lag_columns:
        raise ValueError("No lag columns were found in the prepared feature frame.")
    return lag_columns


def _infer_extra_context_columns(
    feature_df: pd.DataFrame,
    *,
    lag_columns: list[str],
    target_columns: list[str],
) -> list[str]:
    """Collect numeric context columns that are not part of the lag or target sequence."""
    excluded_columns = {
        "unique_id",
        "ds",
        "y",
        "forecast_origin_ds",
        "split",
        "split_reference_ds",
        *lag_columns,
        *target_columns,
        *(f"{target_column}_ds" for target_column in target_columns),
    }
    return [
        column
        for column in feature_df.columns
        if column not in excluded_columns
        and not column.startswith("target_h")
        and pd.api.types.is_numeric_dtype(feature_df[column])
    ]


def _fit_station_normalizer(
    log_lag_values: np.ndarray,
    log_target_values: np.ndarray,
    station_indices: np.ndarray,
    split_labels: np.ndarray,
    station_ids: list[str],
) -> StationNormalizer:
    """Fit one mean and standard deviation per station in log1p space using train rows only."""
    mean_by_station = np.zeros(len(station_ids), dtype=np.float32)
    std_by_station = np.ones(len(station_ids), dtype=np.float32)
    train_mask = split_labels == "train"

    for station_index, _station_id in enumerate(station_ids):
        station_mask = train_mask & (station_indices == station_index)
        if not np.any(station_mask):
            continue

        observed_values = np.concatenate(
            [log_lag_values[station_mask].reshape(-1), log_target_values[station_mask].reshape(-1)]
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


def _standardize_extra_context_features(
    feature_df: pd.DataFrame,
    *,
    feature_columns: list[str],
    split_labels: np.ndarray,
) -> np.ndarray:
    """Standardize optional extra context features using train-split statistics."""
    if not feature_columns:
        return np.zeros((len(feature_df), 0), dtype=np.float32)

    context_df = feature_df.loc[:, feature_columns].copy()
    train_mask = split_labels == "train"
    train_context = context_df.loc[train_mask]

    means = train_context.mean(axis=0).fillna(0.0)
    stds = train_context.std(axis=0).replace(0.0, 1.0).fillna(1.0)
    standardized = ((context_df - means) / stds).fillna(0.0)
    return standardized.to_numpy(dtype=np.float32)


def prepare_neural_window_bundle(feature_df: pd.DataFrame) -> NeuralWindowBundle:
    """Build the normalized sequence, context, target, and baseline tensors for compact neural models."""
    lag_columns = _infer_lag_columns(feature_df)
    target_columns = infer_direct_target_columns(feature_df)
    if not target_columns:
        raise ValueError("No direct target columns found. Prepare the 5-to-3 feature frame before neural training.")

    station_ids = sorted(feature_df["unique_id"].astype(str).unique().tolist())
    station_to_index = {station_id: index for index, station_id in enumerate(station_ids)}
    station_indices = feature_df["unique_id"].astype(str).map(station_to_index).to_numpy(dtype=np.int64)
    split_labels = feature_df["split"].astype(str).to_numpy()
    extra_context_columns = _infer_extra_context_columns(
        feature_df,
        lag_columns=lag_columns,
        target_columns=target_columns,
    )

    lag_values = feature_df.loc[:, lag_columns].to_numpy(dtype=np.float32)
    target_values = feature_df.loc[:, target_columns].to_numpy(dtype=np.float32)
    log_lag_values = np.log1p(lag_values)
    log_target_values = np.log1p(target_values)

    normalizer = _fit_station_normalizer(
        log_lag_values,
        log_target_values,
        station_indices,
        split_labels,
        station_ids,
    )

    means = normalizer.mean_by_station[station_indices][:, None]
    stds = normalizer.std_by_station[station_indices][:, None]
    normalized_sequence = ((log_lag_values - means) / stds).astype(np.float32)
    normalized_targets = ((log_target_values - means) / stds).astype(np.float32)

    window_mean = normalized_sequence.mean(axis=1, keepdims=True)
    window_std = normalized_sequence.std(axis=1, keepdims=True)
    window_min = normalized_sequence.min(axis=1, keepdims=True)
    window_max = normalized_sequence.max(axis=1, keepdims=True)
    deltas = np.diff(normalized_sequence, axis=1)
    extra_context = _standardize_extra_context_features(
        feature_df,
        feature_columns=extra_context_columns,
        split_labels=split_labels,
    )

    context_features = np.concatenate([window_mean, window_std, window_min, window_max, deltas, extra_context], axis=1)
    context_features = context_features.astype(np.float32)
    flat_features = np.concatenate([normalized_sequence, context_features], axis=1).astype(np.float32)
    baseline = np.repeat(normalized_sequence[:, -1:], normalized_targets.shape[1], axis=1).astype(np.float32)

    return NeuralWindowBundle(
        feature_df=feature_df.copy(),
        lag_columns=lag_columns,
        target_columns=target_columns,
        prediction_columns=[f"prediction_h{index}" for index in range(1, len(target_columns) + 1)],
        station_indices=station_indices,
        sequence_features=normalized_sequence[:, :, None],
        flat_features=flat_features,
        context_features=context_features,
        targets=normalized_targets,
        baseline=baseline,
        split_labels=split_labels,
        normalizer=normalizer,
    )


def _build_scale_reference_from_feature_frame(
    feature_df: pd.DataFrame,
    target_columns: list[str],
    *,
    split_column: str,
    group_column: str = "unique_id",
) -> pd.DataFrame:
    """Build the train-based scale reference used later for MASE and RMSSE."""
    train_rows = feature_df.loc[feature_df[split_column] == "train"].copy()
    reference_frames: list[pd.DataFrame] = []
    for target_column in target_columns:
        horizon = int(target_column.removeprefix("target_h"))
        target_ds_column = f"target_h{horizon}_ds"
        reference_frames.append(
            train_rows.loc[:, [group_column, target_ds_column, target_column]]
            .rename(columns={target_ds_column: "target_ds", target_column: "target"})
            .dropna(subset=["target_ds", "target"])
        )

    reference_df = (
        pd.concat(reference_frames, ignore_index=True)
        .drop_duplicates(subset=[group_column, "target_ds"], keep="first")
        .sort_values([group_column, "target_ds"], kind="stable")
        .reset_index(drop=True)
    )
    return build_scale_reference(reference_df, group_column=group_column, time_column="target_ds", target_column="target")


def _slice_loader_dataset(torch: Any, TensorDataset: Any, bundle: NeuralWindowBundle, index_array: np.ndarray) -> Any:
    """Convert a subset of the prepared bundle into one ``TensorDataset``."""
    return TensorDataset(
        torch.tensor(bundle.sequence_features[index_array], dtype=torch.float32),
        torch.tensor(bundle.flat_features[index_array], dtype=torch.float32),
        torch.tensor(bundle.context_features[index_array], dtype=torch.float32),
        torch.tensor(bundle.station_indices[index_array], dtype=torch.long),
        torch.tensor(bundle.targets[index_array], dtype=torch.float32),
        torch.tensor(bundle.baseline[index_array], dtype=torch.float32),
    )


def _build_model(
    torch: Any,
    model_name: str,
    bundle: NeuralWindowBundle,
    config: dict[str, Any],
) -> Any:
    """Instantiate the requested compact neural model from the prepared bundle metadata."""
    del torch
    horizon_count = len(bundle.target_columns)
    station_count = len(bundle.normalizer.station_ids)
    embedding_dim = int(config.get("embedding_dim", 8))
    dropout = float(config.get("dropout", 0.1))
    sequence_length = int(bundle.sequence_features.shape[1])
    sequence_input_dim = int(bundle.sequence_features.shape[2])
    static_input_dim = int(bundle.context_features.shape[1])

    if model_name == "ann":
        return ResidualANNForecaster(
            input_dim=bundle.flat_features.shape[1],
            horizon_count=horizon_count,
            station_count=station_count,
            embedding_dim=embedding_dim,
            hidden_dims=config.get("hidden_dims", [64, 64]),
            dropout=dropout,
        )
    if model_name == "lstm":
        return ResidualBidirectionalLSTMForecaster(
            sequence_input_dim=sequence_input_dim,
            static_input_dim=static_input_dim,
            horizon_count=horizon_count,
            station_count=station_count,
            embedding_dim=embedding_dim,
            hidden_size=int(config.get("hidden_size", 32)),
            num_layers=int(config.get("num_layers", 2)),
            dropout=dropout,
            bidirectional=bool(config.get("bidirectional", True)),
            head_hidden_dim=int(config.get("head_hidden_dim", 64)),
        )
    if model_name == "nhits":
        return ResidualNHiTSForecaster(
            sequence_length=sequence_length,
            static_input_dim=static_input_dim,
            horizon_count=horizon_count,
            station_count=station_count,
            embedding_dim=embedding_dim,
            hidden_dims=config.get("hidden_dims", [256, 256]),
            pool_kernels=config.get("pool_kernels", [1, 2, 4]),
            dropout=dropout,
        )
    if model_name == "patchtst":
        return ResidualPatchTSTForecaster(
            sequence_input_dim=sequence_input_dim,
            sequence_length=sequence_length,
            static_input_dim=static_input_dim,
            horizon_count=horizon_count,
            station_count=station_count,
            embedding_dim=embedding_dim,
            patch_len=int(config.get("patch_len", 4)),
            patch_stride=int(config.get("patch_stride", 2)),
            model_dim=int(config.get("model_dim", 64)),
            num_heads=int(config.get("num_heads", 4)),
            num_layers=int(config.get("num_layers", 2)),
            ff_multiplier=int(config.get("ff_multiplier", 4)),
            dropout=dropout,
            head_hidden_dim=int(config.get("head_hidden_dim", 128)),
        )
    if model_name == "tft":
        return ResidualTemporalFusionTransformerForecaster(
            sequence_input_dim=sequence_input_dim,
            static_input_dim=static_input_dim,
            horizon_count=horizon_count,
            station_count=station_count,
            embedding_dim=embedding_dim,
            hidden_size=int(config.get("hidden_size", 64)),
            lstm_layers=int(config.get("lstm_layers", 1)),
            attention_heads=int(config.get("attention_heads", 4)),
            dropout=dropout,
            head_hidden_dim=int(config.get("head_hidden_dim", 128)),
        )
    if model_name == "xlstm":
        return ResidualXLSTMForecaster(
            sequence_input_dim=sequence_input_dim,
            static_input_dim=static_input_dim,
            horizon_count=horizon_count,
            station_count=station_count,
            embedding_dim=embedding_dim,
            model_dim=int(config.get("model_dim", 64)),
            hidden_size=int(config.get("hidden_size", 64)),
            num_blocks=int(config.get("num_blocks", 3)),
            dropout=dropout,
            head_hidden_dim=int(config.get("head_hidden_dim", 128)),
        )
    if model_name == "mamba":
        return ResidualMambaForecaster(
            sequence_input_dim=sequence_input_dim,
            static_input_dim=static_input_dim,
            horizon_count=horizon_count,
            station_count=station_count,
            embedding_dim=embedding_dim,
            model_dim=int(config.get("model_dim", 64)),
            num_blocks=int(config.get("num_blocks", 3)),
            expand_factor=int(config.get("expand_factor", 2)),
            kernel_size=int(config.get("kernel_size", 3)),
            dropout=dropout,
            head_hidden_dim=int(config.get("head_hidden_dim", 128)),
        )
    raise ValueError(f"Unsupported neural model_name: {model_name!r}")


def _move_batch_to_device(batch: tuple[Any, ...], device: Any) -> tuple[Any, ...]:
    """Move every tensor in one batch to the chosen training device."""
    return tuple(tensor.to(device) for tensor in batch)


def _forward_batch(model_name: str, model: Any, batch: tuple[Any, ...]) -> tuple[Any, Any]:
    """Run one forward pass and return predictions together with the target tensor."""
    sequence_features, flat_features, context_features, station_indices, targets, baseline = batch
    if model_name == "ann":
        predictions = model(sequence_features, flat_features, station_indices, baseline)
    else:
        predictions = model(sequence_features, context_features, station_indices, baseline)
    return predictions, targets


def _create_loss_function(nn: Any, config: dict[str, Any]) -> Any:
    """Create the configured point loss for compact neural training."""
    loss_name = str(config.get("loss_name", "smooth_l1")).lower()
    if loss_name == "mse":
        return nn.MSELoss()
    if loss_name == "smooth_l1":
        beta = float(config.get("smooth_l1_beta", 1.0))
        return nn.SmoothL1Loss(beta=beta)
    raise ValueError(f"Unsupported loss_name: {loss_name!r}")


def _predict_dataset(
    model_name: str,
    model: Any,
    loader: Any,
    *,
    device: Any,
    loss_function: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Run one full loader in evaluation mode and collect predictions, targets, and loss."""
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
            station_indices.append(batch[3].detach().cpu().numpy())

    average_loss = total_loss / total_rows if total_rows else float("nan")
    return (
        np.concatenate(predictions, axis=0),
        np.concatenate(targets, axis=0),
        np.concatenate(station_indices, axis=0),
        average_loss,
    )


def _save_torch_checkpoint(torch: Any, payload: dict[str, Any], path: Path) -> None:
    """Save one PyTorch checkpoint, creating parent folders when needed."""
    ensure_parent_dir(path)
    torch.save(payload, path)


def train_neural_experiment(feature_df: pd.DataFrame, config: dict[str, Any]) -> TrainedNeuralExperiment:
    """Train one compact neural benchmark model on the prepared direct feature frame."""
    torch, nn, DataLoader, TensorDataset = _require_torch()

    model_name = str(config.get("model_name", "")).lower()
    supported_model_names = {"ann", "lstm", "nhits", "patchtst", "tft", "xlstm", "mamba"}
    if model_name not in supported_model_names:
        raise ValueError(f"train_neural_experiment supports only {sorted(supported_model_names)}.")

    artifact_dir = Path(config.get("artifact_dir", f"artifacts/{model_name}"))
    ensure_parent_dir(artifact_dir / "training_summary.json")
    set_global_seed(int(config.get("seed", 42)))

    bundle = prepare_neural_window_bundle(feature_df)
    device = _resolve_device(torch, str(config.get("device", "auto")))
    loss_function = _create_loss_function(nn, config)
    model = _build_model(torch, model_name, bundle, config).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 1e-3)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(config.get("lr_decay_factor", 0.5)),
        patience=int(config.get("lr_patience", 2)),
    )
    max_epochs = int(config.get("max_epochs", 20))
    patience = int(config.get("early_stopping_patience", 5))
    checkpoint_interval = int(config.get("checkpoint_interval", 0))
    gradient_clip_norm = float(config.get("gradient_clip_norm", 1.0))

    train_index = np.where(bundle.split_labels == "train")[0]
    validation_index = np.where(bundle.split_labels == "validation")[0]
    test_index = np.where(bundle.split_labels == "test")[0]
    if train_index.size == 0 or validation_index.size == 0 or test_index.size == 0:
        raise ValueError("Neural training requires non-empty train, validation, and test splits.")

    batch_size = int(config.get("batch_size", 4096))
    eval_batch_size = int(config.get("eval_batch_size", batch_size * 2))

    train_dataset = _slice_loader_dataset(torch, TensorDataset, bundle, train_index)
    validation_dataset = _slice_loader_dataset(torch, TensorDataset, bundle, validation_index)
    test_dataset = _slice_loader_dataset(torch, TensorDataset, bundle, test_index)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    validation_loader = DataLoader(validation_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=0)

    split_loader_map = {"validation": validation_loader, "test": test_loader}
    split_feature_frames = {
        "validation": bundle.feature_df.iloc[validation_index].reset_index(drop=True),
        "test": bundle.feature_df.iloc[test_index].reset_index(drop=True),
    }
    scale_reference_df = _build_scale_reference_from_feature_frame(
        bundle.feature_df,
        bundle.target_columns,
        split_column=str(config.get("split_column", "split")),
    )

    best_validation_rmse = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    best_model_path = artifact_dir / "model.pt"
    best_model_epoch_path = artifact_dir / "model_epoch_0000.pt"

    loss_history_rows: list[dict[str, Any]] = []
    epoch_metric_frames: list[pd.DataFrame] = []

    save_json(config, artifact_dir / "config_snapshot.json")
    save_csv(bundle.normalizer.to_frame(), artifact_dir / "scaler_by_station.csv")

    for epoch in range(1, max_epochs + 1):
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

        epoch_prediction_frames: list[pd.DataFrame] = []
        for split_name, loader in split_loader_map.items():
            normalized_predictions, _normalized_targets, station_indices, split_loss = _predict_dataset(
                model_name,
                model,
                loader,
                device=device,
                loss_function=loss_function,
            )
            loss_history_rows.append({"epoch": epoch, "split": split_name, "loss": split_loss})

            restored_predictions = bundle.normalizer.inverse_transform(normalized_predictions, station_indices)
            prediction_columns_df = pd.DataFrame(restored_predictions, columns=bundle.prediction_columns)
            prediction_frame = build_direct_prediction_frame(
                split_feature_frames[split_name],
                prediction_columns_df,
                split_column=str(config.get("split_column", "split")),
            )
            epoch_prediction_frames.append(prediction_frame)

        epoch_prediction_df = pd.concat(epoch_prediction_frames, ignore_index=True)
        overall_metrics_df, _ = evaluate_direct_prediction_frame(
            epoch_prediction_df,
            split_column=str(config.get("split_column", "split")),
            group_column="unique_id",
            scale_reference_df=scale_reference_df,
        )
        overall_metrics_df["epoch"] = epoch
        epoch_metric_frames.append(overall_metrics_df)

        validation_rows = overall_metrics_df.loc[
            (overall_metrics_df["split"] == "validation") & (overall_metrics_df["aggregation"] == "micro")
        ]
        validation_rmse = float(validation_rows["rmse"].mean())
        scheduler.step(validation_rmse)

        checkpoint_payload = {
            "epoch": epoch,
            "model_name": model_name,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "validation_rmse": validation_rmse,
        }

        if validation_rmse < best_validation_rmse:
            best_validation_rmse = validation_rmse
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
        "train": DataLoader(train_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=0),
        "validation": validation_loader,
        "test": test_loader,
    }
    final_feature_frames = {
        "train": bundle.feature_df.iloc[train_index].reset_index(drop=True),
        "validation": split_feature_frames["validation"],
        "test": split_feature_frames["test"],
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
        "loss_name": str(config.get("loss_name", "smooth_l1")),
        "requested_max_epochs": max_epochs,
        "trained_num_epochs": int(loss_history_df["epoch"].max()),
        "best_epoch": best_epoch,
        "best_validation_rmse": best_validation_rmse,
        "device": str(device),
        "batch_size": batch_size,
        "eval_batch_size": eval_batch_size,
        "gradient_clip_norm": gradient_clip_norm,
        "feature_count_ann": int(bundle.flat_features.shape[1]),
        "feature_count_lstm_context": int(bundle.context_features.shape[1]),
        "window_size": int(bundle.sequence_features.shape[1]),
        "horizon_count": len(bundle.target_columns),
        "train_rows": int(train_index.size),
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
