"""Training helpers for tree-based models and simple deterministic baselines.

This module contains the shared training entry points for the non-neural part
of the benchmark. It handles single-target XGBoost runs, direct multi-horizon
XGBoost runs, and no-train baselines such as persistence or seasonal naive.
It also provides the small inference helpers needed to rebuild the exact
feature view expected by saved XGBoost models.

Main helpers
------------
infer_feature_columns
    Select model-ready input columns from a prepared feature frame.
infer_direct_target_columns
    Find the horizon-specific target columns in a direct setup.
train_xgboost_experiment
    Train one XGBoost model for a single target column.
train_direct_xgboost_experiment
    Train one XGBoost model per forecast horizon.
run_direct_seasonal_naive_experiment
    Build deterministic direct predictions without fitting a model.
predict_with_xgboost / predict_direct_xgboost
    Run inference with one saved booster or a full direct bundle.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.io import ensure_parent_dir, save_csv, save_json
from src.utils.seed import set_global_seed


NON_FEATURE_COLUMNS = {
    "unique_id",
    "ds",
    "y",
    "forecast_origin_ds",
    "target_ds",
    "target",
    "split",
    "y_true",
    "y_pred",
    "residual",
    "split_reference_ds",
}


@dataclass
class TrainedXGBoostExperiment:
    """Return bundle for a single-target XGBoost training run."""

    booster: Any
    feature_frame: pd.DataFrame
    feature_columns: list[str]
    artifact_dir: Path
    training_summary: dict[str, Any]


@dataclass
class TrainedDirectXGBoostExperiment:
    """Return bundle for a direct multi-horizon XGBoost training run.

    ``normalizer`` is only set when the run was configured with
    ``target_transform: log1p_station_z``; it maps booster outputs (per-station
    log1p z-scores) back to physical units at inference time.
    """

    boosters: dict[int, Any]
    feature_frame: pd.DataFrame
    feature_columns: list[str]
    target_columns: list[str]
    artifact_dir: Path
    training_summary: dict[str, Any]
    normalizer: Any = None


@dataclass
class TrainedDirectBaselineExperiment:
    """Return bundle for a deterministic direct baseline with no fitted model."""

    feature_frame: pd.DataFrame
    target_columns: list[str]
    prediction_columns_df: pd.DataFrame
    artifact_dir: Path
    training_summary: dict[str, Any]


def _require_xgboost():
    # Some environments load a conflicting OpenMP runtime before XGBoost.
    # This keeps import failures from surfacing on otherwise valid setups.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    try:
        import xgboost as xgb  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise ImportError(
            "xgboost is required for training. Install the project dependencies before running experiments."
        ) from exc
    return xgb


def infer_feature_columns(
    feature_df: pd.DataFrame,
    *,
    extra_excluded_columns: set[str] | None = None,
) -> list[str]:
    """Return usable feature columns after removing known ID, target, and metadata fields."""
    excluded_columns = set(NON_FEATURE_COLUMNS)
    if extra_excluded_columns is not None:
        excluded_columns.update(extra_excluded_columns)

    return [
        column
        for column in feature_df.columns
        if column not in excluded_columns
        and not column.startswith("target_h")
    ]


def infer_direct_target_columns(feature_df: pd.DataFrame) -> list[str]:
    """Return the ordered target columns for a direct multi-horizon setup."""
    return sorted(
        [
            column
            for column in feature_df.columns
            if column.startswith("target_h") and not column.endswith("_ds")
        ],
        key=lambda column: int(column.removeprefix("target_h")),
    )


def run_direct_seasonal_naive_experiment(
    feature_df: pd.DataFrame,
    config: dict[str, Any],
) -> TrainedDirectBaselineExperiment:
    """Build direct baseline predictions using persistence or seasonal naive logic."""
    artifact_dir = Path(config.get("artifact_dir", "artifacts/seasonal_naive"))
    ensure_parent_dir(artifact_dir / "training_summary.json")

    target_columns = infer_direct_target_columns(feature_df)
    if not target_columns:
        raise ValueError("No direct target columns found. Prepare the direct feature frame before running baselines.")

    baseline_type = str(config.get("baseline_type", "persistence")).lower()
    if baseline_type not in {"persistence", "seasonal_naive"}:
        raise ValueError("baseline_type must be one of {'persistence', 'seasonal_naive'}.")

    season_length = int(config.get("season_length", 7))
    prediction_columns_df = pd.DataFrame(index=feature_df.index)

    for target_column in target_columns:
        horizon = int(target_column.removeprefix("target_h"))
        if baseline_type == "persistence":
            if "current_y" in feature_df.columns:
                predictions = feature_df["current_y"]
            elif "lag_1" in feature_df.columns:
                predictions = feature_df["lag_1"]
            else:
                raise ValueError("Persistence baseline requires either 'current_y' or 'lag_1' in the feature frame.")
        else:
            source_offset = season_length - horizon
            if source_offset < 0:
                raise ValueError(
                    f"season_length={season_length} is too short for horizon={horizon}. Increase the seasonal period."
                )
            if source_offset == 0:
                if "current_y" not in feature_df.columns:
                    raise ValueError("Seasonal naive requires 'current_y' when season_length equals the forecast horizon.")
                predictions = feature_df["current_y"]
            else:
                source_column = f"lag_{source_offset}"
                if source_column not in feature_df.columns:
                    raise ValueError(
                        f"Seasonal naive requires feature column {source_column!r} for horizon {horizon}."
                    )
                predictions = feature_df[source_column]

        prediction_columns_df[f"prediction_h{horizon}"] = predictions.to_numpy(dtype=float)

    training_summary = {
        "model_name": str(config.get("model_name", "seasonal_naive")).lower(),
        "baseline_type": baseline_type,
        "season_length": season_length,
        "target_columns": target_columns,
        "train_rows": int((feature_df.get(config.get("split_column", "split")) == "train").sum()),
        "validation_rows": int((feature_df.get(config.get("split_column", "split")) == "validation").sum()),
        "test_rows": int((feature_df.get(config.get("split_column", "split")) == "test").sum()),
    }
    save_json(config, artifact_dir / "config_snapshot.json")
    save_json(training_summary, artifact_dir / "training_summary.json")

    return TrainedDirectBaselineExperiment(
        feature_frame=feature_df.copy(),
        target_columns=target_columns,
        prediction_columns_df=prediction_columns_df.reset_index(drop=True),
        artifact_dir=artifact_dir,
        training_summary=training_summary,
    )


def _prepare_model_frame(
    feature_df: pd.DataFrame,
    feature_columns: list[str],
    *,
    use_station_id_as_feature: bool,
) -> tuple[pd.DataFrame, list[str], bool]:
    """Optionally add station ID as a categorical feature and report the final feature list."""
    model_frame = feature_df.copy()
    used_feature_columns = list(feature_columns)
    enable_categorical = False

    if use_station_id_as_feature:
        model_frame["station_id_feature"] = model_frame["unique_id"].astype("category")
        used_feature_columns.append("station_id_feature")
        enable_categorical = True

    return model_frame, used_feature_columns, enable_categorical


def _build_dmatrix(
    xgb: Any,
    df: pd.DataFrame,
    *,
    feature_columns: list[str],
    target_column: str,
    enable_categorical: bool,
) -> Any:
    """Create the XGBoost ``DMatrix`` used for training or inference."""
    return xgb.DMatrix(
        df.loc[:, feature_columns],
        label=df[target_column],
        enable_categorical=enable_categorical,
    )


def prepare_inference_frame(feature_df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Recreate lightweight derived columns that a saved XGBoost model expects at inference time."""
    model_frame = feature_df.copy()
    if "station_id_feature" in feature_columns and "station_id_feature" not in model_frame.columns:
        if "unique_id" not in model_frame.columns:
            raise ValueError("Cannot rebuild station_id_feature because unique_id is missing.")
        model_frame["station_id_feature"] = model_frame["unique_id"].astype("category")
    return model_frame


def _build_training_checkpoint_callback(xgb: Any, checkpoint_dir: Path, interval: int) -> Any:
    """Create a version-compatible XGBoost checkpoint callback."""
    ensure_parent_dir(checkpoint_dir / ".keep")
    try:
        return xgb.callback.TrainingCheckPoint(
            directory=str(checkpoint_dir),
            name="xgboost",
            interval=interval,
        )
    except TypeError:  # pragma: no cover - version compatibility branch
        return xgb.callback.TrainingCheckPoint(
            directory=str(checkpoint_dir),
            name="xgboost",
            iterations=interval,
        )


def _save_feature_importance(booster: Any, feature_columns: list[str], artifact_dir: Path) -> None:
    """Save gain-based feature importance for one trained XGBoost model."""
    raw_gain_scores = booster.get_score(importance_type="gain")
    feature_importance_df = pd.DataFrame(
        {
            "feature": feature_columns,
            "gain": [float(raw_gain_scores.get(feature_name, 0.0)) for feature_name in feature_columns],
        }
    ).sort_values("gain", ascending=False, kind="stable")
    save_csv(feature_importance_df, artifact_dir / "feature_importance.csv")


def train_xgboost_experiment(
    feature_df: pd.DataFrame,
    config: dict[str, Any],
) -> TrainedXGBoostExperiment:
    """Train one XGBoost regressor on a prepared single-target feature frame."""
    xgb = _require_xgboost()

    split_column = config.get("split_column", "split")
    target_column = config.get("target_column", "target")
    artifact_dir = Path(config.get("artifact_dir", "artifacts/xgboost"))
    ensure_parent_dir(artifact_dir / "model.json")

    set_global_seed(int(config.get("seed", 42)))

    base_feature_columns = infer_feature_columns(feature_df)
    model_frame, feature_columns, enable_categorical = _prepare_model_frame(
        feature_df,
        base_feature_columns,
        use_station_id_as_feature=bool(config.get("use_station_id_as_feature", True)),
    )

    train_df = model_frame.loc[model_frame[split_column] == "train"].reset_index(drop=True)
    validation_df = model_frame.loc[model_frame[split_column] == "validation"].reset_index(drop=True)
    test_df = model_frame.loc[model_frame[split_column] == "test"].reset_index(drop=True)

    if train_df.empty:
        raise ValueError("Training split is empty. Prepare the XGBoost feature frame before training.")
    if validation_df.empty:
        raise ValueError("Validation split is empty. Early stopping requires a validation split.")
    if test_df.empty:
        raise ValueError("Test split is empty. Evaluation requires a held-out test split.")

    dtrain = _build_dmatrix(
        xgb,
        train_df,
        feature_columns=feature_columns,
        target_column=target_column,
        enable_categorical=enable_categorical,
    )
    dvalidation = _build_dmatrix(
        xgb,
        validation_df,
        feature_columns=feature_columns,
        target_column=target_column,
        enable_categorical=enable_categorical,
    )

    eval_metric = config.get("eval_metric", ["rmse", "mae"])
    params = {
        "objective": config.get("objective", "reg:squarederror"),
        "eval_metric": eval_metric,
        "tree_method": config.get("tree_method", "hist"),
        "learning_rate": config.get("learning_rate", 0.05),
        "max_depth": int(config.get("max_depth", 6)),
        "min_child_weight": float(config.get("min_child_weight", 1.0)),
        "subsample": float(config.get("subsample", 0.8)),
        "colsample_bytree": float(config.get("colsample_bytree", 0.8)),
        "reg_lambda": float(config.get("reg_lambda", 1.0)),
        "reg_alpha": float(config.get("reg_alpha", 0.0)),
        "seed": int(config.get("seed", 42)),
        "verbosity": int(config.get("verbosity", 1)),
    }

    callbacks: list[Any] = []
    early_stopping_rounds = int(config.get("early_stopping_rounds", 50))
    if early_stopping_rounds > 0:
        early_stopping_metric = config.get(
            "early_stopping_metric",
            eval_metric[0] if isinstance(eval_metric, list) else eval_metric,
        )
        callbacks.append(
            xgb.callback.EarlyStopping(
                rounds=early_stopping_rounds,
                metric_name=early_stopping_metric,
                data_name="validation",
                save_best=True,
            )
        )

    checkpoint_interval = int(config.get("checkpoint_interval", 0))
    if checkpoint_interval > 0:
        callbacks.append(_build_training_checkpoint_callback(xgb, artifact_dir / "checkpoints", checkpoint_interval))

    evals_result: dict[str, dict[str, list[float]]] = {}
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=int(config.get("num_boost_round", 500)),
        evals=[(dtrain, "train"), (dvalidation, "validation")],
        evals_result=evals_result,
        callbacks=callbacks,
        verbose_eval=config.get("verbose_eval", 25),
    )

    best_iteration = getattr(booster, "best_iteration", None)
    trained_rounds = int(best_iteration) + 1 if best_iteration is not None else booster.num_boosted_rounds()

    booster.save_model(str(artifact_dir / "model.json"))
    booster.save_model(str(artifact_dir / f"model_rounds_{trained_rounds:04d}.json"))

    _save_feature_importance(booster, feature_columns, artifact_dir)

    training_summary = {
        "objective": params["objective"],
        "eval_metric": eval_metric,
        "requested_num_boost_round": int(config.get("num_boost_round", 500)),
        "trained_num_boost_round": trained_rounds,
        "best_iteration": None if best_iteration is None else int(best_iteration),
        "best_score": None if getattr(booster, "best_score", None) is None else float(getattr(booster, "best_score")),
        "feature_count": len(feature_columns),
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(validation_df)),
        "test_rows": int(len(test_df)),
        "use_station_id_as_feature": bool(config.get("use_station_id_as_feature", True)),
        "feature_columns": feature_columns,
    }

    save_json(config, artifact_dir / "config_snapshot.json")
    save_json(training_summary, artifact_dir / "training_summary.json")
    save_json({"evals_result": evals_result}, artifact_dir / "evals_result.json")

    return TrainedXGBoostExperiment(
        booster=booster,
        feature_frame=model_frame,
        feature_columns=feature_columns,
        artifact_dir=artifact_dir,
        training_summary=training_summary,
    )


SUPPORTED_TARGET_TRANSFORMS = {"log1p_station_z"}


def _fit_direct_target_normalizer(
    feature_df: pd.DataFrame,
    *,
    target_columns: list[str],
    split_column: str,
) -> Any:
    """Fit the per-station log1p normalizer used by the scaled-target XGBoost variant.

    This reuses the exact fitting logic of the neural pipeline
    (``src.training.advanced_neural._fit_station_normalizer``), so the scaled
    variant trains in the identical target space as the neural models:
    z = (log1p(y) - mu_station) / sigma_station, with statistics fitted on the
    TRAIN split only from the target history (lag columns plus current
    discharge) together with the train-split targets.
    """
    from src.training.advanced_neural import _fit_station_normalizer, _infer_target_lag_columns

    station_ids = sorted(feature_df["unique_id"].astype(str).unique().tolist())
    station_to_index = {station_id: index for index, station_id in enumerate(station_ids)}
    station_indices = feature_df["unique_id"].astype(str).map(station_to_index).to_numpy(dtype=np.int64)
    split_labels = feature_df[split_column].astype(str).to_numpy()

    history_columns = list(_infer_target_lag_columns(feature_df))
    if "current_y" in feature_df.columns:
        history_columns.append("current_y")

    log_history_values = np.log1p(feature_df.loc[:, history_columns].to_numpy(dtype=np.float32))
    log_target_values = np.log1p(feature_df.loc[:, target_columns].to_numpy(dtype=np.float32))
    return _fit_station_normalizer(
        log_history_values,
        log_target_values,
        station_indices,
        split_labels,
        station_ids,
    )


def _transform_direct_targets_to_station_z(
    df: pd.DataFrame,
    *,
    target_columns: list[str],
    normalizer: Any,
) -> pd.DataFrame:
    """Return a copy of ``df`` whose direct target columns are per-station log1p z-scores."""
    transformed = df.copy()
    station_indices = transformed["unique_id"].astype(str).map(normalizer.station_to_index).to_numpy(dtype=np.int64)
    means = normalizer.mean_by_station[station_indices][:, None]
    stds = normalizer.std_by_station[station_indices][:, None]
    log_targets = np.log1p(transformed.loc[:, target_columns].to_numpy(dtype=np.float64))
    transformed.loc[:, target_columns] = (log_targets - means) / stds
    return transformed


def train_direct_xgboost_experiment(
    feature_df: pd.DataFrame,
    config: dict[str, Any],
) -> TrainedDirectXGBoostExperiment:
    """Train a separate XGBoost model for each direct forecast horizon."""
    xgb = _require_xgboost()

    split_column = config.get("split_column", "split")
    artifact_dir = Path(config.get("artifact_dir", "artifacts/xgboost"))
    ensure_parent_dir(artifact_dir / "training_summary.json")
    set_global_seed(int(config.get("seed", 42)))

    target_transform = config.get("target_transform")
    if target_transform is not None and target_transform not in SUPPORTED_TARGET_TRANSFORMS:
        raise ValueError(
            f"Unsupported target_transform: {target_transform!r}. "
            f"Omit it for raw targets or use one of {sorted(SUPPORTED_TARGET_TRANSFORMS)}."
        )

    target_columns = infer_direct_target_columns(feature_df)
    if not target_columns:
        raise ValueError("No direct target columns found. Prepare the direct feature frame before training.")

    base_feature_columns = infer_feature_columns(feature_df)
    model_frame, feature_columns, enable_categorical = _prepare_model_frame(
        feature_df,
        base_feature_columns,
        use_station_id_as_feature=bool(config.get("use_station_id_as_feature", True)),
    )

    train_df = model_frame.loc[model_frame[split_column] == "train"].reset_index(drop=True)
    validation_df = model_frame.loc[model_frame[split_column] == "validation"].reset_index(drop=True)
    test_df = model_frame.loc[model_frame[split_column] == "test"].reset_index(drop=True)

    if train_df.empty:
        raise ValueError("Training split is empty. Prepare the XGBoost feature frame before training.")
    if validation_df.empty:
        raise ValueError("Validation split is empty. Early stopping requires a validation split.")
    if test_df.empty:
        raise ValueError("Test split is empty. Evaluation requires a held-out test split.")

    normalizer = None
    if target_transform == "log1p_station_z":
        # Scaled-target variant: each horizon's booster regresses per-station
        # log1p z-scores instead of raw pooled m3/s. The raw-target baseline
        # optimizes pooled squared error in physical units, so scale-minority
        # stations contribute almost nothing to the split objective and can be
        # served by cross-station feature values that are wildly wrong for
        # their own scale. Training in the neural models' target space removes
        # exactly that asymmetry while leaving the tree inputs untouched:
        # features stay RAW because trees are invariant to monotone feature
        # transforms — only the target space matters here. Early stopping still
        # monitors validation RMSE, which is now z-space RMSE and therefore
        # weights every station uniformly in relative terms.
        normalizer = _fit_direct_target_normalizer(
            feature_df,
            target_columns=target_columns,
            split_column=split_column,
        )
        save_csv(normalizer.to_frame(), artifact_dir / "scaler_by_station.csv")
        train_df = _transform_direct_targets_to_station_z(
            train_df, target_columns=target_columns, normalizer=normalizer
        )
        validation_df = _transform_direct_targets_to_station_z(
            validation_df, target_columns=target_columns, normalizer=normalizer
        )

    eval_metric = config.get("eval_metric", ["rmse", "mae"])
    base_params = {
        "objective": config.get("objective", "reg:squarederror"),
        "eval_metric": eval_metric,
        "tree_method": config.get("tree_method", "hist"),
        "learning_rate": config.get("learning_rate", 0.05),
        "max_depth": int(config.get("max_depth", 6)),
        "min_child_weight": float(config.get("min_child_weight", 1.0)),
        "subsample": float(config.get("subsample", 0.8)),
        "colsample_bytree": float(config.get("colsample_bytree", 0.8)),
        "reg_lambda": float(config.get("reg_lambda", 1.0)),
        "reg_alpha": float(config.get("reg_alpha", 0.0)),
        "seed": int(config.get("seed", 42)),
        "verbosity": int(config.get("verbosity", 1)),
    }

    boosters: dict[int, Any] = {}
    per_horizon_summary: dict[str, Any] = {}

    for target_column in target_columns:
        horizon = int(target_column.removeprefix("target_h"))
        dtrain = _build_dmatrix(
            xgb,
            train_df,
            feature_columns=feature_columns,
            target_column=target_column,
            enable_categorical=enable_categorical,
        )
        dvalidation = _build_dmatrix(
            xgb,
            validation_df,
            feature_columns=feature_columns,
            target_column=target_column,
            enable_categorical=enable_categorical,
        )

        callbacks: list[Any] = []
        early_stopping_rounds = int(config.get("early_stopping_rounds", 30))
        if early_stopping_rounds > 0:
            early_stopping_metric = config.get(
                "early_stopping_metric",
                eval_metric[0] if isinstance(eval_metric, list) else eval_metric,
            )
            callbacks.append(
                xgb.callback.EarlyStopping(
                    rounds=early_stopping_rounds,
                    metric_name=early_stopping_metric,
                    data_name="validation",
                    save_best=True,
                )
            )

        checkpoint_interval = int(config.get("checkpoint_interval", 0))
        horizon_dir = artifact_dir / f"h{horizon}"
        if checkpoint_interval > 0:
            callbacks.append(
                _build_training_checkpoint_callback(xgb, horizon_dir / "checkpoints", checkpoint_interval)
            )

        evals_result: dict[str, dict[str, list[float]]] = {}
        booster = xgb.train(
            params=base_params,
            dtrain=dtrain,
            num_boost_round=int(config.get("num_boost_round", 300)),
            evals=[(dtrain, "train"), (dvalidation, "validation")],
            evals_result=evals_result,
            callbacks=callbacks,
            verbose_eval=config.get("verbose_eval", 25),
        )

        best_iteration = getattr(booster, "best_iteration", None)
        trained_rounds = int(best_iteration) + 1 if best_iteration is not None else booster.num_boosted_rounds()

        ensure_parent_dir(horizon_dir / "model.json")
        booster.save_model(str(horizon_dir / "model.json"))
        booster.save_model(str(horizon_dir / f"model_rounds_{trained_rounds:04d}.json"))
        _save_feature_importance(booster, feature_columns, horizon_dir)

        horizon_summary = {
            "horizon": horizon,
            "target_column": target_column,
            "requested_num_boost_round": int(config.get("num_boost_round", 300)),
            "trained_num_boost_round": trained_rounds,
            "best_iteration": None if best_iteration is None else int(best_iteration),
            "best_score": None
            if getattr(booster, "best_score", None) is None
            else float(getattr(booster, "best_score")),
            "eval_metric": eval_metric,
            "artifact_dir": str(horizon_dir),
        }
        save_json(horizon_summary, horizon_dir / "training_summary.json")
        save_json({"evals_result": evals_result}, horizon_dir / "evals_result.json")

        boosters[horizon] = booster
        per_horizon_summary[f"h{horizon}"] = horizon_summary

    training_summary = {
        "objective": base_params["objective"],
        "eval_metric": eval_metric,
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "target_columns": target_columns,
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(validation_df)),
        "test_rows": int(len(test_df)),
        "use_station_id_as_feature": bool(config.get("use_station_id_as_feature", True)),
        "per_horizon": per_horizon_summary,
    }
    if target_transform is not None:
        # With log1p_station_z the per-horizon best_score values (and the
        # early-stopping signal) are validation RMSE in z-space, not m3/s.
        training_summary["target_transform"] = target_transform
        training_summary["early_stopping_target_space"] = "log1p_station_z"
    save_json(config, artifact_dir / "config_snapshot.json")
    save_json(training_summary, artifact_dir / "training_summary.json")

    return TrainedDirectXGBoostExperiment(
        boosters=boosters,
        feature_frame=model_frame,
        feature_columns=feature_columns,
        target_columns=target_columns,
        artifact_dir=artifact_dir,
        training_summary=training_summary,
        normalizer=normalizer,
    )


def predict_with_xgboost(
    booster: Any,
    feature_df: pd.DataFrame,
    *,
    feature_columns: list[str],
    target_column: str = "target",
    enable_categorical: bool = False,
    iteration_range: tuple[int, int] | None = None,
) -> pd.Series:
    """Run inference with one XGBoost booster and return predictions as a series."""
    xgb = _require_xgboost()
    model_frame = prepare_inference_frame(feature_df, feature_columns)
    inference_matrix = _build_dmatrix(
        xgb,
        model_frame,
        feature_columns=feature_columns,
        target_column=target_column,
        enable_categorical=enable_categorical,
    )
    prediction_kwargs = {}
    if iteration_range is not None:
        prediction_kwargs["iteration_range"] = iteration_range
    predictions = booster.predict(inference_matrix, **prediction_kwargs)
    return pd.Series(predictions, index=model_frame.index, dtype=float)


def predict_direct_xgboost(
    boosters: dict[int, Any],
    feature_df: pd.DataFrame,
    *,
    feature_columns: list[str],
    enable_categorical: bool = False,
    normalizer: Any = None,
) -> pd.DataFrame:
    """Run inference with a direct XGBoost bundle and return one prediction column per horizon.

    When ``normalizer`` is given (scaled-target runs), booster outputs are
    per-station log1p z-scores and are inverse-transformed back to physical
    units (expm1, clipped at zero) before the frame is returned, so the
    downstream evaluation pipeline always scores in m3/s.
    """
    prediction_frame = pd.DataFrame(index=feature_df.index)
    for horizon, booster in sorted(boosters.items()):
        prediction_frame[f"prediction_h{horizon}"] = predict_with_xgboost(
            booster,
            feature_df,
            feature_columns=feature_columns,
            target_column=f"target_h{horizon}",
            enable_categorical=enable_categorical,
        )
    if normalizer is not None:
        station_indices = (
            feature_df["unique_id"].astype(str).map(normalizer.station_to_index).to_numpy(dtype=np.int64)
        )
        restored = normalizer.inverse_transform(prediction_frame.to_numpy(dtype=float), station_indices)
        prediction_frame = pd.DataFrame(restored, columns=prediction_frame.columns, index=prediction_frame.index)
    return prediction_frame.reset_index(drop=True)
