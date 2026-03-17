"""Prepare an XGBoost-ready tabular dataset from canonical discharge data."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.tabular import (
    assign_groupwise_time_split,
    build_enriched_direct_feature_frame,
    build_xgboost_feature_frame,
    build_xgboost_direct_feature_frame,
    drop_incomplete_direct_rows,
    drop_incomplete_tabular_rows,
    load_canonical_data,
    load_reanalysis_data,
    load_station_metadata_from_geojson,
)
from src.utils.config import load_yaml_config
from src.utils.io import save_parquet
from src.utils.logging import get_logger


def _resolve_config_path(argv: list[str]) -> Path:
    if len(argv) > 1:
        return (PROJECT_ROOT / argv[1]).resolve()
    return PROJECT_ROOT / "configs" / "xgboost.yaml"


def main(argv: list[str] | None = None) -> None:
    """Load configs and write the feature table used by XGBoost baselines."""
    active_argv = argv or sys.argv
    logger = get_logger("prepare_xgboost_data")
    model_config = load_yaml_config(_resolve_config_path(active_argv))

    canonical_data_path = PROJECT_ROOT / model_config["canonical_data_path"]
    feature_frame_path = PROJECT_ROOT / model_config["feature_frame_path"]

    canonical_df = load_canonical_data(canonical_data_path)

    if "horizons" in model_config:
        if model_config.get("use_reanalysis_features", False) or model_config.get("use_flow_context", False):
            reanalysis_df = None
            if model_config.get("use_reanalysis_features", False):
                reanalysis_df = load_reanalysis_data(PROJECT_ROOT / model_config["reanalysis_data_path"])

            feature_df, feature_columns, target_columns, required_feature_columns = build_enriched_direct_feature_frame(
                canonical_df,
                window_size=model_config["window_size"],
                horizons=model_config["horizons"],
                include_window_stats=model_config.get("include_window_stats", True),
                include_window_deltas=model_config.get("include_window_deltas", True),
                include_current_observation=model_config.get("include_current_observation", True),
                reanalysis_df=reanalysis_df,
                reanalysis_variables=model_config.get("reanalysis_variables", ()),
                reanalysis_lags=model_config.get("reanalysis_lags", ()),
                reanalysis_windows=model_config.get("reanalysis_windows", ()),
                include_future_reanalysis=model_config.get("include_future_reanalysis", False),
                flow_context_station_ids=model_config.get("flow_context_station_ids"),
                flow_context_lags=model_config.get("flow_context_lags", ()),
            )
        else:
            feature_df, feature_columns, target_columns = build_xgboost_direct_feature_frame(
                canonical_df,
                window_size=model_config["window_size"],
                horizons=model_config["horizons"],
                include_window_stats=model_config.get("include_window_stats", True),
                include_window_deltas=model_config.get("include_window_deltas", True),
            )
            required_feature_columns = feature_columns

        feature_df = drop_incomplete_direct_rows(
            feature_df,
            feature_columns,
            target_columns,
            required_feature_columns=required_feature_columns,
        )
        feature_df = assign_groupwise_time_split(
            feature_df,
            train_fraction=model_config["train_fraction"],
            validation_fraction=model_config["validation_fraction"],
            time_column="split_reference_ds",
        )
        usable_feature_count = len(feature_columns)
    else:
        station_metadata = None
        if model_config.get("include_station_metadata", True):
            station_metadata_path = PROJECT_ROOT / model_config["station_metadata_path"]
            station_metadata = load_station_metadata_from_geojson(station_metadata_path)

        feature_df, feature_columns = build_xgboost_feature_frame(
            canonical_df,
            horizon=model_config["horizon"],
            lags=model_config["lags"],
            rolling_windows=model_config["rolling_windows"],
            add_calendar_features=model_config["include_calendar_features"],
            station_metadata=station_metadata,
        )
        feature_df = drop_incomplete_tabular_rows(feature_df, feature_columns)
        feature_df = assign_groupwise_time_split(
            feature_df,
            train_fraction=model_config["train_fraction"],
            validation_fraction=model_config["validation_fraction"],
        )
        usable_feature_count = len(feature_columns)

    save_parquet(feature_df, feature_frame_path)
    logger.info(
        "Wrote %s feature rows with %s usable features to %s",
        len(feature_df),
        usable_feature_count,
        feature_frame_path,
    )


if __name__ == "__main__":
    main()
