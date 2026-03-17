from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.tabular import (
    assign_groupwise_time_split,
    build_enriched_direct_feature_frame,
    build_xgboost_direct_feature_frame,
    build_xgboost_feature_frame,
    drop_incomplete_direct_rows,
    drop_incomplete_tabular_rows,
)


def test_build_xgboost_feature_frame_creates_supervised_target_and_lags() -> None:
    df = pd.DataFrame(
        {
            "unique_id": ["station_a"] * 10,
            "ds": pd.date_range("2020-01-01", periods=10, freq="D"),
            "y": list(range(10)),
        }
    )

    feature_df, feature_columns = build_xgboost_feature_frame(
        df,
        horizon=2,
        lags=[0, 1, 2],
        rolling_windows=[3],
        add_calendar_features=False,
    )
    usable = drop_incomplete_tabular_rows(feature_df, feature_columns)

    sample = usable.loc[usable["forecast_origin_ds"] == pd.Timestamp("2020-01-04")].iloc[0]

    assert sample["target_ds"] == pd.Timestamp("2020-01-06")
    assert sample["target"] == 5
    assert sample["lag_0"] == 3
    assert sample["lag_1"] == 2
    assert sample["lag_2"] == 1
    assert sample["rolling_mean_3"] == 2.0
    assert sample["rolling_min_3"] == 1.0
    assert sample["rolling_max_3"] == 3.0


def test_assign_groupwise_time_split_respects_station_boundaries() -> None:
    df = pd.DataFrame(
        {
            "unique_id": ["station_a"] * 4 + ["station_b"] * 4,
            "target_ds": pd.date_range("2020-01-01", periods=4, freq="D").tolist() * 2,
            "target": [1.0] * 8,
            "lag_0": [1.0] * 8,
        }
    )

    split_df = assign_groupwise_time_split(df, train_fraction=0.5, validation_fraction=0.25)

    station_a_splits = split_df.loc[split_df["unique_id"] == "station_a", "split"].astype(str).tolist()
    station_b_splits = split_df.loc[split_df["unique_id"] == "station_b", "split"].astype(str).tolist()

    assert station_a_splits == ["train", "train", "validation", "test"]
    assert station_b_splits == ["train", "train", "validation", "test"]


def test_build_xgboost_direct_feature_frame_creates_5_to_3_structure() -> None:
    df = pd.DataFrame(
        {
            "unique_id": ["station_a"] * 12,
            "ds": pd.date_range("2020-01-01", periods=12, freq="D"),
            "y": list(range(12)),
        }
    )

    feature_df, feature_columns, target_columns = build_xgboost_direct_feature_frame(
        df,
        window_size=5,
        horizons=[1, 2, 3],
        include_window_stats=True,
        include_window_deltas=True,
    )
    usable = drop_incomplete_direct_rows(feature_df, feature_columns, target_columns)

    sample = usable.loc[usable["forecast_origin_ds"] == pd.Timestamp("2020-01-06")].iloc[0]

    assert feature_columns[:5] == ["lag_1", "lag_2", "lag_3", "lag_4", "lag_5"]
    assert target_columns == ["target_h1", "target_h2", "target_h3"]
    assert sample["lag_1"] == 4
    assert sample["lag_5"] == 0
    assert sample["lag_mean"] == 2.0
    assert sample["delta_1"] == 1.0
    assert sample["target_h1"] == 6
    assert sample["target_h2"] == 7
    assert sample["target_h3"] == 8
    assert sample["split_reference_ds"] == pd.Timestamp("2020-01-09")


def test_build_enriched_direct_feature_frame_adds_weather_and_flow_context() -> None:
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    discharge_df = pd.DataFrame(
        {
            "unique_id": ["station_a"] * 10 + ["station_b"] * 10,
            "ds": list(dates) * 2,
            "y": list(range(10)) + list(range(10, 20)),
        }
    )
    reanalysis_df = pd.DataFrame(
        {
            "unique_id": ["station_a"] * 10 + ["station_b"] * 10,
            "ds": list(dates) * 2,
            "era5_precipitation_sum": [0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 3.0, 0.0, 1.0] * 2,
            "era5_temperature_2m_mean": [1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0] * 2,
        }
    )

    feature_df, feature_columns, target_columns, required_feature_columns = build_enriched_direct_feature_frame(
        discharge_df,
        window_size=3,
        horizons=[1, 2],
        include_current_observation=True,
        reanalysis_df=reanalysis_df,
        reanalysis_variables=["era5_precipitation_sum", "era5_temperature_2m_mean"],
        reanalysis_lags=[0, 1],
        reanalysis_windows=[2],
        include_future_reanalysis=True,
        flow_context_station_ids=["station_a", "station_b"],
        flow_context_lags=[0],
    )
    usable = drop_incomplete_direct_rows(
        feature_df,
        feature_columns,
        target_columns,
        required_feature_columns=required_feature_columns,
    )

    sample = usable.loc[
        (usable["unique_id"] == "station_a") & (usable["forecast_origin_ds"] == pd.Timestamp("2020-01-04"))
    ].iloc[0]

    assert "current_y" in feature_columns
    assert "era5_precipitation_sum_lag_0" in feature_columns
    assert "era5_precipitation_sum_future_h1" in feature_columns
    assert "flow_context_station_b_lag_0" in feature_columns
    assert sample["current_y"] == 3.0
    assert sample["era5_precipitation_sum_lag_0"] == 2.0
    assert sample["era5_temperature_2m_mean_mean_2"] == 3.5
    assert sample["era5_precipitation_sum_future_h1"] == 0.0
    assert sample["flow_context_station_b_lag_0"] == 13.0
