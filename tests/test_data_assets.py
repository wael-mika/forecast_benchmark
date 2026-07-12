"""Schema and integrity checks for the shipped processed data parquets.

These tests validate the distributed artifacts (discharge and the two reanalysis
parquets) without relying on any raw-ingestion code. They are skipped when the
parquets are not present (e.g. a code-only checkout).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

DISCHARGE = DATA_DIR / "discharge_daily.parquet"
REANALYSIS = DATA_DIR / "reanalysis_daily.parquet"
REANALYSIS_HYDRO = DATA_DIR / "reanalysis_hydro_daily.parquet"

WEATHER_VARS = [
    "era5_precipitation_sum",
    "era5_rain_sum",
    "era5_snowfall_sum",
    "era5_precipitation_hours",
    "era5_temperature_2m_mean",
    "era5_temperature_2m_max",
    "era5_temperature_2m_min",
]
HYDRO_VARS = [
    "era5_shortwave_radiation_sum",
    "era5_wind_speed_10m_mean",
    "era5_et0_fao_evapotranspiration",
    "era5l_soil_temperature_0_to_7cm_mean",
    "era5l_soil_temperature_7_to_28cm_mean",
    "era5l_soil_temperature_28_to_100cm_mean",
    "era5l_soil_temperature_100_to_255cm_mean",
    "era5l_soil_moisture_0_to_7cm_mean",
    "era5l_soil_moisture_7_to_28cm_mean",
    "era5l_soil_moisture_28_to_100cm_mean",
    "era5l_soil_moisture_100_to_255cm_mean",
]


def _load(path: Path) -> pd.DataFrame:
    if not path.exists():
        pytest.skip(f"data asset not present: {path.name}")
    return pd.read_parquet(path)


def _assert_continuous_daily_calendar(df: pd.DataFrame) -> None:
    for _uid, grp in df.groupby("unique_id"):
        ds = pd.to_datetime(grp["ds"]).sort_values()
        gaps = ds.diff().dropna().dt.days
        assert (gaps == 1).all(), f"non-daily calendar gap in station {_uid}"


def test_discharge_schema_and_calendar() -> None:
    df = _load(DISCHARGE)
    assert list(df.columns) == ["unique_id", "ds", "y"]
    assert pd.api.types.is_datetime64_any_dtype(df["ds"])
    assert pd.api.types.is_float_dtype(df["y"])
    assert df["unique_id"].nunique() == 21
    _assert_continuous_daily_calendar(df)


def test_reanalysis_weather_schema_and_calendar() -> None:
    df = _load(REANALYSIS)
    for column in ["unique_id", "ds", *WEATHER_VARS]:
        assert column in df.columns, f"missing weather column {column}"
    assert df["unique_id"].nunique() == 21
    _assert_continuous_daily_calendar(df)


def test_reanalysis_hydro_schema_and_calendar() -> None:
    df = _load(REANALYSIS_HYDRO)
    for column in ["unique_id", "ds", *HYDRO_VARS]:
        assert column in df.columns, f"missing hydro column {column}"
    assert df["unique_id"].nunique() == 21
    _assert_continuous_daily_calendar(df)
