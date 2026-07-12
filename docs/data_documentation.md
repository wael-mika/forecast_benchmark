# Data Documentation — Forecast Benchmark

This document describes all datasets used in the Slovak river discharge forecast benchmark:
the hydrological target variable, ERA5-seamless weather reanalysis, ERA5-Land hydro-physical
reanalysis, and the flow-context features derived from upstream station discharge.

---

## 1. Target Variable — River Discharge (GRDC)

| Attribute | Value |
|---|---|
| **Source** | Global Runoff Data Centre (GRDC) daily station files |
| **Country** | Slovakia |
| **Stations in raw data** | 21 gauging stations |
| **Stations used in benchmark** | 20 (station 6144500 excluded — only 3 years of records) |
| **Temporal resolution** | Daily |
| **Benchmark period** | 1984-01-01 onwards (station-dependent end date) |
| **Processed file** | `data/processed/discharge_daily.parquet` |

### Schema

| Column | Type | Description |
|---|---|---|
| `unique_id` | string | GRDC station identifier |
| `ds` | date | Observation date |
| `y` | float | Mean daily discharge (m³/s) |

Missing-value sentinel `−999.000` is replaced with `NaN`. Exact duplicate rows are dropped.

### Station List

All 21 stations present in `discharge_daily.parquet`. **In Benchmark** indicates whether the
station participates in model training and evaluation. **In Context** marks the 17 stations
whose discharge is used as upstream flow-context features for every other station.

| GRDC ID | River | Station Name | Basin | Lat | Lon | Alt (m) | Area (km²) | Data Period | In Benchmark | In Context |
|---|---|---|---|---|---|---|---|---|---|---|
| 6142150 | Morava | Moravský Ján | Morava | 48.602 | 16.936 | 146 | 24,129 | 1921–2024 | ✓ | ✓ |
| 6142200 | Dunaj (Danube) | Bratislava | Danube | 48.140 | 17.109 | 128 | 131,331 | 1900–2024 | ✓ | ✓ |
| 6142500 | Nitra | Bánovce (Bánovská Kesa) | Nitra | 48.050 | 18.200 | 116 | 3,103 | 1978–1988 | ✓ | — |
| 6142520 | Nitra | Nitrianska Streda | Nitra | 48.524 | 18.173 | 158 | 2,094 | 1930–2024 | ✓ | ✓ |
| 6142551 | Kysuca | Kysucké Nové Mesto | Váh | 49.297 | 18.786 | 346 | 955 | 1931–2024 | ✓ | ✓ |
| 6142600 | Ipeľ | Ipeľský Sokolec | Ipeľ | 48.040 | 18.820 | 115 | 4,838 | 1930–1992 | ✓ | — |
| 6142601 | Ipeľ | Holiša | Ipeľ | 48.298 | 19.742 | 172 | 686 | 1931–2024 | ✓ | ✓ |
| 6142620 | Váh | Šaľa | Váh | 48.161 | 17.883 | 109 | 11,218 | 1920–2024 | ✓ | ✓ |
| 6142640 | Turiec | Martin | Váh | 49.070 | 18.913 | 390 | 827 | 1931–2024 | ✓ | ✓ |
| 6142650 | Hron | Banská Bystrica | Hron | 48.730 | 19.130 | 334 | 1,766 | 1930–2008 | ✓ | ✓ |
| 6142660 | Hron | Brehy | Hron | 48.407 | 18.647 | 195 | 3,821 | 1930–2024 | ✓ | ✓ |
| 6142680 | Váh | Liptovský Mikuláš | Váh | 49.087 | 19.603 | 568 | 1,107 | 1920–2024 | ✓ | ✓ |
| 6144100 | Slaná | Lenartovce | Slaná | 48.304 | 20.314 | 150 | 1,830 | 1930–2024 | ✓ | ✓ |
| 6144150 | Bodva | Turňa nad Bodvou | Bodva | 48.593 | 20.893 | 171 | 663 | 1941–2023 | ✓ | ✓ |
| 6144200 | Bodrog | Streda nad Bodrogom | Bodrog | 48.396 | 21.750 | 91 | 11,474 | 1950–2023 | ✓ | ✓ |
| 6144300 | Hernád | Ždáňa | Hernád | 48.603 | 21.340 | 167 | 4,232 | 1958–2023 | ✓ | ✓ |
| 6144350 | Torysa | Košické Olšany | Hernád | 48.733 | 21.337 | 186 | 1,298 | 1930–2023 | ✓ | ✓ |
| 6144400 | Topľa | Hanusovcé | Bodrog | 49.033 | 21.500 | — | 1,050 | 1930–2008 | ✓ | ✓ |
| 6144490 | Udava | Udavské | Bodrog | 48.983 | 21.967 | — | 211 | 1976–1988 | ✓ | — |
| 6144500 | Udava | Adidovce | Bodrog | 49.017 | 22.050 | 202 | 176 | 1978–1980 | — | — |
| 6158100 | Poprad | Chmelnica | Poprad | 49.289 | 20.730 | 507 | 1,262 | 1930–2024 | ✓ | ✓ |

**Notes on station coverage:**

- **6144500** (Adidovce, 1978–1980) — only 3 years of records; excluded from all benchmark
  feature frames because there are insufficient rows to form valid 30-day windows with
  train/val/test splits.
- **6142500** (Bánovce, 1978–1988) and **6144490** (Udavské, 1976–1988) — fewer than 15 years
  of post-1984 data; included in training where their record overlaps 1984, but excluded from
  the 17-station flow-context set.
- **6142600** (Ipeľský Sokolec, 1930–1992) — record ends before 2008; excluded from the
  flow-context set but participates in training for 1984–1992.
- **6142650** (Banská Bystrica, 1930–2008) and **6144400** (Hanusovcé, 1930–2008) — record
  ends 2008 but is long enough to be included in the flow-context set.
- **6144200** (Streda nad Bodrogom) — GRDC quality rating "Medium"; all other stations rated "High".
- Station geometry and catchment polygons: `data/raw/Data_slovakia/stationbasins.geojson`.

---

## 2. Weather Features — ERA5-Seamless (Daily)

| Attribute | Value |
|---|---|
| **Source** | Open-Meteo Archive API — `era5_seamless` model |
| **Spatial resolution** | ~28 km (ERA5 native grid) |
| **Temporal resolution** | Daily |
| **Period** | 1984-01-01 – 2024-12-31 |
| **Processed file** | `data/processed/reanalysis_daily.parquet` |
| **Download script** | `scripts/download_reanalysis_data.py` |

Coordinates for the API call are the GRDC station coordinates. Data are fetched per station;
rate-limit pauses are applied between batches to respect Open-Meteo free-tier limits.

### Variables

| Column name | ERA5 variable | Unit | Physical role |
|---|---|---|---|
| `era5_precipitation_sum` | `precipitation_sum` | mm/day | Total precipitation (rain + snow equivalent) — primary runoff driver |
| `era5_rain_sum` | `rain_sum` | mm/day | Liquid-phase precipitation — direct runoff contribution |
| `era5_snowfall_sum` | `snowfall_sum` | cm/day | Solid-phase precipitation — snowpack accumulation |
| `era5_temperature_2m_mean` | `temperature_2m_mean` | °C | Daily mean air temperature at 2 m |
| `era5_temperature_2m_max` | `temperature_2m_max` | °C | Daily maximum temperature — snowmelt indicator |
| `era5_temperature_2m_min` | `temperature_2m_min` | °C | Daily minimum temperature — frost / freeze indicator |
| `era5_precipitation_hours` | `precipitation_hours` | hours | Hours with measurable precipitation — intensity proxy |

---

## 3. Hydro-Physical Features — ERA5-Land & ERA5-Seamless Supplement

ERA5-Land provides soil-column and surface energy-balance variables that represent soil
moisture dynamics and the thermal state of the catchment. Three additional variables from
the ERA5-seamless endpoint (radiation, wind, evapotranspiration) complete the hydro-physical
picture. Together these 11 variables form the **hydro-weather data level**.

| Attribute | Value |
|---|---|
| **Sources** | Open-Meteo Archive API — `era5_seamless` and `era5_land` models |
| **ERA5-Land spatial resolution** | ~9 km (~3× finer than ERA5) |
| **Temporal resolution** | Daily |
| **Period** | 1984-01-01 – 2024-12-31 |
| **Processed file** | `data/processed/reanalysis_hydro_daily.parquet` |
| **Download scripts** | `scripts/download_hydro_era5l_daily.py`, `scripts/download_hydro_era5l_hourly.py` |

### 3a. ERA5-Seamless Supplement (3 variables)

| Column name | Variable | Unit | Physical role |
|---|---|---|---|
| `era5_shortwave_radiation_sum` | `shortwave_radiation_sum` | MJ/m²/day | Incoming solar radiation — drives snowmelt and reference evapotranspiration |
| `era5_wind_speed_10m_mean` | `wind_speed_10m_mean` | km/h | Daily mean wind speed at 10 m — secondary evapotranspiration driver |
| `era5_et0_fao_evapotranspiration` | `et0_fao_evapotranspiration` | mm/day | FAO-56 Penman–Monteith reference ET₀ — basin water loss estimate |

### 3b. ERA5-Land Soil Temperature Profile (4 variables)

Soil temperature provides the thermal state of the four standard ERA5-Land depth layers.
Critical for detecting frozen-ground conditions that suppress infiltration.

| Column name | Depth layer | Unit | Physical role |
|---|---|---|---|
| `era5l_soil_temperature_0_to_7cm_mean` | 0–7 cm | °C | Surface temperature — controls top-layer freeze/thaw |
| `era5l_soil_temperature_7_to_28cm_mean` | 7–28 cm | °C | Shallow subsurface temperature |
| `era5l_soil_temperature_28_to_100cm_mean` | 28–100 cm | °C | Root-zone temperature |
| `era5l_soil_temperature_100_to_255cm_mean` | 100–255 cm | °C | Deep thermal inertia — slow seasonal signal |

### 3c. ERA5-Land Soil Moisture Profile (4 variables)

| Column name | Depth layer | Unit | Physical role |
|---|---|---|---|
| `era5l_soil_moisture_0_to_7cm_mean` | 0–7 cm | m³/m³ | Top-layer volumetric moisture — fast runoff trigger |
| `era5l_soil_moisture_7_to_28cm_mean` | 7–28 cm | m³/m³ | Root-zone moisture — primary active storage layer |
| `era5l_soil_moisture_28_to_100cm_mean` | 28–100 cm | m³/m³ | Sub-root zone moisture — intermediate subsurface storage |
| `era5l_soil_moisture_100_to_255cm_mean` | 100–255 cm | m³/m³ | Deep moisture — slow groundwater proxy |

**Total hydro-physical variables: 11** (3 ERA5-seamless + 4 ERA5-Land soil temperature + 4 ERA5-Land soil moisture).

### ERA5-Land Coverage Caveat

Station **6158100** (Poprad at Chmelnica) falls outside the ERA5-Land spatial domain.
Its ERA5-Land feature columns contain NaN values (~6 % of rows), which are imputed
with the per-column training-set mean before model training.

---

## 4. Context Features — Upstream Flow Context

| Attribute | Value |
|---|---|
| **Source** | Derived from `discharge_daily.parquet` |
| **Temporal resolution** | Daily |
| **Lags used** | lag 0 (same day) and lag 1 (one day prior) |

### Description

"Context" refers to the observed discharge at the **17 long-record benchmark stations**,
used as upstream or hydrologically connected signals. For each target station, the discharge
at all 17 context stations lagged by 0 and 1 day is appended as additional input features.
This encodes inter-basin connectivity and routing delays without requiring an explicit
river network topology.

The 17 context stations (all records extending to at least 2008):

| GRDC ID | River | Station Name | Basin | Data Period |
|---|---|---|---|---|
| 6142150 | Morava | Moravský Ján | Morava | 1921–2024 |
| 6142200 | Dunaj (Danube) | Bratislava | Danube | 1900–2024 |
| 6142520 | Nitra | Nitrianska Streda | Nitra | 1930–2024 |
| 6142551 | Kysuca | Kysucké Nové Mesto | Váh | 1931–2024 |
| 6142601 | Ipeľ | Holiša | Ipeľ | 1931–2024 |
| 6142620 | Váh | Šaľa | Váh | 1920–2024 |
| 6142640 | Turiec | Martin | Váh | 1931–2024 |
| 6142650 | Hron | Banská Bystrica | Hron | 1930–2008 |
| 6142660 | Hron | Brehy | Hron | 1930–2024 |
| 6142680 | Váh | Liptovský Mikuláš | Váh | 1920–2024 |
| 6144100 | Slaná | Lenartovce | Slaná | 1930–2024 |
| 6144150 | Bodva | Turňa nad Bodvou | Bodva | 1941–2023 |
| 6144200 | Bodrog | Streda nad Bodrogom | Bodrog | 1950–2023 |
| 6144300 | Hernád | Ždáňa | Hernád | 1958–2023 |
| 6144350 | Torysa | Košické Olšany | Hernád | 1930–2023 |
| 6144400 | Topľa | Hanusovcé | Bodrog | 1930–2008 |
| 6158100 | Poprad | Chmelnica | Poprad | 1930–2024 |

### Column Naming

```
flow_context_{station_id}_lag_{k}    k ∈ {0, 1}
```

Example: `flow_context_6142150_lag_0` — Moravský Ján discharge on the forecast date;
`flow_context_6142150_lag_1` — one day prior.
**Total: 34 context columns** (17 stations × 2 lags).

---

## 5. Feature Engineering

Three feature frames are produced, one per benchmark data level. All rows start from
**1984-01-01** onward and require a complete discharge history window (NaN lag rows are dropped).

### 5a. Context Level — `features_context_w30_h3.parquet`

Built by `scripts/prepare_features_w30.py`.

| Feature group | # Columns | Description |
|---|---|---|
| `unique_id`, `ds` | 2 | Station ID and observation date |
| `current_y` | 1 | Discharge on the forecast origin date (lag 0 alias) |
| `lag_1` … `lag_30` | 30 | Past 30 daily discharge observations |
| `delta_1` … `delta_29` | 29 | First differences `lag_k − lag_{k+1}` (rate of change) |
| `lag_mean`, `lag_std`, `lag_min`, `lag_max` | 4 | Summary statistics over the 30-day window |
| `flow_context_{id}_lag_{0,1}` | 34 | Discharge at 17 context stations × 2 lags |
| `target_h1`, `target_h2`, `target_h3` | 3 | Forecast targets: discharge at h+1, h+2, h+3 days |
| `target_h1_ds`, `target_h2_ds`, `target_h3_ds` | 3 | Corresponding target dates |
| `forecast_origin_ds`, `split_reference_ds`, `split` | 3 | Metadata and train/val/test label |
| **Total** | **~110** | |

### 5b. Weather Level — `features_weather_plus_w30_h3.parquet`

Built by `scripts/prepare_features_w30.py` (extends the context frame in-place).

For each of the **7 ERA5-seamless variables**, the following columns are added:

| Feature sub-group | # Columns | Description |
|---|---|---|
| `{var}` | 1 | Current-day value (lag-0 alias) |
| `{var}_lag_0` … `{var}_lag_30` | 31 | Daily lags spanning 30 days into the past |
| `{var}_roll_3`, `_roll_7`, `_roll_14`, `_roll_21` | 4 | Rolling means over 3, 7, 14, 21 days |

7 variables × 36 columns = **252 ERA5 columns** added on top of the context frame.

**Total: ~362 columns.**

### 5c. Hydro-Weather Level — `features_hydro_weather_w30_h3.parquet`

Built by `scripts/prepare_hydro_features.py` (extends the weather frame with hydro variables).

> **Note:** This frame is constructed from the **14-day discharge-window** weather base
> (`features_weather_plus_w14_h3.parquet`) and therefore retains a **14-day discharge
> context window** (lags 1–14 for discharge) rather than the 30-day window used at the
> context and weather levels. The ERA5 and ERA5-Land lag features still span 0–30 days.

For each of the **11 hydro-physical variables**, the following columns are added:

| Feature sub-group | # Columns | Aggregation |
|---|---|---|
| `{var}` | 1 | Current-day value |
| `{var}_lag_0` … `{var}_lag_30` | 31 | Daily lags |
| `{var}_sum_3/7/14/21` | 4 | Rolling sum (flux variables: radiation, ET) |
| `{var}_mean_3/7/14/21` | 4 | Rolling mean (state variables: soil temperature, soil moisture, wind) |

**Total: ~607 columns.**

### 5d. Feature Frame Summary

All three files live in `data/processed/xgboost/`.

| Parquet file | Discharge window | ERA5 lags | Hydro vars | Columns | Rows | Stations |
|---|---|---|---|---|---|---|
| `features_context_w30_h3.parquet` | 30 days | — | — | 110 | 248,139 | 20 |
| `features_weather_plus_w30_h3.parquet` | 30 days | 0–30 + rolling [3,7,14,21] | — | 362 | 248,139 | 20 |
| `features_hydro_weather_w30_h3.parquet` | **14 days** | 0–30 + rolling [3,7,14,21] | 11 | 607 | 248,139 | 20 |

---

## 6. Data Split

| Split | Fraction | Purpose |
|---|---|---|
| `train` | 70 % | Model fitting |
| `validation` | 15 % | Hyper-parameter tuning and early stopping |
| `test` | 15 % | Final held-out evaluation |

Splits are applied **chronologically per station** to preserve temporal ordering.
The `split` column is present in all feature frames.

---

## 7. Normalisation

Neural network models apply **station-wise log1p + z-score normalisation**:

1. `y_norm = (log1p(y) − μ_train) / σ_train`
2. `μ_train` and `σ_train` are estimated on the training split only and stored alongside
   model artifacts.
3. Predictions are inverse-transformed before metric computation.

XGBoost models operate on raw discharge values without normalisation.

---

## 8. Forecast Setup

| Parameter | Value |
|---|---|
| Forecast horizon | 3 days (h+1, h+2, h+3) |
| Context window | 30 days — context & weather levels; 14 days — hydro-weather level |
| Forecasting strategy | Direct multi-step (separate output head per horizon) |
| Loss function | Weighted SmoothL1 + first-difference + curvature regularisation |
| Horizon loss weights | [1.0, 1.2, 1.45] (higher weight on later steps) |
| Timezone | Europe/Bratislava (CET/CEST) |

---

## 9. Data Flow

```
GRDC station files  (data/raw/Data_slovakia/**/*_Q_Day*.txt)
    └─► scripts/prepare_data.py
          └─► data/processed/discharge_daily.parquet
                  21 stations · 1900–2024 · ~604,068 rows

Open-Meteo ERA5-seamless daily API  (per station, 1984–2024)
    └─► scripts/download_reanalysis_data.py
          └─► data/processed/reanalysis_daily.parquet
                  7 weather variables · 21 stations

Open-Meteo ERA5-seamless daily API  ─┐
Open-Meteo ERA5-Land daily API      ─┤─► scripts/download_hydro_era5l_daily.py
Open-Meteo ERA5-Land hourly API     ─┘    scripts/download_hydro_era5l_hourly.py
                                           scripts/download_hydro_enrichment.py
          └─► data/processed/reanalysis_hydro_daily.parquet
                  11 hydro variables · 21 stations · 1984–2024

discharge_daily.parquet
  + reanalysis_daily.parquet
          └─► scripts/prepare_features_w30.py
                └─► features_context_w30_h3.parquet       (110 cols · 248,139 rows · 20 stations)
                └─► features_weather_plus_w30_h3.parquet  (362 cols · 248,139 rows · 20 stations)

features_weather_plus_w14_h3.parquet
  + reanalysis_hydro_daily.parquet
          └─► scripts/prepare_hydro_features.py
                └─► features_hydro_weather_w30_h3.parquet (607 cols · 248,139 rows · 20 stations)

features_*_w*_h3.parquet
    └─► scripts/run_experiment.py  (training & evaluation)
            └─► runs/*/predictions.parquet
            └─► runs/*/metrics_summary.csv
```

---

## 10. Evaluation Metrics

All models are evaluated on the held-out test split:

| Metric | Description |
|---|---|
| Bias | Mean signed error |
| MAE | Mean Absolute Error |
| RMSE | Root Mean Squared Error |
| R² | Coefficient of determination |
| NSE | Nash–Sutcliffe Efficiency (standard hydrological benchmark) |
| MAPE | Mean Absolute Percentage Error |
| SMAPE | Symmetric MAPE |
| WAPE | Weighted APE |
| MASE | Mean Absolute Scaled Error |
| RMSSE | Root Mean Squared Scaled Error |

Metrics are reported **per horizon** (h+1, h+2, h+3) and aggregated both micro
(across all station-steps) and macro (mean over stations).
