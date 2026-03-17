# XGBoost

## Status

Implemented baseline.

Current default experiment:
- input window: 5 daily discharge values
- forecast horizon: direct prediction of the next 3 daily discharges
- model family: one XGBoost regressor per horizon

Runnable path:
- `python scripts/prepare_data.py`
- `python scripts/prepare_xgboost_data.py`
- `python scripts/run_experiment.py`

## Model

This baseline uses gradient-boosted decision trees for direct multi-horizon forecasting.

Instead of one broad feature table with many long lags, the current setup stays intentionally simple:
- use the last 5 observed daily discharges as the main signal
- derive a few short-window summaries from those same 5 values
- train separate models for `t+1`, `t+2`, and `t+3`

This makes the experiment easier to understand and much easier to debug.

## Data

Raw source:
- Slovakia GRDC daily discharge files under `data/raw/Data_slovakia`

Canonical dataset:
- `unique_id`: station id
- `ds`: daily timestamp
- `y`: observed daily discharge

Prepared direct feature table:
- `data/processed/xgboost/features_w5_h3.parquet`

## Features

Current input features are all derived from the previous 5 daily discharge values:

Core lag inputs:
- `lag_1`
- `lag_2`
- `lag_3`
- `lag_4`
- `lag_5`

Short-window summaries:
- `lag_mean`
- `lag_std`
- `lag_min`
- `lag_max`

Short-window dynamics:
- `delta_1 = lag_1 - lag_2`
- `delta_2 = lag_2 - lag_3`
- `delta_3 = lag_3 - lag_4`
- `delta_4 = lag_4 - lag_5`

Additional context:
- categorical station identifier feature rebuilt during training and inference

Targets:
- `target_h1`
- `target_h2`
- `target_h3`

## Training

Current default objective:
- `reg:squarederror`

Tracked validation metrics during fitting:
- `rmse`
- `mae`

Current training behavior:
- one global feature table across all stations
- one direct XGBoost model for each horizon
- early stopping on the validation split
- checkpoint saving during boosting for each horizon

Artifact layout:
- `artifacts/xgboost_w5_h3_simple/h1/`
- `artifacts/xgboost_w5_h3_simple/h2/`
- `artifacts/xgboost_w5_h3_simple/h3/`
- `artifacts/xgboost_w5_h3_simple/training_summary.json`

## Splitting

Splits are created independently within each station and are chronological.

Current fractions:
- train: earliest 70%
- validation: next 15%
- test: final 15%

For the direct setup, split assignment is based on the furthest target date in the 3-step output window so the future horizon does not leak backward across split boundaries.

## Testing And Evaluation

Evaluation is performed separately for horizons 1, 2, and 3.

Reported metrics:
- bias
- mae
- mse
- rmse
- r2
- nse
- mape
- smape
- wape
- mase
- rmsse

Saved outputs:
- `artifacts/xgboost_w5_h3_simple/predictions.parquet`
- `artifacts/xgboost_w5_h3_simple/metrics_summary.csv`
- `artifacts/xgboost_w5_h3_simple/metrics_by_station.csv`

## Current Results

Latest simple baseline test metrics from `artifacts/xgboost_w5_h3_simple/metrics_summary.csv`:

- Horizon 1 test micro RMSE: about `127.61`
- Horizon 2 test micro RMSE: about `155.53`
- Horizon 3 test micro RMSE: about `172.90`
- Horizon 1 test micro MAE: about `27.19`
- Horizon 2 test micro MAE: about `34.36`
- Horizon 3 test micro MAE: about `39.67`

Average test micro scores across the 3 horizons:
- RMSE: about `152.01`
- MAE: about `33.74`

## Why This Setup

The earlier broader feature setup performed poorly because:
- station scales differ by orders of magnitude
- the feature space was larger than needed for a first baseline
- the experiment was harder to interpret

The current 5-to-3 framing is stronger because it:
- matches a simple forecasting question directly
- keeps feature engineering local to the recent history
- remains easy to compare with persistence and future neural baselines
