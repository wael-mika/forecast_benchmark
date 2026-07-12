# ANN

## Status

Implemented baseline.

Runnable path:
- `python scripts/prepare_xgboost_data.py`
- `python scripts/run_experiment.py configs/ann.yaml`
- `python scripts/plot_neural_results.py configs/ann.yaml`

## Model

This is a simple feed-forward neural baseline for the shared 5-to-3 discharge task.

The ANN predicts the next 3 daily discharges from:
- the previous 5 daily discharge values
- 4 window summary statistics
- 4 first differences
- one learned station embedding

The model predicts residual corrections on top of a persistence baseline.

## Data

Prepared feature table:
- `data/processed/xgboost/features_w5_h3.parquet`

The ANN uses exactly the same rows, splits, and targets as XGBoost so the comparison stays fair.

## Features

Raw window:
- `lag_5` to `lag_1` in oldest-to-newest order

Derived context:
- window mean
- window std
- window min
- window max
- 4 consecutive deltas

Additional context:
- station embedding

## Normalization

The ANN uses train-only station-wise normalization:
- apply `log1p` to discharge values
- compute one mean/std pair per station on the training split
- normalize all lag and target values with those station-specific statistics
- invert back to discharge units after prediction

This is the main reason the ANN behaves much better than the earlier unscaled broad-feature setup.

## Training

Current defaults:
- loss: `SmoothL1Loss`
- optimizer: `AdamW`
- learning-rate schedule: `ReduceLROnPlateau`
- gradient clipping
- early stopping on average validation RMSE across the 3 horizons
- checkpoint saving every 5 epochs

Artifacts:
- `artifacts/ann_w5_h3_simple/model.pt`
- `artifacts/ann_w5_h3_simple/model_epoch_0011.pt`
- `artifacts/ann_w5_h3_simple/loss_history.csv`
- `artifacts/ann_w5_h3_simple/epoch_metrics.csv`
- `artifacts/ann_w5_h3_simple/scaler_by_station.csv`

## Splitting

Splits are inherited from the shared 5-to-3 frame:
- train: earliest 70% within each station
- validation: next 15%
- test: last 15%

## Evaluation

The ANN uses the same direct multi-horizon evaluation pipeline as XGBoost.

Reported outputs:
- `predictions.parquet`
- `metrics_summary.csv`
- `metrics_by_station.csv`
- diagnostic plots under `artifacts/ann_w5_h3_simple/plots/`

## Current Results

Latest test micro metrics:
- Horizon 1 RMSE: about `121.54`
- Horizon 2 RMSE: about `151.85`
- Horizon 3 RMSE: about `171.05`
- Horizon 1 MAE: about `24.76`
- Horizon 2 MAE: about `31.27`
- Horizon 3 MAE: about `35.52`

Average across horizons:
- RMSE: about `148.15`
- MAE: about `30.51`
- R2: about `0.9284`

## Takeaway

On this short-window task, the ANN is currently the strongest baseline in the repo.

That suggests the combination of:
- small local input windows
- station-wise normalization
- persistence-residual learning

is more important here than using a more complex sequence encoder.
