# LSTM

## Status

Implemented baseline.

Runnable path:
- `python scripts/prepare_xgboost_data.py`
- `python scripts/run_experiment.py configs/lstm.yaml`
- `python scripts/plot_neural_results.py configs/lstm.yaml`

## Model

This baseline uses a bidirectional LSTM on the same 5-day discharge window as the ANN and XGBoost runs.

The model:
- reads the 5 normalized lag values as a short sequence
- encodes them with a bidirectional recurrent layer stack
- combines that sequence state with summary features and a station embedding
- predicts 3 residual corrections on top of a persistence baseline

## Data

Prepared feature table:
- `data/processed/xgboost/features_w5_h3.parquet`

The LSTM reuses the same direct multi-horizon rows and split labels as the ANN and XGBoost runs.

## Features

Sequence input:
- normalized `lag_5` to `lag_1`

Context features:
- window mean
- window std
- window min
- window max
- 4 consecutive deltas

Additional context:
- station embedding

## Normalization

The LSTM uses the same train-only station-wise normalization as the ANN:
- `log1p`
- one mean/std pair per station from the training split
- inverse-transform back to discharge units after prediction

## Training

Current defaults:
- loss: `SmoothL1Loss`
- optimizer: `AdamW`
- bidirectional LSTM with 2 layers
- gradient clipping
- learning-rate decay on validation plateau
- early stopping on average validation RMSE
- checkpoint saving every 5 epochs

Artifacts:
- `artifacts/lstm_w5_h3_simple/model.pt`
- `artifacts/lstm_w5_h3_simple/model_epoch_0003.pt`
- `artifacts/lstm_w5_h3_simple/loss_history.csv`
- `artifacts/lstm_w5_h3_simple/epoch_metrics.csv`
- `artifacts/lstm_w5_h3_simple/scaler_by_station.csv`

## Splitting

Splits are the same chronological 70/15/15 per-station split used for the other 5-to-3 baselines.

## Evaluation

The evaluation pipeline is the same as for XGBoost and ANN:
- micro and macro metrics by split and horizon
- per-station summaries
- prediction exports
- plot bundle under `artifacts/lstm_w5_h3_simple/plots/`

## Current Results

Latest test micro metrics:
- Horizon 1 RMSE: about `122.59`
- Horizon 2 RMSE: about `152.65`
- Horizon 3 RMSE: about `171.54`
- Horizon 1 MAE: about `24.90`
- Horizon 2 MAE: about `31.40`
- Horizon 3 MAE: about `35.74`

Average across horizons:
- RMSE: about `148.93`
- MAE: about `30.68`
- R2: about `0.9277`

## Takeaway

The bidirectional LSTM improves on XGBoost, but it does not currently beat the simpler ANN on this benchmark.

That is a useful result:
- the recurrent model is competitive
- the sequence structure helps
- but with only 5 input days, the simpler dense model is slightly easier to optimize
