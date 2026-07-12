# Daily River Discharge Forecasting Benchmark

This repository is a clean starting point for a daily river discharge forecasting benchmark on Slovakia GRDC stations.

The long-term benchmark will compare:
- seasonal naive
- xgboost on lag features
- ann on normalized windows
- lstm on normalized windows
- nhits
- patchtst
- tft
- xlstm
- one mamba-based model

The repository now covers two data-preparation layers:
- raw GRDC daily ingestion into one canonical long-format dataset
- a simple XGBoost-ready direct forecasting table built from the previous 5 daily discharges

## Canonical schema

All downstream data processing will use:
- `unique_id`: station identifier
- `ds`: timestamp
- `y`: daily discharge target

## Repository layout

```text
forecast_benchmark/
  configs/
  docs/
  data/
    raw/
    processed/
  scripts/
  src/
    data/
    evaluation/
    models/
    training/
    utils/
  tests/
```

## Current milestone

Implemented now:
- repository skeleton
- YAML config loading
- defensive GRDC-style file ingestion from nested folders under `data/raw/`
- daily-file filtering for the Slovakia GRDC dump
- canonical schema validation
- GRDC station-id extraction from headers/filenames
- missing-value sentinel handling for `-999.000`
- exact duplicate row removal
- parquet export to `data/processed/discharge_daily.parquet`
- simple 5-lag feature generation for direct 3-day XGBoost forecasting
- lightweight window summaries and lag deltas derived from those 5 inputs
- chronological per-station train/validation/test split labels for tabular baselines
- direct multi-horizon XGBoost training with one model per forecast horizon
- station-wise `log1p + z-score` normalization for neural baselines
- ANN baseline on the same 5-to-3 framing
- bidirectional LSTM baseline on the same 5-to-3 framing
- prediction export plus metric summaries for trained experiments
- plot generation for XGBoost, ANN, LSTM, and model-comparison diagnostics
- per-model benchmark notes under `docs/models/`
- pytest coverage for schema checks and ingestion

Not implemented yet:
- seasonal naive benchmark
- transformer-style sequence baselines

## Raw data expectations

Place raw daily station files in `data/raw/`.

Assumptions for the current data-prep layer:
- files are delimited text tables such as `.csv`, `.txt`, `.tsv`, or `.dat`
- column names may vary slightly across station files
- the ingestion code will try multiple parsing strategies and match columns using configured candidate names
- if no station id column is found, the GRDC header or filename prefix is used as `unique_id`
- missing `y` values are preserved
- GRDC missing-value sentinels such as `-999.000` are converted to nulls
- exact duplicate rows are removed
- conflicting duplicate `(unique_id, ds)` rows are treated as an error instead of being silently dropped
- the default config ingests only daily discharge files (`*_Q_Day*.txt`) for the benchmark target

Files that cannot be parsed are skipped with a warning so one bad file does not stop the whole directory scan.

## Usage

From the repository root:

```bash
cd forecast_benchmark
python scripts/prepare_data.py
python scripts/prepare_xgboost_data.py
python scripts/run_experiment.py
python scripts/run_experiment.py configs/ann.yaml
python scripts/run_experiment.py configs/lstm.yaml
python scripts/plot_xgboost_results.py artifacts/xgboost_w5_h3_simple
python scripts/plot_neural_results.py configs/ann.yaml
python scripts/plot_neural_results.py configs/lstm.yaml
python scripts/compare_model_results.py
```

This produces:
- `data/processed/discharge_daily.parquet`
- `data/processed/xgboost/features_w5_h3.parquet`
- `artifacts/xgboost_w5_h3_simple/h1/model.json`
- `artifacts/xgboost_w5_h3_simple/h2/model.json`
- `artifacts/xgboost_w5_h3_simple/h3/model.json`
- `artifacts/xgboost_w5_h3_simple/predictions.parquet`
- `artifacts/xgboost_w5_h3_simple/metrics_summary.csv`
- `artifacts/xgboost_w5_h3_simple/metrics_by_station.csv`
- `artifacts/xgboost_w5_h3_simple/plots/`
- `artifacts/ann_w5_h3_simple/model.pt`
- `artifacts/ann_w5_h3_simple/metrics_summary.csv`
- `artifacts/ann_w5_h3_simple/plots/`
- `artifacts/lstm_w5_h3_simple/model.pt`
- `artifacts/lstm_w5_h3_simple/metrics_summary.csv`
- `artifacts/lstm_w5_h3_simple/plots/`
- `artifacts/model_comparison/`

## XGBoost framing

The default XGBoost prep step now uses a deliberately simple direct setup:
- inputs: the previous 5 daily discharge values
- derived features: 5-day mean, std, min, max
- derived features: 4 first-order lag deltas
- one categorical station identifier feature
- outputs: direct forecasts for day `t+1`, `t+2`, and `t+3`

This gives 13 numeric/categorical input features in total before model-specific handling.

## Neural Baselines

The ANN and LSTM baselines reuse the exact same 5-step input window and 3-step direct targets.

Shared preprocessing:
- `log1p` transform on discharge values
- per-station mean/std fit on the training split only
- inverse-transform back to discharge units for evaluation
- residual prediction on top of a persistence baseline

ANN baseline:
- flattened normalized window
- 4 window summary statistics
- 4 first differences
- station embedding
- dense prediction head

LSTM baseline:
- normalized 5-step sequence
- bidirectional recurrent encoder
- window summary statistics and deltas as context features
- station embedding
- dense residual head

## Training And Evaluation

The experiment runner currently trains the XGBoost baseline using:
- `reg:squarederror` as the default objective
- three separate direct models for horizons 1, 2, and 3
- early stopping on the validation split for each horizon
- periodic model checkpoints during boosting

The evaluation pipeline writes:
- full long-format prediction rows for train, validation, and test
- micro metrics over all rows in a split for each horizon
- macro metrics averaged across stations for each horizon
- per-station metric tables for each horizon

Included metrics:
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

## Current Test Results

Test micro metrics on the shared 5-to-3 setup:

- XGBoost average RMSE across horizons: about `152.01`
- ANN average RMSE across horizons: about `148.15`
- LSTM average RMSE across horizons: about `148.93`
- XGBoost average MAE across horizons: about `33.74`
- ANN average MAE across horizons: about `30.51`
- LSTM average MAE across horizons: about `30.68`

Per-horizon test RMSE:
- XGBoost: `127.61`, `155.53`, `172.90`
- ANN: `121.54`, `151.85`, `171.05`
- LSTM: `122.59`, `152.65`, `171.54`

On this short 5-day context, the ANN is currently the strongest overall baseline, with the bidirectional LSTM close behind and both neural models improving on XGBoost.

## Model Docs

Model notes live under `docs/models/`.

Implemented now:
- `docs/models/xgboost.md`
- `docs/models/ann.md`
- `docs/models/lstm.md`

Planned benchmark notes:
- `docs/models/seasonal_naive.md`
- `docs/models/nhits.md`
- `docs/models/patchtst.md`
- `docs/models/tft.md`
- `docs/models/xlstm.md`
- `docs/models/mamba.md`

## Testing

```bash
cd forecast_benchmark
pytest
```
