# Daily River Discharge Forecasting Benchmark

A reproducible benchmark for 1-to-3-day-ahead daily river-discharge forecasting on
Slovak GRDC stations. It compares tabular and neural sequence models on one shared
set of feature frames, splits, and training settings so that stage (context vs.
weather vs. hydro) and architecture comparisons are controlled.

## Canonical schema

All processed data uses the long format:
- `unique_id` — station identifier
- `ds` — daily timestamp
- `y` — mean daily discharge (m3/s)

## Models

Eleven models, each trained at three input levels (`context`, `weather`,
`hydro_weather`) and two lookback windows (`w14`, `w30`):

`xgboost`, `ann`, `lstm` (unidirectional), `bilstm` (bidirectional), `nhits`,
`patchtst`, `tft`, `xlstm`, `mamba`, `hybrid`, `flownet`.

All neural models share one training block (persistence-residual baseline,
trajectory loss, cosine schedule with warmup, 150 epochs / patience 25, seed 42).
They differ only in architecture hyperparameters; XGBoost differs additionally in
its raw-unit target space.

## Feature frames

`scripts/prepare_features.py` builds six frames plus a split table from one master
index — 20 stations (station `6144500` dropped), forecast origins from 1984-01-01,
rows with all 30 discharge lags and the h1-h3 targets present — and one shared
per-station chronological 70/15/15 split:

```
data/processed/features/{context,weather,hydro}_{w14,w30}_h3.parquet
data/processed/features/split_boundaries.csv
```

Every frame carries the identical `(unique_id, forecast_origin_ds, split)` rows;
the frames differ only in their columns (window length and stage variables). All
rolling features are strictly causal (`shift(1)` then rolling over t-1..t-w). No
frame contains any known-future weather column.

## Quickstart (macOS / Apple Silicon)

```bash
cd forecast_benchmark

# 1. Environment (torch MPS wheel from PyPI)
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt torch==2.10.0
.venv/bin/pip install -e .
.venv/bin/python -c "import torch; print(torch.backends.mps.is_available())"   # True

# 2. Build the six feature frames (needs the processed parquets in data/processed/)
.venv/bin/python scripts/prepare_features.py

# 3. Tests
.venv/bin/python -m pytest

# 4. Smoke run (a few models, 3 epochs, context/w14)
.venv/bin/python scripts/run_train.py --run-name smoke --window w14 \
    --levels context --models xgboost ann lstm bilstm mamba patchtst xlstm \
    --max-epochs 3

# 5. Full suite for one window (all 11 models x 3 levels)
.venv/bin/python scripts/run_train.py --run-name final_w14 --window w14
.venv/bin/python scripts/run_train.py --run-name final_w30 --window w30
```

`run_train.py` resolves each frame path from `--window` and the data level, writes
artifacts to `runs/{run_name}/{model}_{level}_{window}/`, and skips already-complete
runs so suites are resumable.

## Data availability

- The processed parquets used by the pipeline are the authoritative distributed
  artifacts:
  - `data/processed/discharge_daily.parquet`
  - `data/processed/reanalysis_daily.parquet` (ERA5 weather)
  - `data/processed/reanalysis_hydro_daily.parquet` (ERA5 / ERA5-Land hydrology)
- Raw GRDC discharge exports are **not** redistributed (GRDC terms). Given the raw
  daily files under `data/raw/`, `scripts/prepare_data.py` rebuilds
  `discharge_daily.parquet` (parses the GRDC `;`-delimited daily format, converts
  `-999.0` to NaN, reindexes onto a gap-free daily calendar).
- The ERA5 / ERA5-Land reanalysis parquets are assembled from Open-Meteo caches by
  the `scripts/download_*` helpers.
- Built feature frames are git-ignored (rebuild with `scripts/prepare_features.py`);
  `split_boundaries.csv` is tracked for reference.
- Station `6158100` has no ERA5-Land soil coverage; its hydro columns are imputed
  with the per-column training-split mean of the other stations (disclosed).

## Repository layout

```text
forecast_benchmark/
  configs/          # {model}_{level}.yaml (33 configs) + data/reanalysis configs
  data/
    raw/            # GRDC exports (git-ignored, not redistributed)
    processed/      # canonical parquets + built features/
  docs/
  scripts/
    prepare_data.py       # raw GRDC -> discharge_daily.parquet
    prepare_features.py   # -> six unified feature frames + split_boundaries.csv
    run_train.py          # window-aware suite runner
    run_experiment.py     # single config end-to-end
  src/
    evaluation/     # metrics + evaluation pipeline
    models/         # advanced neural architectures
    training/       # training loops (advanced neural, xgboost)
    utils/
  tests/
```

## Metrics

Per split x horizon x station, then aggregated. Reported: bias, mae, mse, rmse,
r2, nse, mape, smape, wape, mase, rmsse. Macro (unweighted per-station mean) is the
primary aggregation; micro (pooled) is retained for reference.

## Testing

```bash
cd forecast_benchmark
.venv/bin/python -m pytest
```

`tests/test_frame_integrity.py` is the acceptance suite: lag/target alignment,
rolling causality, shared rows/split/boundaries across all six frames, absence of
future columns, ERA5 lag alignment, the Mamba parallel-vs-sequential scan, and the
PatchTST / (bi)LSTM forward shapes. `tests/test_data_assets.py` checks the shipped
parquets. Frame tests skip when the frames are not built.
