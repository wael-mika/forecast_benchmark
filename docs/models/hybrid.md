# Hybrid

## Status

Implemented advanced model.

## Model

This architecture combines:
- temporal convolutions for short-range local response patterns
- a bidirectional recurrent encoder for hydrological memory
- cross-attention from future-known covariates into the encoded history
- a residual decoder head on top of a persistence baseline

## Data

The hybrid model uses the matched advanced feature frames:
- `data/processed/xgboost/features_context_w14_h3.parquet`
- `data/processed/xgboost/features_weather_plus_w14_h3.parquet`

## Features

Historical inputs:
- discharge history
- dense weather lag channels when available

Static inputs:
- window statistics
- lag deltas
- any non-sequential numeric context columns
- station embedding

Future-known inputs:
- future weather horizons when available
- future calendar features derived from target timestamps

## Training

Recommended path:
- `python scripts/run_experiment.py configs/hybrid_context.yaml`
- `python scripts/run_experiment.py configs/hybrid_weather.yaml`
- `python scripts/plot_neural_results.py configs/hybrid_context.yaml`
- `python scripts/plot_neural_results.py configs/hybrid_weather.yaml`

## Evaluation

The model writes the same artifact bundle as the other neural models:
- `predictions.parquet`
- `metrics_summary.csv`
- `metrics_by_station.csv`
- `loss_history.csv`
- `epoch_metrics.csv`
- plots under `artifacts/advanced/hybrid_*`
