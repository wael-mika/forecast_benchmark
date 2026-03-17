# N-HiTS

## Status

Planned model, not implemented yet.

## Model

N-HiTS is a hierarchical interpolation-based neural forecasting model designed for long-horizon univariate and multivariate forecasting.

## Data

Expected base input:
- canonical daily panel with `unique_id`, `ds`, `y`

Likely training representation:
- sliding windows with context length and forecast horizon
- one tensorized batch per station window

## Features

The first N-HiTS version in this repo should probably use:
- past target history
- calendar covariates
- optional static station metadata

External meteorological covariates are not available in the repo yet, so the first version should stay target-driven.

## Training

Expected setup:
- mini-batch training over supervised windows
- optimizer such as Adam
- early stopping on validation loss
- checkpoint saving by epoch

## Splitting

Use the same chronological split logic as the XGBoost baseline, but apply it before window generation to avoid leakage.

## Testing And Evaluation

Evaluate on the same held-out test span and report the same metric bundle as XGBoost.

## Current Notes

Before implementation, the repo needs a real window builder and a sequence training loop.
