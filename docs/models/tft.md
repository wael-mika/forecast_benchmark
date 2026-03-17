# Temporal Fusion Transformer

## Status

Planned model, not implemented yet.

## Model

Temporal Fusion Transformer combines recurrent sequence encoding, attention, gating, and explicit support for static, known-future, and observed-past covariates.

## Data

Expected input groups:
- static features per station
- observed past target values
- known calendar covariates
- forecast horizon labels

## Features

The most natural first TFT feature set in this repo would be:
- past discharge target
- calendar features known in advance
- station metadata as static covariates

Without weather forcings, the first version would still be a target-plus-calendar model.

## Training

Expected setup:
- supervised window batches
- validation loss monitoring
- checkpointing by epoch
- probabilistic loss only if the repo later decides to model quantiles

## Splitting

Same chronological split rule as the other models, applied before window extraction.

## Testing And Evaluation

Should use the same benchmark metrics as the existing XGBoost baseline, with optional probabilistic metrics added later if quantile outputs are enabled.

## Current Notes

TFT becomes more valuable when richer covariates are available, so this model may make more sense after weather or snow inputs are added.
