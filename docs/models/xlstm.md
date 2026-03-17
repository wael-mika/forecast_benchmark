# xLSTM

## Status

Planned model, not implemented yet.

## Model

xLSTM is a modern recurrent architecture that extends classical LSTM ideas with larger-scale gating and memory design for long-sequence modeling.

## Data

Expected input:
- canonical daily panel
- fixed context windows
- direct or sequence forecast targets

## Features

The first xLSTM version should likely start with:
- past discharge sequence
- calendar covariates
- optional station metadata

## Training

Expected setup:
- batched sequence training
- checkpointing by epoch
- validation monitoring and early stopping

## Splitting

Use the same per-station chronological train, validation, and test split logic as the rest of the benchmark.

## Testing And Evaluation

Use the same deterministic evaluation bundle as XGBoost unless the architecture is extended to probabilistic outputs later.

## Current Notes

This model needs the same windowing and neural training infrastructure that N-HiTS and PatchTST need.
