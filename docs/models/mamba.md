# Mamba

## Status

Planned model, not implemented yet.

## Model

The benchmark plan includes one selective state-space style model from the Mamba family as a long-sequence baseline.

## Data

Expected input:
- canonical daily panel
- sequence windows with context length and horizon

## Features

A first repo version should likely use:
- past discharge sequence
- calendar covariates
- optional static station metadata

## Training

Expected setup:
- sequence minibatches
- validation checkpointing
- epoch-based learning curves

## Splitting

Chronological station-wise train, validation, and test partitioning should remain identical to the other models for fair comparison.

## Testing And Evaluation

Testing should produce:
- prediction tables
- split-level metric summaries
- per-station diagnostics
- the same benchmark metrics already used by XGBoost

## Current Notes

The exact Mamba variant should be chosen only after the sequence data pipeline is in place.
