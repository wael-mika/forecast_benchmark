# PatchTST

## Status

Planned model, not implemented yet.

## Model

PatchTST is a Transformer-based time-series model that converts long sequences into smaller temporal patches before attention is applied.

## Data

Expected input:
- canonical daily panel
- fixed-length historical windows
- forecast horizon labels

## Features

A first PatchTST version here should likely use:
- past discharge sequence
- calendar features aligned to each timestep
- optional static station metadata through conditioning or concatenation

## Training

Expected setup:
- patch-based sequence batching
- validation-based checkpoint selection
- learning-rate scheduling
- saved checkpoints by epoch

## Splitting

Chronological per-station train, validation, and test splits should be created before sequence windows are extracted.

## Testing And Evaluation

Testing should mirror the XGBoost pipeline:
- one held-out test span
- the same metric bundle
- saved prediction tables and summary plots

## Current Notes

This model will need a stronger sequence data pipeline than the repository has today.
