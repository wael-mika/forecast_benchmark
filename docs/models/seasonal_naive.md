# Seasonal Naive

## Status

Planned baseline, not implemented yet.

## Model

Seasonal naive is the simplest benchmark: forecast the future value using the most recent value from the matching seasonal position.

For daily river discharge, two useful variants are likely:
- persistence: predict `y(t + h)` from `y(t)`
- annual seasonal naive: predict from the same day-of-year in the previous year when enough history exists

## Data

Input should be the canonical daily dataset:
- `unique_id`
- `ds`
- `y`

No additional static metadata is required.

## Features

This model does not learn feature weights. Its only inputs are the historical target values needed to copy the baseline forecast.

## Training

No parameter fitting is required beyond selecting the seasonal period and forecast rule.

## Splitting

Use the same per-station chronological train, validation, and test splits as the learned models.

## Testing And Evaluation

This baseline should be evaluated with the exact same metric bundle as the learned models so it remains comparable.

## Current Notes

This should be added before or alongside the first neural model, because it is the most important sanity-check baseline.
