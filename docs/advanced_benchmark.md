# Advanced Hydrology Benchmark

This note documents the advanced benchmark extension added after the first weather-aware baseline pass.

## Goals

The advanced pass focuses on two things:
- scale and optimize the existing model families without breaking the original baseline code path
- study the effect of weather variables with a fairer matched setup

## Matched Weather Ablation

The earlier comparison mixed two changes at once:
- the no-weather setup used a shorter context
- the weather setup used a longer context plus ERA5 variables

That makes the weather contribution hard to isolate.

The advanced suite fixes that by using two matched feature frames:
- `data/processed/xgboost/features_context_w14_h3.parquet`
- `data/processed/xgboost/features_weather_plus_w14_h3.parquet`

Both use:
- a 14-day discharge window plus `current_y`
- the same chronological 70/15/15 split
- the same flow-context features
- the same 3-step direct forecast horizon

Only the weather frame adds:
- ERA5-style reanalysis lag features out to 14 days
- 3, 7, and 14 day rolling weather summaries
- known-future weather inputs for horizons 1 to 3

## Matched Baseline Results

The first completed matched runs were:
- advanced XGBoost on the 14-day context frame
- advanced XGBoost on the matched weather frame
- advanced residual ANN on the 14-day context frame
- advanced residual ANN on the matched weather frame

Test micro metrics:

| Model | Regime | Avg RMSE | Avg MAE | Avg R2 |
| --- | --- | ---: | ---: | ---: |
| XGBoost | Context only | 118.400 | 25.556 | 0.9518 |
| XGBoost | + Weather | 90.273 | 19.117 | 0.9646 |
| ANN | Context only | 114.477 | 23.344 | 0.9546 |
| ANN | + Weather | 83.681 | 15.124 | 0.9691 |

Per-horizon test RMSE:

| Model | Regime | H1 | H2 | H3 |
| --- | --- | ---: | ---: | ---: |
| XGBoost | Context only | 76.115 | 124.271 | 154.815 |
| XGBoost | + Weather | 63.469 | 94.977 | 112.371 |
| ANN | Context only | 71.750 | 120.886 | 150.794 |
| ANN | + Weather | 55.159 | 88.184 | 107.700 |

Matched weather-ablation gains already verified:
- ANN: `+30.796` average RMSE gain and `+8.219` average MAE gain
- XGBoost: `+28.128` average RMSE gain and `+6.440` average MAE gain

These aggregate comparison artifacts are available under:
- `artifacts/advanced/model_comparison_context`
- `artifacts/advanced/model_comparison_weather`
- `artifacts/advanced/weather_ablation`

These numbers remain useful as a reference point, but the early advanced neural path still used a persistence-style residual target:
- the last observed discharge was repeated across all forecast horizons
- the network only learned a correction around that baseline

That residual shortcut helped some scores, but it also encouraged the lagging "copy the latest step forward" behavior visible in the forecast-window plots.

## Sequential Debiased Suite

To address the lagging effect, the advanced neural stack was updated to support direct-output training and then rerun sequentially with a separate config for every model and regime.

Main changes:
- `baseline_strategy: zero` removes the hard persistence residual shortcut
- `loss_name: trajectory` combines point loss, first-difference loss, and curvature loss to penalize delayed response shapes
- every model now has its own context and weather YAML config under `configs/`
- `scripts/run_advanced_suite.py` runs models sequentially and resumes cleanly if a run is interrupted
- completed outputs are written to dedicated folders under `artifacts/advanced_seq/`

All 16 neural runs finished in the sequential debiased suite:
- `ann`
- `lstm`
- `nhits`
- `patchtst`
- `tft`
- `xlstm`
- `mamba`
- `hybrid`

Final test micro RMSE leaderboard:

| Model | Context only | + Weather | Weather gain |
| --- | ---: | ---: | ---: |
| ANN | 119.673 | 86.939 | 32.734 |
| N-HiTS | 139.104 | 113.495 | 25.609 |
| TFT | 168.822 | 152.150 | 16.672 |
| LSTM | 173.622 | 157.195 | 16.427 |
| Mamba | 181.917 | 160.794 | 21.123 |
| PatchTST | 185.774 | 170.198 | 15.576 |
| Hybrid | 193.958 | 173.771 | 20.187 |
| xLSTM | 197.126 | 165.219 | 31.907 |

Key takeaways from the sequential rerun:
- weather improved every neural architecture in the matched setup
- ANN and N-HiTS remained the strongest neural models after removing the persistence shortcut
- several heavier sequence models scored worse than the earlier residual ANN runs, which suggests the old formulation was flattering them with a lag-prone baseline rather than learning truly anticipatory dynamics
- the new suite is a more honest benchmark even when the raw RMSE is worse

Each completed model artifact now includes:
- `metrics_summary.csv`
- `training_summary.json`
- `predictions.parquet`
- a `plots/` folder with loss curves, evaluation plots, and `test_forecast_windows.png` prediction-vs-target windows

Sequential-suite aggregate artifacts are available under:
- `artifacts/advanced_seq/model_comparison_context`
- `artifacts/advanced_seq/model_comparison_weather`
- `artifacts/advanced_seq/weather_ablation`

## Advanced Model Path

The original models remain unchanged in:
- `src/models/neural.py`
- `src/training/neural.py`

The scaled path lives in:
- `src/models/advanced_neural.py`
- `src/training/advanced_neural.py`

The experiment runner switches to the advanced trainer when:
- `model_variant: advanced`
- or `model_name: hybrid`

## What Changed

Shared data upgrades:
- multivariate temporal tensors instead of target-only sequences
- future-known covariates from weather forecasts already present in the feature frame
- future calendar covariates derived from the horizon timestamps
- station-wise target normalization for discharge
- standardized static, historical exogenous, and future exogenous channels

Shared optimization upgrades:
- AdamW
- plateau learning-rate schedule
- gradient clipping
- early stopping on validation RMSE
- larger hidden sizes and deeper residual heads
- direct-output forecasting via configurable baseline handling
- trajectory-aware losses that penalize delayed step changes

Plotting upgrades:
- the existing loss, metric, scatter, residual, and station plots remain
- a new `test_forecast_windows.png` plot shows horizon-wise prediction paths against the target path for sampled forecast origins

## Model Notes

ANN:
- now uses a deeper residual MLP over flattened history, static covariates, and future-known covariates

LSTM:
- now uses multivariate history
- adds a local temporal convolution front-end before the bidirectional recurrent encoder
- uses attention pooling conditioned on static and future context

N-HiTS:
- keeps the interpolation-style backcast and forecast structure
- conditions blocks on static features, future-known features, and compressed exogenous history

PatchTST:
- follows the official PatchTST design direction more closely by using channel-independent patching and RevIN-style normalization

TFT:
- uses historical encoding plus future-conditioned attention instead of a single static query
- explicitly consumes historical and future exogenous groups

xLSTM:
- uses a lightweight xLSTM-inspired residual stack with convolutional memory mixing
- this is an approximation, not a drop-in reproduction of the full official kernels

Mamba:
- replaces the earlier cumulative-average proxy with a selective state-space-inspired block
- this is also a CPU-friendly approximation rather than the exact CUDA-focused reference implementation

Hybrid:
- new architecture combining temporal convolutions, a bidirectional recurrent encoder, cross-attention over future covariates, and a residual decoder head
- intended to capture short-range runoff responses, longer hydrological memory, and horizon-specific weather forcing in one model

## Reproducible Commands

Prepare the matched feature frames:

```bash
python scripts/prepare_xgboost_data.py configs/advanced_data_context.yaml
python scripts/prepare_xgboost_data.py configs/advanced_data_weather.yaml
```

Run the full suite:

```bash
python scripts/run_advanced_suite.py
```

The suite is resumable:
- completed training runs are skipped automatically
- completed plot bundles are skipped automatically
- use `python scripts/run_advanced_suite.py --force` to regenerate everything

Run a subset:

```bash
python scripts/run_advanced_suite.py ann lstm nhits patchtst tft hybrid
```

Generate comparison artifacts after training:

```bash
python scripts/compare_advanced_results.py
python scripts/compare_weather_ablations.py
```

## Reference Implementations And Papers

These guided the advanced implementation choices:
- PatchTST official repository: <https://github.com/yuqinie98/PatchTST>
- N-HiTS paper: <https://arxiv.org/abs/2201.12886>
- Temporal Fusion Transformer paper: <https://arxiv.org/abs/1912.09363>
- RevIN paper: <https://openreview.net/forum?id=cGDAkQo1C0p>
- NeuralHydrology project: <https://github.com/neuralhydrology/neuralhydrology>
- xLSTM repository: <https://github.com/NX-AI/xlstm>
- Mamba repository: <https://github.com/state-spaces/mamba>

## Practical Limitation

The advanced xLSTM and Mamba variants are deliberately lightweight.

Reason:
- the official high-performance implementations are optimized around kernels and environments that are not a clean fit for this CPU-only macOS benchmark workspace

So the benchmark now follows the architectural ideas of those families while staying runnable inside the current repository and environment.
