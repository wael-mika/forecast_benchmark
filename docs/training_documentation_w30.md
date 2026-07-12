# Training Documentation — Forecast Benchmark (w30)

This document covers the full training pipeline for the **w30 benchmark**: how samples
are built from the feature frames, how models are normalised, trained, and evaluated,
and what artefacts are produced.  It applies to all neural models (basic and advanced).
XGBoost is noted where it differs.

> **Note.** The original w14 benchmark is documented in `training_documentation.md`.
> Key changes in this version: **30-day input window** for context and weather levels,
> **cosine LR schedule** as default for advanced models, **MSE trajectory loss** (was
> SmoothL1), **150 max epochs / 25 patience** for advanced models, and `run_train.py`
> as the unified launch script replacing the old suite runners.

---

## 1. Sample Construction

### Feature Frames

All neural models consume one of three pre-built Parquet files
(see [data documentation](data_documentation.md) §5d for details):

| Frame | Path | Discharge window |
|---|---|---|
| Context-only | `data/processed/xgboost/features_context_w30_h3.parquet` | 30 days |
| Weather-plus | `data/processed/xgboost/features_weather_plus_w30_h3.parquet` | 30 days |
| Hydro-weather | `data/processed/xgboost/features_hydro_weather_w30_h3.parquet` | 14 days¹ |

¹ The hydro-weather frame is built on top of the w14 weather base and retains a
14-day discharge window; ERA5 and ERA5-Land lag features still span 0–30 days.

Each row in a feature frame corresponds to one forecast origin — one station on one day.

| Group | Column pattern | Description |
|---|---|---|
| Autoregressive lags | `lag_1` … `lag_30` (or `lag_14` for hydro) | Past daily discharge values |
| Reanalysis lags | `{var}_lag_{k}` | ERA5 value k days prior (k = 0…30) |
| Reanalysis rolling | `{var}_roll_{w}` | Rolling mean over w ∈ {3, 7, 14, 21} days |
| Flow context | `flow_context_{id}_lag_{0,1}` | Discharge at 17 context stations, lags 0 and 1 |
| Targets | `target_h1`, `target_h2`, `target_h3` | Discharge at h+1, h+2, h+3 |
| Split label | `split` | `"train"` / `"validation"` / `"test"` |
| Station identity | `unique_id` | GRDC station string |

### Covariate Groups Passed to Models

Each sample is split into four tensors at dataset construction time:

| Tensor | Shape | Content |
|---|---|---|
| `sequence_features` | `(T, C)` | T=30 (or 14) timesteps, C channels — past discharge + reanalysis lags organised as a time series |
| `flat_features` | `(F,)` | Flattened rolling aggregations and any non-sequence static context |
| `context_features` | `(S,)` | Static features (station statistics computed from training split) |
| `future_features` | `(H, 0)` | H=3 horizons; future reanalysis is **disabled** (`include_future_reanalysis: false`) — horizon position embeddings are still generated inside the model |

The **baseline** tensor `(H,)` holds the persistence forecast in normalised space
(last observed log-discharge repeated across horizons).  Advanced configs use
`baseline_strategy: zero` (correction-only) or `baseline_strategy: persistence`
(Hybrid, FlowNet).

> **Why future reanalysis is disabled.** Shifting ERA5 by the forecast horizon
> would use ground-truth future weather as a model input, creating data leakage.
> Future ERA5 values are not available in real-time forecasting.

### Minimum Sequence Coverage

Reanalysis channels with fewer than 75 % non-NaN entries in the window are
dropped from the sequence tensor before training (`min_sequence_coverage: 0.75`).

### Split Assignment

Rows are assigned to `train / validation / test` via the `split` column created
during feature-frame preparation (70 % / 15 % / 15 %, chronological per station).
The dataset class filters rows by split label — no random re-splitting at training time.

---

## 2. Normalisation

### Method

All discharge values (lags and targets) are normalised with **per-station log1p
z-score** normalisation:

```
z = (log1p(x) − μ_station) / σ_station
```

### Fitting

Parameters `μ` and `σ` are estimated **on the training split only**:

```python
observed = concat(log1p(lag_values[train]), log1p(target_values[train]))
μ_station = mean(observed)
σ_station = std(observed)
# Guard against degenerate stations:
if σ_station < 1e-6:
    σ_station = 1.0
```

Both lag history and target values enter the fit together, so the normaliser
covers the full range of discharge seen at the station.

### Exogenous Features

Reanalysis features are standardised independently:
the per-column mean and std are computed from the training rows and applied to all
splits.  After standardisation, any remaining `NaN`, `+inf`, or `−inf` is replaced
with `0.0`.

### Inverse Transform

At evaluation time, model outputs (in normalised space) are converted back to
physical discharge (m³/s):

```
x̂ = expm1(z × σ_station + μ_station)
x̂ = clip(x̂, min=0)          # discharge is non-negative
```

### Stored Artefact

Normalisation parameters are saved per experiment to:

```
{artifact_dir}/scaler_by_station.csv
```

Columns: `unique_id`, `station_index`, `log1p_mean`, `log1p_std`.

---

## 3. DataLoader

### Batching

```python
DataLoader(train_dataset,      batch_size=batch_size,      shuffle=True,  num_workers=0)
DataLoader(validation_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=0)
DataLoader(test_dataset,       batch_size=eval_batch_size, shuffle=False, num_workers=0)
```

| Config tier | `batch_size` | `eval_batch_size` |
|---|---|---|
| Basic models | 4 096 | 8 192 |
| Advanced models | 512 | 1 024 |

Training batches are **shuffled**; validation and test batches preserve row order
(important for reproducible per-station predictions).

`num_workers=0` — data loading runs in the main process (avoids Windows multiprocessing
issues with PyTorch).

### NaN Handling at Batch Level

After the standardisation step, the arrays stored in the dataset are guaranteed to
contain no NaN or infinite values (`nan_to_num(..., nan=0.0, posinf=0.0, neginf=0.0)`).
No further masking is applied inside the model forward pass.

---

## 4. Optimiser

All neural models use **AdamW**:

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr           = config["learning_rate"],   # see table below
    weight_decay = config["weight_decay"],    # default 1e-4
    betas        = config.get("adam_betas", (0.9, 0.999)),
)
```

| Tier | Learning rate | Weight decay | β₁ | β₂ |
|---|---|---|---|---|
| Basic | 0.0005 – 0.0007 | 1e-4 | 0.9 | 0.999 |
| Advanced | 0.001 | 1e-4 | 0.9 | 0.999 |

---

## 5. Learning-Rate Schedule

Two schedulers are available, selected via `scheduler_name` in the config.

### CosineAnnealingLR (default for advanced models)

```python
scheduler = CosineAnnealingLR(
    optimizer,
    T_max   = config["max_epochs"],
    eta_min = config.get("min_learning_rate", 1e-5),
)
```

Smoothly decays the learning rate from `lr` to `eta_min` following a cosine curve
over the full training duration.  Advanced models default to cosine in the w30
benchmark (the w14 benchmark used plateau as default).

### ReduceLROnPlateau (available, default for basic models)

```python
scheduler = ReduceLROnPlateau(
    optimizer, mode="min",
    factor   = config.get("lr_decay_factor", 0.5),
    patience = config.get("lr_patience", 3),
)
```

Monitors **validation loss**.  When the loss does not improve for `patience` epochs
the learning rate is multiplied by `factor` (halved by default).

| Parameter | Basic | Advanced |
|---|---|---|
| `factor` | 0.5 | 0.5 |
| `patience` | 2 – 3 | 5 |
| `min_lr` | — | 1e-5 |

### Linear Warm-Up (advanced models only)

For the first `warmup_epochs` (**10** in the w30 benchmark, was 5) the learning rate
is linearly ramped from 0 to the base learning rate before the main scheduler takes over:

```python
if epoch <= warmup_epochs:
    for pg in optimizer.param_groups:
        pg["lr"] = base_lr * (epoch / warmup_epochs)
```

This stabilises early gradient steps when the model is randomly initialised, and is
especially helpful with the longer 150-epoch training runs.

---

## 6. Training Loop

```
for epoch in 1 … max_epochs:

    ① LR warm-up override (advanced, first warmup_epochs epochs)

    ② Training pass  (model.train())
        for batch in train_loader:
            forward → loss
            loss.backward()
            clip_grad_norm_(parameters, max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

    ③ Validation pass  (model.eval(), torch.no_grad())
        collect predictions on validation split
        inverse-transform → physical discharge
        compute micro/macro metrics

    ④ Scheduler step
        cosine:  scheduler.step()
        plateau: scheduler.step(val_loss)

    ⑤ Early stopping check
        metric monitored: val macro-RMSE (primary) or val macro-NSE
        if metric improved → save checkpoint, reset counter
        else              → increment counter
        if counter ≥ patience → break

    ⑥ Periodic checkpoint  (every checkpoint_interval epochs)
```

### Gradient Clipping

Applied before every optimiser step:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

This prevents exploding gradients in recurrent and attention layers and is active
for all neural models (`gradient_clip_norm: 1.0` in all configs).

### Early Stopping

| Parameter | Basic | Advanced |
|---|---|---|
| Monitored metric | Val macro-NSE | Val macro-RMSE |
| Patience | 5 – 6 epochs | 25 epochs |
| Max epochs | 20 – 25 | 150 |

> **Change from w14:** Advanced models now monitor **macro-RMSE** (minimise) rather
> than macro-NSE (maximise), and patience increased from 15 → 25 to allow models
> more time to converge with the larger 30-day input and cosine schedule.

---

## 7. Loss Functions

### Basic Models — MSE

```yaml
loss_name: mse
```

`L = mean((ŷ − y)²)`

### Advanced Models — Trajectory Loss

A three-component composite loss that penalises not just point errors but also
the shape of the 3-day forecast trajectory.

```yaml
loss_name:             trajectory
trajectory_point_loss: mse
loss_horizon_weights:  [1.0, 1.15, 1.35]   # or [1.0, 1.2, 1.5] for Hybrid/FlowNet
loss_diff_weight:      0.45                 # or 0.5 for Hybrid/FlowNet
loss_curvature_weight: 0.12                 # or 0.15 for Hybrid/FlowNet
```

> **Change from w14:** The point-loss component switched from `SmoothL1 (β=0.4)` to
> plain `MSE` for all advanced models.

#### Component 1 — Weighted Point Loss

```
L_point = (1/H) Σ_h  w_h · MSE(ŷ_h, y_h)
```

`w_h` are the horizon weights, normalised so their mean is 1.
Default weights `[1.0, 1.15, 1.35]` penalise later horizons more — this encourages
the model to extend accuracy beyond the next-day step.

#### Component 2 — First-Difference (Trajectory Slope) Loss

```
L_diff = λ_diff · L_point(Δŷ, Δy,  weights=w[1:])

where  Δŷ_h = ŷ_{h+1} − ŷ_h   (predicted step-to-step change)
       Δy_h = y_{h+1} − y_h    (true step-to-step change)
```

Penalises getting the **direction** of change wrong.

Default weight `λ_diff = 0.45` (0.50 for Hybrid/FlowNet).

#### Component 3 — Curvature (Second-Difference) Loss

```
L_curv = λ_curv · L_point(Δ²ŷ, Δ²y,  weights=w[2:])

where  Δ²ŷ_h = ŷ_{h+2} − 2ŷ_{h+1} + ŷ_h   (discrete second derivative)
```

Penalises sharp kinks or sudden reversals in the 3-day trajectory.

Default weight `λ_curv = 0.12` (0.15 for Hybrid/FlowNet).

#### Total Loss

```
L = L_point  +  λ_diff · L_diff  +  λ_curv · L_curv
```

All three components operate in **normalised log-discharge space** (z-scores),
so the scale is consistent across stations.

---

## 8. Checkpointing

The following files are written to `runs/{run_name}/{model}_{level}/` for every experiment:

| File | When written | Content |
|---|---|---|
| `model.pt` | Each time the monitored metric improves | Best model `state_dict` + optimiser state + epoch metadata |
| `model_epoch_{NNNN}.pt` | Same as above | Epoch-stamped copy of the best checkpoint (backup) |
| `checkpoints/model_epoch_{NNNN}.pt` | Every `checkpoint_interval` epochs | Periodic snapshot for resuming |
| `loss_history.csv` | End of training | Train / val / test loss per epoch |
| `epoch_metrics.csv` | End of training | Full micro/macro metric table per epoch |
| `config_snapshot.json` | Start of training | Full resolved config |
| `training_summary.json` | End of training | Row counts, paths, best epoch, final metrics |
| `scaler_by_station.csv` | Before training | Normalisation parameters |

The checkpoint payload stored in `.pt` files:

```python
{
    "epoch":                  int,
    "model_name":             str,
    "model_variant":          "basic" | "advanced",
    "model_state_dict":       dict,          # model.state_dict()
    "optimizer_state_dict":   dict,
    "validation_macro_nse":   float,
    "validation_micro_rmse":  float,
}
```

---

## 9. Inference and Prediction Collection

After training, the best checkpoint is loaded and a full inference pass is run over
all three splits (train, val, test):

```python
model.eval()
with torch.no_grad():
    for batch in loader:
        ŷ_normalised = model(batch)
ŷ_physical = normalizer.inverse_transform(ŷ_normalised, station_indices)
```

Predictions from all batches are concatenated and written to:

```
runs/{run_name}/{model}_{level}/predictions.parquet
```

Columns: `unique_id`, `ds`, `split`, `horizon`, `y` (true), `y_hat` (predicted),
plus `y_hat_raw` (normalised-space output before inverse transform).

---

## 10. Evaluation Metrics

All metrics are computed in **original discharge units (m³/s)** after inverse
transformation.

### Metric Definitions

| Metric | Formula |
|---|---|
| Bias | `mean(ŷ − y)` |
| MAE | `mean(|ŷ − y|)` |
| MSE | `mean((ŷ − y)²)` |
| RMSE | `sqrt(MSE)` |
| R² / NSE | `1 − Σ(ŷ−y)² / Σ(y−ȳ)²` |
| MAPE | `mean(|ŷ−y| / |y|)` for y ≠ 0 |
| SMAPE | `mean(2|ŷ−y| / (|ŷ|+|y|))` |
| WAPE | `Σ|ŷ−y| / Σ|y|` |
| MASE | `mean(|ŷ−y|) / mean(|Δy_train|)` |
| RMSSE | `sqrt(mean((ŷ−y)²) / mean(Δy_train²))` |

MASE and RMSSE denominators are computed from the **training split only** to prevent
data leakage into the scale reference.

### Aggregation Levels

Metrics are computed at three levels and stored in `metrics_summary.csv` and
`metrics_by_station.csv`:

| Level | How |
|---|---|
| **Micro** | All samples pooled across stations |
| **Macro** | Per-station metrics averaged across stations (equal weight per station) |
| **Per-station** | Individual station metrics (one row per station per horizon) |

Each level is broken down by **split** (train / val / test) and **horizon** (h1 / h2 / h3).

### Output Files

| File | Content |
|---|---|
| `metrics_summary.csv` | Micro + macro metrics for every split × horizon combination |
| `metrics_by_station.csv` | Full per-station × horizon breakdown |
| `predictions.parquet` | Raw predictions aligned with ground truth |

---

## 11. Experiment Configuration

Experiments are fully specified by YAML config files in `configs/`.  A representative
advanced model config (ANN, context level):

```yaml
# --- identity ---
model_name:     ann
model_variant:  advanced

# --- data ---
feature_frame_path:    data/processed/xgboost/features_context_w30_h3.parquet
artifact_dir:          artifacts/advanced_seq/ann_context_w30_h3
split_column:          split
min_sequence_coverage: 0.75

# --- hardware ---
device: auto     # "cuda" if GPU available, else "cpu"
seed:   42

# --- loss ---
loss_name:              trajectory
trajectory_point_loss:  mse
baseline_strategy:      zero
loss_horizon_weights:   [1.0, 1.15, 1.35]
loss_diff_weight:       0.45
loss_curvature_weight:  0.12

# --- training ---
batch_size:               512
eval_batch_size:          1024
max_epochs:               150
early_stopping_patience:  25
early_stopping_metric:    rmse
checkpoint_interval:      10
validation_eval_fraction: 1.0
test_eval_interval:       0

# --- optimiser ---
learning_rate:      0.001
min_learning_rate:  1.0e-05
weight_decay:       0.0001
scheduler_name:     cosine
warmup_epochs:      10
lr_decay_factor:    0.5
lr_patience:        5
gradient_clip_norm: 1.0

# --- architecture (model-specific) ---
embedding_dim:  16
hidden_dim:     640
num_blocks:     4
dropout:        0.1
```

### Launching Experiments

**Single experiment:**
```bash
.venv/Scripts/python scripts/run_experiment.py configs/ann_advanced_context.yaml
```

**Full benchmark suite (all models × all data levels):**
```bash
.venv/Scripts/python scripts/run_train.py
```

**Named subset — specific models and/or levels:**
```bash
.venv/Scripts/python scripts/run_train.py --run-name w30_v2 --models ann lstm --levels weather
```

**Resume a run, skipping already-completed artifacts:**
```bash
.venv/Scripts/python scripts/run_train.py --run-name full_suite
```

**Force retraining even if outputs already exist:**
```bash
.venv/Scripts/python scripts/run_train.py --run-name full_suite --force
```

`run_train.py` automatically prepares missing feature parquets, runs all configured
model/level pairs sequentially, writes logs to `logs/{run_name}/`, and calls plot
scripts after each model completes.

---

## 12. Quick-Reference Parameter Table

| Parameter | Basic | Advanced |
|---|---|---|
| Optimiser | AdamW | AdamW |
| β₁, β₂ | 0.9, 0.999 | 0.9, 0.999 |
| Weight decay | 1e-4 | 1e-4 |
| Learning rate | 5e-4 – 7e-4 | 1e-3 |
| Min LR | — | 1e-5 |
| LR scheduler | Plateau | **Cosine** (was Plateau in w14) |
| Plateau factor | 0.5 | 0.5 |
| Plateau patience | 2 – 3 | 5 |
| LR warmup | No | **10 epochs** (was 5 in w14) |
| Grad clip norm | 1.0 | 1.0 |
| Batch size (train) | 4 096 | 512 |
| Batch size (eval) | 8 192 | 1 024 |
| Max epochs | 20 – 25 | **150** (was 100 in w14) |
| Early stop patience | 5 – 6 | **25** (was 15 in w14) |
| Early stop metric | Val macro-NSE | **Val macro-RMSE** (was NSE in w14) |
| Loss | MSE | Trajectory: **MSE** + diff (0.45) + curv (0.12) |
| Horizon weights | — | [1.0, 1.15, 1.35] or [1.0, 1.2, 1.5] |
| Normalisation | log1p + z-score (per station, fit on train) | same |
| NaN fill (features) | 0.0 | 0.0 |
| Baseline strategy | Persistence | Zero or Persistence |
| Shuffle (train) | Yes | Yes |
| Shuffle (val/test) | No | No |
