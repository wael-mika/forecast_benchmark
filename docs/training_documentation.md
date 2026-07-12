# Training Documentation — Forecast Benchmark

This document covers the full training pipeline: how samples are built from the feature
frames, how models are normalised, trained, and evaluated, and what artefacts are
produced.  It applies to all neural models (basic and advanced).  XGBoost and the
seasonal naive baseline are noted where they differ.

---

## 1. Sample Construction

### Feature Frames

All neural models consume one of three pre-built Parquet files
(see [data documentation](data_documentation.md) §5d for details):

| Frame | Path |
|---|---|
| Context-only | `data/processed/xgboost/features_context_w14_h3.parquet` |
| Weather-plus | `data/processed/xgboost/features_weather_plus_w14_h3.parquet` |
| Hydro-weather | `data/processed/xgboost/features_hydro_weather_w14_h3.parquet` |

The suffix `w14_h3` encodes: **14-day history window**, **3-day forecast horizon**.

Each row in a feature frame corresponds to one forecast origin — one station on one day.
The columns encode:

| Group | Column pattern | Description |
|---|---|---|
| Autoregressive lags | `lag_1` … `lag_14` | Past 14 daily discharge values |
| Reanalysis lags | `{var}_lag_{k}` | Past k-day value of ERA5 variable |
| Reanalysis rolling | `{var}_roll{w}_{agg}` | Rolling sum/mean over w days |
| Targets | `target_h1`, `target_h2`, `target_h3` | Discharge at h+1, h+2, h+3 |
| Future covariates | `{var}_future_h{i}` | Reanalysis value known for forecast day i |
| Split label | `split` | `"train"` / `"val"` / `"test"` |
| Station identity | `unique_id` | GRDC station string |

### Covariate Groups Passed to Models

Each sample is split into four tensors at dataset construction time:

| Tensor | Shape | Content |
|---|---|---|
| `sequence_features` | `(T, C)` | T=14 timesteps, C channels — past discharge + reanalysis lags organised as a time series |
| `flat_features` | `(F,)` | Flattened rolling aggregations and any non-sequence static context |
| `context_features` | `(S,)` | Static features (station statistics computed from training split) |
| `future_features` | `(H, F_f)` | H=3 horizons, F_f future-known reanalysis features per horizon |

The **baseline** tensor `(H,)` holds the persistence forecast in normalised space
(last observed log-discharge repeated across horizons).
In advanced configs it is set to zero (`baseline_strategy: zero`) and the model
is trained to predict the correction directly.

### Minimum Sequence Coverage

Reanalysis channels with fewer than 75 % non-NaN entries in the 14-step window are
dropped from the sequence tensor before training (`min_sequence_coverage: 0.75`).
This prevents sparse channels from dominating the input.

### Split Assignment

Rows are assigned to `train / val / test` via the `split` column that was created
during feature-frame preparation (70 % / 15 % / 15 %, chronological per station).
The dataset class filters rows by split label — no random re-splitting is done at
training time.

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
| Advanced | 0.001 – 0.002 | 1e-4 | 0.9 | 0.999 |

---

## 5. Learning-Rate Schedule

Two schedulers are available, selected via `scheduler_name` in the config.

### ReduceLROnPlateau (default for basic models)

```python
scheduler = ReduceLROnPlateau(
    optimizer, mode="min",
    factor   = config.get("lr_decay_factor", 0.5),
    patience = config.get("lr_patience", 3),
)
```

The scheduler monitors **validation loss**.  When the loss does not improve for
`patience` epochs the learning rate is multiplied by `factor` (halved by default).

| Parameter | Basic | Advanced |
|---|---|---|
| `factor` | 0.5 | 0.5 |
| `patience` | 2 – 3 | 3 – 5 |
| `min_lr` | — | 1e-5 |

### CosineAnnealingLR (advanced models, optional)

```python
scheduler = CosineAnnealingLR(
    optimizer,
    T_max   = config["max_epochs"],
    eta_min = config.get("min_learning_rate", 1e-5),
)
```

Smoothly decays the learning rate from `lr` to `eta_min` following a cosine curve
over the full training duration.

### Linear Warm-Up (advanced models only)

For the first `warmup_epochs` (default **5**) the learning rate is linearly ramped
from 0 to the base learning rate before the main scheduler takes over:

```python
if epoch <= warmup_epochs:
    for pg in optimizer.param_groups:
        pg["lr"] = base_lr * (epoch / warmup_epochs)
```

This stabilises early gradient steps when the model is randomly initialised.

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
        plateau: scheduler.step(val_loss)
        cosine:  scheduler.step()

    ⑤ Early stopping check
        if val macro-NSE improved → save checkpoint, reset counter
        else                      → increment counter
        if counter ≥ patience     → break

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
| Monitored metric | Validation macro-NSE | Validation macro-NSE |
| Patience | 5 – 6 epochs | 15 epochs |
| Max epochs | 20 – 25 | 100 |

The monitored metric is the **macro-averaged NSE** (mean NSE across all stations) on
the validation split — the primary hydrological performance indicator.  If macro-NSE
is unavailable (e.g. all-NaN station), negative micro-RMSE is used as a fallback.

---

## 7. Loss Functions

### Basic Models — MSE / Smooth L1

```yaml
loss_name: mse        # or smooth_l1
smooth_l1_beta: 1.0   # only for smooth_l1
```

**MSE:**  `L = mean((ŷ − y)²)`

**Smooth L1 (Huber):**
```
L = mean(ℓ(ŷ − y))

ℓ(e) = 0.5 × e² / β        if |e| < β
      = |e| − 0.5 × β       otherwise
```

### Advanced Models — Trajectory Loss

A three-component composite loss that penalises not just point errors but also
the shape of the 3-day forecast trajectory.

```yaml
loss_name:             trajectory
trajectory_point_loss: smooth_l1
smooth_l1_beta:        0.4
loss_horizon_weights:  [1.0, 1.15, 1.35]
loss_diff_weight:      0.45
loss_curvature_weight: 0.12
```

#### Component 1 — Weighted Point Loss

```
L_point = (1/H) Σ_h  w_h · SmoothL1(ŷ_h, y_h)
```

`w_h` are the horizon weights, normalised so their mean is 1 to keep the loss
on the same scale regardless of the number of horizons.
Default weights `[1.0, 1.15, 1.35]` penalise later horizons more — this encourages
the model to extend accuracy beyond the next-day step.

#### Component 2 — First-Difference (Trajectory Slope) Loss

```
L_diff = λ_diff · L_point(Δŷ, Δy,  weights=w[1:])

where  Δŷ_h = ŷ_{h+1} − ŷ_h   (predicted step-to-step change)
       Δy_h = y_{h+1} − y_h    (true step-to-step change)
```

Penalises getting the **direction** of change wrong.  For example, if the
true discharge is rising but the model predicts a flat or falling trajectory,
this term adds a penalty even if the point values happen to be close.

Default weight `λ_diff = 0.45`.

#### Component 3 — Curvature (Second-Difference) Loss

```
L_curv = λ_curv · L_point(Δ²ŷ, Δ²y,  weights=w[2:])

where  Δ²ŷ_h = ŷ_{h+2} − 2ŷ_{h+1} + ŷ_h   (discrete second derivative)
```

Penalises forecasts that have the right level and direction but unrealistic
acceleration — sharp kinks or sudden reversals in the 3-day trajectory.

Default weight `λ_curv = 0.12`.

#### Total Loss

```
L = L_point  +  0.45 · L_diff  +  0.12 · L_curv
```

All three components operate in **normalised log-discharge space** (z-scores),
so the scale is consistent across stations.

---

## 8. Checkpointing

The following files are written to `{artifact_dir}/` for every experiment:

| File | When written | Content |
|---|---|---|
| `model.pt` | Each time val NSE improves | Best model `state_dict` + optimiser state + epoch metadata |
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
{artifact_dir}/predictions.parquet
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
advanced model config looks like:

```yaml
# --- identity ---
model_name:     ann
model_variant:  advanced         # "basic" omits this key

# --- data ---
feature_frame_path: data/processed/xgboost/features_hydro_weather_w14_h3.parquet
artifact_dir:       artifacts/advanced_seq/ann_hydro_weather_w14_h3
split_column:       split
min_sequence_coverage: 0.75      # drop channels with < 75 % non-NaN in window

# --- hardware ---
device: auto                     # "cuda" if GPU available, else "cpu"
seed:   42

# --- loss ---
loss_name:              trajectory
trajectory_point_loss:  smooth_l1
smooth_l1_beta:         0.4
baseline_strategy:      zero
loss_horizon_weights:   [1.0, 1.15, 1.35]
loss_diff_weight:       0.45
loss_curvature_weight:  0.12

# --- training ---
batch_size:               512
eval_batch_size:          1024
max_epochs:               100
early_stopping_patience:  15
checkpoint_interval:      10
validation_eval_fraction: 1.0    # fraction of val set used per epoch (1.0 = full)
test_eval_interval:       0      # 0 = evaluate test only at end

# --- optimiser ---
learning_rate:      0.001
min_learning_rate:  1.0e-05
weight_decay:       0.0001
scheduler_name:     plateau      # or "cosine"
lr_decay_factor:    0.5
lr_patience:        5
gradient_clip_norm: 1.0
warmup_epochs:      5

# --- architecture (model-specific) ---
embedding_dim:  16
hidden_dim:     512
num_blocks:     4
dropout:        0.1
```

### Launching Experiments

**Single experiment:**
```bash
python scripts/run_experiment.py configs/ann_hydro_weather.yaml
```

**Full advanced suite (all models, all feature frames):**
```bash
python scripts/run_advanced_suite.py
```

**Hydro suite (hydro-weather frame only):**
```bash
python scripts/run_hydro_suite.py
```

The suite scripts run experiments sequentially, write a `run_suite.log`, and call
the plotting and comparison scripts automatically after training completes.

---

## 12. Quick-Reference Parameter Table

| Parameter | Basic | Advanced |
|---|---|---|
| Optimiser | AdamW | AdamW |
| β₁, β₂ | 0.9, 0.999 | 0.9, 0.999 |
| Weight decay | 1e-4 | 1e-4 |
| Learning rate | 5e-4 – 7e-4 | 1e-3 – 2e-3 |
| Min LR | — | 1e-5 |
| LR scheduler | Plateau | Plateau or Cosine |
| Plateau factor | 0.5 | 0.5 |
| Plateau patience | 2 – 3 | 3 – 5 |
| LR warmup | No | 5 epochs (linear) |
| Grad clip norm | 1.0 | 1.0 |
| Batch size (train) | 4 096 | 512 |
| Batch size (eval) | 8 192 | 1 024 |
| Max epochs | 20 – 25 | 100 |
| Early stop patience | 5 – 6 | 15 |
| Early stop metric | Val macro-NSE | Val macro-NSE |
| Loss | MSE or Smooth L1 (β=1.0) | Trajectory: Smooth L1 (β=0.4) + diff (0.45) + curv (0.12) |
| Horizon weights | — | [1.0, 1.15, 1.35] |
| Normalisation | log1p + z-score (per station, fit on train) | same |
| NaN fill (features) | 0.0 | 0.0 |
| Baseline strategy | Persistence | Zero (correction-only) |
| Shuffle (train) | Yes | Yes |
| Shuffle (val/test) | No | No |
