# Model Documentation — Forecast Benchmark

This document describes every model in the benchmark at a high level: its architecture,
how it is adapted for multi-step discharge forecasting, and where it departs from the
original literature.

All models share three common design decisions that are described once here and not
repeated per model:

1. **Residual / correction framing.** Every neural model outputs a *correction* that is
   added to a persistence baseline (the last observed log-discharge).  The formula is
   `ŷ = baseline + correction`.  This is implemented via `baseline_strategy: zero` in
   the config — the model is trained to predict the correction from zero, while the
   baseline is handled externally.

2. **Direct multi-step forecasting.** Unless noted otherwise, all models predict horizons
   h+1, h+2, h+3 in a single forward pass (not auto-regressively).  This avoids error
   accumulation but treats each horizon as an independent regression head.

3. **Station-wise log1p + z-score normalisation.** Input discharge is
   `log1p`-transformed and z-score normalised (fit on training split only) before being
   fed to any neural model.  Predictions are inverse-transformed for evaluation.

---

## 1. Seasonal Naive Baseline

**Purpose.** A zero-parameter reference: no model is trained.

**How it works.** For horizon *h*, the prediction is the discharge observed
`season_length − h` days ago (default `season_length = 7`).  If that lag is unavailable
the last observed value (persistence) is used instead.

**Deviations from literature.** This is the standard definition from Hyndman & Athanasopoulos
(*Forecasting: Principles and Practice*); no modifications.

---

## 2. XGBoost

**Purpose.** A strong tabular baseline using gradient-boosted decision trees.

**How it works.**
- One XGBoost model is trained per forecast horizon (direct multi-step).
- Input is a flat feature vector from one of the three feature frames
  (context / weather-plus / hydro-weather; see [data documentation](data_documentation.md)).
- Station identity is passed as an integer feature (`station_id_as_feature`).
- Key hyperparameters: `tree_method=hist`, `max_depth=6`, `learning_rate=0.05`,
  `subsample=0.8`, `colsample_bytree=0.8`, early stopping at 30–50 rounds.

**Deviations from literature.** Standard XGBoost (Chen & Guestrin 2016) — no
modifications.  The direct multi-output framing (separate model per horizon) follows
the convention established by Taieb & Atiya (2016) for multi-step forecasting with
tabular models.

---

## 3. ANN (Basic)

**File.** `src/models/neural.py` → `ResidualANNForecaster`

**Architecture.**
```
flat feature vector
  → station embedding (→ 8 dims)
  → MLP: [64, 64] with ReLU + Dropout(0.1)
  → linear head → [h1, h2, h3] corrections
```

**Deviations from literature.** Standard feedforward MLP — no sequence processing.
The network operates on a flat window of lag features, so there is no notion of
temporal order inside the network.

---

## 4. ANN (Advanced)

**File.** `src/models/advanced_neural.py` → `ResidualAdvancedANNForecaster`

**Architecture.**
```
flat feature vector (lags + static context + future weather)
  → input projection → hidden_dim (512)
  → 4× Residual MLP block: LayerNorm → Linear → GELU → Dropout → Linear + skip
  → linear head → [h1, h2, h3] corrections
```

**Key differences from Basic ANN.**
- 4 deep residual MLP blocks (hidden_dim = 512 vs 64) for richer representation.
- Handles future covariates (known weather for forecast days) by including them in
  the flat input.
- LayerNorm + GELU residual blocks (similar to a tiny Transformer FFN).

**Deviations from literature.** Still a plain MLP — the residual block design follows
He et al. (2016) skip-connection conventions but is not novel.

---

## 5. LSTM (Basic)

**File.** `src/models/neural.py` → `ResidualBidirectionalLSTMForecaster`

**Architecture.**
```
discharge history sequence (T=14, D features)
  → Bidirectional LSTM: hidden=32, 2 layers
  → concatenate forward + backward last hidden states
  → + static features + station embedding (8 dims)
  → dense head: 64 → [h1, h2, h3] corrections
```

**Deviations from literature.**
- **Bidirectional.** Standard LSTMs for forecasting are causal (unidirectional).
  Using a bidirectional LSTM on the *historical window* is valid (the full 14-day
  history is known at forecast time) but differs from the typical usage in the
  literature where the LSTM is applied causally.
- **Small capacity.** `hidden_size=32` is deliberately compact for a basic baseline.
- **Direct output.** The hidden state is projected directly to all horizons, unlike the
  common seq2seq literature formulation where a decoder LSTM generates one step at a time.

---

## 6. LSTM (Advanced)

**File.** `src/models/advanced_neural.py` → `ResidualAdvancedLSTMForecaster`

**Architecture.**
```
multivariate history sequence (T=14, C channels)
  → RevIN (per-sample, per-channel normalisation; affine parameters learned)
  → Temporal Conv block: depthwise conv (kernel=5) + GLU gating
  → Bidirectional LSTM: model_dim=128, hidden=128, 2 layers
  → Attention pooling over sequence (context-weighted summary)
  → Static encoder: GRN(static + station embedding)
  → Future encoder: ResidualFFN(flattened future weather)
  → 2-layer dense head → [h1, h2, h3] corrections
```

**Key differences from Basic LSTM.**
- **RevIN.** Reversible Instance Normalisation (Kim et al. 2022) normalises each sample
  independently, making the LSTM invariant to station-level distributional shifts.
- **Temporal convolution pre-filter.** A depthwise conv with GLU gating smooths the
  multivariate input before the LSTM, providing a local-feature extraction stage.
- **Attention pooling.** Instead of using only the last hidden state, a learned
  attention weight aggregates all timesteps.
- **Multi-source fusion.** Static catchment context and future weather are explicitly
  encoded and fused before the head, matching the spirit of TFT-style conditioning.

**Deviations from literature.** There is no single "Advanced LSTM" paper; this
architecture synthesises components from RevIN (Kim et al. 2022), TCN (Bai et al. 2018),
and multi-source feature fusion.

---

## 7. N-HiTS (Basic)

**File.** `src/models/neural.py` → `ResidualNHiTSForecaster`

**Architecture.**
```
discharge history (T=14)
  → 3 hierarchical blocks (pool kernels: 1, 2, 4):
      each block:
        AvgPool1d(kernel) on residual history
        backcast head → subtract from residual (decomposes history)
        forecast head → add to cumulative forecast
  → cumulative forecast = [h1, h2, h3] corrections
```

**Deviations from literature (Challu et al. 2023).**
- **3 blocks** with kernels `[1, 2, 4]`.  The paper uses more blocks and a larger
  architecture (hundreds of hidden units per block, additional stacks).
- **No basis expansion.** The original N-HiTS uses Fourier or polynomial basis
  functions in its forecast heads.  Here, the forecast heads are plain linear
  projections.
- **Compact hidden dims** `[256, 256]` vs the paper's larger configurations.
- **Residual correction framing** vs the paper's direct point-forecast framing.

---

## 8. N-HiTS (Advanced)

**File.** `src/models/advanced_neural.py` → `ResidualAdvancedNHiTSForecaster`

**Architecture.**
```
multivariate history (T=14, C channels)
  → channel 0 (discharge) separated from exogenous features
  → exogenous features flattened → condition_dim (256) projection
  → static encoder: GRN; future encoder: ResidualFFN
  → 4 hierarchical blocks (MaxPool kernels: 1, 2, 4, 8):
      each: conditional backcast on discharge residual + forecast
  → correction head → [h1, h2, h3]
```

**Key differences from Basic N-HiTS.**
- **4 blocks** with an additional scale (kernel=8) capturing a longer lookback.
- **MaxPool** instead of AvgPool — preserves peak signals (flood peaks) rather than
  smoothing them.
- **Exogenous conditioning** — each block is conditioned on the projected weather,
  static, and future feature context, which is absent from the original paper.

---

## 9. PatchTST (Basic)

**File.** `src/models/neural.py` → `ResidualPatchTSTForecaster`

**Architecture.**
```
discharge history (T=14, 1 channel)
  → patchify: patch_len=4, stride=2 → N patches
  → linear patch projection → model_dim (64)
  → Transformer encoder: 2 layers, 4 heads, FF×4
  → mean pool all patch tokens → sequence summary
  → + static + baseline
  → linear head → [h1, h2, h3] corrections
```

**Deviations from literature (Nie et al. 2023).**
- **Univariate.** The original PatchTST is channel-independent but still handles
  multiple channels simultaneously via a shared encoder.  The basic version here
  treats the input as a single channel (discharge only).
- **No RevIN.** The original paper applies Reversible Instance Normalisation; the
  basic version does not.
- **Direct correction output** rather than full sequence reconstruction.

---

## 10. PatchTST (Advanced)

**File.** `src/models/advanced_neural.py` → `ResidualAdvancedPatchTSTForecaster`

**Architecture.**
```
multivariate history (T=14, C channels)
  → RevIN (per-channel)
  → each channel independently patchified (patch_len=4, stride=2)
  → shared Transformer encoder: model_dim=128, 8 heads, 3 layers
  → per-channel projection head → per-channel horizon forecast
  → learned soft channel mixing (softmax weights over channels)
  → + static encoder (GRN) + future encoder
  → correction head → [h1, h2, h3]
```

**Key differences from Basic PatchTST.**
- **True channel-independent processing** followed by **learned soft channel mixing** —
  closely follows the original paper's intent while adding a mixing step to capture
  inter-channel correlations absent in the original.
- **RevIN** restored.
- Larger model: `model_dim=128`, 8 heads, 3 layers.
- Explicit static and future covariate conditioning.

**Deviations from literature.** The channel mixing layer is an addition not present in
Nie et al. (2023).

---

## 11. TFT (Basic)

**File.** `src/models/neural.py` → `ResidualTemporalFusionTransformerForecaster`

**Architecture.**
```
static features + station embedding
  → GRN → static context vector
history sequence → linear → hidden_size
  → LSTM encoder: hidden=64, 1 layer (init from static context)
  → Multihead attention: query=static_ctx, keys/values=LSTM output
  → GRN fusion: [attention + static + baseline]
  → linear head → [h1, h2, h3] corrections
```

**Deviations from literature (Lim et al. 2021).**
- **No Variable Selection Networks (VSN).** The original TFT learns per-variable
  importance weights at every timestep; the basic version skips this.
- **Single LSTM layer** vs the paper's 2-layer encoder+decoder setup.
- **No future decoder.** The original TFT uses a separate LSTM decoder for future
  known inputs.  Here, future information is not explicitly processed.
- **Compact capacity:** `hidden_size=64`, 4 attention heads.

---

## 12. TFT (Advanced)

**File.** `src/models/advanced_neural.py` → `ResidualAdvancedTemporalFusionTransformerForecaster`

**Architecture.**
```
RevIN normalisation

Static context → GRN → 4 context vectors:
  enrichment, h_init, c_init, decoder_context

Past VSN (Variable Selection Network):
  per-timestep softmax over C input channels → selected past sequence

LSTM encoder (2 layers, hidden=128):
  initialised with h_init, c_init
  → GLU gate on output

Static enrichment: per-timestep GRN(LSTM output, enrichment_ctx)

Future VSN: per-horizon softmax over future channels

LSTM decoder (2 layers):
  processes future tokens, initialised from encoder final state
  → GLU gate

Multi-head attention (8 heads):
  decoder queries attend to enriched encoder sequence

Position-wise GRN on attention output

Output projection: per-horizon correction → [h1, h2, h3]
```

**Key differences from Basic TFT.** This version is the closest to the full Lim et al.
(2021) architecture in the benchmark.

**Deviations from literature.**
- **RevIN** added (not in the original paper) for per-sample distribution shift
  handling.
- The original TFT uses **quantile loss** for probabilistic forecasts; this version
  uses SmoothL1 (point forecast only).
- The original paper is evaluated on longer time series; here the encoder processes
  a 14-day window.

---

## 13. xLSTM (Basic)

**File.** `src/models/neural.py` → `ResidualXLSTMForecaster`

**Architecture.**
```
history sequence (T=14, D)
  → input projection → model_dim (64)
  → 3× Residual LSTM block:
      standard LSTM → linear → GLU gating + residual skip
  → last hidden state
  → + static + baseline
  → dense head: 128 → [h1, h2, h3]
```

**Deviations from literature (Beck et al. 2024).**
The basic version is a *significant simplification*: it replaces the matrix LSTM
(mLSTM) cell with a standard LSTM + GLU gate.  The key innovations of xLSTM —
exponential gating, matrix memory `C ∈ R^{d×d}`, log-space stabilisation — are
absent.  This is effectively a stacked LSTM with gated residual blocks.

---

## 14. xLSTM (Advanced)

**File.** `src/models/advanced_neural.py` → `ResidualAdvancedXLSTMForecaster`

**Architecture.**
```
RevIN normalisation
→ input projection → model_dim (128)
→ 4× mLSTM block (Beck et al. 2024):
    matrix memory  C_t ∈ R^{H × d_h × d_h}
    log-space forget gate with running stabiliser
    exponential input gate
    adaptive normaliser  n_t
    output gate (sigmoid)
    h_t = o_t * (C_t q_t) / max(|n_t^T q_t|, 1)
→ attention pooling
→ static encoder (GRN) + future encoder
→ dense head → [h1, h2, h3]
```

**Deviations from literature.**
- Faithful mLSTM implementation with log-space stabilisation for numerical safety.
- The paper also proposes an sLSTM (scalar LSTM) variant and mixes both in
  "xLSTM[7:1]" stacking — here only mLSTM blocks are used.
- Direct multi-step output vs the language-model auto-regressive usage in Beck et al.

---

## 15. Mamba (Basic)

**File.** `src/models/neural.py` → `ResidualMambaForecaster`

**Architecture.**
```
history sequence
  → input projection → model_dim (64)
  → 3× simplified SSM block:
      LayerNorm
      value projection + gate projection
      cumulative sum (state accumulation proxy)
      optional 1-D convolution (kernel=3) for temporal mixing
      gated output
  → last token as sequence summary
  → + static + baseline
  → dense head → [h1, h2, h3]
```

**Deviations from literature (Gu & Dao 2023).**
The basic Mamba uses a **cumulative sum** as a proxy for the true selective state-space
scan (`S6`).  This is a severe simplification: it loses input-dependent gating of the
state transitions (`Δ`, `B`, `C` projections) and the ZOH (zero-order hold)
discretisation.  It is essentially a linear recurrence with no selectivity.

---

## 16. Mamba (Advanced)

**File.** `src/models/advanced_neural.py` → `ResidualAdvancedMambaForecaster`

**Architecture.**
```
RevIN normalisation
→ input projection → model_dim (128)
→ 4× Mamba block (Gu & Dao 2023):
    LayerNorm
    split: x_branch | z_gate
    causal depthwise conv (kernel=4) on x_branch
    SiLU activation
    Selective SSM (_ssm):
      A: diagonal, log-space HiPPO-like init: log(1…state_dim)
      input-dependent B_t, C_t via low-rank projection
      input-dependent Δ_t via projection + softplus
      ZOH discretisation:
        Ā_t = exp(Δ_t ⊗ A)
        B̄_t = Δ_t ⊗ B_t
      recurrence: h_t = Ā_t * h_{t-1} + B̄_t * u_t
      output:    y_t = C_t · h_t + D * u_t
    SiLU output gate: y = y * silu(z)
    output projection
→ attention pooling
→ static encoder (GRN) + future encoder
→ dense head → [h1, h2, h3]
```

**Deviations from literature.**
- Faithful implementation of the `S6` selective scan from Gu & Dao (2023) including
  ZOH discretisation and HiPPO-inspired `A` initialisation.
- The original Mamba uses a highly optimised parallel CUDA scan kernel; this
  implementation runs a sequential Python loop over timesteps (correct but slower).
- Multi-step direct output vs the original language-model token-by-token generation.

---

## 17. Hybrid (Custom)

**File.** `src/models/advanced_neural.py` → `ResidualHydroHybridForecaster`

**Purpose.** A purpose-built model that combines convolutional local feature extraction,
bidirectional LSTM sequence encoding, and cross-attention between future weather tokens
and the encoded history — motivated by the complementary strengths of each component
for catchment hydrology.

**Architecture.**
```
RevIN normalisation

History encoder (parallel branches):
  Conv branch: 2× TemporalConvBlock (depthwise conv kernel=5 + GLU)
  LSTM branch: Bidirectional LSTM (hidden=96, 2 layers)
  → concatenate → history_memory (T × (2*conv_dim + 2*lstm_hidden))

Static encoder: GRN(static + station embedding)

Future encoder: embedding of horizon position + linear projection

Cross-attention (4 heads):
  queries  = future tokens (one per horizon)
  keys/values = history_memory
  → future_attended

Global pooling: attention-weighted history summary

Per-horizon decoder (for each of h1, h2, h3):
  GRN([future_token_h + future_attended_h + global + static]) → correction
```

**Not based on a single paper.** Components are drawn from:
- TCN gating — Bai et al. (2018)
- Bidirectional LSTM — Schuster & Paliwal (1997)
- Cross-attention — Vaswani et al. (2017)
- GRN — Lim et al. (2021 / TFT)

---

## 18. FlowNet (Custom)

**File.** `src/models/advanced_neural.py` → `ResidualHydroFlowNetForecaster`

**Purpose.** The most complex model in the benchmark.  Designed specifically for
river discharge forecasting, it integrates three inductive biases:
(i) multi-scale temporal structure of hydrological processes (flash runoff, soil
recharge, baseflow),
(ii) long-range memory propagation via Mamba selective state-space encoding,
(iii) smooth inter-horizon dependencies via a seq2seq LSTM decoder.

**Architecture.**
```
RevIN normalisation

Encoder:
  input projection → model_dim (96)
  Multi-scale conv fusion (_MultiScaleFusion):
    3 parallel TemporalConvBlocks with kernels [3, 7, 14]
    (capture flash-runoff / soil-recharge / baseflow timescales)
    learned softmax weights → weighted blend
  3× Mamba S6 blocks (state_dim=32):
    selective scan builds long-range memory representation
  LayerNorm
  Attention pooling → global_context (summary vector)

Context:
  Static encoder: GRN(static + station embedding)
  Future encoder:
    per-horizon embedding + linear projection of future weather
    + static bias injection → future_tokens (H × model_dim)

Decoder (Seq2Seq LSTM, 2 layers):
  LSTM hidden / cell states initialised:
    GRN(global_context) → h_0, c_0
  processes future_tokens one step per horizon
  → decoder_outputs (H × model_dim)
  Cross-attention (4 heads): decoder → encoder sequence
  GLU gate on attended output

Output head (per horizon):
  GRN([decoder_h + static + baseline]) → correction
→ [h1, h2, h3]
```

**Not based on a single paper.**  The design synthesises:
- Multi-scale convolution — inspired by Inception networks (Szegedy et al. 2015) and
  WaveNet (van den Oord et al. 2016)
- Mamba S6 encoder — Gu & Dao (2023)
- Seq2seq LSTM decoder — Sutskever et al. (2014)
- Cross-attention — Vaswani et al. (2017)
- GRN — Lim et al. (2021)

**Key distinction vs other models.** FlowNet is the only model that uses a
*recurrent decoder* (seq2seq), meaning each horizon is predicted sequentially with
the LSTM carrying hidden state from the previous horizon.  All other advanced models
predict all horizons in one shot.  This is intended to produce temporally coherent
forecast trajectories.

---

## Summary Table

| Model | Encoder type | Handles future covariates | Seq2seq decoder | RevIN | Custom |
|---|---|---|---|---|---|
| Seasonal Naive | — | — | — | — | — |
| XGBoost | Tabular trees | Yes (flat input) | No | No | — |
| ANN (Basic) | MLP | No | No | No | No |
| ANN (Advanced) | Residual MLP | Yes | No | No | No |
| LSTM (Basic) | BiLSTM | No | No | No | No |
| LSTM (Advanced) | TCN + BiLSTM | Yes | No | Yes | No |
| N-HiTS (Basic) | Hierarchical pool-MLP | No | No | No | No |
| N-HiTS (Advanced) | Hierarchical pool-MLP | Yes | No | No | No |
| PatchTST (Basic) | Patched Transformer | No | No | No | No |
| PatchTST (Advanced) | CI patched Transformer | Yes | No | Yes | No |
| TFT (Basic) | LSTM + attention | No | No | No | No |
| TFT (Advanced) | VSN + LSTM + attention | Yes | Yes (LSTM) | Yes | No |
| xLSTM (Basic) | LSTM + GLU | No | No | No | No |
| xLSTM (Advanced) | mLSTM (matrix memory) | Yes | No | Yes | No |
| Mamba (Basic) | CumSum SSM (approx.) | No | No | No | No |
| Mamba (Advanced) | S6 selective scan | Yes | No | Yes | No |
| Hybrid | Conv + BiLSTM + cross-attn | Yes | No | Yes | **Yes** |
| FlowNet | Multi-scale conv + Mamba | Yes | Yes (LSTM) | Yes | **Yes** |

---

## Loss Function

All advanced neural models share the same **trajectory loss**:

```
L = Σ_h  w_h · SmoothL1(ŷ_h, y_h)          (point loss, β=0.4)
  + λ_diff · MSE(Δŷ, Δy)                    (first-difference loss)
  + λ_curv · MSE(Δ²ŷ, Δ²y)                 (curvature loss)
```

| Component | Weight | Purpose |
|---|---|---|
| Point loss (per horizon) | `[1.0, 1.2, 1.45]` | Penalises magnitude errors; later horizons weighted higher |
| Diff loss | 0.45 – 0.50 | Penalises wrong step-to-step changes (rise/fall direction) |
| Curvature loss | 0.10 – 0.15 | Penalises physically implausible forecast trajectories |

Basic models use plain MSE or SmoothL1 without the diff/curvature terms.

---

## References

- Beck et al. (2024). xLSTM: Extended Long Short-Term Memory. *arXiv:2405.04517*
- Bai et al. (2018). An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling. *arXiv:1803.01271*
- Challu et al. (2023). N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting. *AAAI 2023*
- Gu & Dao (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces. *arXiv:2312.00752*
- Kim et al. (2022). Reversible Instance Normalization for Accurate Time-Series Forecasting. *ICLR 2022*
- Lim et al. (2021). Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting. *International Journal of Forecasting*
- Nie et al. (2023). A Time Series is Worth 64 Words: Long-term Forecasting with Transformers. *ICLR 2023* (PatchTST)
- Sutskever et al. (2014). Sequence to Sequence Learning with Neural Networks. *NeurIPS 2014*
- Szegedy et al. (2015). Going Deeper with Convolutions. *CVPR 2015*
- van den Oord et al. (2016). WaveNet: A Generative Model for Raw Audio. *arXiv:1609.03499*
- Vaswani et al. (2017). Attention Is All You Need. *NeurIPS 2017*
