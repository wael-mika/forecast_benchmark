"""Full-scale neural forecasting architectures for Slovak river discharge forecasting.

This module contains the production-quality, scaled-up versions of every model
in the benchmark. All models consume a 30-day multivariate history (discharge +
optional ERA5 weather / hydrological covariates), optional future covariates
(known-ahead weather), and a persistence baseline, and output per-horizon
residual corrections.

All sequence models apply RevIN (Reversible Instance Normalization) before the
encoder so predictions are scale-invariant across stations with very different
discharge regimes (e.g., Váh at Trenčín vs. small tributaries).

Models
------
ResidualAdvancedANNForecaster
    Deep residual MLP over flattened history + static features. No recurrence.
ResidualAdvancedLSTMForecaster
    Bidirectional LSTM augmented with a depthwise temporal conv pre-processing
    block and context-conditioned attention pooling.
ResidualAdvancedNHiTSForecaster
    Larger N-HiTS (Challu et al. 2023) with MaxPool blocks, condition vectors
    built from static + future + exogenous history projections.
ResidualAdvancedPatchTSTForecaster
    Channel-independent PatchTST (Nie et al. 2023): each input channel runs
    through a shared Transformer; the forecast is the target channel's head
    output (paper-faithful, no channel mixing).
ResidualAdvancedTemporalFusionTransformerForecaster
    Full TFT (Lim et al. 2021): Variable Selection Networks for past and future,
    static-context-initialized LSTM encoder/decoder, multi-head interpretable
    attention, and GLU-gated skip connections throughout.
ResidualAdvancedXLSTMForecaster
    Matrix LSTM (mLSTM) blocks from xLSTM (Beck et al. 2024) with expand_factor=2:
    head_dim=64, causal-conv q/k, stabilized log-space gates, per-head GroupNorm,
    attention-pooled context fusion.
ResidualAdvancedMambaForecaster
    Faithful Mamba selective SSM (Gu & Dao 2023) with ZOH discretization,
    input-dependent B/C/Δ, and an SE-Net channel gate to suppress irrelevant
    ERA5 channels before the input projection.
ResidualHydroHybridForecaster
    Custom hybrid: depthwise temporal conv blocks → bidirectional LSTM → cross-
    attention between future tokens and historical memory → GRN decoder head.
ResidualHydroFlowNetForecaster
    FlowNet: multi-scale parallel conv (3/7/14/21-day kernels) → stacked Mamba
    SSM → seq2seq LSTM decoder initialized from encoder state → cross-attention
    → GRN output. Designed specifically for daily Slovak streamflow.

Shared building blocks (private, prefixed with _)
--------------------------------------------------
ReversibleInstanceNorm      Per-sample per-channel normalization + denorm (Kim et al. 2022)
_GatedResidualNetwork       GRN from TFT paper (ELU + GLU + skip + LayerNorm)
_VariableSelectionNetwork   VSN from TFT paper (per-variable GRN + softmax selection)
_MambaBlock                 Selective SSM block with ZOH discretization (Gu & Dao 2023)
_mLSTMBlock                 Matrix LSTM block from xLSTM (Beck et al. 2024)
_TemporalConvBlock          Gated depthwise conv + pointwise FFN with residual
_MultiScaleFusion           Parallel conv branches at different kernel sizes, softmax blend
_AttentionPooling           Context-conditioned soft attention pooling over a sequence
_StaticEncoder              Projects static + station embedding → fixed-size context vector
_FutureFeatureEncoder       Horizon embeddings + optional future covariate projection
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn


def _build_feed_forward(
    input_dim: int,
    hidden_dims: Iterable[int],
    output_dim: int,
    *,
    dropout: float,
) -> nn.Sequential:
    """Build a feed-forward stack: Linear → LayerNorm → GELU → Dropout, then a final Linear.

    Preferred over ``_build_mlp`` in ``neural.py`` for the advanced models because
    LayerNorm + GELU trains more stably at larger widths.
    """
    layers: list[nn.Module] = []
    current_dim = int(input_dim)
    for hidden_dim in [int(value) for value in hidden_dims]:
        layers.extend(
            [
                nn.Linear(current_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
        )
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, int(output_dim)))
    return nn.Sequential(*layers)


class _ResidualMLPBlock(nn.Module):
    """Pre-norm residual MLP block: LayerNorm → Linear → GELU → Dropout → Linear → Dropout, then residual add."""

    def __init__(self, input_dim: int, hidden_dim: int, *, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.feed_forward(self.norm(x))


class _ResidualFeedForward(nn.Module):
    """Stacked residual MLP: projects input to ``hidden_dim``, passes through N ``_ResidualMLPBlock``s,
    then projects to ``output_dim`` via LayerNorm + Linear.

    Used as a general-purpose encoder for flat feature vectors (e.g., future covariates,
    exogenous history). Deeper than a plain MLP with better gradient flow.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        *,
        blocks: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [_ResidualMLPBlock(hidden_dim, hidden_dim * 2, dropout=dropout) for _ in range(int(blocks))]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.input_projection(x)
        for block in self.blocks:
            hidden = block(hidden)
        return self.output_projection(self.output_norm(hidden))


class ReversibleInstanceNorm(nn.Module):
    """RevIN: per-sample per-channel input normalization (Kim et al. 2022).

    "Reversible Instance Normalization for Accurate Time-Series Forecasting
    against Distribution Shift".

    In this benchmark RevIN normalizes the encoder inputs only. Predictions are
    denormalized by the per-station :class:`StationNormalizer` fitted in training,
    not by this module, so no reverse pass is provided here.

    forward() normalizes the input sequence per sample and per channel and applies
    the optional affine transform.
    """

    def __init__(self, channel_count: int, *, affine: bool = True, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = float(eps)
        self.affine = bool(affine)
        if self.affine:
            self.weight = nn.Parameter(torch.ones(1, 1, channel_count))
            self.bias = nn.Parameter(torch.zeros(1, 1, channel_count))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time, channels)
        if x.size(-1) == 0:
            return x
        mean = x.mean(dim=1, keepdim=True)                            # (B, 1, C)
        std = x.std(dim=1, keepdim=True, unbiased=False).clamp_min(self.eps)  # (B, 1, C)
        normalized = (x - mean) / std
        if self.affine:
            normalized = normalized * self.weight + self.bias
        return normalized


class _AttentionPooling(nn.Module):
    """Context-conditioned soft attention pooling over a variable-length sequence.

    Computes a query from the context vector, scores each timestep with a tanh dot-product,
    applies softmax, and returns the weighted sum. Effectively learns "which timestep matters
    most given the current static/future context".
    """

    def __init__(self, input_dim: int, context_dim: int) -> None:
        super().__init__()
        self.query = nn.Linear(context_dim, input_dim)
        self.score = nn.Linear(input_dim, 1)

    def forward(self, sequence: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        query = self.query(context).unsqueeze(1)
        scores = self.score(torch.tanh(sequence + query)).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        return torch.sum(sequence * weights.unsqueeze(-1), dim=1)


class _FutureFeatureEncoder(nn.Module):
    """Encodes known-ahead future covariates into per-horizon tokens.

    Adds a learned horizon index embedding to distinguish h=1 from h=2 from h=3.
    If ``input_dim > 0``, projects the covariate values and adds them to the embedding.
    If there are no future features (input_dim=0), returns just the horizon embedding.
    """

    def __init__(self, input_dim: int, horizon_count: int, model_dim: int, dropout: float) -> None:
        super().__init__()
        self.horizon_embedding = nn.Embedding(horizon_count, model_dim)
        self.projection = nn.Linear(input_dim, model_dim) if input_dim > 0 else None
        self.dropout = nn.Dropout(dropout)

    def forward(self, future_features: torch.Tensor) -> torch.Tensor:
        batch_size, horizon_count, _feature_dim = future_features.shape
        horizon_index = torch.arange(horizon_count, device=future_features.device)
        embedded = self.horizon_embedding(horizon_index).unsqueeze(0).expand(batch_size, -1, -1)
        if self.projection is None or future_features.size(-1) == 0:
            return embedded
        return embedded + self.dropout(self.projection(future_features))


class _StaticEncoder(nn.Module):
    """Encodes the concatenation of static features + station embedding into a fixed-size context vector.

    If ``input_dim == 0`` (context-only level with no static features), the station embedding
    is used directly without projection. Output is LayerNorm'd + dropped out.
    """

    def __init__(self, input_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim) if input_dim > 0 else None
        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, static_features: torch.Tensor, station_embedding: torch.Tensor) -> torch.Tensor:
        if static_features.size(1) == 0:
            combined = station_embedding
        else:
            combined = torch.cat([static_features, station_embedding], dim=1)
        if self.projection is not None:
            combined = self.projection(combined)
        return self.dropout(self.norm(combined))


def _make_same_padding_conv(model_dim: int, kernel_size: int) -> nn.Conv1d:
    """Create a depthwise Conv1d with same-length padding (output length == input length)."""
    return nn.Conv1d(
        model_dim,
        model_dim,
        kernel_size=max(1, int(kernel_size)),
        padding=max(1, int(kernel_size)) // 2,
        groups=model_dim,
    )


class _TemporalConvBlock(nn.Module):
    """Gated depthwise temporal conv block with a pointwise feed-forward and residual.

    Flow: pointwise expand (×2) → depthwise conv → GLU gating → pointwise project back
    → residual add → pointwise FFN with GELU → residual add.

    The GLU gate (value * sigmoid(gate)) lets the network selectively suppress or pass
    each temporal feature. Same-length padding preserves sequence length throughout.
    """

    def __init__(self, model_dim: int, *, kernel_size: int, expansion: int, dropout: float) -> None:
        super().__init__()
        inner_dim = model_dim * int(expansion)
        self.norm = nn.LayerNorm(model_dim)
        self.pointwise_in = nn.Linear(model_dim, inner_dim * 2)
        self.depthwise_conv = _make_same_padding_conv(inner_dim * 2, kernel_size)
        self.pointwise_out = nn.Linear(inner_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, model_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(x)
        projected = self.pointwise_in(normalized)
        convolved = self.depthwise_conv(projected.transpose(1, 2)).transpose(1, 2)
        convolved = convolved[:, : x.size(1), :]
        value, gate = convolved.chunk(2, dim=-1)
        x = x + self.dropout(self.pointwise_out(value * torch.sigmoid(gate)))
        x = x + self.feed_forward(x)
        return x


class ResidualAdvancedANNForecaster(nn.Module):
    """Deep residual MLP baseline — the non-sequential reference model.

    Input: flattened feature vector (all lag columns, window stats, etc.) concatenated
    with a station embedding and the persistence baseline scalar. No sequence dimension
    is used — the raw 30-day sequence is ignored.

    Stacks ``num_blocks`` of ``_ResidualMLPBlock`` at width ``hidden_dim`` for better
    gradient flow than a plain deep MLP. Despite its simplicity, this model is often
    competitive because the precomputed lag/stat features already capture most of the
    predictive signal.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        horizon_count: int,
        station_count: int,
        embedding_dim: int = 16,
        hidden_dim: int = 512,
        num_blocks: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.station_embedding = nn.Embedding(station_count, embedding_dim)
        self.network = _ResidualFeedForward(
            input_dim + embedding_dim + 1,
            hidden_dim,
            horizon_count,
            blocks=num_blocks,
            dropout=dropout,
        )

    def forward(
        self,
        sequence_features: torch.Tensor,
        flat_features: torch.Tensor,
        future_features: torch.Tensor,
        station_index: torch.Tensor,
        baseline: torch.Tensor,
    ) -> torch.Tensor:
        del sequence_features, future_features
        inputs = torch.cat([flat_features, self.station_embedding(station_index), baseline[:, :1]], dim=1)
        correction = self.network(inputs)
        return baseline + correction


class ResidualAdvancedLSTMForecaster(nn.Module):
    """LSTM encoder with temporal conv pre-processing and attention-pooled context fusion.

    Runs in either a unidirectional (``bidirectional=False``, the "lstm" benchmark
    model) or bidirectional (``bidirectional=True``, the "bilstm" model) mode; both
    use the same class. Because the encoder reads a fully-past lookback window,
    bidirectionality is not a causality violation — it is a richer encoder.

    Pipeline:
      1. RevIN normalization over all input channels.
      2. Linear projection → ``model_dim``.
      3. One ``_TemporalConvBlock`` with a depthwise kernel smooths local patterns.
      4. (Bi)LSTM encodes the full lookback sequence.
      5. Context-conditioned attention pooling collapses the sequence to a vector.
      6. Static encoder + future encoder produce context vectors.
      7. Pooled + last-hidden + static + future + baseline → MLP correction head.
    """

    def __init__(
        self,
        *,
        sequence_input_dim: int,
        static_input_dim: int,
        future_input_dim: int,
        horizon_count: int,
        station_count: int,
        embedding_dim: int = 16,
        model_dim: int = 128,
        hidden_size: int = 128,
        num_layers: int = 2,
        kernel_size: int = 5,
        dropout: float = 0.15,
        head_hidden_dim: int = 256,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.bidirectional = bool(bidirectional)
        directions = 2 if self.bidirectional else 1
        self.station_embedding = nn.Embedding(station_count, embedding_dim)
        self.input_projection = nn.Linear(sequence_input_dim, model_dim)
        self.input_norm = ReversibleInstanceNorm(sequence_input_dim)
        self.conv_block = _TemporalConvBlock(model_dim, kernel_size=kernel_size, expansion=2, dropout=dropout)
        self.encoder = nn.LSTM(
            input_size=model_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=self.bidirectional,
        )
        context_dim = head_hidden_dim
        self.static_encoder = _StaticEncoder(static_input_dim + embedding_dim, context_dim, dropout)
        self.future_encoder = _ResidualFeedForward(
            max(1, horizon_count * future_input_dim),
            head_hidden_dim,
            context_dim,
            blocks=2,
            dropout=dropout,
        )
        self.pooling = _AttentionPooling(hidden_size * directions, context_dim)
        self.head = _build_feed_forward(
            (hidden_size * directions * 2) + (context_dim * 2) + 1,
            [head_hidden_dim, head_hidden_dim],
            horizon_count,
            dropout=dropout,
        )

    def forward(
        self,
        sequence_features: torch.Tensor,
        static_features: torch.Tensor,
        future_features: torch.Tensor,
        station_index: torch.Tensor,
        baseline: torch.Tensor,
    ) -> torch.Tensor:
        normalized_inputs = self.input_norm(sequence_features)
        projected = self.input_projection(normalized_inputs)
        projected = self.conv_block(projected)
        encoded, (hidden_state, _cell_state) = self.encoder(projected)
        if self.bidirectional:
            # Last layer's forward and backward hidden states.
            sequence_summary = torch.cat([hidden_state[-2], hidden_state[-1]], dim=1)
        else:
            sequence_summary = hidden_state[-1]

        station_embedding = self.station_embedding(station_index)
        static_context = self.static_encoder(static_features, station_embedding)
        future_flat = future_features.flatten(start_dim=1)
        if future_flat.size(1) == 0:
            future_flat = torch.zeros(sequence_features.size(0), 1, device=sequence_features.device, dtype=sequence_features.dtype)
        future_context = self.future_encoder(future_flat)
        pooled = self.pooling(encoded, static_context + future_context)

        head_input = torch.cat(
            [
                pooled,
                sequence_summary,
                static_context,
                future_context,
                baseline[:, :1],
            ],
            dim=1,
        )
        correction = self.head(head_input)
        return baseline + correction


class _AdvancedNHiTSBlock(nn.Module):
    """One block of the advanced N-HiTS model.

    MaxPool-downsamples the target history to a coarser scale (controlled by ``pool_kernel``),
    concatenates with a ``condition_dim``-dimensional context vector (static + future + history),
    then produces a backcast (upsampled back to original length) and a horizon forecast.
    """

    def __init__(
        self,
        *,
        input_length: int,
        condition_dim: int,
        horizon_count: int,
        hidden_dims: Iterable[int],
        pool_kernel: int,
        dropout: float,
    ) -> None:
        super().__init__()
        kernel = max(1, int(pool_kernel))
        self.pool = nn.MaxPool1d(kernel_size=kernel, stride=kernel, ceil_mode=True)
        pooled_length = math.ceil(input_length / kernel)
        head_input_dim = pooled_length + condition_dim
        self.backcast_head = _build_feed_forward(head_input_dim, hidden_dims, pooled_length, dropout=dropout)
        self.forecast_head = _build_feed_forward(head_input_dim, hidden_dims, horizon_count, dropout=dropout)

    def forward(self, history: torch.Tensor, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = self.pool(history.unsqueeze(1)).squeeze(1)
        block_input = torch.cat([pooled, condition], dim=1)
        backcast = self.backcast_head(block_input)
        forecast = self.forecast_head(block_input)
        restored = F.interpolate(
            backcast.unsqueeze(1),
            size=history.size(1),
            mode="linear",
            align_corners=False,
        ).squeeze(1)
        return restored, forecast


class ResidualAdvancedNHiTSForecaster(nn.Module):
    """Scaled N-HiTS forecaster (Challu et al. 2023) conditioned on static, future, and history context.

    Improvements over the compact version in ``neural.py``:
    - MaxPool instead of AvgPool (sharper peak detection for flood events).
    - Separate condition vector from static encoder + future encoder + exogenous history projection,
      all concatenated before each block so every scale sees the same rich conditioning.
    - Larger hidden dims (default 512 × 512) and 4 pool scales [1, 2, 4, 8].
    - Final forecast = sum of all block forecasts + persistence baseline.
    """

    def __init__(
        self,
        *,
        sequence_length: int,
        sequence_input_dim: int,
        static_input_dim: int,
        future_input_dim: int,
        horizon_count: int,
        station_count: int,
        embedding_dim: int = 16,
        hidden_dims: Iterable[int] = (512, 512),
        pool_kernels: Iterable[int] = (1, 2, 4, 8),
        dropout: float = 0.1,
        condition_dim: int = 256,
    ) -> None:
        super().__init__()
        exog_dim = max(0, sequence_input_dim - 1)
        self.station_embedding = nn.Embedding(station_count, embedding_dim)
        self.history_projection = nn.Linear(sequence_length * exog_dim, condition_dim) if exog_dim > 0 else None
        self.static_encoder = _StaticEncoder(static_input_dim + embedding_dim, condition_dim, dropout)
        self.future_encoder = _ResidualFeedForward(
            max(1, horizon_count * future_input_dim),
            condition_dim,
            condition_dim,
            blocks=2,
            dropout=dropout,
        )
        block_condition_dim = condition_dim * 3 + 1
        self.blocks = nn.ModuleList(
            [
                _AdvancedNHiTSBlock(
                    input_length=sequence_length,
                    condition_dim=block_condition_dim,
                    horizon_count=horizon_count,
                    hidden_dims=hidden_dims,
                    pool_kernel=pool_kernel,
                    dropout=dropout,
                )
                for pool_kernel in pool_kernels
            ]
        )

    def forward(
        self,
        sequence_features: torch.Tensor,
        static_features: torch.Tensor,
        future_features: torch.Tensor,
        station_index: torch.Tensor,
        baseline: torch.Tensor,
    ) -> torch.Tensor:
        target_history = sequence_features[:, :, 0]
        station_embedding = self.station_embedding(station_index)
        static_context = self.static_encoder(static_features, station_embedding)

        future_flat = future_features.flatten(start_dim=1)
        if future_flat.size(1) == 0:
            future_flat = torch.zeros(sequence_features.size(0), 1, device=sequence_features.device, dtype=sequence_features.dtype)
        future_context = self.future_encoder(future_flat)

        if self.history_projection is not None and sequence_features.size(-1) > 1:
            history_context = self.history_projection(sequence_features[:, :, 1:].reshape(sequence_features.size(0), -1))
        else:
            history_context = torch.zeros_like(static_context)

        condition = torch.cat([static_context, future_context, history_context, baseline[:, :1]], dim=1)
        residual_history = target_history
        forecast = torch.zeros_like(baseline)
        for block in self.blocks:
            backcast, forecast_update = block(residual_history, condition)
            residual_history = residual_history - backcast
            forecast = forecast + forecast_update
        return baseline + forecast


def _patchify_channels(sequence_features: torch.Tensor, patch_len: int, patch_stride: int) -> torch.Tensor:
    """Split a multivariate sequence into per-channel patches for channel-independent processing.

    Input:  (batch, time, channels)
    Output: (batch * channels, num_patches, patch_len) — all channels as independent samples.

    Each channel is treated as a separate "batch item" so a single shared Transformer
    encoder processes all channels without cross-channel attention (channel-independent design).
    Left-pads with zeros if the sequence is shorter than ``patch_len``.
    """
    batch_size, sequence_length, channel_count = sequence_features.shape
    if sequence_length < patch_len:
        pad_length = patch_len - sequence_length
        sequence_features = F.pad(sequence_features, (0, 0, pad_length, 0))
        sequence_length = patch_len

    channels_first = sequence_features.transpose(1, 2)
    patches = channels_first.unfold(dimension=2, size=patch_len, step=patch_stride)
    patch_count = int(patches.size(2))
    return patches.reshape(batch_size * channel_count, patch_count, patch_len)


class ResidualAdvancedPatchTSTForecaster(nn.Module):
    """PatchTST: channel-independent patched transformer (Nie et al. 2023).

    Key design choices faithful to the paper:
    - Each channel processed independently through a *shared* transformer.
    - Patch tokens are FLATTENED (not averaged) before the prediction head,
      preserving temporal ordering within the patch encoding.
    - A per-channel linear head maps flat patch encodings to horizon predictions.
    - The internal forecast is the TARGET channel's head output only (channel 0 =
      ``target_history`` in the training bundle); exogenous channels condition the
      shared encoder but do not vote on the forecast (paper-faithful, no channel
      mixing).
    - RevIN normalizes the encoder inputs.
    """

    def __init__(
        self,
        *,
        sequence_input_dim: int,
        sequence_length: int,
        static_input_dim: int,
        future_input_dim: int,
        horizon_count: int,
        station_count: int,
        embedding_dim: int = 16,
        patch_len: int = 4,
        patch_stride: int = 2,
        model_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 3,
        ff_multiplier: int = 4,
        dropout: float = 0.1,
        head_hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.patch_len = max(1, int(patch_len))
        self.patch_stride = max(1, int(patch_stride))
        max_length = max(int(sequence_length), self.patch_len)
        patch_count = 1 + max(0, (max_length - self.patch_len) // self.patch_stride)
        self.channel_count = int(sequence_input_dim)

        self.station_embedding = nn.Embedding(station_count, embedding_dim)
        self.revin = ReversibleInstanceNorm(sequence_input_dim)

        # Shared patch input projection
        self.patch_projection = nn.Linear(self.patch_len, model_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, patch_count, model_dim))
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        self.dropout_layer = nn.Dropout(dropout)

        # Shared transformer encoder (channel-independent)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=int(num_heads),
            dim_feedforward=model_dim * int(ff_multiplier),
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(num_layers))

        # Per-channel linear head: flatten(patch_count * model_dim) → horizon_count
        self.channel_head = nn.Linear(patch_count * model_dim, horizon_count)

        # Context encoders for static and future features
        self.static_encoder = _StaticEncoder(static_input_dim + embedding_dim, model_dim, dropout)
        self.future_encoder = _ResidualFeedForward(
            max(1, horizon_count * future_input_dim),
            head_hidden_dim,
            model_dim,
            blocks=2,
            dropout=dropout,
        )
        # Correction head: channel-mixed forecast + static/future context + baseline
        self.head = _build_feed_forward(
            horizon_count + (model_dim * 2) + 1,
            [head_hidden_dim, head_hidden_dim],
            horizon_count,
            dropout=dropout,
        )

    def forward(
        self,
        sequence_features: torch.Tensor,
        static_features: torch.Tensor,
        future_features: torch.Tensor,
        station_index: torch.Tensor,
        baseline: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = int(sequence_features.size(0))

        # RevIN normalization: (B, T, C)
        normalized = self.revin(sequence_features)

        # Patchify: (B*C, N_patches, patch_len)
        patches = _patchify_channels(normalized, self.patch_len, self.patch_stride)

        # Project patches: (B*C, N_patches, model_dim)
        tokens = self.patch_projection(patches)
        tokens = self.dropout_layer(tokens + self.position_embedding[:, : tokens.size(1), :])

        # Shared transformer: (B*C, N_patches, model_dim)
        encoded = self.encoder(tokens)

        # Flatten patch tokens per channel: (B*C, N_patches * model_dim)
        flat = encoded.reshape(batch_size * self.channel_count, -1)

        # Per-channel horizon forecast: (B*C, horizon) → (B, C, horizon)
        channel_forecasts = self.channel_head(flat).reshape(batch_size, self.channel_count, -1)

        # Paper-faithful: take the target channel's forecast only (channel 0).
        target_forecast = channel_forecasts[:, 0, :]  # (B, horizon)

        # Static and future context
        station_embedding = self.station_embedding(station_index)
        static_context = self.static_encoder(static_features, station_embedding)
        future_flat = future_features.flatten(start_dim=1)
        if future_flat.size(1) == 0:
            future_flat = torch.zeros(batch_size, 1, device=sequence_features.device, dtype=sequence_features.dtype)
        future_context = self.future_encoder(future_flat)

        head_input = torch.cat([target_forecast, static_context, future_context, baseline[:, :1]], dim=1)
        correction = self.head(head_input)
        return baseline + correction


class _GatedResidualNetwork(nn.Module):
    """GRN from the TFT paper (Lim et al. 2021).

    GRN(a, c=None) = LayerNorm(a_skip + GLU(η_2))
    η_2 = W_2 * ELU(W_1*a + b_1 + W_c*c) + b_2
    GLU gates η_2 element-wise.

    Uses ELU activation (as in the paper) and supports an optional context vector c.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float,
        *,
        context_dim: int = 0,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_ctx = nn.Linear(context_dim, hidden_dim, bias=False) if context_dim > 0 else None
        self.fc2 = nn.Linear(hidden_dim, output_dim * 2)   # *2 for GLU gating
        self.skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        h = self.fc1(x)
        if self.fc_ctx is not None and context is not None:
            h = h + self.fc_ctx(context)
        h = F.elu(h)
        h = self.dropout(h)
        h = self.fc2(h)
        value, gate = h.chunk(2, dim=-1)
        gated = value * torch.sigmoid(gate)
        return self.norm(self.skip(x) + gated)


class _VariableSelectionNetwork(nn.Module):
    """VSN from TFT paper: softmax weights over variable GRN encodings.

    Each input variable is first processed by its own GRN, then a joint GRN
    over the flattened inputs produces softmax selection weights. The output
    is the weighted sum of per-variable encodings.
    """

    def __init__(
        self,
        num_vars: int,
        var_dim: int,
        hidden_dim: int,
        dropout: float,
        *,
        context_dim: int = 0,
    ) -> None:
        super().__init__()
        self.num_vars = num_vars
        self.var_dim = var_dim
        # One GRN per input variable
        self.var_grns = nn.ModuleList(
            [_GatedResidualNetwork(var_dim, hidden_dim, hidden_dim, dropout) for _ in range(num_vars)]
        )
        # Selection GRN: flattened input → softmax weights
        self.selection_grn = _GatedResidualNetwork(
            num_vars * var_dim, hidden_dim, num_vars, dropout, context_dim=context_dim
        )

    def forward(
        self, x: torch.Tensor, context: torch.Tensor | None = None
    ) -> torch.Tensor:
        # x: (batch, num_vars, var_dim) or (batch, time, num_vars, var_dim)
        var_encodings = torch.stack(
            [self.var_grns[i](x[..., i, :]) for i in range(self.num_vars)], dim=-2
        )  # (..., num_vars, hidden_dim)
        flat = x.flatten(start_dim=-2)  # (..., num_vars * var_dim)
        weights = torch.softmax(self.selection_grn(flat, context), dim=-1)  # (..., num_vars)
        selected = (var_encodings * weights.unsqueeze(-1)).sum(dim=-2)  # (..., hidden_dim)
        return selected


class ResidualAdvancedTemporalFusionTransformerForecaster(nn.Module):
    """Temporal Fusion Transformer (Lim et al. 2021).

    Faithful implementation with:
    - Variable Selection Networks (VSN) for past and future covariates
    - GRNs with ELU activation and optional context vector
    - LSTM encoder/decoder with static-context-initialized hidden state
    - Static enrichment GRN applied to LSTM encoder outputs
    - Multi-head attention (interpretable attention over encoder steps)
    - Gated Add+Norm at every stage
    """

    def __init__(
        self,
        *,
        sequence_input_dim: int,
        static_input_dim: int,
        future_input_dim: int,
        horizon_count: int,
        station_count: int,
        embedding_dim: int = 16,
        hidden_size: int = 128,
        lstm_layers: int = 2,
        attention_heads: int = 8,
        dropout: float = 0.1,
        head_hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        d = hidden_size
        self.station_embedding = nn.Embedding(station_count, embedding_dim)
        self.revin = ReversibleInstanceNorm(sequence_input_dim)

        # Static context: project raw static inputs → 4 context vectors
        static_raw_dim = static_input_dim + embedding_dim
        self.static_input_grn = _GatedResidualNetwork(static_raw_dim, d, d, dropout)
        # Four context vectors: enrichment, h_init, c_init, decoder conditioning
        self.ctx_enrichment = _GatedResidualNetwork(d, d, d, dropout)
        self.ctx_h = _GatedResidualNetwork(d, d, d, dropout)
        self.ctx_c = _GatedResidualNetwork(d, d, d, dropout)

        # Past variable selection
        # Each scalar variable is independently projected to d-dim via a shared linear.
        # VSN then learns per-timestep importance weights over all C variables.
        self.past_var_embed = nn.Linear(1, d)
        self.past_vsn = _VariableSelectionNetwork(
            sequence_input_dim, d, d, dropout, context_dim=d
        )

        # Future variable selection (pad to ≥1 var if no future features)
        self._fut_num_vars = max(1, future_input_dim)
        self.future_var_embed = nn.Linear(1, d)
        self.future_vsn = _VariableSelectionNetwork(
            self._fut_num_vars, d, d, dropout, context_dim=d
        )
        # Horizon positional embedding (future step index)
        self.horizon_embedding = nn.Embedding(horizon_count, d)

        # LSTM encoder (processes past)
        self.encoder_lstm = nn.LSTM(d, d, num_layers=lstm_layers, batch_first=True,
                                    dropout=dropout if lstm_layers > 1 else 0.0)
        self.encoder_gate = nn.Linear(d, d * 2)   # GLU gate after LSTM
        self.encoder_norm = nn.LayerNorm(d)

        # Static enrichment GRN (applied to each encoder timestep)
        self.enrichment_grn = _GatedResidualNetwork(d, d, d, dropout, context_dim=d)

        # LSTM decoder (processes future)
        self.decoder_lstm = nn.LSTM(d, d, num_layers=lstm_layers, batch_first=True,
                                    dropout=dropout if lstm_layers > 1 else 0.0)
        self.decoder_gate = nn.Linear(d, d * 2)
        self.decoder_norm = nn.LayerNorm(d)

        # Multi-head attention (decoder queries, encoder keys/values)
        self.attention = nn.MultiheadAttention(d, attention_heads, dropout=dropout, batch_first=True)
        self.attn_gate = nn.Linear(d, d * 2)
        self.attn_norm = nn.LayerNorm(d)
        self.positionwise_grn = _GatedResidualNetwork(d, d, d, dropout)
        self.pre_output_gate = nn.Linear(d, d * 2)
        self.pre_output_norm = nn.LayerNorm(d)

        self.output_projection = nn.Linear(d, 1)

    @staticmethod
    def _glu_gate(x: torch.Tensor, gate_layer: nn.Linear, norm: nn.LayerNorm,
                  residual: torch.Tensor) -> torch.Tensor:
        h = gate_layer(x)
        value, gate = h.chunk(2, dim=-1)
        return norm(residual + value * torch.sigmoid(gate))

    def forward(
        self,
        sequence_features: torch.Tensor,
        static_features: torch.Tensor,
        future_features: torch.Tensor,
        station_index: torch.Tensor,
        baseline: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = sequence_features.size(0)
        horizon = future_features.size(1)
        d = self.encoder_lstm.hidden_size

        # ── Static context ────────────────────────────────────────────────
        station_emb = self.station_embedding(station_index)
        static_raw = torch.cat([static_features, station_emb], dim=1)
        static_ctx = self.static_input_grn(static_raw)               # (B, d)
        ctx_enrich  = self.ctx_enrichment(static_ctx)
        ctx_h       = self.ctx_h(static_ctx)
        ctx_c       = self.ctx_c(static_ctx)

        # ── Past variable selection ───────────────────────────────────────
        normed = self.revin(sequence_features)                         # (B, T, C)
        B_sz, T_sz, C_sz = normed.shape
        # Embed each scalar variable: (B, T, C, 1) → (B, T, C, d)
        past_embeds = self.past_var_embed(normed.unsqueeze(-1))
        # Fold B*T into batch for VSN (which expects 2D batch)
        past_embeds_bt = past_embeds.reshape(B_sz * T_sz, C_sz, d)
        ctx_bt = static_ctx.unsqueeze(1).expand(-1, T_sz, -1).reshape(B_sz * T_sz, d)
        past_selected = self.past_vsn(past_embeds_bt, ctx_bt).reshape(B_sz, T_sz, d)

        # ── LSTM encoder ─────────────────────────────────────────────────
        h0 = ctx_h.unsqueeze(0).expand(self.encoder_lstm.num_layers, -1, -1).contiguous()
        c0 = ctx_c.unsqueeze(0).expand(self.encoder_lstm.num_layers, -1, -1).contiguous()
        enc_out, (enc_h, enc_c) = self.encoder_lstm(past_selected, (h0, c0))
        enc_out = self._glu_gate(enc_out, self.encoder_gate, self.encoder_norm, past_selected)

        # Static enrichment
        ctx_enrich_expanded = ctx_enrich.unsqueeze(1).expand(-1, enc_out.size(1), -1)
        enriched = self.enrichment_grn(enc_out, ctx_enrich_expanded)

        # ── Future variable selection ─────────────────────────────────────
        if future_features.size(-1) == 0:
            future_features = torch.zeros(batch_size, horizon, 1,
                                          device=sequence_features.device,
                                          dtype=sequence_features.dtype)
        fut_C = future_features.size(-1)
        # Embed each future scalar variable: (B, H, fut_C, 1) → (B, H, fut_C, d)
        fut_embeds = self.future_var_embed(future_features.unsqueeze(-1))
        # Pad to self._fut_num_vars if needed (e.g. future_input_dim=0 → _fut_num_vars=1)
        if fut_C < self._fut_num_vars:
            pad = torch.zeros(batch_size, horizon, self._fut_num_vars - fut_C, d,
                              device=future_features.device, dtype=future_features.dtype)
            fut_embeds = torch.cat([fut_embeds, pad], dim=2)
        # Fold B*H into batch for VSN
        fut_embeds_bh = fut_embeds.reshape(batch_size * horizon, self._fut_num_vars, d)
        ctx_bh = static_ctx.unsqueeze(1).expand(-1, horizon, -1).reshape(batch_size * horizon, d)
        future_selected = self.future_vsn(fut_embeds_bh, ctx_bh).reshape(batch_size, horizon, d)
        # Add horizon positional embedding
        horizon_idx = torch.arange(horizon, device=sequence_features.device)
        future_selected = future_selected + self.horizon_embedding(horizon_idx).unsqueeze(0)

        # ── LSTM decoder ─────────────────────────────────────────────────
        dec_out, _ = self.decoder_lstm(future_selected, (enc_h, enc_c))
        dec_out = self._glu_gate(dec_out, self.decoder_gate, self.decoder_norm, future_selected)

        # ── Multi-head attention ──────────────────────────────────────────
        attn_out, _ = self.attention(dec_out, enriched, enriched)
        attn_out = self._glu_gate(attn_out, self.attn_gate, self.attn_norm, dec_out)
        attn_out = self.positionwise_grn(attn_out)
        attn_out = self._glu_gate(attn_out, self.pre_output_gate, self.pre_output_norm, attn_out)

        correction = self.output_projection(attn_out).squeeze(-1)   # (B, horizon)
        return baseline + correction


class _mLSTMBlock(nn.Module):
    """Matrix LSTM (mLSTM) block from xLSTM (Beck et al. 2024).

    This benchmark stacks mLSTM blocks only (no sLSTM), i.e. the xLSTM[1:0]
    variant that Beck et al. themselves report as a baseline.

    Structure (per Beck et al.):
    - Up-projection (expand_factor=2) split into a main branch and a gate branch z.
    - A causal depthwise Conv1d (kernel 4) + SiLU on the main branch feeds the q
      and k projections; the value v comes from the un-convolved main branch.
    - Matrix memory C ∈ R^{H × d_h × d_h} (H heads, d_h = d_inner / H = 64 for
      model_dim 128, heads 4, expand_factor 2).
    - Stabilized exponential input gate, log-sigmoid forget gate, and the
      max(|n·q|, 1) normalizer (unchanged — already correct).
    - Memory update: C_t = f̃_t * C_{t-1} + ĩ_t * (v_t ⊗ k_t).
    - Normalizer:   n_t = f̃_t * n_{t-1} + ĩ_t * k_t.
    - Scan output h̃_t = (C_t q_t) / max(|n_t^T q_t|, 1); per-head GroupNorm over
      the stacked heads, then output gating h = groupnorm(h̃) * SiLU(z), then
      down-projection to model_dim.
    """

    def __init__(
        self,
        model_dim: int,
        *,
        num_heads: int = 4,
        expand_factor: int = 2,
        conv_kernel: int = 4,
        dropout: float,
    ) -> None:
        super().__init__()
        d_inner = model_dim * expand_factor
        assert d_inner % num_heads == 0, "expanded dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = d_inner // num_heads
        self.conv_kernel = int(conv_kernel)

        self.norm = nn.LayerNorm(model_dim)
        self.up_proj = nn.Linear(model_dim, d_inner * 2)     # main branch + gate branch z
        # Causal depthwise conv feeding the q/k projections.
        self.conv = nn.Conv1d(
            d_inner, d_inner, kernel_size=self.conv_kernel,
            padding=self.conv_kernel - 1, groups=d_inner,
        )
        self.q_proj = nn.Linear(d_inner, d_inner)            # from conv(main) branch
        self.k_proj = nn.Linear(d_inner, d_inner)            # from conv(main) branch
        self.v_proj = nn.Linear(d_inner, d_inner)            # from un-convolved main branch
        self.i_proj = nn.Linear(d_inner, num_heads)          # log input gate (one per head)
        self.f_proj = nn.Linear(d_inner, num_heads)          # log forget gate (one per head)
        self.head_norm = nn.GroupNorm(num_heads, d_inner)    # per-head normalization
        self.out_proj = nn.Linear(d_inner, model_dim)        # project back to model_dim
        self.dropout = nn.Dropout(dropout)
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 4, model_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, d = self.num_heads, self.head_dim

        normed = self.norm(x)
        branch, z = self.up_proj(normed).chunk(2, dim=-1)         # each (B, T, d_inner)

        # Causal depthwise conv (+ SiLU) feeding q and k; truncate the right pad.
        conv_out = self.conv(branch.transpose(1, 2))[..., :T]     # (B, d_inner, T)
        conv_out = F.silu(conv_out).transpose(1, 2)               # (B, T, d_inner)

        q = self.q_proj(conv_out).reshape(B, T, H, d)
        k = self.k_proj(conv_out).reshape(B, T, H, d)
        v = self.v_proj(branch).reshape(B, T, H, d)               # value from un-convolved branch
        # Normalize k so memory updates stay bounded
        k = k / (math.sqrt(d) + 1e-6)
        log_i = self.i_proj(conv_out)                             # (B, T, H)
        log_f = F.logsigmoid(self.f_proj(conv_out))               # (B, T, H)

        # Stabilizer: m_t = max(log_f_t + m_{t-1}, log_i_t)
        C = torch.zeros(B, H, d, d, device=x.device, dtype=x.dtype)
        n = torch.zeros(B, H, d,    device=x.device, dtype=x.dtype)
        m = torch.full((B, H), -1e9, device=x.device, dtype=x.dtype)

        outputs: list[torch.Tensor] = []
        for t in range(T):
            log_f_t = log_f[:, t, :]   # (B, H)
            log_i_t = log_i[:, t, :]   # (B, H)
            m_new = torch.maximum(log_f_t + m, log_i_t)
            f_stable = torch.exp(log_f_t + m - m_new)    # (B, H)
            i_stable = torch.exp(log_i_t - m_new)         # (B, H)
            m = m_new

            v_t = v[:, t, :, :]   # (B, H, d)
            k_t = k[:, t, :, :]   # (B, H, d)
            q_t = q[:, t, :, :]   # (B, H, d)

            f_e = f_stable.unsqueeze(-1).unsqueeze(-1)  # (B, H, 1, 1)
            i_e = i_stable.unsqueeze(-1).unsqueeze(-1)
            # C: (B, H, d, d); outer product v⊗k: (B, H, d, d)
            C = f_e * C + i_e * torch.einsum("bhd,bhe->bhde", v_t, k_t)
            n = f_stable.unsqueeze(-1) * n + i_stable.unsqueeze(-1) * k_t

            # h̃ = C q / max(|n^T q|, 1)
            h_raw = torch.einsum("bhde,bhe->bhd", C, q_t)    # (B, H, d)
            denom = (torch.einsum("bhd,bhd->bh", n, q_t).abs() + 1e-6).clamp_min(1.0)
            h_t = h_raw / denom.unsqueeze(-1)
            outputs.append(h_t)

        # Stack heads: (B, T, H, d) → (B, T, d_inner).
        hidden = torch.stack(outputs, dim=1).reshape(B, T, H * d)
        # Per-head GroupNorm over the stacked head channels.
        hidden = self.head_norm(hidden.reshape(B * T, H * d)).reshape(B, T, H * d)
        # Output gating with SiLU(z), then project back to model_dim.
        hidden = hidden * F.silu(z)
        hidden = self.out_proj(self.dropout(hidden))
        x = x + hidden
        x = x + self.feed_forward(x)
        return x


class ResidualAdvancedXLSTMForecaster(nn.Module):
    """xLSTM forecaster using matrix LSTM (mLSTM) blocks (Beck et al. 2024).

    Each ``_mLSTMBlock`` uses expand_factor=2 (d_inner = 2 × model_dim, head_dim = 64 with
    4 heads), a causal depthwise conv feeding q/k, stabilized log-space input/forget gates,
    per-head GroupNorm, and a matrix memory update (v ⊗ k outer product). Stacks
    ``num_blocks`` blocks over the RevIN-normalized sequence.

    The full sequence is collapsed to a single vector via context-conditioned attention
    pooling (not just the last timestep), which performs better on irregular flood events
    where the most informative timestep may not be the most recent.
    """

    def __init__(
        self,
        *,
        sequence_input_dim: int,
        static_input_dim: int,
        future_input_dim: int,
        horizon_count: int,
        station_count: int,
        embedding_dim: int = 16,
        model_dim: int = 128,
        num_blocks: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        head_hidden_dim: int = 256,
        # kernel_size kept for config compatibility but unused in mLSTM
        kernel_size: int = 4,
    ) -> None:
        super().__init__()
        self.station_embedding = nn.Embedding(station_count, embedding_dim)
        self.revin = ReversibleInstanceNorm(sequence_input_dim)
        self.input_projection = nn.Linear(sequence_input_dim, model_dim)
        self.blocks = nn.ModuleList(
            [_mLSTMBlock(model_dim, num_heads=num_heads, dropout=dropout)
             for _ in range(int(num_blocks))]
        )
        self.pooling = _AttentionPooling(model_dim, model_dim)
        self.static_encoder = _StaticEncoder(static_input_dim + embedding_dim, model_dim, dropout)
        self.future_encoder = _ResidualFeedForward(
            max(1, horizon_count * future_input_dim),
            head_hidden_dim,
            model_dim,
            blocks=2,
            dropout=dropout,
        )
        self.head = _build_feed_forward(
            (model_dim * 3) + 1,
            [head_hidden_dim, head_hidden_dim],
            horizon_count,
            dropout=dropout,
        )

    def forward(
        self,
        sequence_features: torch.Tensor,
        static_features: torch.Tensor,
        future_features: torch.Tensor,
        station_index: torch.Tensor,
        baseline: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.input_projection(self.revin(sequence_features))
        for block in self.blocks:
            hidden = block(hidden)

        station_embedding = self.station_embedding(station_index)
        static_context = self.static_encoder(static_features, station_embedding)
        sequence_summary = self.pooling(hidden, static_context)
        future_flat = future_features.flatten(start_dim=1)
        if future_flat.size(1) == 0:
            future_flat = torch.zeros(sequence_features.size(0), 1,
                                      device=sequence_features.device, dtype=sequence_features.dtype)
        future_context = self.future_encoder(future_flat)
        correction = self.head(torch.cat([sequence_summary, static_context, future_context, baseline[:, :1]], dim=1))
        return baseline + correction


class _MambaBlock(nn.Module):
    """Mamba selective state-space block (Gu & Dao 2023).

    Faithful implementation of the S6 selective scan mechanism:
    - Diagonal A in log-space with HiPPO-like init (log(1..d_state))
    - B, C: input-dependent (low-rank from x via linear projections)
    - Δ: input-dependent via low-rank projection dt_rank + softplus for positivity
    - ZOH discretization: Ā_t = exp(Δ_t ⊗ A), B̄_t = Δ_t ⊗ B_t
    - SSM recurrence: h_t = Ā_t * h_{t-1} + B̄_t * u_t (element-wise)
    - Output:  y_t = (C_t · h_t) + D * u_t
    - Gating:  y = y * silu(z)  where z is the parallel branch from in_proj
    - d_inner = model_dim * expand (expand=2, as in Mamba paper)

    The scan is available in two mathematically-equivalent forms: a per-timestep
    sequential recurrence (``_ssm_sequential``) and a chunked log-space associative
    scan (``_ssm_parallel``, default). The associative scan composes the affine
    per-step maps ``h_t = exp(Δ_t·A)·h_{t-1} + B̄_t u_t`` in log-decay space, so no
    intermediate ``exp(-ΣΔA)`` is ever formed and the computation stays numerically
    stable regardless of the accumulated decay magnitude.
    """

    def __init__(
        self,
        model_dim: int,
        *,
        state_dim: int = 16,
        expand: int = 2,
        kernel_size: int = 4,
        dt_rank: int | None = None,
        dropout: float = 0.0,
        use_parallel_scan: bool = True,
        scan_chunk: int = 16,
    ) -> None:
        super().__init__()
        d_inner = model_dim * expand
        self.d_inner = d_inner
        self.state_dim = state_dim
        self.use_parallel_scan = bool(use_parallel_scan)
        self.scan_chunk = int(scan_chunk)
        dt_rank = dt_rank if dt_rank is not None else max(1, model_dim // 16)

        self.norm = nn.LayerNorm(model_dim)
        # Project model_dim → (d_inner [x branch] + d_inner [z gate])
        self.in_proj = nn.Linear(model_dim, d_inner * 2, bias=False)
        # Causal depthwise conv on x branch
        self.conv1d = nn.Conv1d(
            d_inner, d_inner,
            kernel_size=kernel_size,
            padding=kernel_size - 1,
            groups=d_inner,
            bias=True,
        )
        self.act = nn.SiLU()
        # SSM parameter projections
        self.x_proj = nn.Linear(d_inner, dt_rank + state_dim * 2, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        # Log-space diagonal A: shape (d_inner, d_state); initialized as log(1..state_dim)
        A = torch.arange(1, state_dim + 1, dtype=torch.float32).unsqueeze(0).expand(d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))
        # D skip connection (one scalar per inner channel)
        self.D = nn.Parameter(torch.ones(d_inner))
        # Output projection
        self.out_proj = nn.Linear(d_inner, model_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _ssm_params(self, u: torch.Tensor):
        """Project u into the selective-scan parameters (Δ, A, B_t, C_t)."""
        state_dim = self.state_dim
        x_dbl = self.x_proj(u)                            # (B, T, dt_rank + 2*state_dim)
        dt_rank = x_dbl.size(-1) - 2 * state_dim
        delta_raw = x_dbl[..., :dt_rank]                   # (B, T, dt_rank)
        B_t = x_dbl[..., dt_rank: dt_rank + state_dim]    # (B, T, state_dim)
        C_t = x_dbl[..., dt_rank + state_dim:]            # (B, T, state_dim)
        delta = F.softplus(self.dt_proj(delta_raw))        # (B, T, d_inner)
        A = -torch.exp(self.A_log)                         # (d_inner, state_dim)
        return delta, A, B_t, C_t

    def _ssm(self, u: torch.Tensor) -> torch.Tensor:
        """Dispatch to the parallel (default) or sequential selective scan."""
        if self.use_parallel_scan:
            return self._ssm_parallel(u)
        return self._ssm_sequential(u)

    def _ssm_sequential(self, u: torch.Tensor) -> torch.Tensor:
        """Run the selective scan on u: (B, T, d_inner) as a sequential recurrence."""
        B, T, _ = u.shape
        d_inner, state_dim = self.d_inner, self.state_dim
        delta, A, B_t, C_t = self._ssm_params(u)

        # Selective scan (sequential recurrence over T)
        h = torch.zeros(B, d_inner, state_dim, device=u.device, dtype=u.dtype)
        ys: list[torch.Tensor] = []
        for t in range(T):
            # ZOH: Ā_t = exp(Δ_t[:, None] * A[None, :])  shape (B, d_inner, state_dim)
            A_bar = torch.exp(delta[:, t, :, None] * A[None, :, :])
            # B̄_t = Δ_t[:, :, None] * B_t[:, None, :]  shape (B, d_inner, state_dim)
            B_bar = delta[:, t, :, None] * B_t[:, t, None, :]
            u_t = u[:, t, :]                               # (B, d_inner)
            h = A_bar * h + B_bar * u_t[:, :, None]       # element-wise
            # y_t = Σ_n C_t[n] * h[:, :, n]  + D * u_t
            y_t = (h * C_t[:, t, None, :]).sum(dim=-1) + self.D * u_t  # (B, d_inner)
            ys.append(y_t)

        return torch.stack(ys, dim=1)  # (B, T, d_inner)

    def _ssm_parallel(self, u: torch.Tensor) -> torch.Tensor:
        """Chunked log-space associative scan (numerically stable, default path).

        Each timestep defines an affine map h_t = a_t·h_{t-1} + b_t where the diagonal
        decay a_t = exp(Δ_t·A) ∈ (0, 1] is kept as its log ``g_t = Δ_t·A`` (≤ 0). Within
        a chunk the maps are composed with a Hillis–Steele associative scan that only
        ever exponentiates non-positive cumulative log-decays, so nothing overflows;
        chunks are stitched sequentially through a carried state to bound memory.
        """
        B, T, _ = u.shape
        d_inner, state_dim = self.d_inner, self.state_dim
        delta, A, B_t, C_t = self._ssm_params(u)

        # Per-step log-decay g_t (≤ 0) and additive input b_t, shape (B, T, d_inner, state_dim).
        g = delta[:, :, :, None] * A[None, None, :, :]
        b = delta[:, :, :, None] * B_t[:, :, None, :] * u[:, :, :, None]

        carry = torch.zeros(B, d_inner, state_dim, device=u.device, dtype=u.dtype)
        chunk = max(1, self.scan_chunk)
        ys: list[torch.Tensor] = []
        for start in range(0, T, chunk):
            g_c = g[:, start:start + chunk]                # (B, L, d, n)
            b_c = b[:, start:start + chunk]
            L = g_c.size(1)

            # Inclusive associative scan of the affine maps within the chunk.
            log_a = g_c.clone()   # becomes cumulative log-decay from chunk start
            h_loc = b_c.clone()   # becomes local prefix state (zero initial condition)
            offset = 1
            while offset < L:
                log_a_prev = F.pad(log_a[:, : L - offset], (0, 0, 0, 0, offset, 0))
                h_prev = F.pad(h_loc[:, : L - offset], (0, 0, 0, 0, offset, 0))
                # Compose left (older) map into current: b = b_cur + a_cur * b_left.
                h_loc = h_loc + torch.exp(log_a) * h_prev
                log_a = log_a + log_a_prev
                offset *= 2

            # Add the carried state from earlier chunks: h_i = h_loc_i + exp(Σg)_i * carry.
            h_full = h_loc + torch.exp(log_a) * carry[:, None, :, :]

            # y_i = Σ_n C_i[n] * h_i[:, :, n] + D * u_i.
            C_c = C_t[:, start:start + chunk]              # (B, L, n)
            u_c = u[:, start:start + chunk]                # (B, L, d)
            y_c = (h_full * C_c[:, :, None, :]).sum(dim=-1) + self.D * u_c
            ys.append(y_c)
            carry = h_full[:, -1]

        return torch.cat(ys, dim=1)  # (B, T, d_inner)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm(x)
        xz = self.in_proj(normed)                          # (B, T, 2*d_inner)
        u, z = xz.chunk(2, dim=-1)                        # each (B, T, d_inner)

        # Causal depthwise conv on u
        u_conv = self.conv1d(u.transpose(1, 2))            # (B, d_inner, T+pad)
        u_conv = u_conv[:, :, : x.size(1)].transpose(1, 2)  # (B, T, d_inner)
        u_conv = self.act(u_conv)

        # Selective SSM
        y = self._ssm(u_conv)

        # SiLU gating
        y = y * F.silu(z)

        out = self.out_proj(self.dropout(y))
        return x + out


class ResidualAdvancedMambaForecaster(nn.Module):
    """Mamba forecaster using faithful selective SSM blocks (Gu & Dao 2023).

    Key design choices:
    - SE-Net channel gate: temporal mean → sigmoid → multiplicative gate applied before the
      input projection. Discharge channel naturally dominates; irrelevant ERA5 channels
      (e.g., snowfall in summer) are soft-suppressed without hard feature selection.
    - Stacked ``_MambaBlock``s with ZOH discretization, input-dependent B/C/Δ, and SiLU gating.
    - Attention pooling (not last-timestep) for the sequence summary.
    - Static + future context fused in the prediction head alongside the persistence baseline.
    """

    def __init__(
        self,
        *,
        sequence_input_dim: int,
        static_input_dim: int,
        future_input_dim: int,
        horizon_count: int,
        station_count: int,
        embedding_dim: int = 16,
        model_dim: int = 128,
        state_dim: int = 16,
        num_blocks: int = 4,
        kernel_size: int = 4,
        dropout: float = 0.1,
        head_hidden_dim: int = 256,
        use_parallel_scan: bool = True,
    ) -> None:
        super().__init__()
        self.station_embedding = nn.Embedding(station_count, embedding_dim)
        self.revin = ReversibleInstanceNorm(sequence_input_dim)
        self.channel_gate = nn.Sequential(
            nn.Linear(sequence_input_dim, sequence_input_dim),
            nn.Sigmoid(),
        )
        self.input_projection = nn.Linear(sequence_input_dim, model_dim)
        self.blocks = nn.ModuleList(
            [
                _MambaBlock(model_dim, state_dim=state_dim, kernel_size=kernel_size,
                            use_parallel_scan=use_parallel_scan)
                for _ in range(int(num_blocks))
            ]
        )
        self.pooling = _AttentionPooling(model_dim, model_dim)
        self.static_encoder = _StaticEncoder(static_input_dim + embedding_dim, model_dim, dropout)
        self.future_encoder = _ResidualFeedForward(
            max(1, horizon_count * future_input_dim),
            head_hidden_dim,
            model_dim,
            blocks=2,
            dropout=dropout,
        )
        self.head = _build_feed_forward(
            (model_dim * 3) + 1,
            [head_hidden_dim, head_hidden_dim],
            horizon_count,
            dropout=dropout,
        )

    def forward(
        self,
        sequence_features: torch.Tensor,
        static_features: torch.Tensor,
        future_features: torch.Tensor,
        station_index: torch.Tensor,
        baseline: torch.Tensor,
    ) -> torch.Tensor:
        normed = self.revin(sequence_features)                           # (B, T, C)
        gate = self.channel_gate(normed.mean(dim=1))                     # (B, C)
        hidden = self.input_projection(normed * gate.unsqueeze(1))       # (B, T, model_dim)
        for block in self.blocks:
            hidden = block(hidden)

        station_embedding = self.station_embedding(station_index)
        static_context = self.static_encoder(static_features, station_embedding)
        sequence_summary = self.pooling(hidden, static_context)
        future_flat = future_features.flatten(start_dim=1)
        if future_flat.size(1) == 0:
            future_flat = torch.zeros(sequence_features.size(0), 1,
                                      device=sequence_features.device, dtype=sequence_features.dtype)
        future_context = self.future_encoder(future_flat)
        correction = self.head(torch.cat([sequence_summary, static_context, future_context, baseline[:, :1]], dim=1))
        return baseline + correction


class ResidualHydroHybridForecaster(nn.Module):
    """Conv-recurrent-attention hybrid designed for daily hydrological forecasting.

    Architecture:
      1. RevIN normalization + linear projection → ``model_dim``.
      2. ``conv_blocks`` of gated depthwise ``_TemporalConvBlock`` (local pattern extraction).
      3. Bidirectional LSTM over the projected (not conv-modified) sequence for long-range memory.
      4. Concatenate conv output + LSTM output → project back to ``model_dim`` (history memory).
      5. Future covariate tokens (horizon embeddings + projected future features) + static bias.
      6. Cross-attention: future tokens query the history memory to retrieve relevant context.
      7. GRN decoder: fused [future | attended | global_context | static | baseline] → correction.

    The parallel conv + LSTM branches give complementary views: convolutions detect sharp
    local events (rainfall spikes) while the LSTM tracks slow baseflow trends.
    """

    def __init__(
        self,
        *,
        sequence_input_dim: int,
        static_input_dim: int,
        future_input_dim: int,
        horizon_count: int,
        station_count: int,
        embedding_dim: int = 16,
        model_dim: int = 128,
        conv_blocks: int = 3,
        conv_kernel_size: int = 5,
        recurrent_hidden_size: int = 128,
        recurrent_layers: int = 2,
        attention_heads: int = 8,
        dropout: float = 0.1,
        head_hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.station_embedding = nn.Embedding(station_count, embedding_dim)
        self.revin = ReversibleInstanceNorm(sequence_input_dim)
        self.input_projection = nn.Linear(sequence_input_dim, model_dim)
        self.conv_blocks = nn.ModuleList(
            [
                _TemporalConvBlock(model_dim, kernel_size=conv_kernel_size, expansion=2, dropout=dropout)
                for _ in range(int(conv_blocks))
            ]
        )
        self.recurrent_encoder = nn.LSTM(
            input_size=model_dim,
            hidden_size=recurrent_hidden_size,
            num_layers=recurrent_layers,
            batch_first=True,
            dropout=dropout if recurrent_layers > 1 else 0.0,
            bidirectional=True,
        )
        fusion_dim = model_dim + (recurrent_hidden_size * 2)
        self.history_projection = nn.Linear(fusion_dim, model_dim)
        self.static_encoder = _StaticEncoder(static_input_dim + embedding_dim, model_dim, dropout)
        self.future_encoder = _FutureFeatureEncoder(future_input_dim, horizon_count, model_dim, dropout)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.global_pooling = _AttentionPooling(model_dim, model_dim)
        self.decoder_grn = _GatedResidualNetwork((model_dim * 4) + 1, head_hidden_dim, head_hidden_dim, dropout)
        self.output_projection = nn.Linear(head_hidden_dim, 1)

    def forward(
        self,
        sequence_features: torch.Tensor,
        static_features: torch.Tensor,
        future_features: torch.Tensor,
        station_index: torch.Tensor,
        baseline: torch.Tensor,
    ) -> torch.Tensor:
        sequence_inputs = self.input_projection(self.revin(sequence_features))
        conv_hidden = sequence_inputs
        for block in self.conv_blocks:
            conv_hidden = block(conv_hidden)

        recurrent_hidden, _ = self.recurrent_encoder(sequence_inputs)
        history_memory = self.history_projection(torch.cat([conv_hidden, recurrent_hidden], dim=-1))

        station_embedding = self.station_embedding(station_index)
        static_context = self.static_encoder(static_features, station_embedding)
        future_tokens = self.future_encoder(future_features) + static_context.unsqueeze(1)
        attended_future, _ = self.cross_attention(future_tokens, history_memory, history_memory)
        global_context = self.global_pooling(history_memory, static_context).unsqueeze(1).expand_as(attended_future)
        repeated_static = static_context.unsqueeze(1).expand_as(attended_future)

        fused = self.decoder_grn(
            torch.cat([future_tokens, attended_future, global_context, repeated_static, baseline.unsqueeze(-1)], dim=-1)
        )
        correction = self.output_projection(fused).squeeze(-1)
        return baseline + correction


# ── FlowNet: multi-scale conv + Mamba encoder + seq2seq LSTM decoder ────────


class _MultiScaleFusion(nn.Module):
    """Parallel temporal conv branches at different kernel sizes with learnable weighted merge.

    Each branch independently processes the input with its own kernel size, capturing
    patterns at different temporal scales. A learned softmax weighting blends the outputs.
    Since each _TemporalConvBlock contains an internal residual, the weighted average of
    branch outputs equals x + weighted_blend(per-scale modifications).
    """

    def __init__(self, model_dim: int, kernel_sizes: Iterable[int], dropout: float) -> None:
        super().__init__()
        kernel_list = [max(1, int(k)) for k in kernel_sizes]
        self.branches = nn.ModuleList(
            [_TemporalConvBlock(model_dim, kernel_size=k, expansion=2, dropout=dropout) for k in kernel_list]
        )
        self.branch_weights = nn.Parameter(torch.ones(len(kernel_list)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Each branch(x) returns x + scale-specific modification (residual inside)
        branch_outputs = torch.stack([branch(x) for branch in self.branches], dim=0)  # (n, B, T, d)
        weights = torch.softmax(self.branch_weights, dim=0)                           # (n,)
        return (branch_outputs * weights.view(-1, 1, 1, 1)).sum(dim=0)               # (B, T, d)


class ResidualHydroFlowNetForecaster(nn.Module):
    """FlowNet: multi-scale conv encoder + Mamba SSM + seq2seq LSTM decoder.

    Architecture overview
    ---------------------
    Encoder
      1. RevIN per-sample normalization over all input channels.
      2. Linear projection → model_dim.
      3. Multi-scale parallel conv fusion (default kernels: 3, 7, 14 days):
         three independent TemporalConvBlocks capture flash-event, soil-moisture,
         and baseflow patterns simultaneously; outputs blended with learned softmax weights.
      4. Stacked Mamba SSM blocks: selective state-space encoding that learns which
         historical signals to propagate across long gaps (e.g., antecedent soil moisture
         weeks before a flood).
      5. Context-conditioned attention pooling → global history summary vector.

    Decoder
      6. Future covariate tokens: linear projection + learned horizon embedding + static bias.
      7. 2-layer LSTM decoder initialized from global history summary, processes future tokens
         one step per horizon — produces smooth, physically plausible forecast trajectories.
      8. Cross-attention (decoder queries, encoder sequence keys/values) + GLU gating:
         each decoder step retrieves the most relevant historical context for that horizon.
      9. GRN fusion of attended decoder state + static context + per-step baseline value
         → per-step correction added to the persistence baseline.

    Design motivation for daily Slovak streamflow forecasting
    ---------------------------------------------------------
    - Multi-scale kernels 3/7/14 match flash-runoff / soil recharge / baseflow timescales.
    - Mamba's selective scan outperforms biLSTM on long sparse memory (e.g., dry-spell state).
    - The LSTM decoder with encoder-state initialization learns smooth inter-horizon
      dependencies, unlike decoding all horizons independently.
    - GRN gating stabilizes gradients through deep computation paths.
    """

    def __init__(
        self,
        *,
        sequence_input_dim: int,
        static_input_dim: int,
        future_input_dim: int,
        horizon_count: int,
        station_count: int,
        embedding_dim: int = 16,
        model_dim: int = 128,
        num_mamba_blocks: int = 3,
        mamba_state_dim: int = 32,
        multi_scale_kernels: Iterable[int] = (3, 7, 14),
        attention_heads: int = 8,
        decoder_lstm_layers: int = 2,
        dropout: float = 0.1,
        head_hidden_dim: int = 256,
        use_parallel_scan: bool = True,
    ) -> None:
        super().__init__()
        self.horizon_count = int(horizon_count)
        self._future_input_dim = max(1, int(future_input_dim))
        self._decoder_lstm_layers = int(decoder_lstm_layers)

        # ── Normalization + projection ──────────────────────────────────────
        self.station_embedding = nn.Embedding(station_count, embedding_dim)
        self.revin = ReversibleInstanceNorm(sequence_input_dim)
        self.channel_gate = nn.Sequential(
            nn.Linear(sequence_input_dim, sequence_input_dim),
            nn.Sigmoid(),
        )
        self.input_projection = nn.Linear(sequence_input_dim, model_dim)

        # ── Multi-scale parallel conv ───────────────────────────────────────
        self.multi_scale_conv = _MultiScaleFusion(model_dim, multi_scale_kernels, dropout)

        # ── Mamba encoder blocks ────────────────────────────────────────────
        self.mamba_blocks = nn.ModuleList(
            [
                _MambaBlock(model_dim, state_dim=int(mamba_state_dim), kernel_size=4, dropout=dropout,
                            use_parallel_scan=use_parallel_scan)
                for _ in range(int(num_mamba_blocks))
            ]
        )
        self.encoder_norm = nn.LayerNorm(model_dim)

        # ── Static context ──────────────────────────────────────────────────
        self.static_encoder = _StaticEncoder(static_input_dim + embedding_dim, model_dim, dropout)
        self.global_pooling = _AttentionPooling(model_dim, model_dim)

        # ── Future token encoding ───────────────────────────────────────────
        self.future_projection = _GatedResidualNetwork(
            self._future_input_dim, head_hidden_dim, model_dim, dropout
        )
        self.horizon_embedding = nn.Embedding(self.horizon_count, model_dim)

        # ── Seq2seq LSTM decoder ────────────────────────────────────────────
        # GRN maps pooled history → initial hidden state for decoder LSTM
        self.decoder_init_grn = _GatedResidualNetwork(model_dim, head_hidden_dim, model_dim, dropout)
        self.decoder_lstm = nn.LSTM(
            model_dim,
            model_dim,
            num_layers=self._decoder_lstm_layers,
            batch_first=True,
            dropout=dropout if self._decoder_lstm_layers > 1 else 0.0,
        )

        # ── Cross-attention + GLU gating ────────────────────────────────────
        # Ensure model_dim is divisible by attention_heads
        num_heads = int(attention_heads)
        while model_dim % num_heads != 0 and num_heads > 1:
            num_heads -= 1
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=model_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn_gate = nn.Linear(model_dim, model_dim * 2)
        self.cross_attn_norm = nn.LayerNorm(model_dim)

        # ── Output: GRN fusion → per-horizon correction ─────────────────────
        # input: [dec_out(d) | static_context(d) | baseline_step(1)]
        self.output_grn = _GatedResidualNetwork(
            model_dim + model_dim + 1, head_hidden_dim, model_dim, dropout
        )
        self.output_projection = nn.Linear(model_dim, 1)

    def forward(
        self,
        sequence_features: torch.Tensor,
        static_features: torch.Tensor,
        future_features: torch.Tensor,
        station_index: torch.Tensor,
        baseline: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = int(sequence_features.size(0))

        # ── Encoder ──────────────────────────────────────────────────────────
        normed = self.revin(sequence_features)                           # (B, T, C)
        gate = self.channel_gate(normed.mean(dim=1))                     # (B, C)
        hidden = self.input_projection(normed * gate.unsqueeze(1))       # (B, T, d)
        hidden = self.multi_scale_conv(hidden)                          # (B, T, d)
        for block in self.mamba_blocks:
            hidden = block(hidden)                                       # (B, T, d)
        encoded = self.encoder_norm(hidden)                             # (B, T, d)

        # ── Static context ────────────────────────────────────────────────────
        station_emb = self.station_embedding(station_index)
        static_context = self.static_encoder(static_features, station_emb)  # (B, d)
        global_context = self.global_pooling(encoded, static_context)        # (B, d)

        # ── Future tokens ─────────────────────────────────────────────────────
        if future_features.size(-1) == 0:
            future_features = torch.zeros(
                batch_size, self.horizon_count, self._future_input_dim,
                device=sequence_features.device, dtype=sequence_features.dtype,
            )
        fut = future_features[..., : self._future_input_dim]
        if fut.size(-1) < self._future_input_dim:
            pad = torch.zeros(
                batch_size, self.horizon_count, self._future_input_dim - fut.size(-1),
                device=fut.device, dtype=fut.dtype,
            )
            fut = torch.cat([fut, pad], dim=-1)

        horizon_idx = torch.arange(self.horizon_count, device=sequence_features.device)
        future_tokens = (
            self.future_projection(fut)
            + self.horizon_embedding(horizon_idx).unsqueeze(0)
            + static_context.unsqueeze(1)
        )  # (B, H, d)

        # ── Seq2seq LSTM decoder ──────────────────────────────────────────────
        # Initialize hidden state from pooled encoder summary via GRN
        init_state = self.decoder_init_grn(global_context)                              # (B, d)
        h0 = init_state.unsqueeze(0).expand(self._decoder_lstm_layers, -1, -1).contiguous()  # (L, B, d)
        c0 = torch.zeros_like(h0)
        dec_out, _ = self.decoder_lstm(future_tokens, (h0, c0))                         # (B, H, d)

        # ── Cross-attention: decoder attends to full encoder sequence ─────────
        attn_out, _ = self.cross_attention(dec_out, encoded, encoded)   # (B, H, d)
        gate_h = self.cross_attn_gate(attn_out)                         # (B, H, 2d)
        value, gate = gate_h.chunk(2, dim=-1)
        dec_out = self.cross_attn_norm(dec_out + value * torch.sigmoid(gate))  # (B, H, d)

        # ── Output: fuse decoder + static + baseline → per-horizon correction ─
        static_expanded = static_context.unsqueeze(1).expand_as(dec_out)  # (B, H, d)
        baseline_exp = baseline.unsqueeze(-1)                              # (B, H, 1)
        fused = self.output_grn(
            torch.cat([dec_out, static_expanded, baseline_exp], dim=-1)
        )                                                                   # (B, H, d)
        correction = self.output_projection(fused).squeeze(-1)             # (B, H)
        return baseline + correction
