"""Scaled neural forecasting variants and a hybrid architecture for hydrological benchmarking."""

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


class _StationConditioningMixin:
    def _encode_static_context(
        self,
        static_features: torch.Tensor,
        station_embedding: torch.Tensor,
        *,
        output_dim: int,
    ) -> torch.Tensor:
        batch_size = int(station_embedding.size(0))
        if static_features.size(1) == 0:
            static_input = station_embedding
        else:
            static_input = torch.cat([static_features, station_embedding], dim=1)
        if static_input.size(1) == output_dim:
            return static_input
        projection = nn.Linear(static_input.size(1), output_dim, device=static_input.device, dtype=static_input.dtype)
        return projection(static_input)


class ReversibleInstanceNorm(nn.Module):
    """RevIN: per-sample per-channel normalization with reversible denormalization.

    Kim et al. 2022 "Reversible Instance Normalization for Accurate Time-Series
    Forecasting against Distribution Shift".

    forward() normalizes input and caches (mean, std) per sample per channel.
    reverse() uses those cached stats to map predictions back to original scale.
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
        self._mean: torch.Tensor | None = None
        self._std: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time, channels)
        if x.size(-1) == 0:
            return x
        self._mean = x.mean(dim=1, keepdim=True)                            # (B, 1, C)
        self._std = x.std(dim=1, keepdim=True, unbiased=False).clamp_min(self.eps)  # (B, 1, C)
        normalized = (x - self._mean) / self._std
        if self.affine:
            normalized = normalized * self.weight + self.bias
        return normalized

    def reverse(self, y: torch.Tensor, channel_idx: int = 0) -> torch.Tensor:
        """Denormalize predictions for a single channel back to original scale.

        Args:
            y: (batch, horizon) predictions in normalized space.
            channel_idx: index of the channel whose stats to use (default 0 = target).
        Returns:
            (batch, horizon) predictions in original scale.
        """
        if self._mean is None or self._std is None:
            raise RuntimeError("reverse() called before forward(); no cached statistics.")
        mean = self._mean[:, 0, channel_idx]   # (B,)
        std = self._std[:, 0, channel_idx]     # (B,)
        if self.affine:
            w = self.weight[0, 0, channel_idx]
            b = self.bias[0, 0, channel_idx]
            y = (y - b) / (w + self.eps)
        return y * std.unsqueeze(1) + mean.unsqueeze(1)


class _AttentionPooling(nn.Module):
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
    return nn.Conv1d(
        model_dim,
        model_dim,
        kernel_size=max(1, int(kernel_size)),
        padding=max(1, int(kernel_size)) // 2,
        groups=model_dim,
    )


class _TemporalConvBlock(nn.Module):
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
    """A deeper residual MLP baseline over flattened history, static, and future inputs."""

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
    """A conv-augmented bidirectional LSTM with attention pooling over multivariate history."""

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
    ) -> None:
        super().__init__()
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
            bidirectional=True,
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
        self.pooling = _AttentionPooling(hidden_size * 2, context_dim)
        self.head = _build_feed_forward(
            (hidden_size * 4) + (context_dim * 2) + 1,
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
        sequence_summary = torch.cat([hidden_state[-2], hidden_state[-1]], dim=1)

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
    """A larger N-HiTS-style forecaster conditioned on static and future covariates."""

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
    - Learned soft channel-mixing weights aggregate per-channel forecasts.
    - RevIN normalizes each sample; stored stats enable denormalization if needed.
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

        # Learned channel mixing (softmax over channels)
        self.channel_mix = nn.Parameter(torch.ones(self.channel_count))

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

        # Soft channel mixing with learned weights: (B, horizon)
        channel_weights = torch.softmax(self.channel_mix, dim=0)  # (C,)
        mixed_forecast = (channel_forecasts * channel_weights.unsqueeze(0).unsqueeze(-1)).sum(dim=1)

        # Static and future context
        station_embedding = self.station_embedding(station_index)
        static_context = self.static_encoder(static_features, station_embedding)
        future_flat = future_features.flatten(start_dim=1)
        if future_flat.size(1) == 0:
            future_flat = torch.zeros(batch_size, 1, device=sequence_features.device, dtype=sequence_features.dtype)
        future_context = self.future_encoder(future_flat)

        head_input = torch.cat([mixed_forecast, static_context, future_context, baseline[:, :1]], dim=1)
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

    Key properties:
    - Matrix memory C ∈ R^{H × d_h × d_h} (H heads, d_h = model_dim / H)
    - Exponential input gate: i_t = exp(ṽ_i_t)  (stabilized)
    - Log-space forget gate accumulated with stabilizer m_t
    - Output gate o_t = sigmoid(...)
    - Memory update: C_t = f̃_t * C_{t-1} + ĩ_t * (v_t ⊗ k_t)
    - Normalizer:   n_t = f̃_t * n_{t-1} + ĩ_t * k_t
    - Hidden:       h_t = o_t * (C_t q_t) / max(|n_t^T q_t|, 1)
    """

    def __init__(self, model_dim: int, *, num_heads: int = 4, dropout: float) -> None:
        super().__init__()
        assert model_dim % num_heads == 0, "model_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads

        self.norm = nn.LayerNorm(model_dim)
        # Input projections for q, k, v, output gate, input gate, forget gate
        self.qkv_proj = nn.Linear(model_dim, model_dim * 3)
        self.o_proj = nn.Linear(model_dim, model_dim)
        self.i_proj = nn.Linear(model_dim, num_heads)   # log input gate (one per head)
        self.f_proj = nn.Linear(model_dim, num_heads)   # log forget gate (one per head)
        self.out_proj = nn.Linear(model_dim, model_dim)
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
        qkv = self.qkv_proj(normed).reshape(B, T, 3, H, d)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # each (B, T, H, d)
        # Normalize k so memory updates stay bounded
        k = k / (math.sqrt(d) + 1e-6)
        o = torch.sigmoid(self.o_proj(normed).reshape(B, T, H, d))
        log_i = self.i_proj(normed)   # (B, T, H)
        log_f = F.logsigmoid(self.f_proj(normed))  # (B, T, H) — use log-sigmoid for forget

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

            # h = o ⊙ C q / max(|n^T q|, 1)
            h_raw = torch.einsum("bhde,bhe->bhd", C, q_t)    # (B, H, d)
            denom = (torch.einsum("bhd,bhd->bh", n, q_t).abs() + 1e-6).clamp_min(1.0)
            h_t = o[:, t, :, :] * (h_raw / denom.unsqueeze(-1))
            outputs.append(h_t)

        # Stack: (B, T, H, d) → (B, T, D)
        hidden = torch.stack(outputs, dim=1).reshape(B, T, D)
        hidden = self.out_proj(self.dropout(hidden))
        x = x + hidden
        x = x + self.feed_forward(x)
        return x


class ResidualAdvancedXLSTMForecaster(nn.Module):
    """xLSTM forecaster using matrix LSTM (mLSTM) blocks (Beck et al. 2024)."""

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
    ) -> None:
        super().__init__()
        d_inner = model_dim * expand
        self.d_inner = d_inner
        self.state_dim = state_dim
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

    def _ssm(self, u: torch.Tensor) -> torch.Tensor:
        """Run the selective scan on u: (B, T, d_inner)."""
        B, T, _ = u.shape
        d_inner, state_dim = self.d_inner, self.state_dim

        # Project for Δ (low-rank), B, C
        x_dbl = self.x_proj(u)                            # (B, T, dt_rank + 2*state_dim)
        dt_rank = x_dbl.size(-1) - 2 * state_dim
        delta_raw = x_dbl[..., :dt_rank]                   # (B, T, dt_rank)
        B_t = x_dbl[..., dt_rank: dt_rank + state_dim]    # (B, T, state_dim)
        C_t = x_dbl[..., dt_rank + state_dim:]            # (B, T, state_dim)

        # Δ: (B, T, d_inner), softplus for positivity
        delta = F.softplus(self.dt_proj(delta_raw))        # (B, T, d_inner)

        # A: (d_inner, state_dim), diagonal ← negative to ensure stability
        A = -torch.exp(self.A_log)                         # (d_inner, state_dim)

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
    """Mamba forecaster using selective state-space blocks (Gu & Dao 2023)."""

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
    ) -> None:
        super().__init__()
        self.station_embedding = nn.Embedding(station_count, embedding_dim)
        self.revin = ReversibleInstanceNorm(sequence_input_dim)
        self.input_projection = nn.Linear(sequence_input_dim, model_dim)
        self.blocks = nn.ModuleList(
            [
                _MambaBlock(model_dim, state_dim=state_dim, kernel_size=kernel_size)
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


class ResidualHydroHybridForecaster(nn.Module):
    """A new conv-recurrent-attention hybrid tailored to hydrological forecasting."""

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
