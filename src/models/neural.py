"""Neural forecasting baselines and compact benchmark model variants."""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn


def _build_mlp(
    input_dim: int,
    hidden_dims: Iterable[int],
    output_dim: int,
    *,
    dropout: float,
) -> nn.Sequential:
    hidden_sizes = [int(size) for size in hidden_dims]
    layers: list[nn.Module] = []
    previous_dim = int(input_dim)
    for hidden_dim in hidden_sizes:
        layers.extend(
            [
                nn.Linear(previous_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        )
        previous_dim = hidden_dim
    layers.append(nn.Linear(previous_dim, int(output_dim)))
    return nn.Sequential(*layers)


def _patchify_sequence(sequence_features: torch.Tensor, patch_len: int, patch_stride: int) -> torch.Tensor:
    sequence_length = int(sequence_features.size(1))
    if sequence_length < patch_len:
        padding = patch_len - sequence_length
        sequence_features = F.pad(sequence_features, (0, 0, padding, 0))
    return sequence_features.unfold(dimension=1, size=patch_len, step=patch_stride).contiguous()


class ResidualANNForecaster(nn.Module):
    """A compact MLP that learns corrections on top of a persistence baseline."""

    def __init__(
        self,
        *,
        input_dim: int,
        horizon_count: int,
        station_count: int,
        embedding_dim: int = 8,
        hidden_dims: Iterable[int] = (64, 64),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        hidden_sizes = [int(size) for size in hidden_dims]
        if not hidden_sizes:
            raise ValueError("hidden_dims must contain at least one layer size.")

        self.station_embedding = nn.Embedding(station_count, embedding_dim)
        self.network = _build_mlp(
            input_dim + embedding_dim,
            hidden_sizes,
            horizon_count,
            dropout=dropout,
        )

    def forward(
        self,
        sequence_features: torch.Tensor,
        static_features: torch.Tensor,
        station_index: torch.Tensor,
        baseline: torch.Tensor,
    ) -> torch.Tensor:
        del sequence_features
        model_input = torch.cat([static_features, self.station_embedding(station_index)], dim=1)
        correction = self.network(model_input)
        return baseline + correction


class ResidualBidirectionalLSTMForecaster(nn.Module):
    """A bidirectional LSTM with a dense residual prediction head."""

    def __init__(
        self,
        *,
        sequence_input_dim: int,
        static_input_dim: int,
        horizon_count: int,
        station_count: int,
        embedding_dim: int = 8,
        hidden_size: int = 32,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True,
        head_hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.station_embedding = nn.Embedding(station_count, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=sequence_input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        lstm_output_dim = hidden_size * (2 if bidirectional else 1)
        self.head = _build_mlp(
            lstm_output_dim + static_input_dim + embedding_dim + 1,
            [head_hidden_dim],
            horizon_count,
            dropout=dropout,
        )

    def forward(
        self,
        sequence_features: torch.Tensor,
        static_features: torch.Tensor,
        station_index: torch.Tensor,
        baseline: torch.Tensor,
    ) -> torch.Tensor:
        encoded_sequence, _ = self.lstm(sequence_features)
        sequence_summary = encoded_sequence[:, -1, :]
        head_input = torch.cat(
            [
                sequence_summary,
                static_features,
                self.station_embedding(station_index),
                baseline[:, :1],
            ],
            dim=1,
        )
        correction = self.head(head_input)
        return baseline + correction


class _NHiTSBlock(nn.Module):
    def __init__(
        self,
        *,
        input_length: int,
        static_input_dim: int,
        embedding_dim: int,
        horizon_count: int,
        hidden_dims: Iterable[int],
        pool_kernel: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.pool = nn.AvgPool1d(kernel_size=pool_kernel, stride=pool_kernel, ceil_mode=True)
        pooled_length = math.ceil(input_length / pool_kernel)
        head_input_dim = pooled_length + static_input_dim + embedding_dim + 1
        self.backcast_head = _build_mlp(head_input_dim, hidden_dims, pooled_length, dropout=dropout)
        self.forecast_head = _build_mlp(head_input_dim, hidden_dims, horizon_count, dropout=dropout)

    def forward(
        self,
        residual_history: torch.Tensor,
        static_features: torch.Tensor,
        station_embedding: torch.Tensor,
        baseline_scalar: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = self.pool(residual_history.unsqueeze(1)).squeeze(1)
        head_input = torch.cat([pooled, static_features, station_embedding, baseline_scalar], dim=1)
        backcast_pooled = self.backcast_head(head_input)
        backcast = F.interpolate(
            backcast_pooled.unsqueeze(1),
            size=residual_history.size(1),
            mode="linear",
            align_corners=False,
        ).squeeze(1)
        forecast = self.forecast_head(head_input)
        return backcast, forecast


class ResidualNHiTSForecaster(nn.Module):
    """A compact multi-scale residual MLP inspired by N-HiTS."""

    def __init__(
        self,
        *,
        sequence_length: int,
        static_input_dim: int,
        horizon_count: int,
        station_count: int,
        embedding_dim: int = 8,
        hidden_dims: Iterable[int] = (256, 256),
        pool_kernels: Iterable[int] = (1, 2, 4),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.station_embedding = nn.Embedding(station_count, embedding_dim)
        self.blocks = nn.ModuleList(
            [
                _NHiTSBlock(
                    input_length=sequence_length,
                    static_input_dim=static_input_dim,
                    embedding_dim=embedding_dim,
                    horizon_count=horizon_count,
                    hidden_dims=hidden_dims,
                    pool_kernel=max(1, int(pool_kernel)),
                    dropout=dropout,
                )
                for pool_kernel in pool_kernels
            ]
        )

    def forward(
        self,
        sequence_features: torch.Tensor,
        static_features: torch.Tensor,
        station_index: torch.Tensor,
        baseline: torch.Tensor,
    ) -> torch.Tensor:
        residual_history = sequence_features.squeeze(-1)
        station_embedding = self.station_embedding(station_index)
        baseline_scalar = baseline[:, :1]
        forecast = torch.zeros_like(baseline)

        for block in self.blocks:
            backcast, forecast_update = block(
                residual_history,
                static_features,
                station_embedding,
                baseline_scalar,
            )
            residual_history = residual_history - backcast
            forecast = forecast + forecast_update

        return baseline + forecast


class ResidualPatchTSTForecaster(nn.Module):
    """A small patch Transformer for direct multi-horizon forecasting."""

    def __init__(
        self,
        *,
        sequence_input_dim: int,
        sequence_length: int,
        static_input_dim: int,
        horizon_count: int,
        station_count: int,
        embedding_dim: int = 8,
        patch_len: int = 4,
        patch_stride: int = 2,
        model_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_multiplier: int = 4,
        dropout: float = 0.1,
        head_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.station_embedding = nn.Embedding(station_count, embedding_dim)
        self.patch_len = max(1, int(patch_len))
        self.patch_stride = max(1, int(patch_stride))
        max_sequence_length = max(int(sequence_length), self.patch_len)
        patch_count = 1 + max(0, (max_sequence_length - self.patch_len) // self.patch_stride)

        self.patch_projection = nn.Linear(sequence_input_dim * self.patch_len, model_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, patch_count, model_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=int(num_heads),
            dim_feedforward=model_dim * int(ff_multiplier),
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(num_layers))
        self.head = _build_mlp(
            model_dim + static_input_dim + embedding_dim + 1,
            [head_hidden_dim],
            horizon_count,
            dropout=dropout,
        )

    def forward(
        self,
        sequence_features: torch.Tensor,
        static_features: torch.Tensor,
        station_index: torch.Tensor,
        baseline: torch.Tensor,
    ) -> torch.Tensor:
        patches = _patchify_sequence(sequence_features, self.patch_len, self.patch_stride)
        token_inputs = patches.flatten(start_dim=2)
        tokens = self.patch_projection(token_inputs)
        tokens = tokens + self.position_embedding[:, : tokens.size(1), :]
        encoded_tokens = self.encoder(tokens)
        sequence_summary = encoded_tokens.mean(dim=1)
        head_input = torch.cat(
            [
                sequence_summary,
                static_features,
                self.station_embedding(station_index),
                baseline[:, :1],
            ],
            dim=1,
        )
        correction = self.head(head_input)
        return baseline + correction


class _GatedResidualNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)
        self.gate = nn.Linear(output_dim, output_dim)
        self.skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        transformed = F.elu(self.hidden(x))
        transformed = self.output(self.dropout(transformed))
        gated = torch.sigmoid(self.gate(transformed)) * transformed
        return self.norm(self.skip(x) + gated)


class ResidualTemporalFusionTransformerForecaster(nn.Module):
    """A compact TFT-style model with recurrent encoding and attention fusion."""

    def __init__(
        self,
        *,
        sequence_input_dim: int,
        static_input_dim: int,
        horizon_count: int,
        station_count: int,
        embedding_dim: int = 8,
        hidden_size: int = 64,
        lstm_layers: int = 1,
        attention_heads: int = 4,
        dropout: float = 0.1,
        head_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.station_embedding = nn.Embedding(station_count, embedding_dim)
        static_dim = static_input_dim + embedding_dim
        self.static_encoder = _GatedResidualNetwork(static_dim, hidden_size, hidden_size, dropout)
        self.variable_encoder = nn.Linear(sequence_input_dim, hidden_size)
        self.encoder_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.attention = nn.MultiheadAttention(hidden_size, attention_heads, dropout=dropout, batch_first=True)
        self.fusion = _GatedResidualNetwork(hidden_size + hidden_size + 1, head_hidden_dim, head_hidden_dim, dropout)
        self.head = nn.Linear(head_hidden_dim, horizon_count)

    def forward(
        self,
        sequence_features: torch.Tensor,
        static_features: torch.Tensor,
        station_index: torch.Tensor,
        baseline: torch.Tensor,
    ) -> torch.Tensor:
        station_embedding = self.station_embedding(station_index)
        static_context = self.static_encoder(torch.cat([static_features, station_embedding], dim=1))
        encoded_inputs = self.variable_encoder(sequence_features) + static_context.unsqueeze(1)
        temporal_features, _ = self.encoder_lstm(encoded_inputs)
        query = static_context.unsqueeze(1)
        attended_context, _ = self.attention(query, temporal_features, temporal_features)
        fused = self.fusion(torch.cat([attended_context.squeeze(1), static_context, baseline[:, :1]], dim=1))
        correction = self.head(fused)
        return baseline + correction


class _ResidualXLSTMBlock(nn.Module):
    def __init__(self, model_dim: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.lstm = nn.LSTM(model_dim, hidden_size, batch_first=True)
        self.projection = nn.Linear(hidden_size, model_dim)
        self.gate = nn.Linear(model_dim * 2, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        recurrent_output, _ = self.lstm(x)
        projected = self.dropout(self.projection(recurrent_output))
        gate = torch.sigmoid(self.gate(torch.cat([x, projected], dim=-1)))
        return self.norm(x + (gate * projected))


class ResidualXLSTMForecaster(nn.Module):
    """A residual stacked LSTM benchmark inspired by xLSTM-style scaling."""

    def __init__(
        self,
        *,
        sequence_input_dim: int,
        static_input_dim: int,
        horizon_count: int,
        station_count: int,
        embedding_dim: int = 8,
        model_dim: int = 64,
        hidden_size: int = 64,
        num_blocks: int = 3,
        dropout: float = 0.1,
        head_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.station_embedding = nn.Embedding(station_count, embedding_dim)
        self.input_projection = nn.Linear(sequence_input_dim, model_dim)
        self.blocks = nn.ModuleList(
            [_ResidualXLSTMBlock(model_dim, hidden_size, dropout) for _ in range(int(num_blocks))]
        )
        self.head = _build_mlp(
            model_dim + static_input_dim + embedding_dim + 1,
            [head_hidden_dim],
            horizon_count,
            dropout=dropout,
        )

    def forward(
        self,
        sequence_features: torch.Tensor,
        static_features: torch.Tensor,
        station_index: torch.Tensor,
        baseline: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.input_projection(sequence_features)
        for block in self.blocks:
            hidden = block(hidden)
        sequence_summary = hidden[:, -1, :]
        head_input = torch.cat(
            [
                sequence_summary,
                static_features,
                self.station_embedding(station_index),
                baseline[:, :1],
            ],
            dim=1,
        )
        correction = self.head(head_input)
        return baseline + correction


class _MambaStyleBlock(nn.Module):
    def __init__(self, model_dim: int, expand_factor: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        inner_dim = model_dim * expand_factor
        self.norm = nn.LayerNorm(model_dim)
        self.value_projection = nn.Linear(model_dim, inner_dim)
        self.gate_projection = nn.Linear(model_dim, inner_dim)
        self.out_projection = nn.Linear(inner_dim, model_dim)
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, model_dim),
        )
        self.dropout = nn.Dropout(dropout)
        self.kernel_size = int(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(x)
        values = self.value_projection(normalized)
        gates = torch.sigmoid(self.gate_projection(normalized))
        cumulative_state = torch.cumsum(values, dim=1)
        positions = torch.arange(1, x.size(1) + 1, device=x.device, dtype=x.dtype).view(1, -1, 1)
        mixed = cumulative_state / positions
        if self.kernel_size > 1:
            shifted = torch.roll(mixed, shifts=1, dims=1)
            shifted[:, 0, :] = 0.0
            mixed = mixed + shifted
        x = x + self.dropout(self.out_projection(gates * mixed))
        x = x + self.dropout(self.feed_forward(x))
        return x


class ResidualMambaForecaster(nn.Module):
    """A compact Mamba-inspired sequence forecaster with residual prediction head."""

    def __init__(
        self,
        *,
        sequence_input_dim: int,
        static_input_dim: int,
        horizon_count: int,
        station_count: int,
        embedding_dim: int = 8,
        model_dim: int = 64,
        num_blocks: int = 3,
        expand_factor: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.1,
        head_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.station_embedding = nn.Embedding(station_count, embedding_dim)
        self.input_projection = nn.Linear(sequence_input_dim, model_dim)
        self.blocks = nn.ModuleList(
            [
                _MambaStyleBlock(model_dim, int(expand_factor), int(kernel_size), dropout)
                for _ in range(int(num_blocks))
            ]
        )
        self.head = _build_mlp(
            model_dim + static_input_dim + embedding_dim + 1,
            [head_hidden_dim],
            horizon_count,
            dropout=dropout,
        )

    def forward(
        self,
        sequence_features: torch.Tensor,
        static_features: torch.Tensor,
        station_index: torch.Tensor,
        baseline: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.input_projection(sequence_features)
        for block in self.blocks:
            hidden = block(hidden)
        sequence_summary = hidden[:, -1, :]
        head_input = torch.cat(
            [
                sequence_summary,
                static_features,
                self.station_embedding(station_index),
                baseline[:, :1],
            ],
            dim=1,
        )
        correction = self.head(head_input)
        return baseline + correction
