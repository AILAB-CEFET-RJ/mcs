#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STConvS2S grid models for the RJ_E1_T1_clean tensor dataset.

This module contains two implementations:

* STConvS2SGrid: compact local model used as a sanity baseline.
* STConvS2SOfficialRGrid: adaptation of the STConvS2S-R architecture from
  AILAB-CEFET-RJ/stconvs2s for the dengue grid contract.

The official adaptation preserves the temporal-reversed block, factorized
temporal/spatial kernels, and final 3D convolution pattern from the public
STConvS2S implementation, while wrapping the output to our required
``(B, 1, 28, 11, 21)`` target shape.

Original repository:
https://github.com/AILAB-CEFET-RJ/stconvs2s
License: GPL-3.0 in the upstream repository.
Paper: Castro et al., "STConvS2S: Spatiotemporal Convolutional Sequence to
Sequence Network for Weather Forecasting", Neurocomputing, 2021.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[3]
UPSTREAM_STCONVS2S_ROOT = PROJECT_ROOT / "external" / "stconvs2s"
if UPSTREAM_STCONVS2S_ROOT.exists():
    sys.path.insert(0, str(UPSTREAM_STCONVS2S_ROOT))

try:
    from model.stconvs2s import STConvS2S_R as UpstreamSTConvS2S_R
except Exception:  # pragma: no cover - reported at model construction time.
    UpstreamSTConvS2S_R = None


def _triple_kernel(kernel_size: int | tuple[int, int, int] | list[int]) -> list[int]:
    if isinstance(kernel_size, int):
        return [kernel_size, kernel_size, kernel_size]
    values = list(kernel_size)
    if len(values) != 3:
        raise ValueError(f"kernel_size precisa ter 3 valores, recebido {values}")
    return values


class ConvNormAct(nn.Module):
    """3D convolution followed by BatchNorm and LeakyReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int, int],
        padding: tuple[int, int, int],
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class STConvS2SGrid(nn.Module):
    """
    STConvS2S-style model for grid-only dengue forecasting.

    Expected input:
        X: (B, C_in, T_in, H, W)

    Output:
        Y_hat: (B, 1, T_out, H, W)

    For the current E1 daily-28 setup:
        C_in=45, T_in=28, T_out=28, H=11, W=21.
    """

    def __init__(
        self,
        in_channels: int = 45,
        hidden_channels: int = 32,
        out_channels: int = 1,
        t_in: int = 28,
        t_out: int = 28,
        temporal_layers: int = 2,
        spatial_layers: int = 2,
        temporal_kernel: int = 3,
        spatial_kernel: int = 3,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if temporal_kernel % 2 == 0:
            raise ValueError("temporal_kernel precisa ser impar para preservar T com padding simetrico")
        if spatial_kernel % 2 == 0:
            raise ValueError("spatial_kernel precisa ser impar para preservar H/W com padding simetrico")

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.t_in = t_in
        self.t_out = t_out

        temporal_blocks: list[nn.Module] = []
        c = in_channels
        for _ in range(temporal_layers):
            temporal_blocks.append(
                ConvNormAct(
                    c,
                    hidden_channels,
                    kernel_size=(temporal_kernel, 1, 1),
                    padding=(temporal_kernel // 2, 0, 0),
                )
            )
            c = hidden_channels
        self.temporal_block = nn.Sequential(*temporal_blocks)

        spatial_blocks: list[nn.Module] = []
        for _ in range(spatial_layers):
            spatial_blocks.append(
                ConvNormAct(
                    hidden_channels,
                    hidden_channels,
                    kernel_size=(1, spatial_kernel, spatial_kernel),
                    padding=(0, spatial_kernel // 2, spatial_kernel // 2),
                )
            )
        self.spatial_block = nn.Sequential(*spatial_blocks)

        self.dropout = nn.Dropout3d(dropout)
        self.head = nn.Conv3d(hidden_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Esperado X 5D (B,C,T,H,W), recebido shape={tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Esperado C={self.in_channels}, recebido C={x.shape[1]}")
        if x.shape[2] != self.t_in:
            raise ValueError(f"Esperado T_in={self.t_in}, recebido T={x.shape[2]}")

        z = self.temporal_block(x)
        z = self.spatial_block(z)
        z = self.dropout(z)
        y = self.head(z)

        if y.shape[2] != self.t_out:
            y = F.interpolate(
                y,
                size=(self.t_out, y.shape[-2], y.shape[-1]),
                mode="trilinear",
                align_corners=False,
            )
        return y


class OfficialRNet(nn.Module):
    """
    Temporal reversed convolution unit from STConvS2S-R.

    This follows the upstream RNet logic: a temporal convolution plus a special
    correction on the last temporal slice using a kernel-2 convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: list[int],
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.temporal_kernel_value = int(kernel_size[0])
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, bias=bias),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
        )
        self.conv_k2 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=[2, 1, 1], bias=bias),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
        )
        self.pad_k2 = nn.ReplicationPad3d((0, 0, 0, 0, 0, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.temporal_kernel_value == 2:
            return self.conv_k2(x)
        output_conv = self.conv(x)
        output_conv_k2 = self.conv_k2(x[:, :, -self.temporal_kernel_value :, :, :])
        output_conv_k2 = self.pad_k2(output_conv_k2)
        output_conv_part_1 = output_conv[:, :, :-1, :, :]
        output_conv_part_2 = output_conv[:, :, -1:, :, :]
        output_conv_part_2 = output_conv_part_2 - output_conv_k2
        return torch.cat([output_conv_part_1, output_conv_part_2], dim=2)


class OfficialTemporalReversedBlock(nn.Module):
    """Temporal block from STConvS2S-R, adapted to local imports."""

    def __init__(
        self,
        input_size: tuple[int, int, int, int, int],
        num_layers: int,
        kernel_size: int | tuple[int, int, int],
        in_channels: int,
        out_channels: int,
        dropout_rate: float,
        step: int,
    ) -> None:
        super().__init__()
        self.dropout_rate = dropout_rate
        self.input_length = input_size[2]
        self.step = step
        kernel_values = _triple_kernel(kernel_size)
        temporal_kernel_size = [kernel_values[0], 1, 1]

        self.conv_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        intermed_channels = out_channels
        for i in range(num_layers):
            intermed_channels *= 2
            if i == num_layers - 1:
                intermed_channels = out_channels
            self.conv_layers.append(
                OfficialRNet(
                    in_channels,
                    intermed_channels,
                    kernel_size=temporal_kernel_size,
                    bias=False,
                )
            )
            self.dropout_layers.append(nn.Dropout(dropout_rate))
            in_channels = intermed_channels

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        x = torch.flip(input_, [2])
        if self.dropout_rate > 0.0:
            for conv, drop in zip(self.conv_layers, self.dropout_layers):
                x = drop(conv(x))
        else:
            for conv in self.conv_layers:
                x = conv(x)
        return torch.flip(x, [2])


class OfficialSpatialBlock(nn.Module):
    """Spatial block from STConvS2S, adapted to local imports."""

    def __init__(
        self,
        num_layers: int,
        kernel_size: int | tuple[int, int, int],
        in_channels: int,
        out_channels: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()
        self.dropout_rate = dropout_rate
        kernel_values = _triple_kernel(kernel_size)
        spatial_kernel_size = [1, kernel_values[1], kernel_values[2]]
        spatial_padding_value = kernel_values[1] // 2
        spatial_padding = [0, spatial_padding_value, spatial_padding_value]

        self.conv_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        intermed_channels = out_channels
        for i in range(num_layers):
            intermed_channels *= 2
            if i == num_layers - 1:
                intermed_channels = out_channels
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv3d(
                        in_channels,
                        intermed_channels,
                        kernel_size=spatial_kernel_size,
                        padding=spatial_padding,
                        bias=False,
                    ),
                    nn.BatchNorm3d(intermed_channels),
                    nn.LeakyReLU(inplace=True),
                )
            )
            self.dropout_layers.append(nn.Dropout(dropout_rate))
            in_channels = intermed_channels

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        x = input_
        if self.dropout_rate > 0.0:
            for conv, drop in zip(self.conv_layers, self.dropout_layers):
                x = drop(conv(x))
        else:
            for conv in self.conv_layers:
                x = conv(x)
        return x


class STConvS2SOfficialRGrid(nn.Module):
    """
    STConvS2S-R adaptation for dengue grid forecasting.

    Expected input:
        X: (B, C_in, T_in, H, W)

    Output:
        Y_hat: (B, 1, T_out, H, W)

    Notes:
        The original reversed temporal block can shrink the temporal axis when
        kernel_size > 1. The wrapper aligns the output depth to ``t_out`` using
        trilinear interpolation so it can train against the daily-28 target.
    """

    def __init__(
        self,
        in_channels: int = 45,
        hidden_channels: int = 32,
        out_channels: int = 1,
        t_in: int = 28,
        t_out: int = 28,
        height: int = 11,
        width: int = 21,
        num_layers: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.t_in = t_in
        self.t_out = t_out
        self.input_size = (1, in_channels, t_in, height, width)

        self.temporal_block = OfficialTemporalReversedBlock(
            self.input_size,
            num_layers=num_layers,
            kernel_size=kernel_size,
            in_channels=in_channels,
            out_channels=hidden_channels,
            dropout_rate=dropout,
            step=t_out,
        )
        self.spatial_block = OfficialSpatialBlock(
            num_layers=num_layers,
            kernel_size=kernel_size,
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            dropout_rate=dropout,
        )
        padding = kernel_size // 2
        self.conv_final = nn.Conv3d(
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Esperado X 5D (B,C,T,H,W), recebido shape={tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Esperado C={self.in_channels}, recebido C={x.shape[1]}")
        if x.shape[2] != self.t_in:
            raise ValueError(f"Esperado T_in={self.t_in}, recebido T={x.shape[2]}")

        y = self.temporal_block(x)
        y = self.spatial_block(y)
        y = self.conv_final(y)
        if y.shape[2] != self.t_out or y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(
                y,
                size=(self.t_out, x.shape[-2], x.shape[-1]),
                mode="trilinear",
                align_corners=False,
            )
        return y


class STConvS2SOfficialRHurdleGrid(nn.Module):
    """
    STConvS2S-R backbone with two heads for zero-heavy count forecasting.

    Output dict:
        occ_logits: (B, 1, T_out, H, W), logits for P(y > 0)
        log_lambda: (B, 1, T_out, H, W), log Poisson rate for counts
    """

    def __init__(
        self,
        in_channels: int = 45,
        hidden_channels: int = 32,
        t_in: int = 28,
        t_out: int = 28,
        height: int = 11,
        width: int = 21,
        num_layers: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.t_in = t_in
        self.t_out = t_out
        self.input_size = (1, in_channels, t_in, height, width)

        self.temporal_block = OfficialTemporalReversedBlock(
            self.input_size,
            num_layers=num_layers,
            kernel_size=kernel_size,
            in_channels=in_channels,
            out_channels=hidden_channels,
            dropout_rate=dropout,
            step=t_out,
        )
        self.spatial_block = OfficialSpatialBlock(
            num_layers=num_layers,
            kernel_size=kernel_size,
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            dropout_rate=dropout,
        )
        padding = kernel_size // 2
        self.occ_head = nn.Conv3d(
            in_channels=hidden_channels,
            out_channels=1,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.count_head = nn.Conv3d(
            in_channels=hidden_channels,
            out_channels=1,
            kernel_size=kernel_size,
            padding=padding,
        )

    def _align(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if y.shape[2] != self.t_out or y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(
                y,
                size=(self.t_out, x.shape[-2], x.shape[-1]),
                mode="trilinear",
                align_corners=False,
            )
        return y

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 5:
            raise ValueError(f"Esperado X 5D (B,C,T,H,W), recebido shape={tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Esperado C={self.in_channels}, recebido C={x.shape[1]}")
        if x.shape[2] != self.t_in:
            raise ValueError(f"Esperado T_in={self.t_in}, recebido T={x.shape[2]}")

        z = self.temporal_block(x)
        z = self.spatial_block(z)
        occ_logits = self._align(self.occ_head(z), x)
        log_lambda = self._align(self.count_head(z), x)
        return {"occ_logits": occ_logits, "log_lambda": log_lambda}


def _require_upstream_stconvs2s() -> type[nn.Module]:
    if UpstreamSTConvS2S_R is None:
        raise ImportError(
            "Nao foi possivel importar external/stconvs2s/model/stconvs2s.py. "
            "Verifique se o repositorio luhenr/stconvs2s esta clonado em external/stconvs2s."
        )
    return UpstreamSTConvS2S_R


class STConvS2SUpstreamRGrid(nn.Module):
    """
    Adapter that uses the upstream luhenr/stconvs2s STConvS2S_R implementation.

    The upstream model is kept in ``external/stconvs2s``. This wrapper only
    changes the final head and aligns the temporal depth to the dengue target.
    """

    def __init__(
        self,
        in_channels: int = 45,
        hidden_channels: int = 32,
        out_channels: int = 1,
        t_in: int = 28,
        t_out: int = 28,
        height: int = 11,
        width: int = 21,
        num_layers: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.10,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        upstream_cls = _require_upstream_stconvs2s()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.t_in = t_in
        self.t_out = t_out
        self.input_size = (1, in_channels, t_in, height, width)
        self.backbone = upstream_cls(
            self.input_size,
            num_layers,
            hidden_channels,
            kernel_size,
            device,
            dropout,
            step=t_out,
        )
        padding = kernel_size // 2
        self.backbone.stconvs2s_r.conv_final = nn.Conv3d(
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Esperado X 5D (B,C,T,H,W), recebido shape={tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Esperado C={self.in_channels}, recebido C={x.shape[1]}")
        if x.shape[2] != self.t_in:
            raise ValueError(f"Esperado T_in={self.t_in}, recebido T={x.shape[2]}")
        y = self.backbone(x)
        if y.shape[2] != self.t_out or y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(
                y,
                size=(self.t_out, x.shape[-2], x.shape[-1]),
                mode="trilinear",
                align_corners=False,
            )
        return y


class STConvS2SUpstreamRHurdleGrid(nn.Module):
    """
    Upstream STConvS2S_R backbone with local occurrence and count heads.
    """

    def __init__(
        self,
        in_channels: int = 45,
        hidden_channels: int = 32,
        t_in: int = 28,
        t_out: int = 28,
        height: int = 11,
        width: int = 21,
        num_layers: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.10,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        upstream_cls = _require_upstream_stconvs2s()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.t_in = t_in
        self.t_out = t_out
        self.input_size = (1, in_channels, t_in, height, width)
        upstream = upstream_cls(
            self.input_size,
            num_layers,
            hidden_channels,
            kernel_size,
            device,
            dropout,
            step=t_out,
        )
        self.feature_extractor = upstream.stconvs2s_r.conv
        padding = kernel_size // 2
        self.occ_head = nn.Conv3d(hidden_channels, 1, kernel_size=kernel_size, padding=padding)
        self.count_head = nn.Conv3d(hidden_channels, 1, kernel_size=kernel_size, padding=padding)

    def _align(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if y.shape[2] != self.t_out or y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(
                y,
                size=(self.t_out, x.shape[-2], x.shape[-1]),
                mode="trilinear",
                align_corners=False,
            )
        return y

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 5:
            raise ValueError(f"Esperado X 5D (B,C,T,H,W), recebido shape={tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Esperado C={self.in_channels}, recebido C={x.shape[1]}")
        if x.shape[2] != self.t_in:
            raise ValueError(f"Esperado T_in={self.t_in}, recebido T={x.shape[2]}")
        z = self.feature_extractor(x)
        return {
            "occ_logits": self._align(self.occ_head(z), x),
            "log_lambda": self._align(self.count_head(z), x),
        }


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

