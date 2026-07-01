#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STConvS2S-like grid model for the RJ_E1_T1_clean tensor dataset.

This is a compact PyTorch implementation inspired by the STConvS2S contract:
factorized temporal and spatial 3D convolutions over a regular grid. It is not
a verbatim port of the original CEFET-RJ code; it is the minimal model needed
to validate the E1/T1 grid-only experiment on dengue tensors.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


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


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

