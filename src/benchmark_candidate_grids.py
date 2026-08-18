#!/usr/bin/env python3
"""Identical CASEONLY computational smoke benchmark for candidate grids.

This benchmark is not a model-selection experiment. It uses a fixed minimal
architecture, four-week context and a fixed number of batches solely to compare
runtime, memory and numerical stability of spatial resolutions without 2023.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from models.stconv_s2s import STConvS2SGrid, count_parameters


def peak_working_set_bytes():
    if os.name != "nt":
        return None
    class Counters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                    ("PrivateUsage", ctypes.c_size_t)]
    counters = Counters(); counters.cb = ctypes.sizeof(counters)
    handle = ctypes.windll.kernel32.GetCurrentProcess()
    if ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
        return int(counters.PeakWorkingSetSize)
    return None


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def make_batch(cases, indices, lookback, observation, territory, cnes_scaled, mean, std,
               input_channels):
    batch = []
    for target_index in indices:
        history = np.log1p(cases[target_index - lookback:target_index]).astype(np.float32)
        history = np.where(observation[None], (history - mean) / std, 0.0)
        channels = np.stack([
            history,
            np.broadcast_to(cnes_scaled, history.shape),
            np.broadcast_to(territory.astype(np.float32), history.shape),
            np.broadcast_to(observation.astype(np.float32), history.shape),
        ], axis=1)  # T,C,H,W
        if input_channels < 4:
            raise ValueError("input_channels must be at least the four CASEONLY structural channels")
        if input_channels > 4:
            channels = np.concatenate([
                channels,
                np.zeros((lookback, input_channels - 4, *history.shape[1:]), dtype=np.float32),
            ], axis=1)
        batch.append(channels)
    x = torch.from_numpy(np.stack(batch)).permute(0, 2, 1, 3, 4).contiguous()
    y = torch.from_numpy(np.log1p(cases[np.asarray(indices)]).astype(np.float32))[:, None, None]
    return x, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--lookback", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--train-batches", type=int, default=8)
    parser.add_argument("--val-batches", type=int, default=4)
    parser.add_argument("--hidden-channels", type=int, default=4)
    parser.add_argument("--input-channels", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    seed_all(args.seed); torch.set_num_threads(args.threads)
    artifact = Path(args.artifact)
    cases = np.load(artifact / "cases_weekly.npy", mmap_mode="r")
    dates = np.load(artifact / "week_dates.npy")
    masks = np.load(artifact / "masks.npz")
    observation = masks["observation"].astype(bool)
    territory = masks["territory"].astype(bool)
    cnes_count = masks["cnes_count"].astype(np.float32)
    train_indices = np.load(artifact / "train_indices.npy")
    val_indices = np.load(artifact / "validation_indices.npy")
    assert dates.max() <= np.datetime64("2022-12-31")
    train_indices = train_indices[train_indices >= args.lookback]
    val_indices = val_indices[val_indices >= args.lookback]
    train_values = np.log1p(np.asarray(cases[train_indices])[:, observation])
    mean, std = float(train_values.mean()), float(train_values.std()) or 1.0
    cnes_values = np.log1p(cnes_count[observation])
    cnes_mean, cnes_std = float(cnes_values.mean()), float(cnes_values.std()) or 1.0
    cnes_scaled = np.where(observation, (np.log1p(cnes_count) - cnes_mean) / cnes_std, 0).astype(np.float32)

    model = STConvS2SGrid(in_channels=args.input_channels, hidden_channels=args.hidden_channels,
        t_in=args.lookback, t_out=1, temporal_layers=1, spatial_layers=1,
        dropout=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    mask = torch.from_numpy(observation.astype(np.float32))[None, None, None]
    model.train(); losses = []; train_samples = 0
    started = time.perf_counter()
    for batch_index in range(args.train_batches):
        lo = batch_index * args.batch_size; selected = train_indices[lo:lo + args.batch_size]
        if not len(selected): break
        x, y = make_batch(cases, selected, args.lookback, observation, territory,
                          cnes_scaled, mean, std, args.input_channels)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x)
        error = F.mse_loss(prediction, y, reduction="none")
        loss = (error * mask).sum() / (mask.sum() * len(selected))
        if not torch.isfinite(loss): raise RuntimeError("Non-finite training loss")
        loss.backward(); optimizer.step()
        losses.append(float(loss)); train_samples += len(selected)
    train_seconds = time.perf_counter() - started

    model.eval(); val_losses = []; val_samples = 0
    started = time.perf_counter()
    with torch.no_grad():
        for batch_index in range(args.val_batches):
            lo = batch_index * args.batch_size; selected = val_indices[lo:lo + args.batch_size]
            if not len(selected): break
            x, y = make_batch(cases, selected, args.lookback, observation, territory,
                              cnes_scaled, mean, std, args.input_channels)
            prediction = model(x)
            error = F.mse_loss(prediction, y, reduction="none")
            loss = (error * mask).sum() / (mask.sum() * len(selected))
            if not torch.isfinite(loss): raise RuntimeError("Non-finite validation loss")
            val_losses.append(float(loss)); val_samples += len(selected)
    val_seconds = time.perf_counter() - started

    sample_elements = (args.lookback * args.input_channels + 1) * cases.shape[1] * cases.shape[2]
    report = {
        "status": "PASS", "purpose": "computational smoke only; not model selection",
        "artifact": str(artifact), "grid_shape": list(cases.shape[1:]),
        "observation_cells": int(observation.sum()), "weeks": len(dates),
        "last_date": str(dates.max()), "test_2023_loaded": False,
        "lookback_fixed_for_smoke": args.lookback, "batch_size": args.batch_size,
        "input_channels": args.input_channels,
        "train_batches": len(losses), "val_batches": len(val_losses),
        "train_samples": train_samples, "val_samples": val_samples,
        "train_seconds": train_seconds, "val_seconds": val_seconds,
        "train_samples_per_second": train_samples / train_seconds,
        "val_samples_per_second": val_samples / val_seconds,
        "first_train_loss": losses[0], "last_train_loss": losses[-1],
        "mean_val_loss": float(np.mean(val_losses)),
        "finite_losses": True, "parameters": count_parameters(model),
        "estimated_batch_input_target_mib": sample_elements * args.batch_size * 4 / 2**20,
        "peak_process_working_set_mib": (
            peak_working_set_bytes() / 2**20 if peak_working_set_bytes() is not None else None),
        "device": "CPU", "torch_version": torch.__version__, "threads": args.threads,
        "benchmark_seed": args.seed,
    }
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
