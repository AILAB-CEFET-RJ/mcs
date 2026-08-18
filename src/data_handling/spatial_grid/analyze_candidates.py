#!/usr/bin/env python3
"""Generate structural grid-selection artifacts without consulting test data."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import box
from shapely.ops import transform

from .selection import build_grid, evaluate_candidate

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("grid_selection")


def _load_territory(path: str | None, bbox: list[float] | None, crs: str):
    project = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform
    if path:
        frame = gpd.read_file(path).to_crs(crs)
        return frame.geometry.union_all()
    if not bbox:
        raise ValueError("Provide --territory or --bbox-wgs84")
    return transform(project, box(*bbox))


def _plot(cells: pd.DataFrame, rows: int, cols: int, out: Path, size_km: float):
    territory = cells["territory_mask"].to_numpy().reshape(rows, cols)
    cnes = cells["cnes_count"].to_numpy().reshape(rows, cols)
    cases = cells["train_cases"].to_numpy().reshape(rows, cols)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for ax, data, title, cmap in [
        (axes[0], territory, "Máscara territorial", "Greys"),
        (axes[1], np.where(territory, cnes, np.nan), "Estabelecimentos CNES", "viridis"),
        (axes[2], np.where(territory, cases, np.nan), "Casos no treino", "magma"),
    ]:
        image = ax.imshow(data, cmap=cmap)
        ax.set_title(title)
        ax.set_xlabel("coluna")
        ax.set_ylabel("linha")
        fig.colorbar(image, ax=ax, fraction=0.046)
    fig.suptitle(f"Grade-alvo epidemiológica — célula {size_km:g} km")
    fig.savefig(out, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].hist(cells.loc[cells.cnes_count > 0, "cnes_count"], bins=30)
    axes[0].set(title="CNES por célula ocupada", xlabel="estabelecimentos", ylabel="células")
    axes[1].hist(cells.loc[cells.train_cases > 0, "train_cases"], bins=30)
    axes[1].set(title="Casos por célula ativa (treino)", xlabel="casos", ylabel="células")
    fig.savefig(out.with_name(out.stem + "_histograms.png"), dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True)
    parser.add_argument("--crs", required=True, help="Projected metric CRS, e.g. EPSG:31983")
    parser.add_argument("--cell-km", required=True, nargs="+", type=float)
    parser.add_argument("--territory", help="Official boundary vector file")
    parser.add_argument(
        "--bbox-wgs84", nargs=4, type=float, metavar=("MINLON", "MINLAT", "MAXLON", "MAXLAT"),
        help="Exploratory fallback only; not an official territorial mask",
    )
    parser.add_argument("--cnes", required=True)
    parser.add_argument("--sinan", required=True)
    parser.add_argument("--train-end", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--channels", type=int, default=45)
    parser.add_argument("--lookback", type=int, default=28)
    parser.add_argument("--horizon", type=int, default=28)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    LOG.info("Saída: %s", out.resolve())
    LOG.info("Carregando território (%s)", args.territory or "envelope exploratório")
    territory = _load_territory(args.territory, args.bbox_wgs84, args.crs)
    LOG.info("Carregando CNES: %s", args.cnes)
    cnes = pd.read_parquet(args.cnes, columns=["CNES", "LAT", "LNG"])
    LOG.info("CNES carregado: %s linhas", f"{len(cnes):,}")
    LOG.info("Carregando SINAN com filtro CASES > 0: %s", args.sinan)
    # The consolidated SINAN files contain millions of CASES=0 rows.  They do
    # not affect occupancy or counts, so push this predicate into the Parquet
    # reader before materialising dates/identifiers in memory.
    try:
        sinan = pd.read_parquet(
            args.sinan,
            columns=["ID_UNIDADE", "DT_NOTIFIC", "CASES"],
            filters=[("CASES", ">", 0)],
        )
    except (TypeError, ValueError):
        # Compatibility fallback for engines/files that cannot push filters.
        sinan = pd.read_parquet(
            args.sinan, columns=["ID_UNIDADE", "DT_NOTIFIC", "CASES"]
        )
        positive = pd.to_numeric(sinan["CASES"], errors="coerce").fillna(0) > 0
        sinan = sinan.loc[positive]
    sinan["DT_NOTIFIC"] = pd.to_datetime(sinan["DT_NOTIFIC"], errors="coerce")
    train = sinan.loc[sinan["DT_NOTIFIC"] <= pd.Timestamp(args.train_end)]
    if train.empty:
        raise ValueError("No training cases on or before --train-end")
    LOG.info(
        "SINAN útil no treino até %s: %s linhas",
        args.train_end,
        f"{len(train):,}",
    )

    summary = []
    for size in args.cell_km:
        LOG.info("Processando candidata de %g km", size)
        grid = build_grid(args.domain, args.crs, size, territory)
        metrics, cells, masks = evaluate_candidate(
            grid, territory, cnes, train, channels=args.channels,
            lookback=args.lookback, horizon=args.horizon, batch_size=args.batch_size,
        )
        metrics["territory_source"] = args.territory or "exploratory_bbox_wgs84"
        label = f"{size:g}km".replace(".", "p")
        cells.to_csv(out / f"cells_{label}.csv", index=False)
        np.savez_compressed(out / f"masks_{label}.npz", **masks)
        (out / f"manifest_{label}.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        _plot(cells, grid.rows, grid.cols, out / f"maps_{label}.png", size)
        summary.append(metrics)
        LOG.info(
            "Candidata %g km concluída: %dx%d, %d células ativas no treino",
            size,
            grid.rows,
            grid.cols,
            metrics["cells_with_train_cases"],
        )

    pd.DataFrame(summary).to_csv(out / "candidate_grids.csv", index=False)
    (out / "README.md").write_text(
        "# Análise estrutural das grades candidatas\n\n"
        "Os indicadores dependentes de casos usam exclusivamente registros com "
        f"`DT_NOTIFIC <= {args.train_end}`. Validação e teste não participam da seleção. "
        "Valores ERA5 reamostrados não constituem observações meteorológicas de maior "
        "resolução. `smoke_epoch_seconds` e a inflação zero célula×tempo permanecem "
        "nulos até a montagem/execução dos tensores por candidata.\n",
        encoding="utf-8",
    )
    LOG.info("Concluído. Resumo: %s", (out / "candidate_grids.csv").resolve())


if __name__ == "__main__":
    main()
