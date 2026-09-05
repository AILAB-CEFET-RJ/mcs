"""Regenera as figuras descritivas do Capítulo 4 com o suporte final."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT.parent / "dissertacao" / "figures"


def load_epidemiological_support(
    sinan_file: str, start: str, end: str
) -> pd.DataFrame:
    """Carrega os casos retidos no recorte temporal do SINAN."""
    sinan = pd.read_parquet(
        ROOT / "data" / "processed" / "sinan" / sinan_file,
        columns=["ID_UNIDADE", "DT_NOTIFIC", "CASES"],
    )
    sinan["DT_NOTIFIC"] = pd.to_datetime(sinan["DT_NOTIFIC"])
    sinan["ID_UNIDADE"] = sinan["ID_UNIDADE"].astype(str).str.zfill(7)
    sinan = sinan[
        (sinan.DT_NOTIFIC >= pd.Timestamp(start))
        & (sinan.DT_NOTIFIC <= pd.Timestamp(end))
    ]
    return sinan.rename(columns={"DT_NOTIFIC": "DATE"})


def plot_series(df: pd.DataFrame, title: str, output: str) -> None:
    series = df.groupby("DATE", sort=True)["CASES"].sum()
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(series.index, series.values, color="#2f5597", linewidth=1.25)
    ax.set_title(title)
    ax.set_xlabel("Data do alvo")
    ax.set_ylabel("Casos retidos")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.text(
        0.99,
        0.97,
        f"Total no período: {series.sum():,.0f}".replace(",", "."),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="#444444",
    )
    fig.tight_layout()
    fig.savefig(FIGURES / output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_natal_2016(df: pd.DataFrame) -> None:
    series = df[(df.DATE >= "2016-01-01") & (df.DATE < "2017-01-01")]
    weekly = series.set_index("DATE").groupby(pd.Grouper(freq="W-MON"))["CASES"].sum()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(weekly.index, weekly.values, color="#2f5597", marker="o", markersize=3)
    ax.set_title("Casos semanais nas unidades elegíveis de Natal — 2016")
    ax.set_xlabel("Semana")
    ax.set_ylabel("Casos retidos")
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(FIGURES / "casos_natal_2016.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def weekly_view(df: pd.DataFrame) -> pd.DataFrame:
    weekly = df.copy()
    weekly["DATE"] = weekly.DATE.dt.to_period("W").apply(lambda value: value.start_time)
    return weekly.groupby(["ID_UNIDADE", "DATE"], as_index=False)["CASES"].sum()


def restrict_to_modeled_support(
    df: pd.DataFrame, cnes_file: str, mapping_file: str | None = None
) -> pd.DataFrame:
    """Use exactly the CNES identifiers eligible for the modeled target."""
    totals = df.groupby("ID_UNIDADE", sort=False)["CASES"].sum()
    positive = set(totals[totals > 0].index)
    if mapping_file is not None:
        mapping = pd.read_parquet(ROOT / mapping_file)
        eligible = set(mapping.CNES.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(7))
    else:
        cnes = pd.read_parquet(ROOT / "data" / "processed" / "cnes" / cnes_file)
        cnes["CNES"] = cnes.CNES.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(7)
        cnes["LAT"] = pd.to_numeric(cnes.LAT, errors="coerce")
        cnes["LNG"] = pd.to_numeric(cnes.LNG, errors="coerce")
        eligible = set(cnes.dropna(subset=["LAT", "LNG"]).CNES) & positive
    return df[df.ID_UNIDADE.isin(eligible)].copy()


def plot_zero_percentages(datasets: dict[str, pd.DataFrame]) -> None:
    regions = ["Rio de Janeiro", "Natal"]
    daily = [
        100 * (datasets["RJ"].CASES == 0).mean(),
        100 * (datasets["RN"].CASES == 0).mean(),
    ]
    weekly = [
        100 * (weekly_view(datasets["RJ"]).CASES == 0).mean(),
        100 * (weekly_view(datasets["RN"]).CASES == 0).mean(),
    ]
    x = np.arange(len(regions))
    width = 0.34
    fig, ax = plt.subplots(figsize=(9, 5.2))
    bars_daily = ax.bar(x - width / 2, daily, width, label="Diário", color="#4e79a7")
    bars_weekly = ax.bar(x + width / 2, weekly, width, label="Semanal", color="#f28e2c")
    ax.bar_label(bars_daily, fmt="%.1f%%", padding=3)
    ax.bar_label(bars_weekly, fmt="%.1f%%", padding=3)
    ax.set_ylabel("Observações com zero casos (%)")
    ax.set_title("Proporção de zeros na base epidemiológica")
    ax.set_xticks(x, regions)
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "percentual_zeros.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_units(df: pd.DataFrame, cnes_file: str, title: str, output: str) -> None:
    # O mapa caracteriza unidades que efetivamente notificaram ao menos um caso
    # retido no período, excluindo séries preenchidas apenas por zeros.
    totals = df.groupby("ID_UNIDADE", sort=False)["CASES"].sum()
    units = set(totals[totals > 0].index)
    cnes = pd.read_parquet(ROOT / "data" / "processed" / "cnes" / cnes_file)
    cnes["CNES"] = cnes["CNES"].astype(str).str.zfill(7)
    cnes["LAT"] = pd.to_numeric(cnes["LAT"], errors="coerce")
    cnes["LNG"] = pd.to_numeric(cnes["LNG"], errors="coerce")
    selected = (
        cnes[cnes.CNES.isin(units)]
        .dropna(subset=["LAT", "LNG"])
        .drop_duplicates("CNES")
    )
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.scatter(
        selected.LNG,
        selected.LAT,
        s=14 if len(selected) > 100 else 32,
        color="#2f5597",
        alpha=0.7,
        edgecolors="white",
        linewidths=0.25,
    )
    ax.set_title(f"{title} (n = {len(selected)})")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, linestyle=":", alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(FIGURES / output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    datasets = {
        "RJ": load_epidemiological_support(
            "DENG33.parquet", "2014-01-01", "2023-12-31"
        ),
        "RN": load_epidemiological_support(
            "DENG240810.parquet", "2015-01-01", "2019-12-31"
        ),
    }
    modeled = {
        "RJ": restrict_to_modeled_support(
            datasets["RJ"], "STRJ2401.parquet",
            "data/processed/epidemiology/RJ_STATE_2km_WEEKLY_2014_2022/cnes_cell_mapping.parquet",
        ),
        "RN": restrict_to_modeled_support(datasets["RN"], "STRN2412.parquet"),
    }
    plot_series(
        modeled["RJ"],
        "Casos diários retidos no estado do Rio de Janeiro",
        "casos_rj.png",
    )
    plot_series(
        modeled["RN"],
        "Casos diários retidos em Natal",
        "casos_natal.png",
    )
    plot_natal_2016(modeled["RN"])
    plot_zero_percentages(modeled)
    plot_units(
        datasets["RJ"],
        "STRJ2401.parquet",
        "Unidades notificadoras do RJ com coordenadas disponíveis",
        "mapa_unidades_saude_rj.png",
    )
    plot_units(
        datasets["RN"],
        "STRN2412.parquet",
        "Unidades notificadoras de Natal com coordenadas disponíveis",
        "mapa_unidades_saude_rn.png",
    )
    print(f"Figuras atualizadas em: {FIGURES}")


if __name__ == "__main__":
    main()
