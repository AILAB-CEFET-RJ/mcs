"""Baixa e materializa os dados do SINAN usados no holdout temporal.

Compatível com versões do PySUS que retornam um DataFrame ou uma lista de
arquivos Parquet. Este script deve ser executado no Linux/WSL.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd
import pysus
import pyarrow.parquet as pq
from pysus import sinan


DEFAULT_OUTPUT_DIR = Path(
    "/mnt/c/Users/ramon.garcia/Documents/Claude/Projects/SLR/"
    "arboseer/data/raw/holdout_2024_2025/sinan"
)


def materialize_result(result: object, destination: Path) -> None:
    """Grava em um único Parquet o resultado retornado pelo PySUS."""
    if isinstance(result, pd.DataFrame):
        result.to_parquet(destination, index=False)
        return

    if not isinstance(result, list):
        raise RuntimeError(f"Tipo inesperado retornado pelo PySUS: {type(result)!r}")

    if not result:
        raise RuntimeError("O PySUS retornou uma lista vazia.")

    paths = [Path(str(item)).expanduser() for item in result]
    print("Arquivos retornados pelo PySUS:")
    for path in paths:
        print(f"  - {path}")

    missing = [path for path in paths if not path.exists()]
    if missing:
        details = "\n".join(f"  - {path}" for path in missing)
        raise RuntimeError(f"O PySUS retornou caminhos inexistentes:\n{details}")

    # Evita carregar milhões de registros na memória quando há somente um
    # arquivo físico pronto para uso.
    if len(paths) == 1 and paths[0].is_file():
        if paths[0].resolve() != destination.resolve():
            shutil.copy2(paths[0], destination)
        return

    frames = [pd.read_parquet(path) for path in paths]
    pd.concat(frames, ignore_index=True).to_parquet(destination, index=False)


def download_year(year: int, output_dir: Path) -> Path:
    destination = output_dir / f"DENGBR{year % 100:02d}.parquet"
    print(f"\nBaixando SINAN/Dengue de {year}...")
    result = sinan(disease="DENG", year=year)
    print(f"Tipo retornado: {type(result)}")

    materialize_result(result, destination)

    # Validação pelos metadados, sem carregar os registros na memória.
    metadata = pq.ParquetFile(destination).metadata
    size_mb = destination.stat().st_size / 1024**2
    print(f"Arquivo criado: {destination}")
    print(f"Registros: {metadata.num_rows:,}")
    print(f"Colunas: {metadata.num_columns}")
    print(f"Tamanho: {size_mb:.2f} MB")
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=[2024],
        help="Anos a baixar (padrão: 2024).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Diretório de saída (padrão: {DEFAULT_OUTPUT_DIR}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"PySUS: {getattr(pysus, '__version__', 'desconhecida')}")
    for year in args.years:
        download_year(year, args.output_dir)


if __name__ == "__main__":
    main()
