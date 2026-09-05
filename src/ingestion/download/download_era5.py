#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Baixa dados ERA5 horários (mês a mês) no formato NetCDF.

Suporta três datasets do Copernicus CDS:
  - reanalysis-era5-land            (alta resolução, só superfície terrestre)
  - reanalysis-era5-single-levels   (todas variáveis de superfície globais)
  - reanalysis-era5-pressure-levels (variáveis 3D em níveis de pressão)

A escolha do dataset é via --dataset (default: era5-land). Quando o dataset
exige pressure_level (apenas pressure-levels), o parâmetro --pressure-levels
é incluído no request automaticamente.

Uso típico para ERA5-Land (default — equivalente à versão anterior):
  python download_era5.py \\
    --start-year 2016 --end-year 2019 \\
    --north -5.45 --south -6.15 --west -35.6 --east -34.85 \\
    --file-prefix NATAL \\
    --out-dir ./ERA5

Uso para ERA5-pressure-levels (níveis 200/700/1000 hPa, variáveis r/t/u/v/w):
  python download_era5.py \\
    --dataset pressure-levels \\
    --start-year 2014 --end-year 2023 \\
    --north -22 --south -23 --west -44 --east -42 \\
    --file-prefix RJ \\
    --out-dir arboseer/legacy/data/reanalysis/ERA5-pressure-levels/monthly_data

Uso para ERA5-single-levels:
  python download_era5.py \\
    --dataset single-levels \\
    --start-year 2014 --end-year 2023 \\
    --north -22 --south -23 --west -44 --east -42 \\
    --file-prefix RJ \\
    --out-dir arboseer/legacy/data/reanalysis/ERA5-single-levels/monthly_data

Observações:
- A ordem que o CDS espera internamente é [N, W, S, E].
- Pula automaticamente arquivos que já existem (use --overwrite para substituir).
- Quando --variables não é especificado, usa o preset adequado ao dataset.
- pressure-levels é mais sensível a timeouts e fila CDS mais longa.
- O CDS atual usa `data_format: netcdf`; use --legacy-format apenas se
  estiver preso a uma instalação antiga do CDS API.
"""

import argparse
import calendar
import time
from pathlib import Path

import cdsapi


# ----------------------------------------------------------------------------
# Presets por dataset
# ----------------------------------------------------------------------------

DATASETS = {
    "era5-land": {
        "cds_name": "reanalysis-era5-land",
        "default_variables": [
            "2m_dewpoint_temperature",
            "2m_temperature",
            "total_precipitation",
        ],
        "needs_pressure_levels": False,
        "needs_product_type": False,
        "base_sleep": 5,
    },
    "single-levels": {
        "cds_name": "reanalysis-era5-single-levels",
        "default_variables": [
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "2m_dewpoint_temperature",
            "2m_temperature",
            "skin_temperature",
            "soil_temperature_level_1",
            "soil_temperature_level_2",
            "soil_temperature_level_3",
            "soil_temperature_level_4",
            "surface_pressure",
            "total_precipitation",
        ],
        "needs_pressure_levels": False,
        "needs_product_type": True,
        "base_sleep": 5,
    },
    "pressure-levels": {
        "cds_name": "reanalysis-era5-pressure-levels",
        "default_variables": [
            "relative_humidity",
            "temperature",
            "u_component_of_wind",
            "v_component_of_wind",
            "vertical_velocity",
        ],
        "needs_pressure_levels": True,
        "needs_product_type": True,
        "base_sleep": 10,
    },
}

DEFAULT_PRESSURE_LEVELS = ["200", "700", "1000"]
ALL_HOURS = [f"{h:02d}:00" for h in range(24)]


def days_in_month(year, month):
    _, num_days = calendar.monthrange(year, month)
    return [f"{d:02d}" for d in range(1, num_days + 1)]


def _validate_netcdf(target_path):
    """Verifica que o arquivo é um NetCDF válido (magic bytes).

    Levanta RuntimeError se for vazio, truncado ou formato desconhecido.
    """
    target_path = Path(target_path)
    if not target_path.exists():
        raise RuntimeError(f"{target_path.name}: arquivo não existe após download")
    size = target_path.stat().st_size
    if size < 16:
        raise RuntimeError(f"{target_path.name}: muito pequeno ({size} bytes) — provavelmente truncado")
    with open(target_path, "rb") as fh:
        magic = fh.read(4)
    if magic.startswith(b"\x89HDF"):
        return  # NetCDF4/HDF5
    if magic.startswith(b"CDF"):
        return  # NetCDF3 classic
    if magic.startswith(b"PK\x03\x04"):
        return  # ZIP — será desempacotado por _unzip_if_needed em seguida
    if magic.startswith(b"GRIB"):
        return  # GRIB também ok (raro mas válido)
    raise RuntimeError(
        f"{target_path.name}: formato desconhecido (magic={magic.hex()}, size={size}). "
        f"Possivelmente HTML de erro do CDS ou download cortado."
    )


def _unzip_if_needed(target_path):
    """Se o CDS devolveu um ZIP em vez de NetCDF, extrai o .nc de dentro
    e substitui o arquivo no lugar.

    O novo CDS (2024+) ocasionalmente ignora `download_format: unarchived`
    para algumas combinações de variáveis/níveis e devolve ZIP."""
    import zipfile

    target_path = Path(target_path)
    if not target_path.exists() or target_path.stat().st_size < 8:
        return

    with open(target_path, "rb") as fh:
        magic = fh.read(4)
    if not magic.startswith(b"PK\x03\x04"):
        return  # já é NetCDF, nada a fazer

    print(f"      [UNZIP] {target_path.name} veio como ZIP — extraindo NetCDF interno...")
    with zipfile.ZipFile(target_path) as zf:
        nc_members = [n for n in zf.namelist() if n.lower().endswith(".nc")]
        if not nc_members:
            raise RuntimeError(
                f"ZIP {target_path.name} não contém arquivo .nc. "
                f"Conteúdo: {zf.namelist()}"
            )
        # Extrai o primeiro .nc para um path temporário
        member = nc_members[0]
        tmp_path = target_path.with_suffix(".nc.tmp")
        with zf.open(member) as src, open(tmp_path, "wb") as dst:
            dst.write(src.read())

    # Substitui o ZIP pelo NetCDF extraído
    target_path.unlink()
    tmp_path.replace(target_path)
    print(f"      [UNZIP OK] {target_path.name} ({target_path.stat().st_size / 1e6:.2f} MB)")


def _cleanup_failed_download(target_path):
    """Remove arquivos vazios ou respostas XML/HTML deixadas por download falho."""
    target_path = Path(target_path)
    if not target_path.exists():
        return
    try:
        size = target_path.stat().st_size
        with open(target_path, "rb") as fh:
            prefix = fh.read(128).lstrip()
        if size == 0 or prefix.startswith((b"<?xml", b"<Error", b"<!DOCTYPE", b"<html")):
            target_path.unlink()
    except OSError:
        pass


def _existing_file_is_valid(target_path):
    """Retorna True se um arquivo já existente parece um resultado aproveitável."""
    try:
        _validate_netcdf(target_path)
        _unzip_if_needed(target_path)
        return True
    except Exception as e:
        print(f"  [WARN] Arquivo existente inválido será rebaixado: {Path(target_path).name} ({e})")
        _cleanup_failed_download(target_path)
        return False


def _looks_like_zero_byte_download(error):
    message = str(error).lower()
    return "downloaded 0 byte" in message or "0 byte(s)" in message


def safe_retrieve(
    client,
    dataset,
    request,
    target_path,
    max_retries=5,
    base_sleep=5,
    early_fail_zero_download=False,
):
    """retrieve com tentativas e recuo exponencial.

    Depois do download bem-sucedido, verifica se o CDS devolveu ZIP em vez
    de NetCDF e extrai automaticamente.
    """
    attempt = 0
    while True:
        try:
            client.retrieve(dataset, request, target=str(target_path))
            _validate_netcdf(target_path)
            _unzip_if_needed(target_path)
            return
        except Exception as e:
            _cleanup_failed_download(target_path)
            attempt += 1
            if early_fail_zero_download and attempt >= 2 and _looks_like_zero_byte_download(e):
                raise
            if attempt > max_retries:
                raise
            sleep_s = base_sleep * (2 ** (attempt - 1))
            print(
                f"[WARN] Falha ao baixar ({e}). Tentando novamente em {sleep_s}s "
                f"({attempt}/{max_retries})..."
            )
            time.sleep(sleep_s)


def retrieve_with_fallback(client, dataset, request, target_path, dataset_cfg, max_retries=5):
    """Baixa um request e contorna cache quebrado do CDS para pressure-levels."""
    try:
        safe_retrieve(
            client,
            dataset,
            request,
            target_path,
            max_retries=max_retries,
            base_sleep=dataset_cfg["base_sleep"],
            early_fail_zero_download=True,
        )
        return
    except Exception as e:
        pressure_levels = request.get("pressure_level") or []
        if (
            not dataset_cfg["needs_pressure_levels"]
            or len(pressure_levels) <= 1
            or not _looks_like_zero_byte_download(e)
        ):
            raise

    print(
        "      [FALLBACK] O CDS serviu um objeto de cache inválido. "
        "Tentando baixar por pressure_level..."
    )
    retrieve_pressure_levels_separately(
        client,
        dataset,
        request,
        target_path,
        pressure_levels,
        max_retries=max_retries,
        base_sleep=dataset_cfg["base_sleep"],
    )


def retrieve_pressure_levels_separately(
    client,
    dataset,
    request,
    target_path,
    pressure_levels,
    max_retries=5,
    base_sleep=10,
):
    import xarray as xr

    target_path = Path(target_path)
    split_paths = []

    for level in pressure_levels:
        level_request = dict(request)
        level_request["pressure_level"] = [level]
        level_path = target_path.with_name(
            f"{target_path.stem}__plev_{level}{target_path.suffix}"
        )
        split_paths.append(level_path)

        if level_path.exists() and _existing_file_is_valid(level_path):
            print(f"        [SKIP plev] {level_path.name} já existe.")
            continue

        print(f"        [PLEV] Baixando {level_path.name} ...")
        safe_retrieve(
            client,
            dataset,
            level_request,
            level_path,
            max_retries=max_retries,
            base_sleep=base_sleep,
        )

    print(f"        [MERGE plev] Consolidando {len(split_paths)} níveis em {target_path.name} ...")
    datasets = [xr.open_dataset(str(p)) for p in split_paths]
    try:
        merged = xr.concat(datasets, dim="pressure_level")
        if "pressure_level" in merged.coords:
            try:
                merged = merged.sel(pressure_level=[int(p) for p in pressure_levels])
            except Exception:
                merged = merged.sel(pressure_level=pressure_levels)
        merged.to_netcdf(target_path)
    finally:
        for ds in datasets:
            ds.close()

    _validate_netcdf(target_path)
    for split_path in split_paths:
        try:
            split_path.unlink()
        except OSError:
            pass


def build_request(year, month, area, variables, dataset_cfg, pressure_levels, use_legacy_format=False):
    """Monta o dicionário de request adaptado ao dataset escolhido."""
    req = {
        "variable": variables,
        "year": str(year),
        "month": f"{month:02d}",
        "day": days_in_month(year, month),
        "time": ALL_HOURS,
        "download_format": "unarchived",
        "area": area,
    }
    if use_legacy_format:
        req["format"] = "netcdf"
    else:
        req["data_format"] = "netcdf"
    if dataset_cfg["needs_product_type"]:
        req["product_type"] = "reanalysis"
    if dataset_cfg["needs_pressure_levels"]:
        req["pressure_level"] = pressure_levels
    return req


def main():
    parser = argparse.ArgumentParser(
        description="Baixa ERA5 (Land, single-levels ou pressure-levels) mês a mês"
    )
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()),
        default="era5-land",
        help="Qual variante ERA5 baixar (default: era5-land)",
    )
    parser.add_argument("--start-year", type=int, required=True, help="Ano inicial")
    parser.add_argument("--end-year", type=int, required=True, help="Ano final (inclusive)")
    parser.add_argument("--start-month", type=int, default=1, choices=range(1, 13),
                        help="Mês inicial no primeiro ano (padrão: 1)")
    parser.add_argument("--end-month", type=int, default=12, choices=range(1, 13),
                        help="Mês final no último ano (padrão: 12)")

    parser.add_argument("--north", type=float, required=True, help="Latitude norte (N)")
    parser.add_argument("--south", type=float, required=True, help="Latitude sul (S)")
    parser.add_argument("--east", type=float, required=True, help="Longitude leste (E)")
    parser.add_argument("--west", type=float, required=True, help="Longitude oeste (W)")

    parser.add_argument("--file-prefix", type=str, required=True, help="Prefixo (ex: NATAL, RJ)")
    parser.add_argument("--out-dir", type=str, required=True, help="Diretório de saída")

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebaixar e sobrescrever arquivos existentes",
    )
    parser.add_argument(
        "--variables",
        type=str,
        default=None,
        help="Lista de variáveis separadas por vírgula. Padrão: preset do dataset.",
    )
    parser.add_argument(
        "--pressure-levels",
        type=str,
        default=",".join(DEFAULT_PRESSURE_LEVELS),
        help=(
            "Níveis de pressão (hPa) separados por vírgula. "
            f"Usado só com --dataset pressure-levels. Padrão: {', '.join(DEFAULT_PRESSURE_LEVELS)}"
        ),
    )
    parser.add_argument(
        "--zero-pad-month",
        action="store_true",
        help=(
            "Nomeia arquivos como PREFIX_YYYY_MM.nc (zero-padded). "
            "Por padrão, nomeia como PREFIX_YYYY_M.nc (sem zero-pad), "
            "que é o formato esperado pelo legacy spatiotemporal_builder."
        ),
    )
    parser.add_argument(
        "--per-variable",
        action="store_true",
        help=(
            "Baixa cada variável em uma requisição separada. Necessário para "
            "pressure-levels (cost limit do CDS estoura quando todas variáveis "
            "vão juntas). Arquivos intermediários: PREFIX_YYYY_M__<var>.nc"
        ),
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help=(
            "Quando combinado com --per-variable, NÃO consolida os arquivos "
            "por variável em um único PREFIX_YYYY_M.nc por mês. Útil para inspeção."
        ),
    )
    parser.add_argument(
        "--legacy-format",
        action="store_true",
        help=(
            "Usa o campo antigo 'format=netcdf' em vez de 'data_format=netcdf'. "
            "Só use se uma instalação antiga do cdsapi rejeitar data_format."
        ),
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Número máximo de tentativas por requisição CDS antes de falhar ou acionar fallback.",
    )

    args = parser.parse_args()

    if args.start_year > args.end_year:
        raise ValueError("start-year deve ser menor ou igual a end-year.")
    if args.start_year == args.end_year and args.start_month > args.end_month:
        raise ValueError("start-month deve ser menor ou igual a end-month no mesmo ano")

    dataset_cfg = DATASETS[args.dataset]
    cds_name = dataset_cfg["cds_name"]

    area = [args.north, args.west, args.south, args.east]

    if args.variables is None:
        variables = dataset_cfg["default_variables"]
    else:
        variables = [v.strip() for v in args.variables.replace(";", ",").split(",") if v.strip()]

    pressure_levels = [
        p.strip()
        for p in args.pressure_levels.replace(";", ",").split(",")
        if p.strip()
    ]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset CDS:     {cds_name}")
    print(f"Bbox [N,W,S,E]:  {area}")
    print(f"Variáveis:       {variables}")
    if dataset_cfg["needs_pressure_levels"]:
        print(f"Pressure levels: {pressure_levels} hPa")
    print(f"Período:         {args.start_year} a {args.end_year}")
    print(f"Saída:           {out_dir}")
    print(f"Modo:            {'per-variable' if args.per_variable else 'tudo junto'}")
    print(f"Formato CDS:     {'format=netcdf (legado)' if args.legacy_format else 'data_format=netcdf'}")
    print()

    client = cdsapi.Client()

    months = [
        (y, m)
        for y in range(args.start_year, args.end_year + 1)
        for m in range(1, 13)
        if (y > args.start_year or m >= args.start_month)
        and (y < args.end_year or m <= args.end_month)
    ]
    total_months = len(months)
    completed_months = 0
    errored_files = 0

    for idx, (year, month) in enumerate(months, 1):
        month_part = f"{month:02d}" if args.zero_pad_month else f"{month}"
        consolidated_filename = f"{args.file_prefix}_{year}_{month_part}.nc"
        consolidated_path = out_dir / consolidated_filename

        if consolidated_path.exists() and not args.overwrite and _existing_file_is_valid(consolidated_path):
            print(f"[{idx}/{total_months}] [SKIP mês] {consolidated_filename} já existe.")
            completed_months += 1
            continue

        if not args.per_variable:
            # Modo original: todas as variáveis numa requisição só
            request = build_request(
                year,
                month,
                area,
                variables,
                dataset_cfg,
                pressure_levels,
                use_legacy_format=args.legacy_format,
            )
            print(f"[{idx}/{total_months}] Baixando {consolidated_filename} ...")
            try:
                retrieve_with_fallback(
                    client,
                    cds_name,
                    request,
                    consolidated_path,
                    dataset_cfg,
                    max_retries=args.max_retries,
                )
                if consolidated_path.exists() and consolidated_path.stat().st_size > 0:
                    size_mb = consolidated_path.stat().st_size / 1e6
                    print(f"  [OK] {consolidated_filename} ({size_mb:.1f} MB)")
                    completed_months += 1
                else:
                    raise RuntimeError("Download concluído mas arquivo parece vazio.")
            except Exception as e:
                print(f"  [ERRO] {consolidated_filename}: {e}")
                errored_files += 1
            continue

        # Modo per-variable: uma requisição por variável
        print(f"[{idx}/{total_months}] Mês {year}-{month_part}: {len(variables)} requisições (uma por variável)")
        per_var_files = []
        per_var_errors = 0

        for v_idx, var in enumerate(variables, 1):
            # nome sanitizado da variável (CDS aceita "10m_u_component_of_wind",
            # arquivo no disco usa o mesmo, mas sem espaços)
            safe_var = var.replace(" ", "_")
            var_filename = f"{args.file_prefix}_{year}_{month_part}__{safe_var}.nc"
            var_path = out_dir / var_filename

            if var_path.exists() and not args.overwrite and _existing_file_is_valid(var_path):
                print(f"    [{v_idx}/{len(variables)}] [SKIP] {var_filename} já existe.")
                per_var_files.append(var_path)
                continue

            request = build_request(
                year,
                month,
                area,
                [var],
                dataset_cfg,
                pressure_levels,
                use_legacy_format=args.legacy_format,
            )
            print(f"    [{v_idx}/{len(variables)}] Baixando {var_filename} ...")
            try:
                retrieve_with_fallback(
                    client,
                    cds_name,
                    request,
                    var_path,
                    dataset_cfg,
                    max_retries=args.max_retries,
                )
                if var_path.exists() and var_path.stat().st_size > 0:
                    size_mb = var_path.stat().st_size / 1e6
                    print(f"      [OK] {var_filename} ({size_mb:.1f} MB)")
                    per_var_files.append(var_path)
                else:
                    raise RuntimeError("Download vazio.")
            except Exception as e:
                print(f"      [ERRO] {var_filename}: {e}")
                per_var_errors += 1
                errored_files += 1

        # Merge per-variable em um arquivo único
        if per_var_errors == 0 and len(per_var_files) == len(variables) and not args.no_merge:
            try:
                import xarray as xr
                print(f"    [MERGE] Consolidando {len(per_var_files)} variáveis em {consolidated_filename} ...")
                # Não usar open_mfdataset (exige dask). Abrir cada arquivo
                # individualmente e mesclar com xr.merge — funciona sem dask.
                datasets = [xr.open_dataset(str(p)) for p in per_var_files]
                try:
                    merged = xr.merge(datasets, compat="override")
                    merged.to_netcdf(consolidated_path)
                finally:
                    for d in datasets:
                        d.close()
                size_mb = consolidated_path.stat().st_size / 1e6
                print(f"      [OK] {consolidated_filename} ({size_mb:.1f} MB)")
                completed_months += 1
            except Exception as e:
                print(f"      [ERRO merge] {consolidated_filename}: {e}")
                errored_files += 1
        elif per_var_errors > 0:
            print(f"    [SKIP merge] {per_var_errors} variáveis falharam neste mês.")
        elif args.no_merge:
            print(f"    [SKIP merge] flag --no-merge ativa; arquivos por variável mantidos.")
            completed_months += 1

    print()
    print(f"Concluído: {completed_months}/{total_months} meses OK. Erros em arquivos: {errored_files}")
    if errored_files > 0:
        print("Para retomar arquivos faltantes: rode o mesmo comando — já-existentes são pulados.")


if __name__ == "__main__":
    main()
