#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import xarray as xr

def is_ok(nc_path, engine="netcdf4"):
    try:
        xr.open_dataset(nc_path, engine=engine).close()
        return True
    except Exception:
        return False

def merge_without_dask(input_dir, output_file, engine="netcdf4"):
    p = Path(input_dir)
    files = sorted(p.glob("*.nc"))
    if not files:
        raise FileNotFoundError(f"Nenhum .nc em {p}")

    # filtra corrompidos/inválidos
    good = [f for f in files if is_ok(f, engine=engine)]
    bad = sorted(set(files) - set(good))
    if bad:
        print("[ATENÇÃO] Ignorando arquivos inválidos/corrompidos:")
        for b in bad: print("  -", b.name)

    if not good:
        raise RuntimeError("Todos os arquivos parecem inválidos.")

    # carrega todos e concatena (sem dask)
    dsets = []
    for f in good:
        ds = xr.open_dataset(f, engine=engine)
        # garante ordenação temporal interna (se houver)
        if "time" in ds:
            ds = ds.sortby("time")
        dsets.append(ds)

    merged = xr.concat(dsets, dim="time") if "time" in dsets[0] else xr.merge(dsets)
    if "time" in merged:
        merged = merged.sortby("time")

    # salva
    merged.to_netcdf(output_file)
    print(f"[OK] Merge salvo em: {output_file}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Merge NetCDF sem dask")
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output-file", required=True)
    ap.add_argument("--engine", default="netcdf4", choices=["netcdf4","h5netcdf"])
    args = ap.parse_args()
    merge_without_dask(args.input_dir, args.output_file, engine=args.engine)
