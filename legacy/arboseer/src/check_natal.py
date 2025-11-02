import xarray as xr
import numpy as np
from math import radians, sin, cos, asin
import glob
import os

# Coordenadas de Natal-RN
NATAL_LAT = -5.79448
NATAL_LON = -35.211

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return 2 * R * asin(np.sqrt(a))

def get_coord_names(ds):
    lat_candidates = ["latitude", "lat", "nav_lat", "LAT"]
    lon_candidates = ["longitude", "lon", "nav_lon", "LON"]
    lat_name = next((c for c in lat_candidates if c in ds.coords), None)
    lon_name = next((c for c in lon_candidates if c in ds.coords), None)
    if not lat_name or not lon_name:
        raise KeyError("Não achei coordenadas de latitude/longitude no dataset.")
    return lat_name, lon_name

def to_lon180(lon_values):
    lon = np.asarray(lon_values)
    # Se estiver em 0..360, converte para -180..180
    if lon.min() >= 0 and lon.max() <= 360:
        lon = ((lon + 180) % 360) - 180
    return lon

def open_dataset_smart(path_or_pattern):
    """Abre 1 arquivo (open_dataset) ou vários (open_mfdataset) sem passar args inválidos."""
    if isinstance(path_or_pattern, (list, tuple)):
        paths = list(path_or_pattern)
    elif any(ch in str(path_or_pattern) for ch in "*?[]"):
        paths = sorted(glob.glob(path_or_pattern))
    else:
        paths = [str(path_or_pattern)]

    if not paths:
        raise FileNotFoundError(f"Nenhum arquivo encontrado para: {path_or_pattern}")

    if len(paths) == 1:
        # Arquivo único: NÃO passar 'combine'
        return xr.open_dataset(paths[0])
    else:
        # Multi-arquivos: aqui sim usamos 'combine'
        return xr.open_mfdataset(paths, combine="by_coords")

def verify_era5_covers_natal(path_or_pattern):
    ds = open_dataset_smart(path_or_pattern)

    lat_name, lon_name = get_coord_names(ds)
    lats = ds[lat_name].values
    lons = ds[lon_name].values
    lons180 = to_lon180(lons)

    # Extensão (funciona para 1D; em 2D também funciona pegando min/max)
    lat_min, lat_max = float(np.min(lats)), float(np.max(lats))
    lon_min, lon_max = float(np.min(lons180)), float(np.max(lons180))

    # Resolução aproximada se for 1D (comum no ERA5)
    if lats.ndim == 1 and lats.size > 1:
        dlat = float(np.abs(np.diff(lats)).mean())
    else:
        dlat = float("nan")
    if np.ndim(lons180) == 1 and lons180.size > 1:
        dlon = float(np.abs(np.diff(lons180)).mean())
    else:
        dlon = float("nan")

    print(f"Lat range: {lat_min:.4f} .. {lat_max:.4f} (Δ≈{dlat:.4f}°)")
    print(f"Lon range: {lon_min:.4f} .. {lon_max:.4f} (Δ≈{dlon:.4f}°)  [ajustado p/ -180..180]")
    inside = (min(lat_min, lat_max) <= NATAL_LAT <= max(lat_min, lat_max)) and \
             (min(lon_min, lon_max) <= NATAL_LON <= max(lon_min, lon_max))
    print(f"Natal dentro do bounding box? {'SIM' if inside else 'NÃO'}")

    # Dataset com longitudes ajustadas (só para seleção)
    ds_adj = ds.assign_coords({lon_name: (ds[lon_name].dims, lons180)})

    # Ponto de grade mais próximo
    pt = ds_adj.sel({lat_name: NATAL_LAT, lon_name: NATAL_LON}, method="nearest")
    lat_g = float(pt[lat_name].values)
    lon_g = float(pt[lon_name].values)
    dist_km = haversine_km(NATAL_LAT, NATAL_LON, lat_g, lon_g)
    print(f"Ponto de grade mais próximo: lat={lat_g:.4f}, lon={lon_g:.4f}  (~{dist_km:.1f} km de Natal)")

    # Variáveis necessárias
    needed = ["t2m", "d2m", "tp"]
    missing = [v for v in needed if v not in ds.data_vars]
    if missing:
        print(f"Atenção: variáveis faltando: {missing}")
    else:
        print("Variáveis presentes: t2m, d2m, tp")

    return {
        "inside_bbox": inside,
        "nearest_gridpoint": {"lat": lat_g, "lon": lon_g, "dist_km": dist_km},
        "lat_range": (lat_min, lat_max),
        "lon_range": (lon_min, lon_max),
        "res_deg": (dlat, dlon),
        "missing_vars": missing,
    }

# Exemplos:
# info = verify_era5_covers_natal("data/raw/era5_natal/ERA5_Natal_2016_01.nc")
# info = verify_era5_covers_natal("data/raw/era5_natal/ERA5_Natal_2016_*.nc")
info = verify_era5_covers_natal([
    "data/raw/era5/RJ_1997_2024.nc"])

print(info)