import os
import glob
import math
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd


def circle_latlon(center_lat, center_lon, radius_km, n=360):
    R = 6371.0088
    lat0 = math.radians(center_lat)
    lon0 = math.radians(center_lon)
    ang = radius_km / R
    bearings = np.linspace(0, 2 * math.pi, n, endpoint=True)
    lats, lons = [], []
    for b in bearings:
        lat = math.asin(
            math.sin(lat0) * math.cos(ang) +
            math.cos(lat0) * math.sin(ang) * math.cos(b)
        )
        lon = lon0 + math.atan2(
            math.sin(b) * math.sin(ang) * math.cos(lat0),
            math.cos(ang) - math.sin(lat0) * math.sin(lat)
        )
        lats.append(math.degrees(lat))
        lons.append(math.degrees(lon))
    return np.array(lats), np.array(lons)


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    lat1 = np.radians(lat1); lon1 = np.radians(lon1)
    lat2 = np.radians(lat2); lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def assign_level(dist_km, edges_km):
    e1, e2, e3 = edges_km
    if dist_km <= e1: return 1
    if dist_km <= e2: return 2
    if dist_km <= e3: return 3
    return 4


def km_to_deg_lat(km):
    return km / 110.574


def km_to_deg_lon(km, lat_deg):
    return km / (111.320 * math.cos(math.radians(lat_deg)))


def find_shp_in_folder(folder):
    shps = glob.glob(os.path.join(folder, "*.shp"))
    if not shps:
        raise FileNotFoundError(f"Nenhum .shp encontrado em: {folder}")
    return shps[0]


def main(
    config_path=r"config/build_cluster.yaml",
    bairros_folder=r"data/shapes/Limite_de_Bairros",
    out_png="cluster_zoom_bairros.png",
    dpi=700,
    figsize=(14, 14),
    plot_all_inmet_stations=False,   # no zoom, costuma poluir; ligue se quiser
):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    center_lat = float(cfg["cluster"]["station"]["lat"])
    center_lon = float(cfg["cluster"]["station"]["lon"])
    edges = [float(x) for x in cfg["cluster"]["level_edges_km"]]
    max_r = max(edges)

    sinan_path = cfg["paths"]["sinan"]
    cnes_path = cfg["paths"]["cnes"]
    agg_path = cfg["paths"]["inmet_aggregated_parquet"]
    station_id = str(cfg["paths"]["inmet_station_id"])

    pre = cfg.get("preproc", {})
    min_date = pre.get("MIN_DATE")
    max_date = pre.get("MAX_DATE")

    # --- bairros shapefile ---
    shp_path = find_shp_in_folder(bairros_folder)
    bairros = gpd.read_file(shp_path)

    if bairros.crs is None:
        # comum em dados no Brasil
        bairros = bairros.set_crs(epsg=4674)
    bairros = bairros.to_crs(epsg=4326)

    # --- unidades (SINAN + CNES) ---
    sinan = pd.read_parquet(sinan_path)
    sinan["DT_NOTIFIC"] = pd.to_datetime(sinan["DT_NOTIFIC"], errors="coerce")
    sinan["ID_UNIDADE"] = sinan["ID_UNIDADE"].astype(str)

    if min_date:
        sinan = sinan[sinan["DT_NOTIFIC"] >= pd.to_datetime(min_date)]
    if max_date:
        sinan = sinan[sinan["DT_NOTIFIC"] <= pd.to_datetime(max_date)]

    cnes = pd.read_parquet(cnes_path)
    cnes["CNES"] = cnes["CNES"].astype(str)
    cnes["LAT"] = pd.to_numeric(cnes["LAT"], errors="coerce")
    cnes["LNG"] = pd.to_numeric(cnes["LNG"], errors="coerce")
    cnes_xy = cnes[["CNES", "LAT", "LNG"]].drop_duplicates("CNES").dropna()

    cases_sum = sinan.groupby("ID_UNIDADE", as_index=False)["CASES"].sum().rename(columns={"CASES": "TOTAL_CASES"})
    units = cases_sum.merge(cnes_xy, left_on="ID_UNIDADE", right_on="CNES", how="left").drop(columns=["CNES"])
    units = units.dropna(subset=["LAT", "LNG"])

    units["DIST_KM"] = haversine_km(units["LAT"].to_numpy(), units["LNG"].to_numpy(), center_lat, center_lon)
    units["LEVEL"] = units["DIST_KM"].apply(lambda d: assign_level(d, edges))
    units = units[units["LEVEL"].isin([1, 2, 3])].copy()

    # --- INMET stations ---
    inmet = pd.read_parquet(agg_path)
    stations = (
        inmet[["CD_ESTACAO", "VL_LATITUDE", "VL_LONGITUDE"]]
        .dropna()
        .drop_duplicates()
        .copy()
    )
    stations["CD_ESTACAO"] = stations["CD_ESTACAO"].astype(str)
    stations["VL_LATITUDE"] = pd.to_numeric(stations["VL_LATITUDE"], errors="coerce")
    stations["VL_LONGITUDE"] = pd.to_numeric(stations["VL_LONGITUDE"], errors="coerce")
    stations = stations.dropna(subset=["VL_LATITUDE", "VL_LONGITUDE"])

    chosen = stations[stations["CD_ESTACAO"] == station_id]
    chosen_lat = float(chosen["VL_LATITUDE"].iloc[0]) if not chosen.empty else None
    chosen_lon = float(chosen["VL_LONGITUDE"].iloc[0]) if not chosen.empty else None

    # --- plot ---
    fig, ax = plt.subplots(figsize=figsize)

    # contorno de bairros
    bairros.boundary.plot(ax=ax, linewidth=0.6, alpha=0.8, zorder=1)

    # estações (opcional)
    if plot_all_inmet_stations:
        ax.scatter(stations["VL_LONGITUDE"], stations["VL_LATITUDE"],
                   s=10, marker="o", alpha=0.6, zorder=2, label="Estações INMET")

    # unidades (níveis 1..3)
    ax.scatter(units["LNG"], units["LAT"],
               s=22, marker=".", alpha=0.95, zorder=3, label="Unidades (níveis 1–3)")

    # centro do cluster
    ax.scatter([center_lon], [center_lat],
               s=180, marker="x", zorder=6, label="Centro do cluster")

    # estação escolhida
    if chosen_lat is not None:
        ax.scatter([chosen_lon], [chosen_lat],
                   s=240, marker="*", zorder=7, label=f"Estação ({station_id})")

    # círculos (somente contorno)
    circle_colors = ["red", "green", "blue"]
    for rad_km, col in zip(edges, circle_colors):
        lats, lons = circle_latlon(center_lat, center_lon, rad_km)
        ax.plot(lons, lats, linewidth=2.2, color=col, zorder=5, label=f"Raio {rad_km} km")

    # zoom pelo maior raio
    pad_km = max_r * 0.25
    half_lat = km_to_deg_lat(max_r + pad_km)
    half_lon = km_to_deg_lon(max_r + pad_km, center_lat)
    ax.set_xlim(center_lon - half_lon, center_lon + half_lon)
    ax.set_ylim(center_lat - half_lat, center_lat + half_lat)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Rio — Bairros | Zoom {max_r} km | Unidades + Níveis")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    plt.tight_layout()
    plt.savefig(out_png, dpi=dpi)
    print(f"✅ Salvo: {os.path.abspath(out_png)}")
    print(f"✅ Unidades plotadas: {len(units)} | Bairros: {len(bairros)} | Shapefile: {os.path.basename(shp_path)}")


if __name__ == "__main__":
    main()