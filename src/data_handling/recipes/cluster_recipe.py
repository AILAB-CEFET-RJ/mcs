import numpy as np
import pandas as pd


def haversine_km(lat1, lon1, lat2, lon2):
    """Distância Haversine em km (vetorizada)."""
    R = 6371.0088
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return R * (2 * np.arcsin(np.sqrt(a)))


def assign_level(dist_km: float, edges_km):
    """Mapeia uma distância em nível 1..4.

    edges_km = [e1, e2, e3]
      N1: dist <= e1
      N2: e1 < dist <= e2
      N3: e2 < dist <= e3
      N4: dist > e3 (descartado)
    """
    e1, e2, e3 = edges_km
    if dist_km <= e1:
        return 1
    if dist_km <= e2:
        return 2
    if dist_km <= e3:
        return 3
    return 4


class ClusterRecipe:
    """Cria clusters agregando casos por níveis de proximidade à estação."""

    def __init__(self, station_lat, station_lon, level_edges_km, clusters):
        self.station_lat = float(station_lat)
        self.station_lon = float(station_lon)
        self.level_edges_km = list(level_edges_km)
        self.clusters = clusters  # ex: [[1], [1,2], [1,2,3]]

    def make_clusters(self, sinan_df: pd.DataFrame, cnes_df: pd.DataFrame, weekly: bool):
        """Retorna dict {cluster_name: df}.

        df de saída: DT_NOTIFIC, CASES
        """
        df = sinan_df.copy()
        cnes = cnes_df.copy()

        cnes["CNES"] = cnes["CNES"].astype(str)
        df["ID_UNIDADE"] = df["ID_UNIDADE"].astype(str)

        df = df.merge(
            cnes[["CNES", "LAT", "LNG"]].rename(columns={"CNES": "ID_UNIDADE"}),
            on="ID_UNIDADE",
            how="left",
        )
        df = df.dropna(subset=["LAT", "LNG"]).copy()
        df["LAT"] = pd.to_numeric(df["LAT"], errors="coerce")
        df["LNG"] = pd.to_numeric(df["LNG"], errors="coerce")
        df = df.dropna(subset=["LAT", "LNG"])

        # agrega semanal antes de classificar níveis (mantém lat/lon por unidade)
        if weekly:
            df["DT_SEMANA"] = df["DT_NOTIFIC"].dt.to_period("W").apply(lambda r: r.start_time)
            df = (
                df.groupby(["ID_UNIDADE", "DT_SEMANA"])\
                .agg({"CASES": "sum", "LAT": "first", "LNG": "first"})\
                .reset_index()\
                .rename(columns={"DT_SEMANA": "DT_NOTIFIC"})
            )

        # distância e nível por unidade (constante ao longo do tempo)
        # calcula por linha (unidade-data), mas só depende de lat/lon
        dist = haversine_km(df["LAT"].to_numpy(), df["LNG"].to_numpy(), self.station_lat, self.station_lon)
        df["DIST_KM"] = dist
        df["LEVEL"] = [assign_level(x, self.level_edges_km) for x in df["DIST_KM"]]

        # descarta nível 4
        df = df[df["LEVEL"] <= 3].copy()

        # monta clusters solicitados
        out = {}
        for levels in self.clusters:
            levels_set = set(levels)
            label = "_".join([f"L{l}" for l in sorted(levels_set)])
            name = f"cluster_{label}"

            local = df[df["LEVEL"].isin(levels_set)].copy()
            cluster_df = (
                local.groupby("DT_NOTIFIC")["CASES"].sum().reset_index().sort_values("DT_NOTIFIC")
            )
            out[name] = cluster_df

        return out
