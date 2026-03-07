import pandas as pd


class InmetLegacyAggregatedProvider:
    """Provider que consome o aggregated.parquet gerado pelo legacy.

    O legacy (pasta legacy/arboseer) é responsável por baixar/parsing/limpeza.
    Este provider apenas filtra uma estação e normaliza colunas para o contrato do
    pipeline atual (DT_NOTIFIC + TEM_* + RAIN).

    Colunas esperadas no aggregated.parquet (legacy):
      - CD_ESTACAO
      - DT_MEDICAO
      - TEM_MIN, TEM_MAX, TEM_AVG
      - CHUVA
      - VL_LATITUDE, VL_LONGITUDE (não usadas aqui)
    """

    def __init__(self, aggregated_parquet_path: str, station_id: str):
        self.path = aggregated_parquet_path
        self.station_id = str(station_id)

    def get_series(self, weekly: bool = False) -> pd.DataFrame:
        df = pd.read_parquet(self.path)
        df["CD_ESTACAO"] = df["CD_ESTACAO"].astype(str)
        df = df[df["CD_ESTACAO"] == self.station_id].copy()

        if df.empty:
            raise ValueError(f"Estação {self.station_id} não encontrada em {self.path}")

        df["DT_NOTIFIC"] = pd.to_datetime(df["DT_MEDICAO"], errors="coerce")

        out = pd.DataFrame({
            "DT_NOTIFIC": df["DT_NOTIFIC"],
            "TEM_MIN": pd.to_numeric(df.get("TEM_MIN"), errors="coerce"),
            "TEM_MAX": pd.to_numeric(df.get("TEM_MAX"), errors="coerce"),
            "TEM_AVG": pd.to_numeric(df.get("TEM_AVG"), errors="coerce"),
            "RAIN": pd.to_numeric(df.get("CHUVA"), errors="coerce"),
        }).sort_values("DT_NOTIFIC")

        if weekly:
            out["DT_SEMANA"] = out["DT_NOTIFIC"].dt.to_period("W").apply(lambda r: r.start_time)
            out = (
                out.groupby("DT_SEMANA")
                .agg({
                    "TEM_MIN": "min",
                    "TEM_MAX": "max",
                    "TEM_AVG": "mean",
                    "RAIN": "sum",
                })
                .reset_index()
                .rename(columns={"DT_SEMANA": "DT_NOTIFIC"})
            )

        return out
