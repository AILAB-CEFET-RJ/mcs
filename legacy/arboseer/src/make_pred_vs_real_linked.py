#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_pred_vs_real_linked.py

Liga automaticamente modelos em `models/` aos sidecars em `data/datasets/` com base no nome da pasta.

Estrutura esperada:
- data/datasets/<DATASET_NAME>/{dataset_ids.pickle, dataset_meta.json, dataset.pickle}
- models/<DATASET_NAME>_<RUNID>_<model-id>/predictions.csv

Exemplos:
  # Semanal (soma no tempo) por unidade (padrão):
  python make_pred_vs_real_linked.py models --datasets-root data/datasets --resample W --agg-func sum

  # Agregar entre unidades (média) e depois somar por semana no tempo:
  python make_pred_vs_real_linked.py models --datasets-root data/datasets \
    --units-mode aggregate --units-agg-func mean --resample W --agg-func sum

  # Gerar ambos: por unidade e agregado:
  python make_pred_vs_real_linked.py models --datasets-root data/datasets \
    --units-mode both --units-agg-func sum --resample W --agg-func sum

  # Diário:
  python make_pred_vs_real_linked.py models --datasets-root data/datasets --resample D --agg-func sum

  # Sem datas (índice), agregando a cada 7 passos:
  python make_pred_vs_real_linked.py models --datasets-root data/datasets --bin-size 7 --agg-func sum

  # Salvar CSV enriquecido com DATE/ID_UNIDADE:
  python make_pred_vs_real_linked.py models --datasets-root data/datasets --resample W --write-enriched-csv
"""

import os
import re
import argparse
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CSV_NAME = "predictions.csv"
SIDECAR_NAME = "dataset_ids.pickle"

REAL_CANDS = ["y_true", "real", "y", "y_test", "true", "target", "cases", "cases_real", "observed"]
PRED_CANDS = ["y_pred", "predicted", "prediction", "pred", "yhat", "forecast", "preds"]
X_CANDS    = ["DATE", "date", "Datetime", "datetime", "t_index", "time_index", "ds", "idx", "index"]

# models/<DATASET_NAME>_<RUNID>_<model-id>
MODEL_DIR_RE = re.compile(r"^(?P<ds>.+)_(?P<run>\d+)_.*$")


def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
        for col in df.columns:
            if col.lower() == c.lower():
                return col
    return None


def to_datetime_safe(s):
    try:
        return pd.to_datetime(s, errors="raise", utc=False)
    except Exception:
        return None


def derive_dataset_name(model_dir_name):
    """Extrai <DATASET_NAME> e <RUNID> do nome da pasta do modelo."""
    m = MODEL_DIR_RE.match(model_dir_name)
    if not m:
        return None, None
    return m.group("ds"), m.group("run")


def find_dataset_dir(datasets_root, dataset_name):
    """
    Procura `datasets_root/<dataset_name>`; se não achar, tenta substring match.
    Retorna caminho ou None.
    """
    direct = os.path.join(datasets_root, dataset_name)
    if os.path.isdir(direct):
        return direct

    # fallback por substring
    candidates = [d for d in os.listdir(datasets_root) if os.path.isdir(os.path.join(datasets_root, d))]
    scored = []
    for d in candidates:
        score = 0
        if dataset_name and dataset_name in d:
            score += 2
        if dataset_name and d in dataset_name:
            score += 1
        scored.append((score, d))
    scored.sort(reverse=True)
    if scored and scored[0][0] > 0:
        return os.path.join(datasets_root, scored[0][1])
    return None


def choose_split_by_len(ids_payload, n_rows):
    """
    Determina automaticamente o split cujo comprimento de DATE bate com n_rows.
    Se não houver match único, prefere 'test' se existir.
    """
    matches = []
    for split in ["test", "val", "train"]:
        sid = ids_payload.get(split, {})
        arr = sid.get("DATE", None)
        if arr is not None and len(arr) == n_rows:
            matches.append(split)
    if len(matches) == 1:
        return matches[0]
    if "test" in ids_payload and ids_payload["test"].get("DATE", None) is not None:
        return "test"
    return matches[0] if matches else None


def enrich_with_sidecar(df, dataset_dir, split=None, force=False):
    """
    Injeta DATE/ID_UNIDADE/t_index dos sidecars no DataFrame (por posição).
    Se o CSV já possui e 'force' for False, mantém.
    Retorna (df_enriquecido, split_usado, sidecar_path).
    """
    need_date = "DATE" not in df.columns
    need_unit = "ID_UNIDADE" not in df.columns
    need_tidx = "t_index" not in df.columns
    if not (need_date or need_unit or need_tidx or force):
        return df, split, None

    sidecar_path = os.path.join(dataset_dir, SIDECAR_NAME)
    if not os.path.isfile(sidecar_path):
        print(f"[SIDEcar] Não encontrei {SIDECAR_NAME} em {dataset_dir}")
        return df, split, None

    with open(sidecar_path, "rb") as f:
        ids_payload = pickle.load(f)

    if split is None or split == "auto":
        split = choose_split_by_len(ids_payload, len(df))

    if split not in ids_payload:
        print(f"[SIDEcar] split '{split}' ausente em {sidecar_path}")
        return df, split, sidecar_path

    sid = ids_payload[split]
    m = min(len(df), len(sid.get("DATE", [])))
    if m == 0:
        print(f"[SIDEcar] sidecar para '{split}' vazio ou não compatível.")
        return df, split, sidecar_path

    df = df.iloc[:m].copy()
    if need_date or force:
        if "DATE" in sid and sid["DATE"] is not None:
            df["DATE"] = pd.to_datetime(sid["DATE"][:m])
    if need_unit or force:
        if "ID_UNIDADE" in sid and sid["ID_UNIDADE"] is not None:
            df["ID_UNIDADE"] = sid["ID_UNIDADE"][:m]
    if need_tidx or force:
        if "t_index" in sid and sid["t_index"] is not None:
            df["t_index"] = sid["t_index"][:m]

    print(f"[SIDEcar] OK: {sidecar_path} | split={split} | linhas={m}")
    return df, split, sidecar_path


def aggregate_dataframe(df, x_col, real_col, pred_col, resample_rule=None, bin_size=None, agg_func="sum"):
    """Retorna x, y_real, y_pred, x_label após agregação (temporal/bin)."""
    agg = agg_func
    # com datas: resample
    if resample_rule and x_col and x_col in df.columns:
        dt = to_datetime_safe(df[x_col])
        if dt is not None:
            tmp = df.copy()
            tmp[x_col] = dt
            tmp = tmp.dropna(subset=[real_col, pred_col])
            tmp = tmp.set_index(x_col)
            grouped = tmp[[real_col, pred_col]].resample(resample_rule).agg(agg)
            grouped = grouped.dropna(how="all")
            return grouped.index, grouped[real_col], grouped[pred_col], f"{x_col} (resample={resample_rule})"

    # sem datas: bin numérico
    tmp = df[[real_col, pred_col]].dropna(how="all").reset_index(drop=True)
    if bin_size is None or bin_size <= 1:
        x_vals = np.arange(len(tmp))
        return x_vals, tmp[real_col].to_numpy(), tmp[pred_col].to_numpy(), "index"
    groups = (np.arange(len(tmp)) // int(bin_size))
    grouped = tmp.groupby(groups).agg(agg)
    x_vals = (grouped.index.to_numpy() * int(bin_size)) + (int(bin_size) // 2)
    return x_vals, grouped[real_col].to_numpy(), grouped[pred_col].to_numpy(), f"index (bin_size={bin_size}, agg={agg})"


def plot_per_unit(csv_path, df, folder_label, x_col, real_col, pred_col, args, eff_split, sidecar_path):
    """Gera uma figura por unidade."""
    if "ID_UNIDADE" in df.columns and df["ID_UNIDADE"].nunique() > 1:
        for unit, g in df.groupby("ID_UNIDADE", sort=False):
            x_vals, y_real, y_pred, x_label = aggregate_dataframe(
                g, x_col, real_col, pred_col,
                resample_rule=args.resample,
                bin_size=args.bin_size,
                agg_func=args.agg_func
            )
            if len(y_real) == 0 or len(y_pred) == 0:
                print(f"[IGNORADO] {csv_path} (unidade {unit}) sem dados após agregação.")
                continue
            mae = float(np.mean(np.abs(np.asarray(y_real) - np.asarray(y_pred))))
            rmse = float(np.sqrt(np.mean((np.asarray(y_real) - np.asarray(y_pred))**2)))

            plt.figure(figsize=(12, 5))
            plt.plot(x_vals, y_real, linewidth=1.4, label="Real Values")
            plt.plot(x_vals, y_pred, linewidth=1.2, alpha=0.95, label="y_pred")
            plt.title(f"{folder_label} - Predictions x Real")
            subtitle = f"Unidade: {unit}"
            if eff_split:    subtitle += f" | split: {eff_split}"
            if sidecar_path: subtitle += f" | sidecar: {os.path.basename(sidecar_path)}"
            subtitle += f" | MAE={mae:.3f}  RMSE={rmse:.3f}"
            plt.suptitle(subtitle, y=0.98, fontsize=10)
            plt.xlabel(x_label); plt.ylabel("Cases")
            plt.grid(True, linestyle="--", alpha=0.3); plt.legend(); plt.tight_layout()

            suffix = []
            if args.resample: suffix.append(f"resample-{args.resample}")
            if args.bin_size and args.bin_size > 1: suffix.append(f"bin{args.bin_size}")
            suffix.append(args.agg_func)
            suffix = "_".join(suffix)
            fname = f"predictions_vs_real_{unit}.png" if not suffix else f"predictions_vs_real_{unit}_{suffix}.png"
            out_path = os.path.join(os.path.dirname(csv_path), fname)
            plt.savefig(out_path, dpi=220); plt.close()
            print(f"[OK] {out_path}  (MAE={mae:.3f} RMSE={rmse:.3f})")
    else:
        x_vals, y_real, y_pred, x_label = aggregate_dataframe(
            df, x_col, real_col, pred_col,
            resample_rule=args.resample,
            bin_size=args.bin_size,
            agg_func=args.agg_func
        )
        if len(y_real) == 0 or len(y_pred) == 0:
            print(f"[IGNORADO] {csv_path} sem dados após agregação."); return
        mae = float(np.mean(np.abs(np.asarray(y_real) - np.asarray(y_pred))))
        rmse = float(np.sqrt(np.mean((np.asarray(y_real) - np.asarray(y_pred))**2)))

        plt.figure(figsize=(12, 5))
        plt.plot(x_vals, y_real, linewidth=1.4, label="Real Values")
        plt.plot(x_vals, y_pred, linewidth=1.2, alpha=0.95, label="y_pred")
        plt.title(f"{folder_label} - Predictions x Real")
        subtitle = []
        if eff_split:    subtitle.append(f"split: {eff_split}")
        if sidecar_path: subtitle.append(f"sidecar: {os.path.basename(sidecar_path)}")
        subtitle.append(f"MAE={mae:.3f}  RMSE={rmse:.3f}")
        plt.suptitle(" | ".join(subtitle), y=0.98, fontsize=10)
        plt.xlabel(x_label); plt.ylabel("Cases")
        plt.grid(True, linestyle="--", alpha=0.3); plt.legend(); plt.tight_layout()

        suffix = []
        if args.resample: suffix.append(f"resample-{args.resample}")
        if args.bin_size and args.bin_size > 1: suffix.append(f"bin{args.bin_size}")
        suffix.append(args.agg_func)
        suffix = "_".join(suffix)
        fname = "predictions_vs_real.png" if not suffix else f"predictions_vs_real_{suffix}.png"
        out_path = os.path.join(os.path.dirname(csv_path), fname)
        plt.savefig(out_path, dpi=220); plt.close()
        print(f"[OK] {out_path}  (MAE={mae:.3f} RMSE={rmse:.3f})")


def _agg_name_to_callable(name):
    # pandas aceita strings direto em .agg; manter compatibilidade
    return name


def aggregate_across_units(df, x_col, real_col, pred_col, units_agg="sum", resample_rule=None, time_agg="sum"):
    """
    Agrega entre unidades e, opcionalmente, no tempo.
    - Se houver DATE: 2 etapas -> (1) agrega entre unidades por timestamp; (2) resample temporal.
    - Sem DATE: usa t_index se existir; senão, índice sequencial.
    Retorna x_vals, y_real, y_pred, x_label.
    """
    units_agg = _agg_name_to_callable(units_agg)
    time_agg  = _agg_name_to_callable(time_agg)

    # Caminho com datas
    if x_col and x_col in df.columns:
        dt = to_datetime_safe(df[x_col])
        if dt is not None:
            tmp = df.copy()
            tmp[x_col] = dt
            tmp = tmp.dropna(subset=[real_col, pred_col])
            # (1) agrega entre unidades no timestamp exato
            stage1 = tmp.groupby(x_col)[[real_col, pred_col]].agg(units_agg).sort_index()
            # (2) agrega no tempo via resample (se pedido)
            if resample_rule:
                stage2 = stage1.resample(resample_rule).agg(time_agg).dropna(how="all")
                return stage2.index, stage2[real_col].to_numpy(), stage2[pred_col].to_numpy(), f"{x_col} (units={units_agg}, resample={resample_rule}, time={time_agg})"
            else:
                return stage1.index, stage1[real_col].to_numpy(), stage1[pred_col].to_numpy(), f"{x_col} (units={units_agg})"

    # Sem datas → tentar t_index
    if "t_index" in df.columns:
        tmp = df.dropna(subset=[real_col, pred_col]).copy()
        stage1 = tmp.groupby("t_index")[[real_col, pred_col]].agg(units_agg).sort_index()
        return stage1.index.to_numpy(), stage1[real_col].to_numpy(), stage1[pred_col].to_numpy(), f"t_index (units={units_agg})"

    # Último recurso: índice sequencial
    tmp = df[[real_col, pred_col]].dropna(how="all").reset_index(drop=True)
    x_vals = np.arange(len(tmp))
    return x_vals, tmp[real_col].to_numpy(), tmp[pred_col].to_numpy(), f"index (units={units_agg})"


def plot_aggregate(csv_path, df, folder_label, x_col, real_col, pred_col, args, eff_split, sidecar_path):
    """Gera uma figura agregando todas as unidades."""
    x_vals, y_real, y_pred, x_label = aggregate_across_units(
        df=df,
        x_col=x_col,
        real_col=real_col,
        pred_col=pred_col,
        units_agg=args.units_agg_func,
        resample_rule=args.resample,
        time_agg=args.agg_func
    )
    if len(y_real) == 0 or len(y_pred) == 0:
        print(f"[IGNORADO] {csv_path} (aggregate) sem dados após agregação."); return
    mae = float(np.mean(np.abs(np.asarray(y_real) - np.asarray(y_pred))))
    rmse = float(np.sqrt(np.mean((np.asarray(y_real) - np.asarray(y_pred))**2)))

    plt.figure(figsize=(12, 5))
    plt.plot(x_vals, y_real, linewidth=1.4, label="Real Values (agg)")
    plt.plot(x_vals, y_pred, linewidth=1.2, alpha=0.95, label="y_pred (agg)")
    plt.title(f"{folder_label} - Predictions x Real")
    subtitle = []
    if eff_split:    subtitle.append(f"split: {eff_split}")
    if sidecar_path: subtitle.append(f"sidecar: {os.path.basename(sidecar_path)}")
    subtitle.append(f"units={args.units_agg_func}  time={args.agg_func}  MAE={mae:.3f}  RMSE={rmse:.3f}")
    plt.suptitle(" | ".join(subtitle), y=0.98, fontsize=10)
    plt.xlabel(x_label); plt.ylabel("Cases")
    plt.grid(True, linestyle="--", alpha=0.3); plt.legend(); plt.tight_layout()

    suffix = []
    if args.resample: suffix.append(f"resample-{args.resample}")
    if args.bin_size and args.bin_size > 1: suffix.append(f"bin{args.bin_size}")
    suffix.append(f"units{args.units_agg_func}")
    suffix.append(args.agg_func)
    suffix = "_".join(suffix)
    fname = "predictions_vs_real_UNITS_AGG.png" if not suffix else f"predictions_vs_real_UNITS_AGG_{suffix}.png"
    out_path = os.path.join(os.path.dirname(csv_path), fname)
    plt.savefig(out_path, dpi=220); plt.close()
    print(f"[OK] {out_path}  (MAE={mae:.3f} RMSE={rmse:.3f})")


def process_predictions_csv(csv_path, args):
    model_dir = os.path.basename(os.path.dirname(csv_path))
    dataset_name, run_id = derive_dataset_name(model_dir)
    if not dataset_name:
        print(f"[AVISO] Não consegui extrair DATASET_NAME do diretório '{model_dir}'. Tentarei heurísticas.")
        dataset_name = model_dir  # fallback para substring

    dataset_dir = find_dataset_dir(args.datasets_root, dataset_name)
    if not dataset_dir:
        print(f"[ERRO] Não encontrei diretório do dataset para '{dataset_name}' em {args.datasets_root}.")
        return

    print(f"[MAP] modelo '{model_dir}' → dataset '{os.path.basename(dataset_dir)}' ({dataset_dir})")

    # carrega predictions
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[ERRO] Falha ao ler {csv_path}: {e}")
        return

    # enriquece com sidecar se necessário
    df, eff_split, sidecar_path = enrich_with_sidecar(df, dataset_dir, split="auto", force=args.force_sidecar)

    # detectar colunas
    real_col = pick_col(df, REAL_CANDS)
    pred_col = pick_col(df, PRED_CANDS)
    x_col    = pick_col(df, X_CANDS)
    print(f"[DETECT] {csv_path}")
    print(f"         real={real_col}  pred={pred_col}  x={x_col}  unidades={df['ID_UNIDADE'].nunique() if 'ID_UNIDADE' in df.columns else 'N/A'}  split={eff_split or 'N/A'}")

    if real_col is None or pred_col is None:
        print(f"[IGNORADO] {csv_path} sem colunas detectáveis de real/pred.\n"
              f"Reais: {REAL_CANDS}\nPred:  {PRED_CANDS}")
        return

    # opcional: escrever CSV enriquecido
    if args.write_enriched_csv and (("DATE" in df.columns) or ("ID_UNIDADE" in df.columns)):
        out_enriched = os.path.join(os.path.dirname(csv_path), "predictions_enriched.csv")
        df.to_csv(out_enriched, index=False)
        print(f"[SAVE] CSV enriquecido: {out_enriched}")

    # plot: per-unit e/ou aggregate
    folder_label = os.path.basename(os.path.dirname(csv_path))
    if args.units_mode in ("per-unit", "both"):
        plot_per_unit(csv_path, df, folder_label, x_col, real_col, pred_col, args, eff_split, sidecar_path)
    if args.units_mode in ("aggregate", "both"):
        plot_aggregate(csv_path, df, folder_label, x_col, real_col, pred_col, args, eff_split, sidecar_path)


def main():
    p = argparse.ArgumentParser(description="Gera gráficos Predictions x Real ligando models/ aos sidecars de data/datasets/.")
    p.add_argument("models_root", help="Diretório raiz contendo subpastas de modelos com predictions.csv")
    p.add_argument("--datasets-root", default="data/datasets", help="Raiz onde estão os datasets (com sidecars)")
    p.add_argument("--csv-name", default=CSV_NAME, help="Nome do CSV de predições (padrão: predictions.csv)")
    p.add_argument("--force-sidecar", action="store_true", help="Força usar sidecar mesmo se CSV já tiver DATE/ID_UNIDADE")
    p.add_argument("--resample", default=None, help="Regra de reamostragem (D, W, 7D, M, ...). Use quando existir DATE.")
    p.add_argument("--bin-size", type=int, default=None, help="Tamanho do bin quando não houver DATE (ex.: 7)")
    p.add_argument("--agg-func", default="sum", choices=["sum","mean","median","max","min"], help="Agregação temporal/bin (aplicada após resample/bin)")
    # >>> Controle de agregação por unidade
    p.add_argument("--units-mode", default="per-unit", choices=["per-unit","aggregate","both"],
                   help="Como tratar múltiplas unidades: per-unit (um gráfico por unidade), aggregate (todas agregadas), both (ambos).")
    p.add_argument("--units-agg-func", default="sum", choices=["sum","mean","median","max","min"],
                   help="Agregação entre unidades (padrão: sum).")
    p.add_argument("--write-enriched-csv", action="store_true", help="Salva predictions_enriched.csv com DATE/ID_UNIDADE ao lado do predictions.csv")
    args = p.parse_args()

    any_found = False
    for subdir, _, files in os.walk(args.models_root):
        for f in files:
            if f.lower() == args.csv_name.lower():
                any_found = True
                process_predictions_csv(os.path.join(subdir, f), args)
    if not any_found:
        print(f"[AVISO] Nenhum '{args.csv_name}' encontrado em {args.models_root}")


if __name__ == "__main__":
    main()
