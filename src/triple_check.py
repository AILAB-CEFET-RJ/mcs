#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
triple_check.py

Confere consistência entre:
- dataset: data/datasets/<...>/dataset.pickle
- sidecar: data/datasets/<...>/dataset_ids.pickle
- previsões: models/<...>/predictions.csv

Checks:
1) len(X_test) == len(y_test)
2) len(ids_test) == len(X_test)
3) len(predictions) == len(y_test) == len(ids_test)
"""

import os
import sys
import argparse
import pickle
import pandas as pd
import numpy as np

PRED_CANDIDATES = ["y_pred", "predicted", "prediction", "pred", "yhat", "forecast", "preds"]
REAL_CANDIDATES = ["y_true", "real", "y", "y_test", "true", "target", "cases", "observed"]

def load_dataset(dataset_path):
    with open(dataset_path, "rb") as f:
        obj = pickle.load(f)
    # Aceita tuple padrão ou dict
    if isinstance(obj, (list, tuple)) and len(obj) == 6:
        X_train, y_train, X_val, y_val, X_test, y_test = obj
    elif isinstance(obj, dict):
        X_test = obj.get("X_test")
        y_test = obj.get("y_test")
        if X_test is None or y_test is None:
            raise ValueError("Dict de dataset.pickle não contém chaves 'X_test'/'y_test'.")
    else:
        raise ValueError("Formato de dataset.pickle não reconhecido.")
    # Garantir shapes coerentes
    n_x = int(X_test.shape[0]) if hasattr(X_test, "shape") else len(X_test)
    n_y = int(len(y_test))
    return n_x, n_y

def sidecar_length(ids_path, split="test"):
    with open(ids_path, "rb") as f:
        ids_payload = pickle.load(f)
    if split not in ids_payload:
        raise KeyError(f"Sidecar não contém split '{split}'.")
    ids = ids_payload[split]
    # escolher um vetor presente para medir tamanho (preferindo DATE)
    for key in ["DATE", "ID_UNIDADE", "t_index"]:
        arr = ids.get(key, None)
        if arr is not None:
            return int(len(arr))
    raise ValueError("Sidecar não possui arrays em DATE/ID_UNIDADE/t_index para medir tamanho.")

def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
        for col in df.columns:
            if col.lower() == c.lower():
                return col
    return None

def load_predictions_len(pred_path):
    df = pd.read_csv(pred_path)
    # tamanho = número de linhas úteis (considera ao menos 1 das colunas real/pred)
    real_col = pick_col(df, REAL_CANDIDATES)
    pred_col = pick_col(df, PRED_CANDIDATES)
    if real_col and pred_col:
        n = int(len(df.dropna(subset=[real_col, pred_col], how="all")))
    elif real_col:
        n = int(len(df.dropna(subset=[real_col], how="all")))
    elif pred_col:
        n = int(len(df.dropna(subset=[pred_col], how="all")))
    else:
        # sem colunas detectáveis → contar linhas brutas
        n = int(len(df))
    return n

def status(ok):
    return "✅ OK" if ok else "❌ FALHA"

def main():
    p = argparse.ArgumentParser(description="Triple check: dataset, sidecar e predictions.")
    p.add_argument("--dataset", default="data/datasets/RN_DAILY/dataset.pickle",
                   help="Caminho para dataset.pickle")
    p.add_argument("--sidecar", default="data/datasets/RN_DAILY/dataset_ids.pickle",
                   help="Caminho para dataset_ids.pickle")
    p.add_argument("--pred", default="models/RN_DAILY_FULL_987_rf/predictions.csv",
                   help="Caminho para predictions.csv")
    p.add_argument("--split", default="test", choices=["train","val","test"],
                   help="Split a verificar no sidecar (padrão: test)")
    p.add_argument("--strict-exit", action="store_true",
                   help="Se qualquer checagem falhar, sai com código 1.")
    args = p.parse_args()

    print("=== Triple Check ===")
    print(f"dataset : {args.dataset}")
    print(f"sidecar : {args.sidecar}  (split={args.split})")
    print(f"pred    : {args.pred}")
    print("")

    # 1) dataset: X_test vs y_test
    try:
        n_x, n_y = load_dataset(args.dataset)
        ok1 = (n_x == n_y)
        print(f"[1] len(X_{args.split}) vs len(y_{args.split}) -> {n_x} vs {n_y}  :: {status(ok1)}")
    except Exception as e:
        print(f"[1] Erro ao ler dataset: {e}")
        ok1 = False

    # 2) sidecar: ids_test vs X_test
    try:
        n_ids = sidecar_length(args.sidecar, split=args.split)
        ok2 = ('n_x' in locals()) and (n_ids == n_x)
        print(f"[2] len(ids_{args.split}) vs len(X_{args.split}) -> {n_ids} vs {n_x if 'n_x' in locals() else '??'}  :: {status(ok2)}")
    except Exception as e:
        print(f"[2] Erro ao ler sidecar: {e}")
        ok2 = False

    # 3) predictions: linhas vs y_test/ids_test
    try:
        n_pred = load_predictions_len(args.pred)
        expect = n_y if 'n_y' in locals() else n_ids if 'n_ids' in locals() else None
        ok3 = (expect is not None) and (n_pred == expect)
        print(f"[3] len(predictions) vs esperado -> {n_pred} vs {expect}  :: {status(ok3)}")
    except Exception as e:
        print(f"[3] Erro ao ler predictions: {e}")
        ok3 = False

    all_ok = ok1 and ok2 and ok3
    print("\nRESULTADO FINAL:", status(all_ok))
    if args.strict_exit and not all_ok:
        sys.exit(1)

if __name__ == "__main__":
    main()
