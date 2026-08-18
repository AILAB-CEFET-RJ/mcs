#!/usr/bin/env python3
"""Create compact interpolation targets for cells containing experimental CNES."""
import argparse
from pathlib import Path
import pandas as pd

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--mapping", required=True); p.add_argument("--cells", required=True); p.add_argument("--out", required=True)
args = p.parse_args()
mapping = pd.read_parquet(args.mapping, columns=["cell_id"]).drop_duplicates("cell_id")
cells = pd.read_csv(args.cells, usecols=["cell_id", "latitude", "longitude"])
frame = mapping.merge(cells, on="cell_id", how="left", validate="one_to_one").sort_values("cell_id").reset_index(drop=True)
if frame[["latitude", "longitude"]].isna().any().any(): raise RuntimeError("Célula CNES sem centro geográfico")
# Compact storage coordinates; original cell_id is retained as the semantic key.
frame["row"] = 0; frame["col"] = range(len(frame))
out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
frame[["cell_id", "row", "col", "latitude", "longitude"]].to_csv(out, index=False)
print(f"targets={len(frame)} output={out}")
