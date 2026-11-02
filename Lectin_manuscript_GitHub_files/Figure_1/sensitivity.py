#!/usr/bin/env python3
"""
Sensitivity check: do the headline results change if pairs with few
synonymous substitutions (low cS) are also excluded?

Usage:
  python sensitivity.py kaks_per_gene_ci.csv
  python sensitivity.py kaks_per_gene_ci.csv --labels types.csv --label-col type
      (types.csv needs columns Seq_1, Seq_2 and the label column, e.g. paralog/ortholog)
"""
import argparse
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("file")
ap.add_argument("--labels", help="optional CSV/TSV with Seq_1, Seq_2 and a label column")
ap.add_argument("--label-col", default="type", help="name of the label column (default: type)")
ap.add_argument("--thresholds", type=float, nargs="+", default=[3, 5, 10],
                help="minimum cS values to test (default: 3 5 10)")
a = ap.parse_args()

g = pd.read_csv(a.file, usecols=["Seq_1", "Seq_2", "pass", "cS", "omega"])
g = g[g["pass"]].copy()                       # your current cutoffs = baseline

use_labels = False
if a.labels:
    lab = pd.read_csv(a.labels, sep=None, engine="python")[["Seq_1", "Seq_2", a.label_col]]
    g = g.merge(lab, on=["Seq_1", "Seq_2"], how="left")
    use_labels = True

base_gt1 = int((g["omega"] > 1).sum())
rows = []
for t in [0] + list(a.thresholds):            # 0 = baseline (no cS filter)
    d = g[g["cS"] >= t] if t > 0 else g
    r = {
        "min_cS": "none (current)" if t == 0 else f">= {t:g}",
        "pairs": len(d),
        "pairs_lost_%": round(100 * (1 - len(d) / len(g)), 3),
        "omega>1": int((d["omega"] > 1).sum()),
        "omega>1_%": round(100 * (d["omega"] > 1).mean(), 3),
        "omega>1_lost_%": round(100 * (1 - (d["omega"] > 1).sum() / base_gt1), 2),
        "omega>5": int((d["omega"] > 5).sum()),
        "median_omega": round(d["omega"].median(), 4),
    }
    if use_labels:
        for k, sub in d.groupby(a.label_col):
            r[f"{k}_omega>1_%"] = round(100 * (sub["omega"] > 1).mean(), 3)
            r[f"{k}_omega>5"] = int((sub["omega"] > 5).sum())
    rows.append(r)

out = pd.DataFrame(rows)
print(out.T.to_string(header=False))
out.to_csv("sensitivity_results.csv", index=False)
print("\nSaved: sensitivity_results.csv")