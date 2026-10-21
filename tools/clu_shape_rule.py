"""Study: prefer the BIC-selected Gaussian mixture when the data reject round clusters.

    python tools/clu_shape_rule.py

Rule: start from the most stable method; switch to gmm_bic if its stability is within `delta` of the top
AND BIC prefers a non-spherical shape by at least `tau` per row (round, equal clusters are what k-means
and spectral-on-k-means assume). Grid-searched on the 34 tuning datasets of tools/clu_bench.py, then
reported on 12 validation datasets (judge clustering cases, two seed bases never used before).
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
warnings.filterwarnings("ignore")
from sklearn.metrics import adjusted_rand_score as ari  # noqa: E402

import clu_bench as cb  # noqa: E402
import judge_sim as js  # noqa: E402

DELTAS = [0.03, 0.05, 0.08, 0.1, 0.15]
TAUS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]


def validation_sets():
    for base in (990_000, 1_010_000):
        for i, f in enumerate(js.CLU):
            X, y, k, _ = f(js.R(base + 10_000 + i))
            yield f"valid{base // 10000}/{f.__name__}", X, y, k


def collect(source):
    rows = []
    for name, df, y, k in source:
        r = cb.run_one(df.reset_index(drop=True), k)
        rep = r["report"]
        rows.append(dict(name=name, ari={m: ari(y, l) for m, l in r["candidates"].items()},
                         stab=rep["stability"], ev=rep.get("shape_evidence_per_row"), shape=rep.get("bic_shape"),
                         base_choice=rep["chosen"]))
        print(f"  {name:34s} evidence {rep.get('shape_evidence_per_row')} shape {rep.get('bic_shape')}", flush=True)
    return rows


def choose(d, delta, tau):
    s = d["stab"]
    top = max(s, key=s.get)
    if ("gmm_bic" in s and d["ev"] is not None and d["shape"] != "spherical" and d["ev"] >= tau
            and s["gmm_bic"] >= s[top] - delta):
        return "gmm_bic"
    return top


def evaluate(rows, delta, tau):
    return np.array([d["ari"][choose(d, delta, tau)] for d in rows])


def main():
    print("tuning set (34):")
    tune = collect(cb.datasets())
    print("validation set (12, never used for tuning):")
    valid = collect(validation_sets())
    base_t, base_v = evaluate(tune, -1, 1e9), evaluate(valid, -1, 1e9)   # delta<0: rule never fires
    print(f"\ncurrent rule: tuning {base_t.mean():.4f}   validation {base_v.mean():.4f}")
    grid = []
    for dl in DELTAS:
        for tau in TAUS:
            t, v = evaluate(tune, dl, tau), evaluate(valid, dl, tau)
            grid.append(dict(delta=dl, tau=tau, tune=t.mean(), tune_worse=int((t < base_t - 1e-3).sum()),
                             tune_better=int((t > base_t + 1e-3).sum()), valid=v.mean(),
                             valid_worse=int((v < base_v - 1e-3).sum()), valid_better=int((v > base_v + 1e-3).sum())))
    g = pd.DataFrame(grid).sort_values(["tune", "tune_worse"], ascending=[False, True])
    pd.set_option("display.width", 200)
    print(g.round(4).head(12).to_string(index=False))
    best = g.iloc[0]
    print(f"\nselected on tuning: delta={best.delta} tau={best.tau}: tuning {base_t.mean():.4f} -> {best.tune:.4f}, "
          f"validation {base_v.mean():.4f} -> {best.valid:.4f}")
    for rows, label in ((tune, "tuning"), (valid, "validation")):
        for d in rows:
            a, b = d["ari"][choose(d, -1, 1e9)], d["ari"][choose(d, best.delta, best.tau)]
            if abs(a - b) > 1e-3:
                print(f"   {label:10s} {d['name']:34s} {a:.3f} -> {b:.3f}")


if __name__ == "__main__":
    main()
