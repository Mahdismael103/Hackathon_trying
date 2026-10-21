"""Round-2 research: compare clustering selection rules on the 34-dataset benchmark (offline).

    python tools/clu_rules2.py

Candidate labels, stability and consensus come from the real solve_clustering; the rules below only
re-rank the same candidates, so differences are purely due to the selection rule.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
warnings.filterwarnings("ignore")
from sklearn.metrics import adjusted_rand_score as ari  # noqa: E402

import clu_bench as cb  # noqa: E402
from automl.preprocess import Preparer  # noqa: E402
from automl.profiling import profile_dataset  # noqa: E402


def heldout_gaussian_loglik(X, labels, reps=3, seed=0):
    """Held-out log-likelihood of a Gaussian mixture built from a partition (tested, not adopted)."""
    from scipy.special import logsumexp

    X = np.asarray(X, float)
    n, d = X.shape
    labels = np.asarray(labels)
    groups = np.unique(labels)
    if len(groups) < 2 or d == 0:
        return -np.inf
    rng = np.random.default_rng(seed)
    total_var = X.var(0).mean() + 1e-12
    scores = []
    for _ in range(reps):
        tr = rng.random(n) < 0.5
        te = ~tr
        comp = []
        for g in groups:
            Xg = X[tr & (labels == g)]
            if len(Xg) < 2:
                continue
            mu, reg = Xg.mean(0), 1e-2 * total_var
            cov = (np.cov(Xg, rowvar=False).reshape(d, d) + reg * np.eye(d)) if len(Xg) > d + 1 \
                else np.diag(Xg.var(0) + reg)
            sign, logdet = np.linalg.slogdet(cov)
            if sign <= 0:
                continue
            diff = X[te] - mu
            maha = np.einsum("ij,jk,ik->i", diff, np.linalg.inv(cov), diff)
            comp.append(np.log(len(Xg) / tr.sum()) - 0.5 * (maha + logdet + d * np.log(2 * np.pi)))
        if comp:
            scores.append(float(logsumexp(np.vstack(comp), axis=0).mean()))
    return float(np.mean(scores)) if scores else -np.inf


def collect():
    out = []
    for name, df, y, k in cb.datasets():
        df = df.reset_index(drop=True)
        _, roles = profile_dataset(df, None, "clustering", None, {}, None, k)
        prep = Preparer(roles, clustering=True).fit(df)
        X = prep.transform_linear(df, onehot_weight=0.7)
        cont = np.array(["==" not in nm for nm in prep.linear_names])
        r = cb.run_one(df, k)
        rep = r["report"]
        Xc = X[:, cont] if cont.any() else X
        Xc = Xc[:, Xc.var(0) > 1e-12]
        t = time.time()
        ll = {m: heldout_gaussian_loglik(Xc, lab) for m, lab in r["candidates"].items()}
        out.append(dict(name=name, y=y, cands=r["candidates"], stab=rep["stability"], cons=rep["consensus"],
                        ll=ll, ll_seconds=time.time() - t))
        print(f"{name:34s} ll in {time.time() - t:.2f}s", flush=True)
    return out


def pick(d, rule):
    c = list(d["cands"])
    s, cons, ll = d["stab"], d["cons"], d["ll"]
    if rule == "current (stability, tol .03, consensus)":
        top = max(s.values()); near = [m for m in c if s[m] >= top - .03]
        return max(near, key=lambda m: cons[m])
    if rule == "stability only":
        return max(c, key=lambda m: s[m])
    if rule == "held-out likelihood only":
        return max(c, key=lambda m: ll[m])
    if rule.startswith("stable (>= top-"):
        tol = float(rule.split("-")[1].split(")")[0])
        top = max(s.values()); near = [m for m in c if s[m] >= top - tol]
        return max(near, key=lambda m: ll[m])
    if rule == "rank(stability)+rank(likelihood)":
        rs = pd.Series(s).rank(); rl = pd.Series(ll).rank()
        return max(c, key=lambda m: rs[m] + rl[m] + 1e-3 * s[m])
    raise ValueError(rule)


RULES = ["current (stability, tol .03, consensus)", "stability only", "held-out likelihood only",
         "stable (>= top-0.03) then likelihood", "stable (>= top-0.05) then likelihood",
         "stable (>= top-0.1) then likelihood", "rank(stability)+rank(likelihood)"]


def main():
    data = collect()
    rows = []
    for d in data:
        row = {"dataset": d["name"], "oracle": max(ari(d["y"], l) for l in d["cands"].values())}
        for r in RULES:
            row[r] = ari(d["y"], d["cands"][pick(d, r)])
        rows.append(row)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(df.round(3).to_string(index=False))
    summary = pd.DataFrame({"mean ARI": df[["oracle"] + RULES].mean(), "worst": df[["oracle"] + RULES].min(),
                            "datasets below 0.8": (df[["oracle"] + RULES] < .8).sum()})
    print("\n", summary.sort_values("mean ARI", ascending=False).round(4).to_string())
    print(f"\nlikelihood cost: mean {np.mean([d['ll_seconds'] for d in data]):.2f}s per dataset")


if __name__ == "__main__":
    main()
