"""Clustering benchmark: 34 labelled datasets, run in-process through the real pipeline pieces.

    python tools/clu_bench.py [--tag name]

Sets: the 16-dataset research suite (tools/clu_suite.py) + the 6 clustering cases of the judge
simulation for three seed bases (910000, 950000, 970000). Reports, per dataset, the ARI of the chosen
solution, of every candidate method, and the best candidate (an upper bound for the selection rule).
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
warnings.filterwarnings("ignore")

from sklearn.metrics import adjusted_rand_score as ari  # noqa: E402

import judge_sim as js  # noqa: E402
from automl.clustering import solve_clustering  # noqa: E402
from automl.preprocess import Preparer  # noqa: E402
from automl.profiling import profile_dataset  # noqa: E402
from clu_suite import suite  # noqa: E402


def datasets():
    for name, df, y, k in suite():
        yield f"suite/{name}", df, y, k
    for base in (910_000, 950_000, 970_000):
        for i, f in enumerate(js.CLU):
            X, y, k, _ = f(js.R(base + 10_000 + i))
            yield f"judge{base // 10000}/{f.__name__}", X, y, k


def run_one(df, k):
    _, roles = profile_dataset(df, None, "clustering", None, {}, None, k)
    prep = Preparer(roles, clustering=True).fit(df)
    X = prep.transform_linear(df, onehot_weight=0.7)
    names = prep.linear_names
    return solve_clustering(X, k, time.time() + 40, log=lambda *_: None,
                            onehot_columns=sum("==" in nm for nm in names),
                            continuous_mask=["==" not in nm for nm in names], return_candidates=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag")
    a = ap.parse_args()
    rows = []
    for name, df, y, k in datasets():
        t = time.time()
        res = run_one(df.reset_index(drop=True), k)
        cand = {m: ari(y, lab) for m, lab in res["candidates"].items()}
        row = {"dataset": name, "chosen_ari": ari(y, res["labels"]), "chosen": res["report"]["chosen"],
               "oracle": max(cand.values()), "oracle_method": max(cand, key=cand.get), "seconds": time.time() - t}
        rows.append(row)
        print(f"{name:34s} chosen {row['chosen_ari']:.3f} ({row['chosen']:>18s})  best {row['oracle']:.3f} "
              f"({row['oracle_method']})  {row['seconds']:.1f}s", flush=True)
    df = pd.DataFrame(rows)
    print(f"\nmean chosen ARI {df.chosen_ari.mean():.4f}   mean oracle {df.oracle.mean():.4f}   "
          f"worst chosen {df.chosen_ari.min():.3f}   mean time {df.seconds.mean():.1f}s")
    if a.tag:
        df.to_csv(ROOT / "outputs" / f"clu_bench_{a.tag}.csv", index=False)


if __name__ == "__main__":
    main()
