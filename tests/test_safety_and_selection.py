"""Safety nets (watchdog, interim answers) and selection logic, tested directly."""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from automl import clustering as clu
from automl.metrics import best_binary_macro_f1, fast_macro_f1

ROOT = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(7)


def test_fast_macro_f1_matches_sklearn():
    for _ in range(200):
        n, k = int(rng.integers(5, 300)), int(rng.integers(2, 6))
        y, p = rng.integers(0, k, n), rng.integers(0, k, n)
        assert abs(fast_macro_f1(y, p, k) - f1_score(y, p, average="macro")) < 1e-12


def test_best_threshold_is_exact_optimum_including_ties():
    for _ in range(40):
        n = int(rng.integers(10, 300))
        y = (rng.random(n) < rng.uniform(.05, .5)).astype(int)
        if y.sum() in (0, n):
            continue
        prob = np.round(rng.random(n), int(rng.integers(1, 4)))
        best, thr = best_binary_macro_f1(y, prob)
        brute = max(f1_score(y, (prob >= c).astype(int), average="macro") for c in np.unique(np.r_[prob, 1.1]))
        assert abs(best - brute) < 1e-12
        assert abs(f1_score(y, (prob >= thr).astype(int), average="macro") - best) < 1e-12


def test_clustering_picks_the_most_stable_method(monkeypatch):
    # three well separated blobs: every method agrees; then force distinct stabilities
    from sklearn.datasets import make_blobs
    X, _ = make_blobs(300, 2, centers=3, random_state=0)
    fake = {"kmeans": 0.90, "gmm_full": 0.97, "gmm_bic": 0.96, "ward": 0.80, "spectral": 0.95}
    monkeypatch.setattr(clu, "_stability", lambda m, Z, base, r, dl: fake[m.name])
    res = clu.solve_clustering(X, 3, time.time() + 30, log=lambda *_: None)
    assert res["report"]["chosen"] == "gmm_full"      # highest stability, no tie margin


def _run_with_patch(tmp_path, patch_code, data_dir, limit=20):
    script = tmp_path / "run.py"
    header = textwrap.dedent(f"""\
        import sys, time
        sys.path.insert(0, r"{ROOT}")
        sys.argv = ["main.py", "--data", r"{data_dir}", "--output", r"{tmp_path / 'out'}", "--time-limit", "{limit}"]
        import main
        """)
    script.write_text(header + textwrap.dedent(patch_code) + "\nmain.main()\n")
    t = time.time()
    proc = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=120, check=False)
    assert "Traceback" not in proc.stderr or "watchdog" in proc.stderr, proc.stderr[-2000:]  # script itself ran
    return time.time() - t, proc


def _dataset(tmp_path, task):
    d = tmp_path / "data"
    d.mkdir()
    df = pd.DataFrame({"a": rng.normal(size=300), "b": rng.normal(size=300)})
    if task == "regression":
        df["y"] = df["a"] * 2 + rng.normal(size=300) * .1
        df.iloc[:240].to_csv(d / "train.csv", index=False)
        df.iloc[240:].drop(columns="y").to_csv(d / "test.csv", index=False)
        (d / "meta.json").write_text(json.dumps({"task": "regression", "target": "y"}))
    else:
        df.to_csv(d / "train.csv", index=False)
        (d / "meta.json").write_text(json.dumps({"task": "clustering", "n_clusters": 3}))
    return d


def test_watchdog_stops_a_hung_run_with_useful_predictions(tmp_path):
    d = _dataset(tmp_path, "regression")
    # every boosting / forest / SVM fit hangs; only the linear safety answer can finish
    patch = """
import automl.supervised as s
for name in ("HistGradientBoostingRegressor", "RandomForestRegressor", "ExtraTreesRegressor", "SVR",
             "KNeighborsRegressor", "RidgeCV"):
    cls = getattr(s, name)
    if name == "RidgeCV":
        continue
    setattr(s, name, type(name, (cls,), {"fit": lambda self, *a, **k: time.sleep(300)}))
"""
    secs, proc = _run_with_patch(tmp_path, patch, d, limit=30)   # headroom for a loaded machine
    assert secs < 35, proc.stderr[-2000:]                      # stopped by the watchdog, not hung
    pred = pd.read_csv(tmp_path / "out" / "predictions.csv")["prediction"]
    assert len(pred) == 60 and pred.std() > 0.1                # a real (linear) model, not the constant fallback
    prof = json.loads((tmp_path / "out" / "profile.json").read_text())
    assert prof["model"]["status"] == "interim"


def test_clustering_hang_keeps_quick_kmeans_labels(tmp_path):
    d = _dataset(tmp_path, "clustering")
    patch = """
import automl.clustering as c
orig = c._Method.fit_predict
c._Method.fit_predict = lambda self, Z: time.sleep(300)
"""
    secs, proc = _run_with_patch(tmp_path, patch, d, limit=30)   # headroom for a loaded machine
    assert secs < 35, proc.stderr[-2000:]
    labels = pd.read_csv(tmp_path / "out" / "predictions.csv")["prediction"]
    assert labels.nunique() == 3                               # quick k-means groups, not one cluster
