"""Simulated assessment: 18 unseen datasets, scored like the brief describes.

    python tools/judge_sim.py [--main path/to/main.py] [--python path/to/python]

* 6 classification + 6 regression + 6 clustering datasets, generated with seeds never used during
  development, in the style of the brief's Kaggle examples, with the brief's variations injected on
  purpose so the correct profile answers are known.
* Each dataset runs through main.py as a separate process with a hard 60 s kill.
* Score (the brief's weights; the official formula is not published, so this is an approximation):
    quality  85 = mean over the 3 tasks (equal weight) of the mean task score
                  (macro-F1, R2 clipped at 0, ARI clipped at 0)
    profile  10 = mean share of correctly reported variations (missing values, scale differences,
                  categorical, constant and duplicate columns, irrelevant features, imbalance)
    speed     5 = mean of (1 - seconds/60) over datasets whose quality is useful
                  (clearly above a trivial baseline)
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import datasets as skd
from sklearn import metrics as skm

ROOT = Path(__file__).resolve().parents[1]
BASE = Path(tempfile.gettempdir()) / "automl_judge"
VARS = ["missing_values", "differently_scaled_features", "categorical_columns", "constant_columns",
        "duplicate_columns", "irrelevant_features", "imbalanced_classes"]


def R(seed):
    return np.random.default_rng(seed)


# ------------------------------------------------------------------ injection helpers (truth is recorded)
def inject(df, rng, truth, missing=0.0, constant=False, duplicate=False, noise=0, scale=False, cat=False):
    df = df.copy()
    num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if scale and num:
        df[num[0]] = df[num[0]] * 1000.0
    if noise:
        for i in range(noise):
            df[f"noise_{i}"] = rng.normal(0, 1, len(df))
    if cat:
        df["random_category"] = rng.choice(["red", "green", "blue", "grey"], len(df))
    if duplicate and num:
        df[f"{num[-1]}_dup"] = df[num[-1]]
    if constant:
        df["constant_col"] = 7
    if missing:
        for c in num[: max(1, len(num) // 3)]:
            df.loc[rng.random(len(df)) < missing, c] = np.nan
    truth.update({
        "missing_values": truth.get("missing_values", False) or missing > 0,
        "differently_scaled_features": truth.get("differently_scaled_features", False) or scale,
        "categorical_columns": truth.get("categorical_columns", False) or cat,
        "constant_columns": constant, "duplicate_columns": duplicate,
        "irrelevant_features": truth.get("irrelevant_features", False) or noise > 0,
    })
    return df


def standardise(df):
    return (df - df.mean()) / df.std()


# ------------------------------------------------------------------ classification
def c1_telco(rng):
    n = 4000
    tenure = rng.integers(0, 73, n)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[.5, .3, .2])
    internet = rng.choice(["DSL", "Fiber optic", "No"], n, p=[.35, .45, .2])
    monthly = np.round(25 + 50 * (internet == "Fiber optic") + 20 * (internet == "DSL") + rng.normal(0, 10, n), 2)
    total = (monthly * tenure).round(2).astype(str)
    total[tenure == 0] = " "
    logit = -2.6 + 1.6 * (contract == "Month-to-month") - .04 * tenure + 1.0 * (internet == "Fiber optic")
    y = np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), "Yes", "No")
    X = pd.DataFrame({"customerID": [f"C{100000 + i}" for i in range(n)], "Contract": contract,
                      "InternetService": internet, "tenure": tenure, "MonthlyCharges": monthly, "TotalCharges": total})
    truth = {"missing_values": True, "differently_scaled_features": True, "categorical_columns": True,
             "irrelevant_features": False, "constant_columns": False, "duplicate_columns": False}
    return X, y, truth


def c2_ai4i(rng):
    n = 8000
    typ = rng.choice(["L", "M", "H"], n, p=[.6, .3, .1])
    air = rng.normal(300, 2, n); proc = air + 10 + rng.normal(0, 1, n)
    rpm = rng.normal(1540, 180, n); torque = np.clip(rng.normal(40, 10, n), 3, None); wear = rng.integers(0, 250, n)
    power = torque * rpm * 2 * np.pi / 60
    y = ((power < 3500) | (power > 9000) | (wear * torque > 11000) | (rng.random(n) < .003)).astype(int)
    X = pd.DataFrame({"Type": typ, "Air temperature [K]": air, "Process temperature [K]": proc,
                      "Rotational speed [rpm]": rpm, "Torque [Nm]": torque, "Tool wear [min]": wear})
    truth = {"categorical_columns": True, "differently_scaled_features": True}
    X = inject(X, rng, truth, duplicate=True)
    truth["irrelevant_features"] = True  # failure never depends on the temperatures or Type
    return X, y, truth


def c3_breast_cancer(rng):
    b = skd.load_breast_cancer(as_frame=True)
    truth = {"differently_scaled_features": True}
    X = inject(b.data, rng, truth, missing=.05, constant=True, noise=5)
    return X, b.target.values, truth


def c4_multiclass_clean(rng):
    X, y = skd.make_classification(3000, 10, n_informative=5, n_redundant=0, n_classes=3,
                                   n_clusters_per_class=2, random_state=int(rng.integers(1e6)), shuffle=False)
    X = standardise(pd.DataFrame(X).add_prefix("f"))
    truth = {"irrelevant_features": True, "categorical_columns": False, "missing_values": False,
             "differently_scaled_features": False, "constant_columns": False, "duplicate_columns": False}
    return X, y, truth


def c5_nonlinear_moons(rng):
    X, y = skd.make_moons(2500, noise=.25, random_state=int(rng.integers(1e6)))
    X = pd.DataFrame(X, columns=["u", "v"])
    truth = {}
    X = inject(X, rng, truth, noise=3, cat=True)
    return X, y, truth


def c6_imbalanced_multiclass(rng):
    n = 5000
    shop = rng.choice([f"S{i}" for i in range(150)], n)
    eff = {f"S{i}": rng.normal(0, 1.5) for i in range(150)}
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    z = np.vectorize(eff.get)(shop) + 1.2 * x1 + rng.normal(0, .6, n)
    y = np.where(z > 2.0, "high", np.where(z > 0.6, "mid", "low"))
    X = pd.DataFrame({"shop": shop, "x1": x1, "x2": x2})
    truth = {"categorical_columns": True}
    X = inject(X, rng, truth, missing=.08)
    truth["differently_scaled_features"] = False
    truth["irrelevant_features"] = True  # x2 never enters the label
    return X, y, truth


# ------------------------------------------------------------------ regression
def r1_house(rng):
    n = 1800
    hood = rng.choice([f"N{i}" for i in range(20)], n); he = {f"N{i}": rng.normal(0, .25) for i in range(20)}
    area = np.exp(rng.normal(7.3, .35, n)); qual = rng.integers(1, 11, n); year = rng.integers(1900, 2011, n)
    logp = 11 + .55 * np.log(area / 1500) + .1 * qual + .003 * (year - 1970) + np.vectorize(he.get)(hood) + rng.normal(0, .12, n)
    X = pd.DataFrame({"Id": np.arange(1, n + 1), "Neighborhood": hood, "GrLivArea": area.round(),
                      "OverallQual": qual, "YearBuilt": year, "LotFrontage": rng.normal(70, 20, n).round()})
    X.loc[rng.random(n) < .18, "LotFrontage"] = np.nan
    truth = {"missing_values": True, "differently_scaled_features": True, "categorical_columns": True,
             "constant_columns": False, "duplicate_columns": False}
    return X, np.exp(logp).round(), truth


def r2_bike(rng):
    n = 7000
    t = pd.Timestamp("2011-01-01") + pd.to_timedelta(rng.integers(0, 2 * 365 * 24, n), unit="h")
    hour, wd = t.hour.values, (t.dayofweek.values < 5).astype(int)
    temp = 15 + 10 * np.sin((t.dayofyear.values - 100) / 365 * 2 * np.pi) + rng.normal(0, 3, n)
    lam = np.clip((40 + wd * (np.exp(-((hour - 8) ** 2) / 2) * 300 + np.exp(-((hour - 17.5) ** 2) / 3) * 350)
                   + (1 - wd) * np.exp(-((hour - 14) ** 2) / 12) * 250) * (1 + .03 * (temp - 15)), 1, None)
    X = pd.DataFrame({"datetime": t.strftime("%Y-%m-%d %H:%M:%S"), "workingday": wd, "temp": temp.round(2),
                      "humidity": np.clip(rng.normal(60, 18, n), 0, 100).round()})
    truth = {"differently_scaled_features": True, "missing_values": False, "categorical_columns": False,
             "constant_columns": False, "duplicate_columns": False}
    return X, rng.poisson(lam), truth


def r3_friedman(rng):
    X, y = skd.make_friedman1(2500, 10, noise=1.0, random_state=int(rng.integers(1e6)))
    X = standardise(pd.DataFrame(X).add_prefix("x"))
    truth = {"irrelevant_features": True, "differently_scaled_features": False}
    X = inject(X, rng, truth, duplicate=True)
    truth["irrelevant_features"] = True
    return X, y, truth


def r4_diabetes(rng):
    d = skd.load_diabetes(as_frame=True, scaled=False)
    truth = {"differently_scaled_features": True}
    X = inject(d.data, rng, truth, missing=.05, constant=True)
    truth["irrelevant_features"] = None  # unknown for real data without injected noise
    return X, d.target.values, truth


def r5_sparse_linear(rng):
    X, y = skd.make_regression(1500, 20, n_informative=5, noise=5, random_state=int(rng.integers(1e6)), shuffle=False)
    X = standardise(pd.DataFrame(X).add_prefix("r"))
    truth = {"irrelevant_features": True}
    X = inject(X, rng, truth, scale=True)
    truth["irrelevant_features"] = True
    return X, y, truth


def r6_lognormal(rng):
    n = 3000
    X = pd.DataFrame(rng.normal(size=(n, 4))).add_prefix("q")
    X["grade"] = rng.choice(list("ABCD"), n)
    y = np.exp(1.2 + .5 * X["q0"] - .4 * X["q1"] * X["q2"] + .3 * (X["grade"] == "A") + .3 * rng.normal(size=n))
    truth = {"categorical_columns": True, "differently_scaled_features": False, "missing_values": False,
             "constant_columns": False, "duplicate_columns": False, "irrelevant_features": True}
    return X, y.values, truth


# ------------------------------------------------------------------ clustering
def k1_mall(rng):
    centers = np.array([[25, 80], [55, 50], [85, 15], [85, 85], [25, 20]])
    n = 450; lab = rng.integers(0, 5, n)
    X = pd.DataFrame({"CustomerID": np.arange(1, n + 1), "Gender": rng.choice(["Male", "Female"], n),
                      "Annual Income (k$)": (centers[lab, 0] + rng.normal(0, 7, n)).round(),
                      "Spending Score (1-100)": np.clip(centers[lab, 1] + rng.normal(0, 7, n), 1, 100).round()})
    truth = {"categorical_columns": True, "missing_values": False, "constant_columns": False,
             "duplicate_columns": False, "irrelevant_features": True, "differently_scaled_features": False}
    return X, lab, 5, truth


def k2_iris_noise(rng):
    b = skd.load_iris(as_frame=True)
    truth = {"differently_scaled_features": False}
    X = inject(b.data, rng, truth, missing=.04, noise=2)
    return X, b.target.values, 3, truth


def k3_wine(rng):
    b = skd.load_wine(as_frame=True)
    truth = {"differently_scaled_features": True, "missing_values": False, "categorical_columns": False,
             "constant_columns": False, "duplicate_columns": False, "irrelevant_features": None}
    return b.data, b.target.values, 3, truth


def k4_blobs_cat(rng):
    X, y = skd.make_blobs(2000, 6, centers=4, cluster_std=1.5, random_state=int(rng.integers(1e6)))
    X = pd.DataFrame(X).add_prefix("b")
    truth = {"differently_scaled_features": False, "irrelevant_features": False}
    X = inject(X, rng, truth, constant=True)
    X["segment"] = np.where(rng.random(2000) < .8, np.array(list("PQRS"))[y], rng.choice(list("PQRS"), 2000))
    truth["categorical_columns"] = True
    return X, y, 4, truth


def k5_aniso(rng):
    X, y = skd.make_blobs(1500, 2, centers=3, random_state=int(rng.integers(1e6)))
    X = pd.DataFrame(X @ np.array([[0.6, -0.64], [-0.41, 0.85]]), columns=["a", "b"])
    truth = {"differently_scaled_features": False, "irrelevant_features": False}
    X = inject(X, rng, truth, duplicate=True)
    return X, y, 3, truth


def k6_groups_with_noise(rng):
    X, y = skd.make_classification(2000, 8, n_informative=4, n_redundant=0, n_classes=4, n_clusters_per_class=1,
                                   class_sep=2.5, random_state=int(rng.integers(1e6)), shuffle=False)
    X = pd.DataFrame(X[:, :4]).add_prefix("g")
    truth = {"differently_scaled_features": False}
    X = inject(X, rng, truth, noise=4)
    return X, y, 4, truth


CLS = [c1_telco, c2_ai4i, c3_breast_cancer, c4_multiclass_clean, c5_nonlinear_moons, c6_imbalanced_multiclass]
REG = [r1_house, r2_bike, r3_friedman, r4_diabetes, r5_sparse_linear, r6_lognormal]
CLU = [k1_mall, k2_iris_noise, k3_wine, k4_blobs_cat, k5_aniso, k6_groups_with_noise]


def build(seed_base=910_000):
    cases = []
    for i, f in enumerate(CLS + REG):
        rng = R(seed_base + i)                      # seeds never used during development
        X, y, truth = f(rng)
        task = "classification" if f in CLS else "regression"
        if task == "classification":
            vc = pd.Series(y).value_counts()
            truth["imbalanced_classes"] = bool(vc.iloc[0] / vc.iloc[-1] > 3)
        idx = rng.permutation(len(X)); cut = int(.7 * len(X))
        tr, te = idx[:cut], idx[cut:]
        d = BASE / "data" / f.__name__
        shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True)
        X.iloc[tr].assign(target=np.asarray(y)[tr]).to_csv(d / "train.csv", index=False)
        X.iloc[te].to_csv(d / "test.csv", index=False)
        (d / "meta.json").write_text(json.dumps({"task": task, "target": "target"}))
        cases.append((f.__name__, task, d, np.asarray(y)[te], np.asarray(y)[tr], truth))
    for i, f in enumerate(CLU):
        rng = R(seed_base + 10_000 + i)
        X, y, k, truth = f(rng)
        d = BASE / "data" / f.__name__
        shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True)
        X.to_csv(d / "train.csv", index=False)
        (d / "meta.json").write_text(json.dumps({"task": "clustering", "n_clusters": k}))
        cases.append((f.__name__, "clustering", d, y, None, truth))
    return cases


def quality(task, truth, pred, ytrain):
    if task == "classification":
        q = skm.f1_score(pd.Series(truth).astype(str), pd.Series(pred).astype(str), average="macro")
        maj = pd.Series(ytrain).astype(str).mode()[0]
        base = skm.f1_score(pd.Series(truth).astype(str), np.repeat(maj, len(truth)), average="macro")
        return q, q > base + 0.1
    if task == "regression":
        q = max(0.0, skm.r2_score(truth.astype(float), pred.astype(float)))
        return q, q > 0.1
    q = max(0.0, skm.adjusted_rand_score(truth, pred))
    return q, q > 0.1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", default=str(ROOT / "main.py"))
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--seed", type=int, default=910_000,
                    help="dataset seed base: 910000 = development set, 950000 = held-out set")
    ap.add_argument("--tag", help="save the per-dataset table to outputs/judge_<tag>.csv")
    a = ap.parse_args()
    rows = []
    for name, task, d, ytrue, ytrain, truth in build(a.seed):
        out = BASE / "out" / name
        shutil.rmtree(out, ignore_errors=True)
        t0 = time.time()
        try:
            subprocess.run([a.python, a.main, "--data", str(d), "--output", str(out)],
                           capture_output=True, text=True, timeout=60, check=False)
            killed = False
        except subprocess.TimeoutExpired:
            killed = True
        secs = time.time() - t0
        row = {"dataset": name, "task": task, "seconds": round(secs, 1), "killed_at_60s": killed}
        try:
            pred = pd.read_csv(out / "predictions.csv")["prediction"].values
            prof = json.loads((out / "profile.json").read_text())
            if len(pred) != len(ytrue):
                raise ValueError(f"{len(pred)} predictions for {len(ytrue)} rows")
            q, useful = quality(task, ytrue, pred, ytrain)
            issues = prof.get("data_issues", {})
            checked = [v for v in VARS if truth.get(v) is not None and (v != "imbalanced_classes" or task == "classification")]
            right = [v for v in checked if bool(issues.get(v)) == bool(truth[v])]
            wrong = [v for v in checked if v not in right]
            row.update(quality=round(q, 4), useful=useful, status=prof["model"].get("status"),
                       profile_score=round(len(right) / len(checked), 3), profile_wrong=",".join(wrong) or "-",
                       speed=round(max(0.0, 1 - secs / 60) if useful else 0.0, 3))
        except Exception as e:
            row.update(quality=0.0, useful=False, status=f"NO VALID OUTPUT ({e})"[:60], profile_score=0.0,
                       profile_wrong="all", speed=0.0)
        rows.append(row)
        print(json.dumps(row), flush=True)

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print()
    print(df[["dataset", "task", "seconds", "status", "quality", "profile_score", "profile_wrong"]].to_string(index=False))
    per_task = df.groupby("task")["quality"].mean()
    q_pts = 85 * per_task.mean()
    p_pts = 10 * df["profile_score"].mean()
    s_pts = 5 * df["speed"].mean()
    print("\nquality per task:", {k: round(v, 4) for k, v in per_task.items()})
    print(f"failures (no valid output): {int(df['status'].str.startswith('NO VALID').sum())}   "
          f"killed at 60 s: {int(df['killed_at_60s'].sum())}   slowest: {df['seconds'].max()} s")
    if a.tag:
        (ROOT / "outputs").mkdir(exist_ok=True)
        df.to_csv(ROOT / "outputs" / f"judge_{a.tag}.csv", index=False)
    print(f"\nSIMULATED SCORE  quality {q_pts:.1f}/85  +  profiling {p_pts:.1f}/10  +  speed {s_pts:.1f}/5"
          f"  =  {q_pts + p_pts + s_pts:.1f}/100")


if __name__ == "__main__":
    main()
