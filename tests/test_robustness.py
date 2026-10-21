"""Robustness tests: hostile or unusual inputs must still give a valid, complete output.

    python -m pytest tests -q

Each case writes a small dataset to a temporary folder and runs the real pipeline (automl.run).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import automl
from automl import config

rng = np.random.default_rng(0)


def write(folder: Path, train: pd.DataFrame, test: pd.DataFrame | None, meta: dict, **csv_kw):
    folder.mkdir(parents=True, exist_ok=True)
    train.to_csv(folder / "train.csv", index=False, **csv_kw)
    if test is not None:
        test.to_csv(folder / "test.csv", index=False, **csv_kw)
    (folder / "meta.json").write_text(json.dumps(meta))
    return folder


def outputs(out: Path):
    pred = pd.read_csv(out / config.PREDICTIONS_FILE)
    prof = json.loads((out / config.PROFILE_FILE).read_text(encoding="utf-8"))
    return pred, prof


def simple_frame(n=300):
    df = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "c": rng.choice(["x", "y", "z"], n)})
    df["y"] = (df["a"] + (df["c"] == "x") > 0.3).astype(int)
    return df


def test_latin1_encoded_csv(tmp_path):
    df = simple_frame()
    df["city"] = rng.choice(["Zürich", "Málaga", "São Paulo", "Köln"], len(df))
    d = write(tmp_path / "d", df.iloc[:240], df.iloc[240:].drop(columns="y"),
              {"task": "classification", "target": "y"}, encoding="latin-1")
    automl.run(str(d), str(tmp_path / "out"))
    pred, prof = outputs(tmp_path / "out")
    assert len(pred) == 60 and prof["model"]["status"] == "ok"
    assert "city" in prof["feature_types"] and prof["feature_types"]["city"] == "categorical"


def test_utf8_bom_header(tmp_path):
    df = simple_frame()
    d = write(tmp_path / "d", df.iloc[:240], df.iloc[240:].drop(columns="y"),
              {"task": "classification", "target": "y"}, encoding="utf-8-sig")
    automl.run(str(d), str(tmp_path / "out"))
    pred, prof = outputs(tmp_path / "out")
    assert len(pred) == 60 and prof["model"]["status"] == "ok"
    assert "a" in prof["feature_types"]          # no "﻿a" column


def test_empty_test_table(tmp_path):
    df = simple_frame()
    d = write(tmp_path / "d", df, df.iloc[:0].drop(columns="y"), {"task": "classification", "target": "y"})
    automl.run(str(d), str(tmp_path / "out"))
    pred, prof = outputs(tmp_path / "out")
    assert len(pred) == 0 and list(pred.columns)[1] == config.PREDICTION_COLUMN


def test_all_missing_target_gives_fallback_not_crash(tmp_path):
    df = simple_frame()
    df["y"] = np.nan
    d = write(tmp_path / "d", df.iloc[:240], df.iloc[240:].drop(columns="y"), {"task": "regression", "target": "y"})
    automl.run(str(d), str(tmp_path / "out"))
    pred, prof = outputs(tmp_path / "out")
    assert len(pred) == 60 and not pred["prediction"].isna().any()


def test_more_clusters_than_rows(tmp_path):
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 10.0], "b": [1.0, 1.5, 0.5, 9.0]})
    d = write(tmp_path / "d", df, None, {"task": "clustering", "n_clusters": 10})
    automl.run(str(d), str(tmp_path / "out"))
    pred, _ = outputs(tmp_path / "out")
    assert len(pred) == 4


def test_no_usable_features(tmp_path):
    df = pd.DataFrame({"Id": np.arange(200), "y": rng.normal(size=200)})
    d = write(tmp_path / "d", df.iloc[:150], df.iloc[150:].drop(columns="y"), {"task": "regression", "target": "y"})
    automl.run(str(d), str(tmp_path / "out"))
    pred, _ = outputs(tmp_path / "out")
    assert len(pred) == 50 and np.isfinite(pred["prediction"]).all()


def test_duplicate_column_names(tmp_path):
    d = tmp_path / "d"
    d.mkdir()
    lines = ["a,a,b,y"] + [f"{i},{i * 2},{i % 3},{i % 2}" for i in range(200)]
    (d / "train.csv").write_text("\n".join(lines[:161]))
    (d / "test.csv").write_text("\n".join(["a,a,b"] + [ln.rsplit(",", 1)[0] for ln in lines[161:]]))
    (d / "meta.json").write_text(json.dumps({"task": "classification", "target": "y"}))
    automl.run(str(d), str(tmp_path / "out"))
    pred, _ = outputs(tmp_path / "out")
    assert len(pred) == 40


def test_infinite_values_in_regression_target(tmp_path):
    df = simple_frame()
    df["y"] = df["a"] * 3.0
    df.loc[:4, "y"] = np.inf
    d = write(tmp_path / "d", df.iloc[:240], df.iloc[240:].drop(columns="y"), {"task": "regression", "target": "y"})
    automl.run(str(d), str(tmp_path / "out"))
    pred, prof = outputs(tmp_path / "out")
    assert len(pred) == 60 and np.isfinite(pred["prediction"]).all() and prof["model"]["status"] == "ok"


def test_mixed_number_and_text_column(tmp_path):
    df = simple_frame()
    df["mixed"] = [str(v) if i % 7 else "n/a" for i, v in enumerate(rng.integers(0, 100, len(df)))]
    d = write(tmp_path / "d", df.iloc[:240], df.iloc[240:].drop(columns="y"), {"task": "classification", "target": "y"})
    automl.run(str(d), str(tmp_path / "out"))
    pred, prof = outputs(tmp_path / "out")
    assert len(pred) == 60 and prof["model"]["status"] == "ok"


def test_metadata_cannot_point_outside_dataset_folder(tmp_path):
    secret = tmp_path / "secret.csv"
    pd.DataFrame({"a": [1, 2], "y": [0, 1]}).to_csv(secret, index=False)
    df = simple_frame()
    d = write(tmp_path / "d", df.iloc[:240], df.iloc[240:].drop(columns="y"),
              {"task": "classification", "target": "y", "train": "../secret.csv"})
    automl.run(str(d), str(tmp_path / "out"))
    _, prof = outputs(tmp_path / "out")
    assert prof["n_rows"] == 240     # the dataset's own train.csv was used, not ../secret.csv


@pytest.mark.parametrize("task", ["classification", "regression", "clustering"])
def test_profile_contains_required_sections(tmp_path, task):
    df = simple_frame()
    if task == "regression":
        df["y"] = df["a"] * 2 + rng.normal(size=len(df))
    meta = {"task": task, "target": "y"} if task != "clustering" else {"task": "clustering", "n_clusters": 3}
    train = df if task != "clustering" else df.drop(columns="y")
    test = df.iloc[240:].drop(columns="y") if task != "clustering" else None
    d = write(tmp_path / "d", train.iloc[:240] if test is not None else train, test, meta)
    automl.run(str(d), str(tmp_path / "out"))
    _, prof = outputs(tmp_path / "out")
    for key in ("task", "n_rows", "feature_types", "data_issues", "variations_detected", "summary", "model"):
        assert key in prof, key
    assert prof["model"]["status"] == "ok"
