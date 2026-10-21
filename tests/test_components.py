"""Unit tests for the building blocks (fast, no full pipeline)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from automl.io_contract import infer_task, normalise_feature_types, normalise_task
from automl.metrics import classification_score, normalise_metric, regression_score
from automl.preprocess import Preparer
from automl.profiling import infer_type, is_id_like, profile_dataset

rng = np.random.default_rng(1)


def test_task_and_type_aliases():
    assert normalise_task("Binary Classification") == "classification"
    assert normalise_task("unsupervised") == "clustering"
    assert normalise_task("nonsense") is None
    ft = normalise_feature_types([{"name": "a", "type": "float"}, {"name": "b", "type": "nominal"}])
    assert ft == {"a": "numeric", "b": "categorical"}
    assert normalise_feature_types({"numeric": ["x"], "categorical": ["c"]}) == {"x": "numeric", "c": "categorical"}


def test_task_inference():
    df = pd.DataFrame({"x": rng.normal(size=200), "lab": rng.choice(["a", "b"], 200), "val": rng.normal(size=200)})
    assert infer_task(df, "lab", None) == "classification"
    assert infer_task(df, "val", None) == "regression"
    assert infer_task(df, None, 3) == "clustering"


def test_type_inference():
    n = 500
    assert infer_type(pd.Series(rng.normal(size=n)), "x", n) == "numeric"
    assert infer_type(pd.Series([str(v) for v in rng.normal(size=n)]), "x", n) == "numeric"   # numbers as text
    assert infer_type(pd.Series(rng.choice(["a", "b"], n)), "x", n) == "categorical"
    dates = pd.Series(pd.date_range("2020-01-01", periods=n, freq="h").astype(str))
    assert infer_type(dates, "x", n) == "datetime"


def test_id_detection_is_strict():
    n = 300
    assert is_id_like(pd.Series(np.arange(n)), "row_id", "numeric", n)
    assert is_id_like(pd.Series([f"C{i:05d}" for i in range(n)]), "customer", "categorical", n)
    assert not is_id_like(pd.Series(rng.normal(size=n)), "price", "numeric", n)    # unique but continuous


def test_profile_detects_injected_variations():
    n = 400
    df = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n) * 1000, "c": rng.choice(["x", "y"], n)})
    df["a_copy"] = df["a"].astype(float)
    df["const"] = 5
    df.loc[:9, "a"] = np.nan
    df["y"] = np.where(rng.random(n) < 0.1, 1, 0)
    prof, roles = profile_dataset(df, None, "classification", "y", {}, None, None)
    assert prof["constant_columns"] == ["const"]
    assert prof["duplicate_columns"] == [] or "a_copy" not in roles  # a has NaNs, so a_copy differs
    assert prof["missing_by_column"]["a"] == 10
    assert prof["features_differently_scaled"]
    assert prof["target_profile"]["is_imbalanced"]
    assert roles["c"] == "categorical"


def test_target_encoding_is_out_of_fold():
    n = 1000
    cat = rng.choice([f"k{i}" for i in range(50)], n)
    y = rng.normal(size=n)                       # target unrelated to the category
    df = pd.DataFrame({"cat": cat})
    prep = Preparer({"cat": "categorical"}).fit(df, y=y, task="regression")
    X, _ = prep.transform_tree(df, train=True)
    te = X[:, [i for i, nm in enumerate(prep.tree_names) if "__te" in nm]].ravel()
    # out-of-fold encoding must not "explain" a random target (in-fold encoding would, correlation ~0.2+)
    assert abs(np.corrcoef(te, y)[0, 1]) < 0.12


def test_preparer_handles_unseen_categories_and_nans():
    tr = pd.DataFrame({"n": [1.0, 2.0, np.nan, 4.0] * 25, "c": ["a", "b", "a", "b"] * 25})
    te = pd.DataFrame({"n": [np.nan, 100.0], "c": ["zzz", None]})
    prep = Preparer({"n": "numeric", "c": "categorical"}).fit(tr)
    lin = prep.transform_linear(te)
    assert np.isfinite(lin).all()
    tree, mask = prep.transform_tree(te)
    assert tree.shape[0] == 2 and mask.dtype == bool


def test_metrics_are_higher_is_better():
    y = np.array([0, 1, 1, 0])
    proba = np.array([[.9, .1], [.2, .8], [.4, .6], [.7, .3]])
    assert classification_score("f1_macro", y, proba) == 1.0
    assert regression_score("rmse", np.array([1.0, 2.0]), np.array([1.0, 2.0])) == 0.0
    assert regression_score("rmse", np.array([1.0, 2.0]), np.array([2.0, 3.0])) < 0
    assert normalise_metric("Macro F1", "classification", "accuracy") == "f1_macro"
    assert normalise_metric("unknown", "regression", "rmse") == "rmse"
