"""End-to-end orchestration: load -> profile -> prepare -> model -> write outputs."""
from __future__ import annotations

import re
import sys
import time
import traceback

import numpy as np
import pandas as pd

from . import config
from .io_contract import TaskSpec, write_predictions, write_profile
from .metrics import normalise_metric
from .preprocess import Preparer
from .profiling import profile_dataset, to_numeric
from .relevance import original_name


def log(msg: str):
    print(msg, file=sys.stderr, flush=True)


def _ids(spec: TaskSpec, frame: pd.DataFrame):
    if spec.id_column and spec.id_column in frame.columns:
        return frame[spec.id_column].values, str(spec.id_column)
    return np.arange(len(frame)), config.DEFAULT_ID_COLUMN


def _clean_class_target(y: pd.Series) -> pd.Series:
    if pd.api.types.is_float_dtype(y) and np.allclose(y.values, np.round(y.values)):
        return y.astype(np.int64)
    if pd.api.types.is_bool_dtype(y):
        return y.astype(str)
    return y


# "Id", "ID", "row_id", "UDI", "customerID", "PassengerId" - but not "paid" / "valid"
_ID_NAME = re.compile(r"^(id|uuid|guid|udi|index|idx|rowid|row_id|row|record_id)$|[_\s\-.](id|uuid|udi)$", re.I)
_CAMEL_ID = re.compile(r"[a-z](ID|Id)$")


def _detect_id_column(train: pd.DataFrame, pred_frame: pd.DataFrame, target) -> str | None:
    """An identifier column to report predictions against when the metadata names none."""
    for c in pred_frame.columns:
        if c == target or c not in train.columns:
            continue
        name = str(c).strip()
        if not (_ID_NAME.search(name) or _CAMEL_ID.search(name)):
            continue
        s = pred_frame[c]
        if s.notna().all() and s.nunique() == len(s) and train[c].nunique() >= 0.98 * len(train):
            return c
    return None


def peak_memory_mb() -> float | None:
    """Peak resident memory of this process (Windows and Unix), for the report only."""
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class PMC(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            k32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
            k32.K32GetProcessMemoryInfo.restype = wintypes.BOOL
            pmc = PMC()
            pmc.cb = ctypes.sizeof(PMC)
            if not k32.K32GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
                return None
            return round(pmc.PeakWorkingSetSize / 2**20, 1)
        import resource
        kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(kb / 1024 / (1024 if sys.platform == "darwin" else 1), 1)
    except Exception:
        return None


def _environment() -> dict:
    import platform
    import sklearn
    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__, "cpu_count": config.available_cpus(), "platform": platform.platform()}


LINEAR_MODELS = {"logreg", "ridge", "ridge_log"}


def _nonlinearity(report: dict, task: str) -> dict:
    """Compare the best linear and best non-linear model in cross-validation."""
    sc = report.get("candidate_cv_scores", {})
    lin = [v for k, v in sc.items() if k in LINEAR_MODELS]
    non = [v for k, v in sc.items() if k not in LINEAR_MODELS]
    if not lin or not non:
        return {"is_nonlinear": None}
    bl, bn = max(lin), max(non)
    # scores are "higher is better"; error metrics are negative, so use a relative margin for them
    error_metric = report.get("selection_metric") in ("rmse", "mae", "mse", "rmsle")
    margin = 0.05 * abs(bl) if error_metric or bl < 0 else 0.01
    return {"best_linear_cv": bl, "best_nonlinear_cv": bn, "is_nonlinear": bool(bn > bl + margin)}


def _cluster_relevance(X, labels, names) -> dict:
    """eta^2 (between-cluster variance share) per original feature."""
    out = {}
    tot = X.var(0) + 1e-12
    means = np.zeros_like(X)
    for g in np.unique(labels):
        m = labels == g
        means[m] = X[m].mean(0)
    eta = means.var(0) / tot
    for nm, v in zip(names, eta):
        o = nm.split("==")[0].split("__")[0]
        out[o] = round(max(out.get(o, 0.0), float(v)), 5)
    return {"eta_squared": out, "irrelevant_features": sorted(k for k, v in out.items() if v < 0.05)}


def _data_issues(p: dict) -> dict:
    tp = p.get("target_profile", {})
    irr = p.get("irrelevant_features")
    if irr is None:
        irr = p.get("cluster_feature_relevance", {}).get("irrelevant_features", [])
    return {
        "missing_values": bool(p.get("has_missing_values")),
        "differently_scaled_features": bool(p.get("features_differently_scaled", False)),
        "categorical_columns": bool(p.get("categorical_columns")),
        "datetime_columns": bool(p.get("datetime_columns")),
        "text_columns": bool(p.get("text_columns")),
        "constant_columns": bool(p.get("constant_columns")),
        "duplicate_columns": bool(p.get("duplicate_columns")),
        "id_like_columns": bool(p.get("id_like_columns")),
        "duplicate_rows": bool(p.get("duplicate_rows")),
        "irrelevant_features": bool(irr),
        "imbalanced_classes": bool(tp.get("is_imbalanced", False)),
        "skewed_target": bool(tp.get("is_skewed", False)),
        "nonlinear_relationships": p.get("nonlinearity", {}).get("is_nonlinear"),
    }


def _summary(p: dict, spec: TaskSpec) -> dict:
    """Compact profile under the most common key names (the official checker's names are unknown)."""
    tp = p.get("target_profile", {})
    return {
        "task": p.get("task"),
        "target": p.get("target"),
        "n_rows": p.get("n_rows"),
        "n_rows_test": p.get("n_rows_test"),
        "n_columns": p.get("n_columns"),
        "n_features": p.get("n_features"),
        "column_types": p.get("feature_types", {}),
        "numeric_features": p.get("numeric_columns", []),
        "categorical_features": p.get("categorical_columns", []),
        "datetime_features": p.get("datetime_columns", []),
        "text_features": p.get("text_columns", []),
        "missing_values": p.get("missing_by_column", {}),
        "missing_cells": p.get("missing_cells"),
        "constant_columns": p.get("constant_columns", []),
        "duplicate_columns": p.get("duplicate_columns", []),
        "id_columns": p.get("id_like_columns", []),
        "irrelevant_features": p.get("irrelevant_features",
                                     p.get("cluster_feature_relevance", {}).get("irrelevant_features", [])),
        "differently_scaled": p.get("features_differently_scaled", False),
        "imbalanced": tp.get("is_imbalanced", False),
        "class_distribution": tp.get("class_counts"),
        "n_classes": tp.get("n_classes"),
        "n_clusters": spec.n_clusters if p.get("task") == "clustering" else None,
        "nonlinear": p.get("nonlinearity", {}).get("is_nonlinear"),
        "variations": p.get("variations_detected", []),
    }


def solve(spec: TaskSpec, out_dir: str, t0: float | None = None, time_limit: float | None = None) -> dict:
    t0 = t0 or time.time()
    limit = time_limit or config.TIME_LIMIT_SECONDS
    # 12 s margin at the official 60 s limit, proportionally less for shorter limits
    deadline = t0 + limit - min(config.SAFETY_MARGIN_SECONDS, 0.2 * limit)
    task = spec.task
    timings = {"startup_and_loading": round(time.time() - t0, 2)}
    mark = [time.time()]

    def lap(name):
        now = time.time()
        timings[name] = round(now - mark[0], 2)
        mark[0] = now

    log(f"[automl] dataset={spec.name} task={task} target={spec.target} k={spec.n_clusters} "
        f"train={spec.train.shape} test={None if spec.test is None else spec.test.shape}")

    train = spec.train
    missing_target = 0
    if task != "clustering":
        target = train[spec.target]
        if task == "regression":
            target = to_numeric(target).replace([np.inf, -np.inf], np.nan)   # text numbers, infinities
            train = train.assign(**{spec.target: target})
        missing_target = int(target.isna().sum())
        train = train[target.notna()].reset_index(drop=True)
    pred_frame = spec.test if spec.test is not None else train
    if not spec.id_column:
        spec.id_column = _detect_id_column(train, pred_frame, spec.target)
    ids, id_name = _ids(spec, pred_frame)

    # ---------------------------------------------------------------- fallback output first
    if task == "classification":
        y = _clean_class_target(train[spec.target])
        fallback = np.repeat(y.mode().iloc[0] if len(y) else 0, len(pred_frame))
    elif task == "regression":
        fallback = np.repeat(float(train[spec.target].median()) if len(train) else 0.0, len(pred_frame))
    else:
        fallback = np.zeros(len(pred_frame), dtype=int)
    write_predictions(out_dir, ids, fallback, id_name)
    write_profile(out_dir, {"task": task, "target": spec.target, "n_rows": len(train),
                            "n_columns": int(train.shape[1]), "model": {"status": "fallback"},
                            "note": "minimal profile written at start; replaced when profiling finishes"})

    try:
        profile, roles = profile_dataset(train, spec.test, task, spec.target, spec.feature_types,
                                         spec.id_column, spec.n_clusters)
    except Exception as e:  # never lose a dataset to an unexpected column: fall back to simple roles
        log("[automl] profiling failed, using simple column roles:\n" + traceback.format_exc())
        roles = {c: ("numeric" if pd.api.types.is_numeric_dtype(train[c]) else "categorical")
                 for c in train.columns if c not in (spec.target, spec.id_column)
                 and (spec.test is None or c in spec.test.columns)}
        profile = {"task": task, "target": spec.target, "n_rows": len(train), "dropped_columns": {},
                   "profiling_error": f"{type(e).__name__}: {e}"}
    if task != "clustering":
        profile["rows_without_target_removed"] = missing_target
        profile["n_rows_train_original"] = len(spec.train)
        if "target_profile" in profile:
            profile["target_profile"]["missing_target"] = missing_target
    if spec.id_column:
        roles.pop(spec.id_column, None)
    feats = list(roles)
    log(f"[automl] roles: {sum(r == 'numeric' for r in roles.values())} numeric, "
        f"{sum(r == 'categorical' for r in roles.values())} categorical, "
        f"{sum(r == 'datetime' for r in roles.values())} datetime, "
        f"{sum(r == 'text' for r in roles.values())} text; dropped {list(profile['dropped_columns'])}")
    lap("profiling")
    profile["environment"] = _environment()
    profile["model"] = {"status": "fallback"}
    write_profile(out_dir, profile)

    try:
        if task == "clustering":
            from .clustering import solve_clustering  # lazy: import only what the task needs
            frames =[train[feats]] + ([spec.test[feats]] if spec.test is not None else [])
            all_df = pd.concat(frames, ignore_index=True)
            prep = Preparer(roles, clustering=True).fit(all_df)
            X = prep.transform_linear(all_df, onehot_weight=0.7)
            lap("preparation")
            def interim_clusters(all_labels):
                lab = all_labels[len(train):] if spec.test is not None else all_labels
                write_predictions(out_dir, ids, lab, id_name)
                profile["model"] = {"status": "interim", "interim_model": "kmeans (quick)"}
                write_profile(out_dir, profile)
                log("[automl] interim cluster labels saved (quick k-means)")

            res = solve_clustering(X, spec.n_clusters, deadline, log=log, on_interim=interim_clusters,
                                   onehot_columns=sum("==" in nm for nm in prep.linear_names),
                                   continuous_mask=["==" not in nm for nm in prep.linear_names])
            labels = res["labels"]
            profile["cluster_feature_relevance"] = _cluster_relevance(X, labels, prep.linear_names)
            if spec.test is not None:
                all_ids = np.concatenate([_ids(spec, train)[0], ids]) if spec.id_column else np.arange(len(labels))
                write_predictions(out_dir, all_ids, labels, id_name, filename=config.ALL_ROWS_PREDICTIONS_FILE)
                labels = labels[len(train):]
            write_predictions(out_dir, ids, labels, id_name)
            model_report = dict(res["report"], status="ok")
            lap("clustering")
        else:
            from .supervised import solve_supervised
            metric = normalise_metric(spec.metric, task, config.DEFAULT_CLASSIFICATION_METRIC
                                      if task == "classification" else config.DEFAULT_REGRESSION_METRIC)
            y = _clean_class_target(train[spec.target]) if task == "classification" \
                else to_numeric(train[spec.target])
            prep = Preparer(roles).fit(train[feats], y=y.values, task=task)
            test = spec.test if spec.test is not None else train
            vtr = {"tree": prep.transform_tree(train[feats], train=True),
                   "linear": prep.transform_linear(train[feats], train=True)}
            vte = {"tree": prep.transform_tree(test[feats]), "linear": prep.transform_linear(test[feats])}
            lap("preparation")
            log(f"[automl] prepared: tree view {vtr['tree'][0].shape}, linear view {vtr['linear'].shape}; "
                f"metric={metric}")
            from .relevance import SCREEN_MIN_COLUMNS, screen_features
            if vtr["tree"][0].shape[1] > SCREEN_MIN_COLUMNS:
                # very wide table: rank columns once and give every model only the strongest ones
                scr = screen_features(vtr["tree"][0], y.values, task, list(prep.tree_names))
                keep = set(scr["keep"])
                tcols = np.array([original_name(nm) in keep for nm in prep.tree_names])
                lcols = np.array([original_name(nm) in keep for nm in prep.linear_names])
                vtr = {"tree": (vtr["tree"][0][:, tcols], vtr["tree"][1][tcols]), "linear": vtr["linear"][:, lcols]}
                vte = {"tree": (vte["tree"][0][:, tcols], vte["tree"][1][tcols]), "linear": vte["linear"][:, lcols]}
                prep.tree_names = [nm for nm, k in zip(prep.tree_names, tcols) if k]
                prep.linear_names = [nm for nm, k in zip(prep.linear_names, lcols) if k]
                profile["feature_screening"] = {"kept": len(scr["keep"]), "screened_out": scr["screened_out"],
                                                "method": "Extra-Trees importance, top features kept"}
                log(f"[automl] wide table: kept {len(scr['keep'])} strongest features, "
                    f"screened out {len(scr['screened_out'])}")
            try:
                from .relevance import feature_relevance
                rel = feature_relevance(vtr["linear"], y.values, prep.linear_names, task,
                                        tree_view=vtr["tree"], tree_names=list(prep.tree_names),
                                        budget_seconds=config.RELEVANCE_BUDGET_SECONDS)
                profile["irrelevant_features"] = sorted(set(rel["irrelevant_features"]) | set(
                    profile.get("feature_screening", {}).get("screened_out", [])))
                profile["feature_relevance_mi"] = rel["scores"]
                profile["feature_permutation_importance"] = rel["permutation_importance"]
                mask = rel["relevant_mask"]
                if 0 < mask.sum() < len(mask):
                    vtr["linear_sel"], vte["linear_sel"] = vtr["linear"][:, mask], vte["linear"][:, mask]
                log(f"[automl] relevance: {len(rel['irrelevant_features'])} irrelevant feature(s) "
                    f"{rel['irrelevant_features'][:10]}")
            except Exception as e:  # relevance is advisory only
                log(f"[automl] relevance skipped: {e}")
            lap("feature_relevance")
            def interim(pred, proba, classes, model_name):
                write_predictions(out_dir, ids, pred, id_name, proba=proba, class_labels=classes)
                profile["model"] = {"status": "interim", "interim_model": model_name}
                write_profile(out_dir, profile)
                log(f"[automl] interim predictions saved ({model_name})")

            res = solve_supervised(task, vtr, vte, y.values, metric, deadline, on_interim=interim,
                                   # soft budget from start, but never less than MIN_SEARCH_SECONDS of
                                   # search when loading / profiling were slow (busy or slow CPU)
                                   search_deadline=min(deadline, max(t0 + config.SOFT_SEARCH_SECONDS,
                                                                     time.time() + config.MIN_SEARCH_SECONDS)),
                                   imbalanced=bool(profile.get("is_imbalanced", False)), log=log)
            write_predictions(out_dir, ids, res["pred"], id_name, proba=res["proba"], class_labels=res["classes"])
            model_report = dict(res["report"], status="ok")
            profile["nonlinearity"] = _nonlinearity(res["report"], task)
            lap("model_search_and_fit")
        model_report["preprocessing"] = {
            "dropped_columns": profile["dropped_columns"],
            "numeric_imputation": "median (+ missing indicators for linear models); NaN native for trees",
            "categorical_encoding": "ordinal/native for boosting, top-20 one-hot + frequency for linear/distance",
            "scaling": "0.1-99.9% clipping, log1p for skewed non-negative, standardisation",
            "log_transformed_columns": getattr(prep, "log_cols", []),
        }
    except Exception as e:
        log("[automl] modelling stopped early, keeping the last saved predictions:\n" + traceback.format_exc())
        prev = profile.get("model", {})
        model_report = {"status": "interim" if prev.get("status") == "interim" else "fallback",
                        "interim_model": prev.get("interim_model"), "error": f"{type(e).__name__}: {e}"}

    profile["model"] = model_report
    profile["data_issues"] = _data_issues(profile)
    profile["variations_detected"] = [k for k, v in profile["data_issues"].items() if v]
    profile["summary"] = _summary(profile, spec)
    profile["runtime_seconds"] = round(time.time() - t0, 3)
    timings["total"] = profile["runtime_seconds"]
    profile["peak_memory_mb"] = peak_memory_mb()
    profile["timings"] = timings
    write_profile(out_dir, profile)
    log(f"[automl] done in {profile['runtime_seconds']}s -> {out_dir}")
    return profile
