"""Time-aware model selection for classification and regression.

1. Build candidate models (gradient boosting, random forest, extra trees, linear, kNN, SVM).
2. Score each with K-fold out-of-fold predictions, aborting candidates that would blow the budget.
3. Greedy (Caruana) ensemble selection on the out-of-fold predictions.
4. Tune the decision rule (threshold / prior correction) for the target metric.
5. Refit the chosen models on all training rows and predict the test rows.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import (ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

from . import config
from .metrics import best_binary_macro_f1, classification_score, regression_score, tuned_classification_score

warnings.filterwarnings("ignore")
RS = config.RANDOM_STATE


# ----------------------------------------------------------------------------- candidates
def _hgb_params(n: int, cat_mask, deep: bool = False, n_classes: int = 2) -> dict:
    cats = cat_mask if cat_mask is not None and cat_mask.any() else None
    # fewer histogram bins on small tables: same accuracy, much cheaper histograms
    # (native categoricals need max_bins >= their cardinality, so keep 255 when present)
    max_bins = 255 if cats is not None else int(np.clip(n // 4, 32, 255))
    if n < 1000:
        p = dict(learning_rate=0.07, max_iter=160, max_leaf_nodes=15, min_samples_leaf=8,
                 l2_regularization=1.0, early_stopping=False)
    else:
        # (15 leaves won a CV-only comparison, tools/tune_hgb.py, but needs many more rounds and was then
        #  aborted by the time budget on 3 regression sets: rejected, see EXPERIMENTS.md, experiment B)
        p = dict(learning_rate=0.07, max_iter=700, max_leaf_nodes=31, min_samples_leaf=20,
                 l2_regularization=1.0, early_stopping=True, validation_fraction=0.12,
                 n_iter_no_change=30, tol=1e-5)
    if deep:
        p.update(max_leaf_nodes=63 if n >= 1000 else 31, learning_rate=0.05, min_samples_leaf=5 if n < 1000 else 10,
                 max_features=0.6, l2_regularization=0.3)
        if n >= 1000:
            p["max_iter"] = 1000
    if n_classes >= 5:
        # one tree per class per iteration: use bigger steps so the cost stays bounded
        p["learning_rate"] = max(p["learning_rate"], 0.12)
        p["max_iter"] = min(p["max_iter"], 300)
        if p.get("early_stopping"):
            p["n_iter_no_change"] = 15
    return dict(p, max_bins=max_bins, categorical_features=cats, random_state=RS)


CORE_MODELS = {"hgb", "logreg", "ridge", "hgb_poisson"}  # always allowed to use the hard time budget
# (hgb_poisson only exists for count targets, where it is usually the strongest model)


class FastSVC:
    """RBF SVM without sklearn's internal 5-fold probability calibration (2-3x faster).

    Probabilities come from a softmax (sigmoid for two classes) of the decision values. The ensemble
    step scores every model on out-of-fold predictions, so a rougher probability scale cannot sneak
    a weak model into the blend."""

    def __init__(self, C=10.0):
        self.C = C

    def fit(self, X, y):
        self.model = SVC(C=self.C, gamma="scale", decision_function_shape="ovr", random_state=RS).fit(X, y)
        self.classes_ = self.model.classes_
        return self

    def predict_proba(self, X):
        d = self.model.decision_function(X)
        if d.ndim == 1:
            p1 = 1.0 / (1.0 + np.exp(-2.0 * d))
            return np.column_stack([1 - p1, p1])
        z = np.exp(2.0 * (d - d.max(1, keepdims=True)))
        return z / z.sum(1, keepdims=True)

    def predict(self, X):
        return self.model.predict(X)


def classification_candidates(n, d_lin, cat_mask, imbalanced, n_classes=2):
    """Cheap models first (they cost < 1 s), then boosting, then forests and variants."""
    n_trees, ms = _forest_size(n)
    c = [("logreg", "linear", lambda: LogisticRegression(C=0.5, max_iter=3000))]
    if n <= 20000 and d_lin <= 200:
        k = int(np.clip(np.sqrt(n) / 2, 5, 25))
        c.append(("knn", "linear", lambda: KNeighborsClassifier(n_neighbors=k, weights="distance")))
    if n <= 4000 and d_lin <= 500:
        c.append(("svc", "linear", lambda: SVC(C=1.0, gamma="scale", probability=True, random_state=RS)))
        # a tighter fit: often better on clean, well-separated data (digits-like)
        if n <= 2500:   # calibration is cheap here, and calibrated probabilities blend better
            c.append(("svc_c10", "linear", lambda: SVC(C=10.0, gamma="scale", probability=True, random_state=RS)))
        else:           # larger tables: skip the 5 internal calibration fits (2-3x faster)
            c.append(("svc_c10", "linear", lambda: FastSVC(C=10.0)))
    if n <= 20000 and d_lin <= 200 and n >= 200:
        c.append(("knn_local", "linear", lambda: KNeighborsClassifier(n_neighbors=5, weights="distance")))
    c += [
        ("hgb", "tree", lambda: HistGradientBoostingClassifier(**_hgb_params(n, cat_mask, n_classes=n_classes))),
        ("et", "linear", lambda: ExtraTreesClassifier(n_estimators=n_trees, max_features="sqrt",
                                                      min_samples_leaf=1, max_samples=ms, n_jobs=config.N_JOBS, random_state=RS)),
        ("rf", "tree", lambda: RandomForestClassifier(n_estimators=n_trees, max_features="sqrt",
                                                       min_samples_leaf=1, max_samples=ms, n_jobs=config.N_JOBS, random_state=RS)),
    ]
    if imbalanced:
        c.append(("hgb_balanced", "tree", lambda: HistGradientBoostingClassifier(
            **_hgb_params(n, cat_mask, n_classes=n_classes), class_weight="balanced")))
    c.append(("hgb_deep", "tree", lambda: HistGradientBoostingClassifier(**_hgb_params(n, cat_mask, deep=True, n_classes=n_classes))))
    return c


def regression_candidates(n, d_lin, cat_mask, y):
    n_trees, ms = _forest_size(n)
    log_ok = bool(np.min(y) >= 0 and abs(_skew(y)) > 0.75)
    c = [("ridge", "linear", lambda: RidgeCV(alphas=np.logspace(-3, 3, 13)))]
    if log_ok:
        c.append(("ridge_log", "linear", lambda: TransformedTargetRegressor(
            RidgeCV(alphas=np.logspace(-3, 3, 13)), func=np.log1p, inverse_func=_safe_expm1)))
    if d_lin <= POLY_MAX_COLUMNS and n <= 50000:
        c.append(("ridge_poly", "linear", lambda: _poly(RidgeCV(alphas=np.logspace(-2, 4, 13)))))
        if log_ok:
            c.append(("ridge_log_poly", "linear", lambda: TransformedTargetRegressor(
                _poly(RidgeCV(alphas=np.logspace(-2, 4, 13))), func=np.log1p, inverse_func=_safe_expm1)))
    if n <= 20000 and d_lin <= 200:
        k = int(np.clip(np.sqrt(n) / 2, 5, 25))
        c.append(("knn", "linear", lambda: KNeighborsRegressor(n_neighbors=k, weights="distance")))
    if n <= 4000 and d_lin <= 500:
        c.append(("svr", "linear", lambda: TransformedTargetRegressor(
            SVR(C=3.0, gamma="scale", epsilon=0.05), transformer=StandardScaler())))
        c.append(("svr_c30", "linear", lambda: TransformedTargetRegressor(
            SVR(C=30.0, gamma="scale", epsilon=0.05), transformer=StandardScaler())))
    c.append(("hgb", "tree", lambda: HistGradientBoostingRegressor(**_hgb_params(n, cat_mask))))
    y_arr = np.asarray(y, float)
    if np.min(y_arr) >= 0 and np.sum(y_arr) > 0 and np.allclose(y_arr, np.round(y_arr)) and len(np.unique(y_arr)) > 10:
        # non-negative integer target = counts (rentals, visits...): Poisson deviance fits them best
        c.append(("hgb_poisson", "tree", lambda: HistGradientBoostingRegressor(
            **dict(_hgb_params(n, cat_mask), loss="poisson"))))
    if log_ok:
        c.append(("hgb_log", "tree", lambda: TransformedTargetRegressor(
            HistGradientBoostingRegressor(**_hgb_params(n, cat_mask)), func=np.log1p, inverse_func=_safe_expm1)))
    c += [
        ("et", "linear", lambda: ExtraTreesRegressor(n_estimators=n_trees, max_features=0.7,
                                                     min_samples_leaf=1, max_samples=ms, n_jobs=config.N_JOBS, random_state=RS)),
        ("rf", "tree", lambda: RandomForestRegressor(n_estimators=n_trees, max_features=0.5,
                                                      min_samples_leaf=1, max_samples=ms, n_jobs=config.N_JOBS, random_state=RS)),
    ]
    c.append(("hgb_deep", "tree", lambda: HistGradientBoostingRegressor(**_hgb_params(n, cat_mask, deep=True))))
    return c


POLY_MAX_COLUMNS = 40   # pairwise products: 40 columns -> 860 features, still fast for linear models


def _poly(model):
    """Linear model on all pairwise products of the (standardised) linear view: captures interactions
    such as quality x area that plain linear models cannot represent and trees only approximate."""
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures
    return make_pipeline(PolynomialFeatures(2, include_bias=False), StandardScaler(), model)


def _forest_size(n):
    if n <= 5000:
        return 200, None
    return 120, (0.5 if n > 10000 else None)


def _safe_expm1(x):
    return np.expm1(np.clip(x, -50, 50))


def _skew(y):
    y = np.asarray(y, float)
    s = y.std()
    return 0.0 if s == 0 else float(np.mean(((y - y.mean()) / s) ** 3))


# ----------------------------------------------------------------------------- time-adaptive boosting
def _inner_hgb(model):
    """The HistGradientBoosting estimator inside a candidate (plain or wrapped in a target transform)."""
    inner = getattr(model, "regressor", model)
    return inner if hasattr(inner, "max_iter") else None


def _fit_boosting_budget(make, X_fold, y_fold, n_folds, refit_factor, time_left, log, name):
    """Probe a boosting candidate with 10 rounds and shrink it if its full version cannot finish in time.

    Returns (factory, note). The factory is unchanged when the model fits the budget. Otherwise rounds
    are reduced and the learning rate raised in proportion (same total shrinkage), so a strong model is
    still evaluated on a slow machine instead of being aborted. Returns (None, note) when even a short
    version does not fit."""
    probe = make()
    inner = _inner_hgb(probe)
    if inner is None:
        return make, None
    full_iters, lr = inner.max_iter, inner.learning_rate
    uses_early_stopping = inner.early_stopping not in (False, "off")  # read BEFORE the probe changes it
    # two probes separate the fixed cost of a fit (binning the data) from the cost per round
    times = []
    for k in (1, 10):
        inner.set_params(max_iter=k, early_stopping=False)
        t = time.time()
        probe.fit(X_fold, y_fold)
        times.append(time.time() - t)
    per_iter = max((times[1] - times[0]) / 9, 1e-6)
    fixed = max(times[0] - per_iter, 0.0)
    fits = n_folds + refit_factor
    # early stopping usually ends long before max_iter; 250 rounds is a realistic planning figure
    planned = min(full_iters, 250) if uses_early_stopping else full_iters
    need = fits * (fixed + per_iter * planned)
    if need <= time_left:
        return make, None
    iters = int((time_left / fits - fixed) / per_iter)
    if iters < 25:
        return None, f"too slow even shortened ({per_iter * 1000:.0f} ms/round)"
    new_lr = min(0.3, lr * planned / iters)
    note = f"shortened to {iters} rounds, learning rate {new_lr:.3f} (time budget)"
    log(f"  {name:13s} {note}")

    def shortened():
        m = make()
        _inner_hgb(m).set_params(max_iter=iters, learning_rate=new_lr)
        return m
    return shortened, note


# ----------------------------------------------------------------------------- helpers
def _proba_full(model, X, n_classes):
    p = model.predict_proba(X)
    out = np.zeros((X.shape[0], n_classes))
    out[:, model.classes_.astype(int)] = p
    return out


def _splits(y, task, n):
    if n < 4:
        raise ValueError(f"only {n} labelled rows: too few to fit and validate a model")
    k = min(5 if n <= 2000 else 3, n // 2)
    if task == "classification":
        counts = np.bincount(y)
        m = counts[counts > 0].min()
        if m >= 2:
            k = int(min(k, m))
            return list(StratifiedKFold(k, shuffle=True, random_state=RS).split(np.zeros(n), y))
    return list(KFold(k, shuffle=True, random_state=RS).split(np.zeros(n)))


def _greedy_ensemble(oofs: dict, score_fn, max_steps=15):
    names = sorted(oofs, key=lambda k: -score_fn(oofs[k]))
    if not names:
        return {}, -np.inf
    chosen = [names[0]]
    best = score_fn(oofs[names[0]])
    current = oofs[names[0]].copy()
    for _ in range(max_steps):
        step_best, step_name = best, None
        for nm in names:
            cand = (current * len(chosen) + oofs[nm]) / (len(chosen) + 1)
            s = score_fn(cand)
            if s > step_best + 1e-6:
                step_best, step_name = s, nm
        if step_name is None:
            break
        current = (current * len(chosen) + oofs[step_name]) / (len(chosen) + 1)
        chosen.append(step_name)
        best = step_best
    weights = {nm: chosen.count(nm) / len(chosen) for nm in set(chosen)}
    return weights, best


def _blend(preds: dict, weights: dict):
    return sum(preds[k] * w for k, w in weights.items())


# ----------------------------------------------------------------------------- decision rules
def tune_decision(proba, y, metric, priors):
    """Return (rule, score) where rule maps probabilities to labels."""
    base = ("argmax", None)
    best = (base, classification_score(metric, y, proba))
    if metric not in ("f1_macro", "balanced_accuracy", "f1_weighted"):
        return best
    for a in (0.25, 0.5, 0.75, 1.0):
        pred = apply_decision(proba, ("prior", a), priors)
        s = classification_score(metric, y, proba, pred)
        if s > best[1] + 1e-6:
            best = (("prior", a), s)
    if proba.shape[1] == 2:
        if metric == "f1_macro":
            f1, t = best_binary_macro_f1(y, proba[:, 1])   # exact optimum over all thresholds
            if f1 > best[1] + 1e-6:
                best = (("threshold", t), f1)
        else:
            for t in np.arange(0.05, 0.951, 0.01):
                pred = (proba[:, 1] >= t).astype(int)
                s = classification_score(metric, y, proba, pred)
                if s > best[1] + 1e-6:
                    best = (("threshold", float(t)), s)
    return best


def apply_decision(proba, rule, priors):
    kind, val = rule
    if kind == "threshold":
        return (proba[:, 1] >= val).astype(int)
    if kind == "prior":
        return (proba / np.power(np.clip(priors, 1e-9, None), val)).argmax(1)
    return proba.argmax(1)


# ----------------------------------------------------------------------------- main entry
def solve_supervised(task, views_train, views_test, y_raw, metric, deadline, search_deadline=None,
                     imbalanced=False, log=print, on_interim=None):
    """views_*: dict(tree=(X, cat_mask), linear=X). Returns dict with predictions and a report."""
    t_start = time.time()
    Xtr_tree, cat_mask = views_train["tree"]
    Xtr_lin = views_train["linear"]
    Xte = {"tree": views_test["tree"][0], "linear": views_test["linear"]}
    Xtr = {"tree": Xtr_tree, "linear": Xtr_lin}
    if "linear_sel" in views_train:  # noise columns removed (see relevance.py)
        Xtr["linear_sel"], Xte["linear_sel"] = views_train["linear_sel"], views_test["linear_sel"]
    n_full = len(y_raw)

    if task == "classification":
        classes, y = np.unique(np.asarray(y_raw).astype(str) if np.asarray(y_raw).dtype == object else y_raw,
                               return_inverse=True)
        n_classes = len(classes)
        priors = np.bincount(y, minlength=n_classes) / len(y)
    else:
        y = np.asarray(y_raw, dtype=float)
        classes, n_classes, priors = None, None, None

    # degenerate target
    if task == "classification" and n_classes == 1:
        n_te = Xte["tree"].shape[0]
        return dict(pred=np.repeat(classes[0], n_te), proba=np.ones((n_te, 1)), classes=classes,
                    report={"note": "single class in training data"})

    # subsample for selection on large tables
    sel = np.arange(n_full)
    if n_full > config.MAX_SELECTION_ROWS:
        strat = y if task == "classification" and np.bincount(y).min() >= 2 else None
        sel, _ = train_test_split(sel, train_size=config.MAX_SELECTION_ROWS, random_state=RS, stratify=strat)
        sel = np.sort(sel)
    ys = y[sel]
    n = len(sel)
    splits = _splits(ys, task, n)

    if task == "classification":
        cands = classification_candidates(n, Xtr_lin.shape[1], cat_mask, imbalanced, n_classes)
        # score every candidate at the decision rule it will be used with (best threshold / prior
        # correction for macro-F1): with argmax, an imbalanced target makes well-calibrated models look
        # bad and over-confident ones look good, so the wrong model was being selected
        score_fn = lambda p: tuned_classification_score(metric, ys, p, priors)
    else:
        cands = regression_candidates(n, Xtr_lin.shape[1], cat_mask, ys)
        score_fn = lambda p: regression_score(metric, ys, p)
    if "linear_sel" in Xtr:
        # distance-based models are the ones hurt by irrelevant columns
        cands = [(nm, "linear_sel" if nm.startswith(("knn", "svc", "svr")) else v, mk) for nm, v, mk in cands]

    if on_interim is not None:
        # safety net before any search: the first (linear) candidate costs milliseconds; its predictions
        # are saved at once, so a slow machine or the watchdog never leaves only the constant fallback
        name0, view0, make0 = cands[0]
        try:
            m = make0()
            m.fit(Xtr[view0], y)
            if task == "classification":
                p0 = _proba_full(m, Xte[view0], n_classes)
                on_interim(classes[p0.argmax(1)], p0, classes, name0)
            else:
                on_interim(m.predict(Xte[view0]), None, None, name0)
        except Exception as e:
            log(f"  interim prediction skipped: {type(e).__name__}: {e}")

    oofs, fit_times, scores, skipped, adjusted = {}, {}, {}, {}, {}
    search_deadline = search_deadline or deadline
    best_ens, stall = -np.inf, 0
    for name, view, make in cands:
        # converged: the ensemble stopped improving for several candidates in a row (speed points)
        # (small tables have noisy CV and cheap models, so they get more patience)
        patience = config.SEARCH_PATIENCE + (2 if n < 1000 else 0)
        if stall >= patience and CORE_MODELS & set(oofs) and any(k.startswith("hgb") for k in oofs):
            skipped[name] = "search converged"
            continue
        # core models (boosting + linear) may use the hard budget, others only the soft one
        limit = deadline if name in CORE_MODELS else search_deadline
        left = limit - time.time()
        best_refit = max([fit_times[k] for k in oofs], default=0.0) * 1.5 * (n_full / max(n, 1))
        if left < 2.0 + best_refit:
            skipped[name] = "time budget"
            continue
        X = Xtr[view][sel]
        oof = np.zeros((n, n_classes)) if task == "classification" else np.zeros(n)
        t0 = time.time()
        ok = True
        # the main boosting model is always probed (it must never be aborted: it is usually the best);
        # the variants only when the table is large, has many classes, or time is short
        at_risk = (name == "hgb" or n >= 5000 or (task == "classification" and n_classes >= 5)
                   or limit - time.time() < 20)
        if name.startswith("hgb") and at_risk:
            # the final refit falls back to the selection sample when time is short, so plan with
            # at most twice the selection rows rather than the whole (possibly huge) table
            refit_factor = 1.2 * min(n_full, 2 * n) / max(n * (1 - 1 / len(splits)), 1)
            try:
                make, note = _fit_boosting_budget(make, X[splits[0][0]], ys[splits[0][0]], len(splits),
                                                  refit_factor, limit - time.time() - 1.5 - best_refit, log, name)
            except Exception as e:
                make, note = None, f"probe failed: {type(e).__name__}"
            if make is None:
                skipped[name] = note
                continue
            if note:
                adjusted[name] = note
                cands = [(nm, v, make if nm == name else mk) for nm, v, mk in cands]
            t0 = time.time()
        try:
            for i, (tr, va) in enumerate(splits):
                m = make()
                m.fit(X[tr], ys[tr])
                if task == "classification":
                    oof[va] = _proba_full(m, X[va], n_classes)
                else:
                    oof[va] = m.predict(X[va])
                per_fold = (time.time() - t0) / (i + 1)
                remaining_folds = len(splits) - i - 1
                if time.time() + per_fold * remaining_folds > limit - 1.5 - best_refit:
                    ok = False
                    skipped[name] = "aborted (too slow)"
                    break
        except Exception as e:  # a failing candidate must never kill the run
            ok = False
            skipped[name] = f"error: {type(e).__name__}: {e}"[:200]
        if not ok:
            continue
        if task == "regression" and not np.all(np.isfinite(oof)):
            skipped[name] = "non-finite predictions"
            continue
        fit_times[name] = (time.time() - t0) / len(splits)
        oofs[name] = oof
        scores[name] = float(score_fn(oof))
        log(f"  {name:13s} cv {metric}={scores[name]:.4f}  ({fit_times[name]:.2f}s/fold)")
        ens = _greedy_ensemble(oofs, score_fn, max_steps=8)[1]
        if ens > best_ens + 1e-6:  # any real gain resets the patience counter
            best_ens, stall = ens, 0
        else:
            stall += 1

    if not oofs:
        raise RuntimeError("no candidate model could be evaluated in time")

    weights, ens_score = _greedy_ensemble(oofs, score_fn)
    blend_oof = _blend(oofs, weights)
    rule, final_cv = None, ens_score
    if task == "classification":
        rule, final_cv = tune_decision(blend_oof, ys, metric, priors)

    # ---- refit on all rows (or as many as time allows) and predict
    models_by_name = {nm: (view, make) for nm, view, make in cands}
    est_refit = sum(fit_times[k] for k in weights) * (n_full / max(n * (1 - 1 / len(splits)), 1)) * 1.2
    fit_idx = np.arange(n_full)
    if time.time() + est_refit > deadline:
        # too slow to refit everything on all rows: keep the single fastest good model
        best_single = max(weights, key=lambda k: scores[k] - 1e-3 * fit_times[k])
        weights = {best_single: 1.0}
        est = fit_times[best_single] * (n_full / max(n, 1)) * 1.2
        if time.time() + est > deadline:
            fit_idx = sel
    test_preds, used = {}, {}
    for nm in sorted(weights, key=lambda k: -weights[k]):
        if test_preds and time.time() + fit_times[nm] * (len(fit_idx) / max(n, 1)) * 1.2 > deadline:
            continue
        view, make = models_by_name[nm]
        m = make()
        m.fit(Xtr[view][fit_idx], y[fit_idx])
        if task == "classification":
            test_preds[nm] = _proba_full(m, Xte[view], n_classes)
        else:
            test_preds[nm] = m.predict(Xte[view])
        used[nm] = weights[nm]
    tot = sum(used.values())
    used = {k: v / tot for k, v in used.items()}
    blended = _blend(test_preds, used)

    report = {
        "selection_metric": metric,
        "cv_folds": len(splits),
        "selection_rows": int(n),
        "candidate_cv_scores": scores,
        "candidate_fit_seconds_per_fold": fit_times,
        "skipped_candidates": skipped,
        "time_adjusted_candidates": adjusted,
        "ensemble_weights": used,
        "ensemble_cv_score": float(ens_score),
        "final_cv_score": float(final_cv),
        "best_single_model": max(scores, key=scores.get),
        "supervised_seconds": round(time.time() - t_start, 3),
    }
    if task == "classification":
        report["decision_rule"] = {"kind": rule[0], "value": rule[1]}
        idx = apply_decision(blended, rule, priors)
        return dict(pred=classes[idx], proba=blended, classes=classes, report=report)
    if np.min(y) >= 0 and abs(_skew(y)) > 0.75:
        blended = np.clip(blended, 0, None)
    lo, hi = np.min(y), np.max(y)
    span = hi - lo
    blended = np.clip(blended, lo - 0.5 * span, hi + 0.5 * span)
    return dict(pred=blended, proba=None, classes=None, report=report)
