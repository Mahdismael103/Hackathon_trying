"""Scoring helpers. All scores are returned as *higher is better*."""
from __future__ import annotations

import numpy as np
from sklearn import metrics as skm

CLS_METRICS = {"f1_macro", "f1", "accuracy", "balanced_accuracy", "roc_auc", "log_loss", "f1_weighted"}
REG_METRICS = {"rmse", "mse", "mae", "r2", "rmsle"}

_ALIASES = {
    "f1macro": "f1_macro", "macro_f1": "f1_macro", "f1-macro": "f1_macro", "f1": "f1_macro",
    "acc": "accuracy", "balanced_acc": "balanced_accuracy", "bacc": "balanced_accuracy",
    "auc": "roc_auc", "roc": "roc_auc", "logloss": "log_loss", "cross_entropy": "log_loss",
    "weighted_f1": "f1_weighted",
    "root_mean_squared_error": "rmse", "mean_squared_error": "mse", "mean_absolute_error": "mae",
    "r2_score": "r2", "r^2": "r2", "rmsle": "rmsle", "root_mean_squared_log_error": "rmsle",
}


def normalise_metric(metric: str | None, task: str, default: str) -> str:
    if not metric:
        return default
    m = str(metric).strip().lower().replace(" ", "_")
    m = _ALIASES.get(m, m)
    valid = CLS_METRICS if task == "classification" else REG_METRICS
    return m if m in valid else default


def classification_score(metric: str, y: np.ndarray, proba: np.ndarray, pred: np.ndarray | None = None) -> float:
    if pred is None:
        pred = proba.argmax(1)
    if metric == "accuracy":
        return skm.accuracy_score(y, pred)
    if metric == "balanced_accuracy":
        return skm.balanced_accuracy_score(y, pred)
    if metric == "f1_weighted":
        return skm.f1_score(y, pred, average="weighted")
    if metric == "roc_auc":
        try:
            if proba.shape[1] == 2:
                return skm.roc_auc_score(y, proba[:, 1])
            return skm.roc_auc_score(y, proba, multi_class="ovr", labels=np.arange(proba.shape[1]))
        except ValueError:
            return skm.accuracy_score(y, pred)
    if metric == "log_loss":
        p = np.clip(proba, 1e-7, 1)
        p = p / p.sum(1, keepdims=True)
        return -skm.log_loss(y, p, labels=np.arange(proba.shape[1]))
    return skm.f1_score(y, pred, average="macro")


def regression_score(metric: str, y: np.ndarray, pred: np.ndarray) -> float:
    if metric == "mae":
        return -skm.mean_absolute_error(y, pred)
    if metric == "r2":
        return skm.r2_score(y, pred)
    if metric == "mse":
        return -skm.mean_squared_error(y, pred)
    if metric == "rmsle":
        return -float(np.sqrt(np.mean((np.log1p(np.clip(pred, 0, None)) - np.log1p(np.clip(y, 0, None))) ** 2)))
    return -float(np.sqrt(skm.mean_squared_error(y, pred)))


# ----------------------------------------------------------------------------- fast macro-F1
def fast_macro_f1(y: np.ndarray, pred: np.ndarray, n_classes: int) -> float:
    """Macro-F1 from a confusion matrix (same value as sklearn's, ~100x faster on repeated calls)."""
    cm = np.bincount(y * n_classes + pred, minlength=n_classes * n_classes).reshape(n_classes, n_classes)
    tp = np.diag(cm).astype(float)
    denom = cm.sum(0) + cm.sum(1)
    present = denom > 0                      # labels that occur in y or in the predictions
    return float(np.mean(2 * tp[present] / denom[present]))


def best_binary_macro_f1(y: np.ndarray, p1: np.ndarray) -> tuple[float, float]:
    """Best macro-F1 over all decision thresholds on P(class 1), in one sorted pass. Returns (f1, threshold)."""
    order = np.argsort(-p1, kind="mergesort")
    ps, ys = p1[order], y[order]
    pos = ys.sum()
    neg = len(ys) - pos
    tp = np.cumsum(ys)                       # predicting the top k rows as positive, k = 1..n
    k = np.arange(1, len(ys) + 1)
    fp = k - tp
    fn = pos - tp
    tn = neg - fp
    last = np.r_[ps[1:] != ps[:-1], True]    # only cut between different probabilities
    with np.errstate(divide="ignore", invalid="ignore"):
        f1_pos = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), 0.0)
        f1_neg = np.where(2 * tn + fn + fp > 0, 2 * tn / (2 * tn + fn + fp), 0.0)
    macro = np.where(last, (f1_pos + f1_neg) / 2, -1.0)
    i = int(np.argmax(macro))
    # also "predict nobody positive" (threshold above every probability)
    none_pos = (0.0 + (2 * neg / (2 * neg + pos) if neg else 0.0)) / 2
    if none_pos > macro[i]:
        return float(none_pos), float(ps[0] + 1e-9)
    nxt = ps[i + 1] if i + 1 < len(ps) else 0.0
    return float(macro[i]), float((ps[i] + nxt) / 2)


def tuned_classification_score(metric: str, y: np.ndarray, proba: np.ndarray, priors: np.ndarray) -> float:
    """Score at the decision rule the model will actually be used with (see supervised.tune_decision).

    For macro-F1 this is the best threshold (binary) or the best prior correction (multiclass); other
    metrics are threshold-free or use argmax, so they are scored as is."""
    if metric != "f1_macro":
        return classification_score(metric, y, proba)
    k = proba.shape[1]
    if k == 2:
        return max(best_binary_macro_f1(y, proba[:, 1])[0], fast_macro_f1(y, proba.argmax(1), 2))
    best = fast_macro_f1(y, proba.argmax(1), k)
    for a in (0.25, 0.5, 0.75, 1.0):
        best = max(best, fast_macro_f1(y, (proba / np.power(np.clip(priors, 1e-9, None), a)).argmax(1), k))
    return best
