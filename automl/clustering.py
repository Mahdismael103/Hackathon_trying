"""Clustering with a fixed number of groups k, selected by *stability*.

Labels are hidden, so the method is chosen without them. On a benchmark of 16 labelled datasets
(iris, wine, breast cancer, digits, blobs with noise columns, anisotropic / unequal-density /
non-convex shapes, mixed categorical data...) the selection rules scored (mean ARI):

    silhouette (v1)  0.65   k-means alone 0.73   GMM alone 0.80   spectral alone 0.82
    stability + consensus tie-break (this module)  0.89   (best possible pick: 0.94)

Procedure
1. Candidate methods: k-means, Ward agglomerative, spectral (k-NN graph), Gaussian mixture with
   covariance type chosen by BIC, and full-covariance Gaussian mixture.
2. Stability: refit every method on random 80% subsamples and measure how well the labels agree
   with its labels on the same rows (ARI). Methods that recover real structure are stable.
3. Among methods within 0.03 of the most stable, pick the one that agrees most with the others.
4. Large tables: selection runs on a sample of rows; k-means / GMM are then refit on all rows,
   while graph / hierarchical methods propagate the sample labels with a k-NN classifier.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier

from . import config

warnings.filterwarnings("ignore")
RS = config.RANDOM_STATE
SELECTION_SAMPLE = 2500
STABILITY_REPS = 3
STABILITY_FRAC = 0.8
# Highest stability wins. A 0.03 tie margin with a consensus tie-break was used before; on the
# 34-dataset benchmark (tools/clu_bench.py) it scored 0.914 mean ARI against 0.931 for the plain
# rule (see EXPERIMENTS.md, round 2), so ties are now only exact ties.
TIE_TOLERANCE = 0.0
# Switch to the BIC-selected Gaussian mixture when BIC prefers non-round clusters by >= SHAPE_EVIDENCE_MIN
# per row and its stability is within SHAPE_SWITCH_DELTA of the top (tools/clu_shape_rule.py: 34 tuning
# datasets 0.931 -> 0.942; 12 unseen validation datasets 0.911 -> 0.929, none worse).
SHAPE_SWITCH_DELTA = 0.08
SHAPE_EVIDENCE_MIN = 0.2


class _Method:
    """A clustering method that can be fitted on any subset and (optionally) extended to all rows."""

    def __init__(self, name, k, reg_covar, gmm_cols=None):
        self.name, self.k, self.reg = name, k, reg_covar
        # Gaussian mixtures only see continuous columns: one-hot indicators are not Gaussian and
        # make the covariance estimates collapse
        self.gmm_cols = gmm_cols

    def _gmm_space(self, Z):
        if Z.shape[1] > 20:
            self._pca = PCA(10, random_state=RS).fit(Z)
            return self._pca.transform(Z)
        self._pca = None
        return Z

    def fit_predict(self, Z):
        k = self.k
        if self.name.startswith("gmm") and self.gmm_cols is not None:
            Z = Z[:, self.gmm_cols]
        if self.name == "kmeans":
            self.model = KMeans(k, n_init=10 if len(Z) <= 20000 else 3, random_state=RS).fit(Z)
            return self.model.labels_
        if self.name == "ward":
            return AgglomerativeClustering(k, linkage="ward").fit_predict(Z)
        if self.name == "spectral":
            nn = int(min(len(Z) - 1, max(10, 2.5 * np.log(len(Z)))))
            return SpectralClustering(k, affinity="nearest_neighbors", n_neighbors=nn,
                                      assign_labels="cluster_qr", random_state=RS).fit_predict(Z)
        if self.name == "gmm_bic":
            G = self._gmm_space(Z)
            best = None
            self.bic_by_cov = {}
            for cov in ("full", "tied", "diag", "spherical"):
                g = GaussianMixture(k, covariance_type=cov, n_init=2, reg_covar=self.reg, random_state=RS).fit(G)
                b = g.bic(G)
                self.bic_by_cov[cov] = float(b)
                if best is None or b < best[0]:
                    best = (b, g)
            self.bic_n = len(G)
            self.model = best[1]
            return self.model.predict(G)
        if self.name == "gmm_full":
            cov = "full" if Z.shape[1] <= 20 else "diag"
            self.model = GaussianMixture(k, covariance_type=cov, n_init=2, reg_covar=self.reg,
                                         random_state=RS).fit(Z)
            self._pca = None
            return self.model.predict(Z)
        raise ValueError(self.name)

    @property
    def scalable(self):
        return self.name in ("kmeans", "gmm_bic", "gmm_full")


def _stability(method, Z, base, rng, deadline):
    n = len(Z)
    scores = []
    for _ in range(STABILITY_REPS):
        if time.time() > deadline:
            break
        idx = np.sort(rng.choice(n, int(STABILITY_FRAC * n), replace=False))
        try:
            lab = _Method(method.name, method.k, method.reg, method.gmm_cols).fit_predict(Z[idx])
        except Exception:
            return 0.0
        scores.append(adjusted_rand_score(base[idx], lab))
    return float(np.mean(scores)) if scores else 0.0


def solve_clustering(X, k, deadline, log=print, onehot_columns=0, continuous_mask=None, on_interim=None,
                     return_candidates=False):
    t_start = time.time()
    n, d = X.shape
    k = int(max(1, min(k, n)))
    if k == 1 or d == 0:
        return dict(labels=np.zeros(n, dtype=int), report={"note": "trivial clustering"})
    v = X.var(0)
    keep = v > 1e-12 if (v > 1e-12).any() else np.ones(d, bool)
    X = X[:, keep]
    gmm_cols = None
    if continuous_mask is not None:
        cm = np.asarray(continuous_mask, bool)[keep]
        if 0 < cm.sum() < len(cm):
            gmm_cols = np.where(cm)[0]
    d = X.shape[1]
    rng = np.random.default_rng(RS)
    sample = np.sort(rng.choice(n, SELECTION_SAMPLE, replace=False)) if n > SELECTION_SAMPLE else np.arange(n)
    Z = X[sample]
    reg = 1e-2 if onehot_columns else 1e-3

    if on_interim is not None:
        # safety net before the method search: a quick k-means on every row (seconds even for 100k rows),
        # so a slow machine or the watchdog never leaves only the single-cluster fallback on disk
        try:
            quick = KMeans(k, n_init=1, max_iter=100, random_state=RS).fit(X[sample])
            on_interim(quick.predict(X))
        except Exception as e:
            log(f"  interim clustering skipped: {type(e).__name__}: {e}")

    names = ["kmeans", "gmm_full", "gmm_bic", "ward", "spectral"]
    labels, stab, secs = {}, {}, {}
    shape_evidence, bic_shape = None, None
    for nm in names:
        if labels and time.time() > deadline - 3:
            break
        t0 = time.time()
        m = _Method(nm, k, reg, gmm_cols)
        try:
            lab = np.asarray(m.fit_predict(Z))
        except Exception as e:
            log(f"  {nm} failed: {type(e).__name__}: {e}")
            continue
        if len(np.unique(lab)) < min(k, 2):
            log(f"  {nm} degenerate ({len(np.unique(lab))} groups)")
            continue
        labels[nm] = lab
        if getattr(m, "bic_by_cov", None):
            b = m.bic_by_cov
            # per-row BIC advantage of the best non-spherical shape over spherical clusters (the k-means
            # assumption); large values mean the data clearly contradicts round, equal-size clusters
            shape_evidence = (b["spherical"] - min(b["full"], b["tied"], b["diag"])) / max(m.bic_n, 1)
            bic_shape = min(b, key=b.get)
        stab[nm] = _stability(m, Z, lab, rng, deadline - 3)
        if len(np.unique(lab)) < k:
            stab[nm] -= 0.1  # did not use all requested groups
        secs[nm] = round(time.time() - t0, 3)
        log(f"  {nm:9s} stability={stab[nm]:.4f} ({secs[nm]:.2f}s)")

    if not labels:
        lab = KMeans(k, n_init=3, random_state=RS).fit_predict(X)
        return dict(labels=lab, report={"note": "fallback k-means"})

    cons = {p: float(np.mean([adjusted_rand_score(labels[p], labels[o]) for o in labels if o != p]))
            if len(labels) > 1 else 0.0 for p in labels}
    top = max(stab.values())
    near = [p for p in labels if stab[p] >= top - TIE_TOLERANCE]
    best = max(near, key=lambda p: cons[p])
    shape_override = False
    if (best != "gmm_bic" and "gmm_bic" in stab and shape_evidence is not None and bic_shape != "spherical"
            and shape_evidence >= SHAPE_EVIDENCE_MIN and stab["gmm_bic"] >= top - SHAPE_SWITCH_DELTA):
        # The most stable method may be stable *because* it consistently imposes round clusters
        # (k-means, spectral) on elongated / overlapping ones. When BIC clearly rejects round clusters
        # and the mixture is nearly as stable, the mixture's partition is the better-founded one.
        best, shape_override = "gmm_bic", True

    # labels for every row
    if len(sample) == n:
        final = labels[best]
    else:
        m = _Method(best, k, reg, gmm_cols)
        if m.scalable:
            final = np.asarray(m.fit_predict(X))
        else:
            knn = KNeighborsClassifier(n_neighbors=10, weights="distance").fit(Z, labels[best])
            final = knn.predict(X)

    order = np.argsort(-np.bincount(final, minlength=k))
    remap = np.empty_like(order)
    remap[order] = np.arange(len(order))
    final = remap[final]
    report = {
        "selection_criterion": "highest stability (subsample ARI); BIC shape evidence can switch to gmm_bic",
        "chosen": best,
        "stability": {p: round(s, 5) for p, s in stab.items()},
        "consensus": {p: round(c, 5) for p, c in cons.items()},
        "near_ties": near,
        "bic_shape": bic_shape,
        "shape_override": shape_override,
        "shape_evidence_per_row": None if shape_evidence is None else round(float(shape_evidence), 5),
        "selection_rows": int(len(sample)),
        "n_features_in_space": int(d),
        "cluster_sizes": np.bincount(final).tolist(),
        "method_seconds": secs,
        "clustering_seconds": round(time.time() - t_start, 3),
    }
    out = dict(labels=final, report=report)
    if return_candidates and len(sample) == n:   # benchmark hook: every candidate's labels on all rows
        out["candidates"] = labels
    return out

