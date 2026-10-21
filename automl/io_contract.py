"""Input / output adapter.

Everything that depends on the organisers' file contract lives here, so it can be adapted at
kickoff without touching the modelling code. The loader is deliberately forgiving: it discovers
train/test tables and a JSON metadata file inside a dataset folder, accepts many key spellings,
and every value can also be overridden from the command line.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import config

MISSING_MARKERS = ["", "?"]
TABLE_EXTS = (".csv", ".tsv", ".txt", ".parquet", ".json")
META_PRIORITY = ("meta.json", "metadata.json", "task.json", "spec.json", "info.json",
                 "dataset.json", "config.json", "schema.json")

TASK_ALIASES = {
    "classification": "classification", "classify": "classification", "class": "classification",
    "binary": "classification", "multiclass": "classification", "binary_classification": "classification",
    "multiclass_classification": "classification", "clf": "classification",
    "regression": "regression", "regress": "regression", "reg": "regression",
    "clustering": "clustering", "cluster": "clustering", "segmentation": "clustering",
    "unsupervised": "clustering",
}

KEYS = {
    "task": ("task", "task_type", "problem_type", "problem", "type", "kind"),
    "target": ("target", "target_column", "target_col", "label", "label_column", "label_col",
               "y", "y_column", "target_name"),
    "n_clusters": ("n_clusters", "num_clusters", "number_of_clusters", "k", "n_groups",
                   "num_groups", "groups", "clusters", "required_clusters", "n_components"),
    "feature_types": ("feature_types", "column_types", "features", "schema", "columns", "dtypes",
                      "types", "feature_schema"),
    "id_column": ("id_column", "id_col", "id", "index_column", "row_id"),
    "metric": ("metric", "eval_metric", "evaluation_metric", "scoring", "score"),
    "train": ("train", "train_file", "train_path", "train_csv", "training_data", "data", "data_file"),
    "test": ("test", "test_file", "test_path", "test_csv", "eval", "eval_file", "predict_file"),
}

TYPE_ALIASES = {
    "numeric": "numeric", "numerical": "numeric", "number": "numeric", "float": "numeric",
    "int": "numeric", "integer": "numeric", "continuous": "numeric", "real": "numeric",
    "double": "numeric", "float64": "numeric", "int64": "numeric", "ordinal": "numeric",
    "categorical": "categorical", "category": "categorical", "nominal": "categorical",
    "string": "categorical", "str": "categorical", "object": "categorical", "bool": "categorical",
    "boolean": "categorical", "binary": "categorical", "discrete": "categorical",
    "datetime": "datetime", "date": "datetime", "time": "datetime", "timestamp": "datetime",
    "text": "text", "id": "id", "identifier": "id", "ignore": "id",
}


@dataclass
class TaskSpec:
    task: str
    train: pd.DataFrame
    test: pd.DataFrame | None
    target: str | None = None
    n_clusters: int | None = None
    id_column: str | None = None
    metric: str | None = None
    feature_types: dict[str, str] = field(default_factory=dict)
    name: str = "dataset"
    meta: dict[str, Any] = field(default_factory=dict)


# ----------------------------------------------------------------------------------- reading
def _detect_encoding(path: Path) -> str:
    """UTF-8 (with or without byte-order mark) when the file decodes as such, else Windows-1252.

    cp1252 decodes nearly every byte and is what Excel writes on European systems (Latin-1 superset)."""
    with open(path, "rb") as f:
        head = f.read(4_000_000)
    if head.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    try:
        head.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError as e:
        if e.start >= len(head) - 4:   # cut inside a multi-byte character at the 4 MB boundary
            return "utf-8"
        return "cp1252"


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".parquet":
        return pd.read_parquet(path)
    if ext == ".json":
        return pd.read_json(path)
    sep = "\t" if ext == ".tsv" else ","
    enc = _detect_encoding(path)
    df = pd.read_csv(path, sep=sep, low_memory=False, skipinitialspace=True, encoding=enc)
    if df.shape[1] == 1 and ";" in str(df.columns[0]):
        df = pd.read_csv(path, sep=";", low_memory=False, skipinitialspace=True, encoding=enc)
    df.columns = [str(c).strip() for c in df.columns]
    # Whitespace-only cells (e.g. Telco "TotalCharges") and the UCI marker "?" are missing values.
    for c in df.columns:
        if df[c].dtype == object:
            try:
                mask = df[c].str.strip().isin(MISSING_MARKERS).fillna(False).astype(bool)
            except AttributeError:
                continue
            if mask.any():
                df.loc[mask, c] = np.nan
                # a column that was numeric apart from the markers becomes numeric again
                conv = pd.to_numeric(df[c], errors="coerce")
                if conv.notna().sum() == df[c].notna().sum():
                    df[c] = conv
    return df


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "_", str(s).strip().lower())


def _get(meta: dict, name: str):
    lowered = {_norm(k): v for k, v in meta.items()}
    for k in KEYS[name]:
        if k in lowered and lowered[k] not in (None, ""):
            return lowered[k]
    # one level of nesting, e.g. {"task": {"type": ..., "target": ...}}
    for v in meta.values():
        if isinstance(v, dict):
            sub = {_norm(k): vv for k, vv in v.items()}
            for k in KEYS[name]:
                if k in sub and sub[k] not in (None, "") and not isinstance(sub[k], dict):
                    return sub[k]
    return None


def normalise_task(value: str | None) -> str | None:
    if value is None:
        return None
    return TASK_ALIASES.get(_norm(value), None)


def normalise_feature_types(raw) -> dict[str, str]:
    out: dict[str, str] = {}
    if raw is None:
        return out
    if isinstance(raw, dict):
        items = []
        for k, v in raw.items():
            # {"numeric": [cols], "categorical": [cols]} form
            if isinstance(v, (list, tuple)) and _norm(k) in TYPE_ALIASES:
                items += [(c, k) for c in v]
            elif isinstance(v, dict):
                items.append((k, v.get("type") or v.get("dtype") or v.get("kind")))
            else:
                items.append((k, v))
    elif isinstance(raw, (list, tuple)):
        items = []
        for e in raw:
            if isinstance(e, dict):
                name = e.get("name") or e.get("column") or e.get("feature")
                typ = e.get("type") or e.get("dtype") or e.get("kind")
                if name is not None:
                    items.append((name, typ))
    else:
        return out
    for name, typ in items:
        t = TYPE_ALIASES.get(_norm(typ), None) if typ is not None else None
        if t:
            out[str(name)] = t
    return out


def _find_meta(folder: Path) -> Path | None:
    for n in META_PRIORITY:
        p = folder / n
        if p.exists():
            return p
    js = sorted(folder.glob("*.json"))
    return js[0] if js else None


def _find_tables(folder: Path) -> dict[str, Path]:
    files = [p for p in sorted(folder.iterdir()) if p.is_file() and p.suffix.lower() in TABLE_EXTS
             and p.suffix.lower() != ".json"]
    found: dict[str, Path] = {}
    bad = ("solution", "submission", "answer", "truth", "private", "ground")
    for p in files:
        stem = p.stem.lower()
        if any(b in stem for b in bad):
            continue
        is_label = bool(re.search(r"(^y_|_y$|label|target)", stem))
        if "train" in stem:
            found["train_labels" if is_label else "train"] = p
        elif "test" in stem or "eval" in stem or "predict" in stem:
            if not is_label:
                found["test"] = p
    if "train" not in found:
        rest = [p for p in files if p not in found.values()
                and not any(b in p.stem.lower() for b in bad)]
        if rest:
            found["train"] = rest[0]
    return found


# ----------------------------------------------------------------------------------- loading
def load_task(data: str | None = None, train: str | None = None, test: str | None = None,
              meta: str | None = None, task: str | None = None, target: str | None = None,
              n_clusters: int | None = None, id_column: str | None = None,
              metric: str | None = None) -> TaskSpec:
    folder = Path(data) if data else None
    if folder is not None and folder.is_file():
        # a single table was given directly
        if folder.suffix.lower() == ".json":
            meta = meta or str(folder)
            folder = folder.parent
        else:
            train = train or str(folder)
            folder = folder.parent

    meta_dict: dict[str, Any] = {}
    meta_path = Path(meta) if meta else (_find_meta(folder) if folder else None)
    if meta_path is not None and meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            meta_dict = loaded

    tables = _find_tables(folder) if folder else {}
    base = meta_path.parent if meta_path is not None else folder

    def resolve(cli_value, key):
        if cli_value:
            return Path(cli_value)
        v = _get(meta_dict, key)
        if isinstance(v, str) and base is not None:
            cand = (base / v).resolve()
            # metadata may only name files inside the dataset folder (no "../" or absolute paths)
            if cand.is_file() and cand.is_relative_to(base.resolve()):
                return cand
        return tables.get(key)

    train_path = resolve(train, "train")
    test_path = resolve(test, "test")
    if train_path is None:
        raise FileNotFoundError("Could not find a training/data table. Pass --train or --data.")

    train_df = read_table(train_path)
    test_df = read_table(test_path) if test_path is not None and Path(test_path) != Path(train_path) else None

    task_name = normalise_task(task) or normalise_task(_get(meta_dict, "task"))
    tgt = target or _get(meta_dict, "target")
    if isinstance(tgt, (list, tuple)):
        tgt = tgt[0] if tgt else None
    k = n_clusters or _get(meta_dict, "n_clusters")
    try:
        k = int(k) if k is not None else None
    except (TypeError, ValueError):
        k = None

    # separate label file (y_train.csv / train_labels.csv)
    if "train_labels" in tables and (tgt is None or tgt not in train_df.columns):
        ylab = read_table(tables["train_labels"])
        ycol = tgt if tgt in ylab.columns else ylab.columns[-1]
        tgt = tgt or str(ycol)
        train_df[tgt] = ylab[ycol].values[: len(train_df)]

    if task_name is None:
        task_name = infer_task(train_df, tgt, k)
    if task_name != "clustering" and (tgt is None or tgt not in train_df.columns):
        # heuristic: target is the train column absent from test, else the last column
        if test_df is not None:
            missing = [c for c in train_df.columns if c not in test_df.columns]
            tgt = missing[-1] if missing else train_df.columns[-1]
        else:
            tgt = train_df.columns[-1]
    if task_name == "clustering" and k is None:
        k = 3  # last resort; datasets are supposed to provide it

    ftypes = normalise_feature_types(_get(meta_dict, "feature_types"))
    idc = id_column or _get(meta_dict, "id_column")
    if isinstance(idc, (list, tuple)):
        idc = idc[0] if idc else None
    if idc is not None and idc not in train_df.columns and (test_df is None or idc not in test_df.columns):
        idc = None
    for c, t in ftypes.items():
        if t == "id" and idc is None:
            idc = c

    name = (folder.name if folder else Path(train_path).stem) or "dataset"
    return TaskSpec(task=task_name, train=train_df, test=test_df, target=tgt if task_name != "clustering" else None,
                    n_clusters=k, id_column=idc, metric=metric or _get(meta_dict, "metric"),
                    feature_types=ftypes, name=name, meta=meta_dict)


def infer_task(df: pd.DataFrame, target: str | None, k) -> str:
    if k is not None and (target is None or target not in df.columns):
        return "clustering"
    if target is None or target not in df.columns:
        return "clustering"
    y = df[target].dropna()
    if not pd.api.types.is_numeric_dtype(y):
        return "classification"
    n_unique = y.nunique()
    is_int = np.allclose(y.values, np.round(y.values))
    if n_unique <= 2 or (is_int and n_unique <= min(20, max(3, int(0.05 * len(y))))):
        return "classification"
    return "regression"


# ----------------------------------------------------------------------------------- writing
def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        return None if not np.isfinite(v) else v
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    return str(o)


def _clean(o):
    """Replace NaN/inf by None recursively so the JSON is strictly valid."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, float) and not np.isfinite(o):
        return None
    if isinstance(o, np.floating):
        return None if not np.isfinite(o) else float(o)
    return o


def write_predictions(out_dir: str | Path, ids, preds, id_name: str, proba=None, class_labels=None,
                      filename: str | None = None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    filename = filename or config.PREDICTIONS_FILE
    df = pd.DataFrame({config.OUTPUT_ID_COLUMN or id_name: ids, config.PREDICTION_COLUMN: preds})
    tmp = out / (filename + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(out / filename)
    if proba is not None and class_labels is not None:
        pdf = pd.DataFrame(proba, columns=[f"proba_{c}" for c in class_labels])
        pdf.insert(0, config.OUTPUT_ID_COLUMN or id_name, ids)
        pdf.to_csv(out / config.PROBABILITIES_FILE, index=False)


def write_profile(out_dir: str | Path, profile: dict):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / (config.PROFILE_FILE + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_clean(profile), f, indent=2, default=_json_default)
    tmp.replace(out / config.PROFILE_FILE)
