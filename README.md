# Unknown Dataset Hackathon: AutoML for unseen tables

One program that takes a table it has never seen, works out what is in it, and solves the
task it comes with: **classification**, **regression** or **clustering**. It returns
predictions plus a structured data profile, and stays within the 60-second limit per dataset.

## Team

| Name | Role |
|------|------|
| Mahdi Ismael | Sole member: design, implementation, evaluation |

## Quick start

```bash
python -m pip install -r requirements.txt          # Python 3.10 - 3.14 (pins verified with pip for each)
python tools/smoke_test.py                         # ~30 s self-test: one dataset per task
python main.py --data path/to/dataset_folder --output path/to/output_folder
```

Other ways to call it (every field can be given explicitly):

```bash
python main.py --train train.csv --test test.csv --task classification --target Churn --output out
python main.py --train train.csv --test test.csv --task regression --target SalePrice --metric rmse -o out
python main.py --train mall.csv --task clustering --n-clusters 5 --output out
python main.py path/to/dataset_folder path/to/output_folder          # positional form
```

From Python:

```python
import automl
profile = automl.run("path/to/dataset_folder", "out")
profile = automl.run(train="train.csv", test="test.csv", task="regression", target="SalePrice", output="out")
```

### Outputs (in `--output`)

| File | Content |
|------|---------|
| `predictions.csv` | `<id>,prediction`: one row per test row, in the test table's order (or per row for clustering without a test file). The id is the metadata's id column, else a detected id column (`Id`, `customerID`, `row_id`...), else a 0..n-1 row index. |
| `profile.json` | Structured data profile, `data_issues` / `variations_detected` summaries and a model report (see below). |
| `probabilities.csv` | Classification only: class probabilities. |
| `predictions_all_rows.csv` | Clustering with a separate test table only: labels for every row (train rows, then test rows). |

**Matching the official output contract without code changes** (the contract is released at kickoff):

```bash
python main.py --data DATASET --output OUT --predictions-file submission.csv --prediction-column label --output-id-column Id --profile-file data_profile.json --metric accuracy
```

The same names are constants in `automl/config.py`. `profile.json` also has a compact `summary` block that
uses common key names (`column_types`, `missing_values`, `class_distribution`, `variations`, ...).

**A dataset can never end without output, and never overruns:**
* A valid fallback prediction file (majority class / median / single cluster) and a minimal
  `profile.json` are written before profiling and modelling start, and atomically replaced by better versions.
* Before the model search, a linear model is fitted (milliseconds) and its predictions saved, and for
  clustering a quick k-means. So a slow machine can never leave only the constant fallback: in simulated
  hangs the saved answer had R² 0.30 instead of −0.06, and ARI 0.62 instead of 0.
* **Time-adaptive boosting**: before cross-validating a boosting model, 1- and 10-round probes measure
  its fixed and per-round cost. If the full model cannot finish in the time left, rounds are reduced and
  the learning rate raised instead of dropping the model. Size tests: 300,000 rows F1 0.68 → 0.84,
  1,000 columns 0.80 → 0.91, 20 classes 0.72 (watchdog stop) → 0.91.
* Profiling errors fall back to simple column roles, and modelling errors keep the fallback.
* All model searches are time-aware (hard internal stop at 48 s), and a **watchdog** ends the
  process at 57 s whatever is running. It starts before pandas / scikit-learn are imported, because on
  a slow or swapping machine the imports alone can take many seconds. A simulated 120 s hang ended cleanly with valid outputs.
* Thread counts follow the CPUs the process may actually use (container / affinity limits).
  Pinned to 2 cores, the heaviest stress test (100k rows) takes 41 s.

### The data profile (`profile.json`)

| Key | Content |
|-----|---------|
| `variations_detected` | List of every variation found, e.g. `["missing_values", "differently_scaled_features", "categorical_columns", "constant_columns", "duplicate_columns", "irrelevant_features", "imbalanced_classes", "nonlinear_relationships"]` |
| `data_issues` | The same variations as true/false flags (also: datetime/text/id columns, duplicate rows, skewed target) |
| `feature_types` | column → `numeric` / `categorical` / `datetime` / `text` / `dropped (reason)` |
| `missing_by_column`, `missing_by_column_test`, `missing_cells`, `missing_fraction` | Missing values (training and test tables) |
| `rows_without_target_removed`, `target_profile.missing_target` | Training rows that had no label |
| `constant_columns`, `duplicate_columns`, `id_like_columns`, `dropped_columns` | Removed columns and why |
| `irrelevant_features`, `feature_relevance_mi`, `feature_permutation_importance` | Relevance evidence (supervised) |
| `cluster_feature_relevance` | η² per feature and irrelevant features (clustering) |
| `numeric_scale_ratio`, `features_differently_scaled` | Scale differences |
| `target_profile` | Class counts / fractions / imbalance ratio, or target mean / std / skew |
| `nonlinearity` | Best linear vs best non-linear cross-validated score |
| `columns` | Per-column details: dtype, inferred and declared type, missing, cardinality, statistics, top values |
| `model` | Chosen models, CV scores, ensemble weights, decision rule, timings |
| `feature_screening` | Very wide tables: how many columns were kept for modelling |
| `timings` | Seconds spent in each phase (loading, profiling, preparation, relevance, models) |
| `environment`, `peak_memory_mb` | Python / library versions, usable CPU count, peak memory of the run |

## Results and how to reproduce them

```bash
python tools/smoke_test.py           # ~30 s: one dataset per task, checks outputs, quality and time
python -m pip install -r requirements-dev.txt && python -m pytest tests -q   # 27 tests (~2-4 min)
python tools/make_practice_data.py   # 40 practice datasets in practice/ (+ hidden labels in practice_solutions/)
python tools/check.py                # runs main.py on each like the assessment agent; times and scores it
python tools/check.py --only clustering --tag mytest      # one task family, results saved to outputs/
python tools/stress_test.py          # edge cases: 100k rows, wide tables, text, rare classes, odd metadata...
python tools/stress_test2.py         # messy encodings: "?" markers, bool / text targets, tz dates, unseen categories...
python tools/stress_test3.py         # size limits: 1,000 columns, 20 classes, 300,000 rows, 80-column house prices...
python tools/judge_sim.py --seed 1100000   # simulated assessment: 18 unseen datasets, 60 s kill, brief's weights
```

Mean test-set quality over the 40 labelled practice datasets (mean time per dataset 13 / 15 / 6 s for
classification / regression / clustering). Both versions were run on identical data
on an i7-12700H laptop; times include the ~4 s Python start-up.

| Task (datasets) | Metric | Version 1 | **This version** |
|---|---|---|---|
| Classification (10) | macro-F1 | 0.878 | **0.888** |
| Regression (9) | R² | 0.833 | **0.873** |
| Clustering (21) | ARI | 0.773 | **0.938** |
| Slowest dataset | seconds | 44.5 | **27** (43 with the laptop under heavy load) |

Largest gains: iris clustering 0.62 → 0.90, anisotropic clusters 0.61 → 1.00, mall customers with
noise columns 0.66 → 0.99, high-cardinality regression R² 0.76 → 0.94, digits classification
0.970 → 0.991 (44 s → 20 s), bike-style counts RMSE 13.9 → 13.1 (Poisson model).

Size limits (`tools/stress_test3.py`, final version): 1,000 columns F1 0.916 in 22 s, 20 classes F1 0.907 in
39 s, 300,000 rows F1 0.841 in 40 s, 80-column house prices R² 0.82, clustering with 30 noise columns ARI 0.999.

Robustness: all 11 stress tests and all 6 messy-encoding tests pass; the slowest (100,000 rows) takes 42 s.
Pinned to 2 CPU cores, the earlier version took 41 s on the same test. A clean install from `requirements.txt` into a new
virtual environment passes the smoke test.

### Simulated assessment (`tools/judge_sim.py`)

Five independent experiments, each with 18 datasets never seen during development (6 per task,
Kaggle-style, with the brief's variations injected so the correct profile answers are known). Every dataset
runs through `main.py` as a separate process with a hard kill at 60 s, like an assessment agent would.

| Experiment (seed) | Classification F1 | Regression R² | Clustering ARI | Failures / killed | Score (approx.) |
|---|---|---|---|---|---|
| 1 (1100000) | 0.862 | 0.869 | 0.793* | 0 / 0 | 85.1 |
| 2 (1200000) | 0.854 | 0.851 | 0.962 | 0 / 0 | 89.4 |
| 3 (1300000) | 0.858 | 0.848 | 0.963 | 0 / 0 | 89.4 |
| 4 (1400000) | 0.862 | 0.865 | 0.967 | 0 / 0 | 90.2 |
| 5 (1500000) | 0.863 | 0.848 | 0.928 | 0 / 0 | 88.6 |
| **Average** | **0.860** | **0.856** | **0.923** | **0 / 0 (90 datasets)** | **88.5** |

Score = quality 74.8/85 + profiling 9.9/10 + speed 3.8/5, using the brief's weights. The official formula is
not published, so this is an approximation. \*Experiment 1 contains a generated dataset whose groups overlap
completely (every clustering method scores ARI 0.004).

## Approach

```
load (io_contract) → profile + column roles (profiling) → prepare two views (preprocess)
   → classification / regression: relevance filter, CV model search, greedy ensemble, decision tuning
   → clustering: 5 methods, the most stable on 80% subsamples is chosen, unless BIC rejects round clusters
   → predictions.csv + profile.json
```

### 1. Inspecting the data (`automl/profiling.py`, `automl/relevance.py`)
* **Type inference** per column: numeric (including numbers stored as text, e.g. Telco `TotalCharges`),
  categorical, datetime (parsed strings → year/month/day/weekday/hour/day-of-year/weekend/elapsed),
  free text (length / word-count features + frequency encoding). Declared feature types override.
* **Dropped automatically:** constant columns, exact duplicate columns, ID-like columns (unique per
  row and named like an id, unique short strings, or running counters), and any training column
  missing from the test table (prevents leakage such as Bike Sharing's `casual`/`registered`).
* **Irrelevant features:** a feature is relevant if either of two tests fires:
  (1) its mutual information with the target beats what the estimator reports for a *shuffled*
  target, which catches marginal effects; (2) shuffling it clearly lowers the held-out score of a
  quick boosting model (above 2× the noise and at least 3% of the strongest feature), which catches
  effects that only appear through *interactions*. On practice data with known truth the result is
  exact: Friedman #1 (x5–x14 noise), machine failure (`Type` unused) and the high-cardinality set.
  Using MI alone gave three false alarms there. For clustering, features with between-cluster
  variance share η² < 0.05 are reported, which is exact on the blobs-plus-noise set.
* **Nonlinearity:** best linear vs best non-linear model in cross-validation.
* `data_issues` gives one flag per variation in the rules: missing values, scale differences,
  categorical / datetime / text columns, constant, duplicate and id columns, duplicate rows,
  irrelevant features, imbalanced classes, skewed target and nonlinear relationships.
  The profile also has per-column type / missing count / cardinality / statistics / top values,
  class counts and imbalance ratio, and target statistics.

### 2. Preparation (`automl/preprocess.py`)
Fitted on training rows only. Two views:
* **Reading**: file encoding is detected (UTF-8, UTF-8 with byte-order mark, or Windows-1252/Latin-1 as
  written by Excel); column names are trimmed; blank cells and the UCI marker `?` are missing values;
  columns that are numeric apart from those markers stay numeric. Infinite or text-number values in a
  regression target are cleaned; rows without a usable label are removed and counted in the profile.
* **Safety**: file names taken from the metadata must stay inside the dataset folder (`../` paths and
  absolute paths are ignored).
* **Dates**: year, month, day, weekday, hour, day-of-year, weekend and elapsed time, plus cyclical
  sin/cos of hour, weekday and month (so 23:00 is next to 00:00 for linear and distance models).
* **Tree view**: raw numbers with NaN kept (handled natively), categoricals as ordinal codes;
  ≤250 categories use HistGradientBoosting's native categorical splits.
* **Linear/distance view**: median imputation + missing indicators, 0.1–99.9% clipping, `log1p` for
  heavily skewed non-negative columns, standardisation, top-20 one-hot.
* **Target encoding** for categoricals with more than 10 levels: smoothed mean target (per class for
  ≤10 classes), computed **out-of-fold** on training rows so a row never sees its own label.
* A third view without the irrelevant columns is given to k-NN and SVM, the models noise hurts most.
* **Very wide tables** (> 200 prepared columns): one fast Extra-Trees model ranks all columns and every
  model then uses the strongest 150. A 1,000-column test went from a time-out (F1 0.33) to F1 0.91–0.93 in 25–30 s.

### 3. Supervised models (`automl/supervised.py`)
* Candidates, cheapest first: Logistic/Ridge regression (+ log target), **pairwise-interaction ridge**
  (raw and log target, tables up to 40 prepared columns), k-NN (two neighbourhood sizes),
  RBF SVM / SVR at two strengths (n ≤ 4000; the strong SVM skips sklearn's internal probability
  calibration on tables over 2,500 rows, which is 2.6× faster with identical predictions),
  then HistGradientBoosting (default, class-balanced and deeper variants, log-target variant for
  skewed regression, **Poisson** variant for count targets), Extra Trees, Random Forest. Running the cheap models first matters: v1 spent
  the whole budget boosting 10-class digits and never tried the SVM that wins there.
* 5-fold (n ≤ 2000) or 3-fold CV out-of-fold predictions, stratified for classification; tables
  above 30k rows are subsampled for the search; the final model is refit on all rows, or on the
  selection sample when the time left is too short for a full refit.
* **Greedy ensemble selection** (Caruana) over out-of-fold predictions.
* **Decision-aware model selection**: for macro-F1, every candidate and every ensemble step is scored at
  its *best* decision rule (exact optimal threshold for binary targets, found in one sorted pass;
  prior correction for multiclass) rather than at argmax. With argmax, imbalanced targets made
  well-calibrated models look bad and over-confident ones look good. Telco-style churn (11% positives):
  F1 0.570 → 0.620, against a best possible 0.628 from the true probabilities.
* **Decision tuning**: the chosen rule (threshold or prior correction) is applied to the final predictions.
* **Time control:** candidates are aborted when their projected CV time would exceed the budget, the
  search **stops once 3 candidates in a row fail to improve the ensemble** (5 on tables under 1,000 rows,
  where CV is noisy and models are cheap), new candidates stop after ~22 s, but the search always gets at
  least 12 s even when loading was slow (a busy or slow CPU), boosting settings
  scale with the number of classes,
  and refit cost is estimated before refitting. Hard stop is 48 s (12 s margin under the 60 s limit).

### 4. Clustering (`automl/clustering.py`)
Choosing among clusterings without labels is the hard part. We benchmarked selection rules on 16
labelled datasets (iris, wine, breast cancer, digits; blobs with noise columns; anisotropic,
unequal-density, imbalanced and non-convex shapes; mixed categorical data with missing values) using
`tools/clu_experiment*.py`. Mean ARI against the hidden labels:

| Selection rule | Mean ARI |
|---|---|
| silhouette (version 1) | 0.65 |
| k-means only / Ward only | 0.73 / 0.74 |
| Gaussian mixture only | 0.80 |
| spectral only | 0.82 |
| stability within 0.03 + consensus tie-break (round 1) | 0.89–0.91 (16 sets); 0.914 (34 sets) |
| highest stability (round 2) | 0.931 (34 sets) |
| **highest stability + BIC shape switch (used)** | **0.942 (34 sets); 0.911 → 0.929 on 12 unseen validation sets** |
| best possible choice (oracle) | 0.94 |

* Methods: k-means, Ward agglomerative, spectral (k-NN graph), Gaussian mixture with covariance type
  chosen by BIC (PCA to 10 dims for wide data), and full-covariance Gaussian mixture. The mixtures only
  see continuous columns, because one-hot indicators make their covariances collapse.
* **Stability**: every method is refit on random 80% subsamples. Its score is the agreement (ARI) with
  its own labels on the shared rows. Methods that find real structure are stable.
* The most stable method wins (consensus only breaks exact ties). A 0.03 tie margin with a consensus
  tie-break, chosen on 16 datasets in round 1, scored 0.914 against 0.931 for the plain rule on the larger
  34-dataset benchmark (`tools/clu_bench.py`), so it was removed. A held-out Gaussian-likelihood criterion
  was also tested and rejected (0.873 alone, 0.927 combined); see `EXPERIMENTS.md`.
* **Shape check (k-means vs Gaussian mixture)**: a method can be *stable because it is consistently wrong*:
  k-means and spectral clustering impose round, equal-size clusters, and on elongated or overlapping
  groups (iris-like data) they find the same wrong split every time. The Gaussian mixture compares cluster
  shapes by BIC; when BIC prefers non-round clusters by at least 0.2 per row and the mixture's stability is
  within 0.08 of the top, the mixture is chosen. Tuned on 34 datasets (0.931 → 0.942), then checked on 12
  validation datasets never used for tuning (0.911 → 0.929, none worse). Iris + noise columns: 0.536 → 0.922.
* Big tables: selection runs on 2,500 sampled rows. k-means / mixtures are then refit on all rows,
  and graph / hierarchical labels are spread to all rows with k-NN.
* Train and test rows are clustered together; exactly k groups are always requested.

## Assumptions (written before the official kickoff contract was available)

1. **Input layout:** a dataset folder containing a training table (`*train*.csv`), an optional
   test table (`*test*.csv`) and a JSON metadata file (`meta.json`, `metadata.json`, `task.json`,
   or any `.json`). Many key spellings are accepted (`task`/`task_type`/`problem_type`,
   `target`/`label`, `n_clusters`/`k`, `feature_types` as dict or list). A separate `y_train.csv`
   label file, `;` separators, `.tsv` and `.parquet` are also handled. Everything can be overridden from the CLI.
2. If the task is missing, it is inferred (no target → clustering; non-numeric or few integer
   values → classification; otherwise regression). If the target is missing, it is the training
   column absent from the test table.
3. **Output:** `predictions.csv` with an id column and `prediction`; `profile.json` with the profile.
   For clustering without a test table, every row gets a label.
4. **Selection metric:** macro-F1 for classification (robust under imbalance), RMSE for regression.
   If the metadata names a metric (`accuracy`, `balanced_accuracy`, `roc_auc`, `log_loss`, `mae`,
   `r2`, `rmsle`, ...) it is used instead.
5. Cluster labels are arbitrary integers 0..k-1 (assumed scored with a permutation-invariant
   measure such as ARI/NMI).
6. CPU-only environment, Python 3.10–3.14, no internet at run time, 60 s per dataset including start-up.
7. The selection metric is the one most robust to imbalance until the official scoring rules are known;
   switching it is `--metric`, one line in `automl/config.py`, or automatic if the metadata names the metric.
   We measured the trade-off on four practice sets. If grading uses accuracy, tuning for macro-F1 costs at
   most 0.021 accuracy (and was 0.007 *better* on one set). If grading uses macro-F1, tuning for accuracy
   costs up to 0.041 F1. Macro-F1 is therefore the safer default.
8. The brief asks for three students per team; this entry has one member (see Team).

**Adapting to the official contract** takes minutes: file names, column names and defaults are
all in `automl/config.py` and `automl/io_contract.py`. The modelling code does not need to change.

## Repository layout

```
main.py                 standard entry point (CLI)
automl/config.py        time budget, default metrics, output names
automl/io_contract.py   loading inputs / writing outputs
automl/profiling.py     type inference and the data profile
automl/relevance.py     irrelevant-feature detection (mutual information + permutation importance)
automl/preprocess.py    tree view, linear view, out-of-fold target encoding
automl/supervised.py    classification / regression search and ensembling
automl/clustering.py    stability-based clustering selection
automl/metrics.py       scoring helpers
tests/                  pytest suite: components, hostile inputs, watchdog / safety nets, selection logic
tools/                  smoke test, practice-data generator, local checker, stress tests, judge simulation, research
REVIEW.md               engineering / ML / judge review and what was changed
EXPERIMENTS.md          optimisation council: baseline, plan, every experiment with its measured result
requirements.txt        pinned dependencies (requirements-dev.txt: pytest, ruff)
```
