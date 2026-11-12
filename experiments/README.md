# QUANT feature-budget experiments

This directory evaluates a substantive feature-selection stage on top of the
unchanged QUANT transform. The final classifier always has 200 trees. The
immutable [protocol](PROTOCOL.md) was written before extension test evaluation;
`frozen_settings.json` records the training-only development decision and hashes.

## Methods

| Method | Selection stage | Retained columns |
|---|---|---|
| `baseline` | None | All |
| `uniform_half` | Evenly spaced original columns | Half |
| `fisher_half` | Training-only ANOVA F ranking | Half |
| `pilot_half` | 32-tree pilot importance, 80% ranked + 20% coverage reserve | Half |
| `adaptive_pilot` | Same ranking/reserve, importance-mass budget | 25–75% |

The adaptive mass is **0.90**, selected from 0.90/0.95/0.99 by three internal
folds on the original TRAIN splits of 24 representative datasets. Candidates
within 0.5 percentage points of the best validation mean are compared by mean
feature fraction. This tolerance is an engineering rule, not equivalence testing.

## Run from the project root

First install the locked environment, download data and run the reproduction
pipeline described in `../reproduction/README.md`. This prepares the original
QUANT feature caches required here. Then run these commands sequentially:

```powershell
.\.venv\Scripts\python.exe experiments/develop.py
.\.venv\Scripts\python.exe experiments/evaluate.py
.\.venv\Scripts\python.exe experiments/analyze.py
.\.venv\Scripts\python.exe experiments/robustness.py
```

Development contains 24 × 3 × 7 = 504 validation rows. Evaluation contains
142 datasets × 5 official splits × 5 methods = 3550 rows. Both runners resume
existing rows. Evaluation refuses to append if the protocol/settings/method
hash or recorded configuration has changed. To repeat timings from scratch,
use a separate output directory; the supplied results preserve the completed run.

```powershell
# Small functional run; do not mix its timings with the complete evaluation.
.\.venv\Scripts\python.exe experiments/evaluate.py --datasets Coffee GunPoint --output experiments/new_smoke
```

Run the paired timing benchmark without other model-training jobs. Each method
uses eight threads, and method order is deterministically randomized within a
dataset/split. Full pilot cost is charged to both pilot variants even when the
pilot computation is reused. Time includes feature selection, copying selected
training columns and final fitting; it excludes common QUANT transformation and
loading. Prediction time includes extracting selected test columns.

## Reading the results

- `results/<dataset>.csv`: accuracy, feature counts, timing, matrix/tree bytes,
  selected-column hash and seed for each method/split.
- `results/all_runs.csv`, `dataset_means.csv`: complete and aggregated results.
- `results/summary.csv` / `.json`: macro accuracy, paired W/T/L, Wilcoxon and
  Holm-adjusted p-values, total and median timing ratios.
- `results/REPORT.md`: numerical summary and interpretation limits.
- `results/robustness.csv`, `ROBUSTNESS.md`: explicitly additional overlap checks.
- `development/`: dataset list, internal-fold scores and summary.

The main evaluation retains every method, including unsuccessful ones. Its
five-split baseline must agree with the corresponding 710 reproduction results.
Compare methods on those matched five splits, not to the paper's 30-split mean.

## Limits that matter

Original TRAIN labels are used for development. Later resamples can put those
cases in TEST on the 24 development datasets. Therefore the full five-split
evaluation is not a completely independent parameter holdout. The additional
`robustness.py` analysis reports original split zero and the 118 datasets outside
development, without tuning further. This qualification was recognized while
the frozen evaluation was running; the original protocol remains unmodified.

The full feature transform is still computed before selection. Selected-matrix
bytes and tree-array bytes are not peak process RAM. Impurity importance can be
unstable or biased, and the coverage reserve is not separately ablated. Keeping
`max_features=0.1` means fewer retained columns also reduces the absolute number
of candidate columns at each tree node. Conclusions concern the whole proposed
pipeline, not feature identity alone. Five resamples give less sampling coverage
than the 30-resample main reproduction.
