# QUANT primary accuracy reproduction

This directory reproduces the main QUANT accuracy experiment in Section 4.1 of
the supplied 2023 preprint. The author's `quant/code/quant.py` and
`quant/code/run.py` are unchanged. Added download, execution and analysis scripts
live here. Completion and results are recorded in `results/REPORT.md` and
`results/summary.json`; a successful smoke run is not treated as a full reproduction.

## Published protocol

- **142 univariate, equal-length time series classification datasets**, listed in
  `../quant/code/ucr142.txt`: 112 original benchmark datasets and 30 Bake off redux
  additions. Match the exact Eq, Nmv and Discrete variants rather than substituting
  an arbitrary UCR collection.
- **30 splits per dataset, numbered 0-29: 4,260 runs.** Split zero is the original
  TRAIN/TEST split. The original 112 datasets use official Java resampling indices;
  the additional 30 use official Python indices.
- QUANT: `depth=6, div=4`. Representations are the raw series, smoothed first
  difference, second difference and real-FFT magnitude.
- ExtraTrees: `n_estimators=200, max_features=0.1, criterion="entropy"`.
- No additional standardization, imputation or test-based model tuning is applied.
- Average over the 30 splits within each dataset, then average equally over
  datasets. Do not weight the archive mean by test-set size.
- The paper reports approximately **0.8568** on all 142 datasets and **0.8671** on
  the original 112. Its sub-15-minute claim concerns the average time for one
  complete archive round, not all 4,260 runs plus installation and downloading.

## Data sources and versions

1. [QUANT documentation](https://github.com/angus924/quant/blob/main/doc.md).
2. [Original benchmark data](https://www.timeseriesclassification.com/).
3. [Official downloads, corrections and resampling instructions](https://github.com/time-series-machine-learning/tsml-eval/blob/main/tsml_eval/publications/y2023/tsc_bakeoff/tsc_bakeoff_2023.ipynb).
4. [Additional 30 datasets](https://drive.google.com/file/d/1vuh6mgNrNKjHr9MMRQP0J0_gGA4dE7E3/view).
5. [Java indices](https://drive.google.com/file/d/1FsG7Fp74y_TpaPhJ7U066ot8A07BPhr3/view).
6. [Python indices](https://drive.google.com/file/d/1aLBP_nhnoqz075puKg30zuF3F_QBOXYM/view).

The benchmark notebook documents corrections to Covid3Month_disc and the three
FloodModeling_disc datasets. Their current test sizes are 61, 202, 201 and 184,
respectively. Analyze these data-version differences separately rather than
automatically attributing every discrepancy to the classifier.

The downloader writes URLs, archive sizes and SHA256 hashes to
`downloads/manifest.json`. Each `features/<dataset>/metadata.json` also records
TRAIN/TEST file hashes, classes, dimensions, feature counts and the transform
environment. The portable repository preserves the recorded metadata in
`dataset_metadata.json` and archive provenance in `data_provenance.json`.
Legacy suffixes such as `_eq`, `_nmv` and `_disc` are mapped to the author's
canonical dataset names without changing data contents. Official indices are
read directly from their ZIP archives.

## Environment

The recorded environment uses Python 3.11, CPU PyTorch 2.5.1, NumPy 1.26.4 and
scikit-learn 1.5.2 on an Intel Core i7-12700H with approximately 16 GB RAM.
See `hardware.json` and `requirements-lock.txt`. This is a documented working
environment, not a claim to recover unspecified historical dependency versions.

Run the following commands in PowerShell from the **repository root**. Skip
installation if the environment already exists.

```powershell
uv venv --python 3.11 .venv
uv pip install --python .venv/Scripts/python.exe --cache-dir .cache/uv --link-mode copy -r reproduction/requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
```

## Downloads and the original runner

```powershell
.\.venv\Scripts\python.exe reproduction/download_data.py
.\.venv\Scripts\python.exe reproduction/download_data.py --datasets --drive java_indices python_indices new_datasets
New-Item -ItemType Directory -Force reproduction/original_smoke
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
.\.venv\Scripts\python.exe quant/code/run.py -i reproduction/data -o reproduction/original_smoke -d reproduction/smoke_datasets.txt -r 1
```

The final command invokes the unchanged author runner on Coffee, GunPoint,
ItalyPowerDemand and Chinatown. `quant.py` defines the algorithm; executing it
alone does not download data or start experiments. The original runner does not
set a classifier seed, so repeated smoke accuracies can vary.

## Complete reproduction and resuming

The supplied result directories contain the recorded study. For a fresh run on
another machine, preserve them under different names in a separate project copy,
as explained in the root README, or use a new `--output` directory.

```powershell
# Optional first pass: original split only, saved separately.
.\.venv\Scripts\python.exe reproduction/run_reproduction.py --stop 1 --n-jobs 4 --output reproduction/baseline_results

# All 30 splits; existing dataset/split rows are skipped.
.\.venv\Scripts\python.exe reproduction/run_reproduction.py --stop 30 --n-jobs 8

# A smaller run can use the same entry point.
.\.venv\Scripts\python.exe reproduction/run_reproduction.py --datasets Coffee GunPoint ItalyPowerDemand Chinatown --stop 30 --n-jobs 8

# Aggregate records and generate tables, statistics and figures.
.\.venv\Scripts\python.exe reproduction/summarize.py

# Inspect progress without importing PyTorch or interrupting training.
python reproduction/status.py
```

The runner checks index counts, ranges, uniqueness and stratified class counts.
Each completed fit appends one CSV row; failures are recorded in `failures.json`.
Repeat the same command to resume. Use a new output directory after changing
model settings or environment; do not mix configurations in one result set.

## Implementation differences and validation

- We use `random_state=resample`; the original runner does not set this seed.
- Current QUANT includes the 2025 multivariate extension. The historical file
  `reference/quant_2023.py` comes from commit
  `5956b7252464d2c17e3b541e051b05f03c7442ff` and retains the upstream GPL v3 license
  in `../quant/LICENSE.txt`.
- Features are extracted in batches of 64 and cached for reuse across splits.
  The transform uses each case and its representation lengths, without learning
  labels or across-case statistics. Each classifier still fits only its current
  training indices.
- Floating-point batching differences are possible. On four real datasets,
  maximum absolute feature differences were approximately `3.8e-6`, and checked
  seeded predictions were identical. This does not guarantee bitwise identity
  on every dataset.
- Caching supports the accuracy reproduction; it does not establish the paper's
  single-core runtime claim. Main fits use eight classifier threads, with feature
  preparation recorded separately. An initial original-split run used four threads.
- During part of the main run, two disjoint groups of 71 datasets ran concurrently,
  each using eight classifier threads. CPU competition prevents interpreting those
  fit times as isolated performance measurements. The single-process command above
  follows the same accuracy protocol.
- When caches are complete, the training runner avoids importing PyTorch, reducing
  process memory use without changing the cached transform.

```powershell
.\.venv\Scripts\python.exe reproduction/validate_pipeline.py
```

Validation checks the original and new `.ts` loaders, historical and current
univariate transforms, batched-cache numerical agreement, and direct-versus-cached
predictions on official resample 1. Results are saved in `validation.json`.

## Result files

- `original_smoke/results.csv`: four-dataset execution of the original runner.
- `results/<dataset>.csv`: each split's accuracy, author comparison, timings,
  dimensions, seed and index source.
- `results/all_runs.csv`: all primary experiment records.
- `results/comparison.csv`: dataset means and matched-split comparisons.
- `results/summary.json`: coverage, group statistics and paired Wilcoxon tests.
- `results/REPORT.md`: English result summary and interpretation limits.
- `results/accuracy_comparison.png` / `.pdf`: author-versus-reproduction plot.
- `results/environment.json`: settings, dependencies and original source hashes.

The optional `baseline_results/` directory is generated by the preliminary
four-thread run and is not distributed. The reproduced scope is QUANT's main
accuracy experiment. Competing classifiers, QUANTFAST and the paper's complete
sensitivity study were not rerun.

## Separate label-order diagnostic

The author identifies a historical Python resampler that consumes one RNG stream
class by class. For numeric labels such as 1 through 10, string and numeric sorting
can differ, so a seed alone does not determine the same training membership.

The diagnostic checks all 30 x 29 current Python splits, then runs the numeric-order
alternative on all ten datasets whose class order differs: 290 additional fits.
Run it after the paired extension timing experiment to avoid competing for CPU.

```powershell
.\.venv\Scripts\python.exe reproduction/diagnose_label_order.py
```

Outputs are in `label_order_diagnostic/`. Correlations and per-split absolute
accuracy errors use resamples 1-29; original split zero is reused. Only resampler
class iteration changes, while classifier label encoding remains fixed. This
sensitivity analysis does not replace the primary results or prove the author's
historical label dtype. Assumptions, source commit and index hash are recorded in
`label_order_diagnostic/configuration.json`.
