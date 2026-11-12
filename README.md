# Revisiting QUANT: Reproduction and Adaptive Feature Budgeting

AI4TS \(COMP41850\) reproduction project studying the 2023 QUANT preprint by Dempster,
Schmidt and Webb. The original author files in `quant/` are unchanged.

This directory is a self-contained repository copy: commands below are run from
this directory, not from its parent. Raw datasets, feature caches, virtual
environments and temporary build files are excluded. Public data can be fetched
with the included download script.

## Repository layout

| Path | Contents |
|---|---|
| `quant/` | Unchanged upstream implementation, published results and GPL v3 license |
| `reproduction/` | Data download, primary reproduction, validation, diagnostics and recorded results |
| `experiments/` | Feature-budget implementation, frozen protocol/settings, development and evaluation results |
| `verify_artifacts.py` | Offline checks of result coverage, matching baselines and source/settings hashes |

For local review before uploading, see [Repository review and upload notes](UPLOAD_GUIDE.md).

## Start here

- [Reproduction results](reproduction/results/REPORT.md)
- [Extension results](experiments/results/REPORT.md)
- [Reproduction instructions](reproduction/README.md)
- [Extension experiments](experiments/README.md)

The main reproduction covers **142 datasets × 30 official splits = 4260 fits**.
Macro accuracy is **85.5100%**, against the author's **85.6774%**. On the original
112 datasets it is **86.7108% versus 86.7131%**. The full-archive difference is
concentrated in four datasets with documented data corrections; excluding those
four gives **85.4780% versus 85.4635%**. All datasets remain in the primary result.

The extension compares baseline QUANT with uniform, ANOVA, fixed pilot and
adaptive pilot selection on five matched official splits. Its mass parameter
is selected using original-training-only internal validation and then frozen.
The full result table, including negative results and additional independence
checks, is in [experiments/results/REPORT.md](experiments/results/REPORT.md).

## Reproduce the study

To check the supplied records after installing the environment, without
downloading data or training models:

```powershell
.\.venv\Scripts\python.exe verify_artifacts.py
```

This checks the recorded evidence; rerunning the experiments requires the
download and training steps below.

Run from this directory in PowerShell. Python 3.11 and a CPU PyTorch environment
are used. `reproduction/requirements-lock.txt` records exact installed packages.

The delivered CSVs are the recorded study, so the runners normally resume/skip
them. For a **fresh full run on another machine**, first make a separate copy of
the project and rename these supplied folders within that copy:

| Supplied folder | Preserve it as |
|---|---|
| `reproduction/results` | `reproduction/recorded_results` |
| `reproduction/label_order_diagnostic` | `reproduction/recorded_label_order_diagnostic` |
| `experiments/results` | `experiments/recorded_results` |
| `experiments/development` | `experiments/recorded_development` |

The following commands then recreate the default output folders. This avoids
mixing this machine's recorded configuration with a new platform or timing run.
Do not delete the supplied records merely to rerun the study. For an accuracy-only
rerun, `--output reproduction/new_results` and
`reproduction/summarize.py --results reproduction/new_results` are also supported.

```powershell
uv venv --python 3.11 .venv
uv pip install --python .venv/Scripts/python.exe --cache-dir .cache/uv --link-mode copy -r reproduction/requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match

.\.venv\Scripts\python.exe reproduction/download_data.py
.\.venv\Scripts\python.exe reproduction/download_data.py --datasets --drive java_indices python_indices new_datasets
.\.venv\Scripts\python.exe reproduction/run_reproduction.py --stop 30 --n-jobs 8
.\.venv\Scripts\python.exe reproduction/validate_pipeline.py
.\.venv\Scripts\python.exe reproduction/summarize.py

.\.venv\Scripts\python.exe experiments/develop.py
.\.venv\Scripts\python.exe experiments/evaluate.py
.\.venv\Scripts\python.exe experiments/analyze.py
.\.venv\Scripts\python.exe experiments/robustness.py

.\.venv\Scripts\python.exe reproduction/diagnose_label_order.py
.\.venv\Scripts\python.exe verify_artifacts.py
```

Keep these model-training stages sequential for meaningful paired timings.
Existing completed CSV rows are reused. To independently rerun experiments,
use a clean project copy or a new output directory supported by the runners;
do not mix configurations in the supplied results. Downloaded datasets and
feature caches are large and are excluded from the portable source package.
Network downloads are required when rebuilding them in a new environment.

The original author's executable entry point is `quant/code/run.py`;
`quant/code/quant.py` defines the method. See the reproduction README for an
original-code smoke run and a short four-dataset run.

## License

Original QUANT code and the historical reference retain the upstream GNU GPL v3
license in [quant/LICENSE.txt](quant/LICENSE.txt). Cite the original paper and
dataset providers when reusing this study.
