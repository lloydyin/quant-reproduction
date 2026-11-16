# QUANT Reproduction and Adaptive Feature Budgeting

Course project for AI4TS (COMP41850), University College Dublin.

We reproduce the main accuracy experiment from the 2023 QUANT paper by
Dempster, Schmidt and Webb, then test whether feature selection can reduce
training time while maintaining accuracy. The original code in `quant/` is
unchanged.

## Results

The reproduction covers 142 datasets with 30 official train/test splits per
dataset, giving 4,260 runs. Accuracy is averaged over splits within each dataset,
then equally across datasets.

| Datasets | Our accuracy (%) | Published accuracy (%) |
|---|---:|---:|
| All 142 | 85.5100 | 85.6774 |
| Original 112 | 86.7108 | 86.7131 |
| Additional 30 | 81.0272 | 81.8104 |
| Excluding four corrected datasets | 85.4780 | 85.4635 |

Most of the overall difference is concentrated in four datasets with documented
data corrections. They remain in the main result; the last row is an additional
comparison. See the [reproduction results](reproduction/results/REPORT.md) for
the details.

For feature selection, we compare full QUANT with uniform selection, ANOVA
ranking, a fixed pilot budget and an adaptive pilot budget. These experiments
use five matched splits per dataset. Adaptive selection keeps 33.6% of features
on average and increases mean accuracy from 85.11% to 85.42%. Total selection
and fitting time falls by 34.8%, although 81 of the 142 datasets become slower.
These timings exclude data loading and the QUANT transform.

Some cases used to choose the threshold reappear in later test resamples.
We report separate checks using the original splits and the 118 datasets not
used for threshold development. See the [feature-selection results](experiments/results/REPORT.md)
and [additional checks](experiments/results/ROBUSTNESS.md).

## Repository contents

| Path | Contents |
|---|---|
| `quant/` | Original QUANT code, published results and license |
| `reproduction/` | Data download, reproduction scripts, diagnostic checks and saved results |
| `experiments/` | Feature-selection methods, validation, evaluation and saved results |
| `verify_artifacts.py` | Checks that the saved results are complete and consistent with the recorded settings and source files |

Raw datasets and feature caches are not included. The download commands below
fetch the data needed to run the experiments.

## Setup

We ran the experiments on Windows with Python 3.11 and CPU PyTorch 2.5.1.
Exact package versions are listed in
[`reproduction/requirements-lock.txt`](reproduction/requirements-lock.txt).

The commands below use PowerShell. Install Git and
[uv](https://docs.astral.sh/uv/getting-started/installation/) first, then clone
the repository and create the environment:

```powershell
git clone https://github.com/lloydyin/quant-reproduction.git
cd quant-reproduction
uv venv --python 3.11 .venv
uv pip install --python .venv/Scripts/python.exe --cache-dir .cache/uv --link-mode copy -r reproduction/requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
```

Run all remaining commands from the repository root.

## Check the saved results

To check the saved results without downloading data or training models:

```powershell
.\.venv\Scripts\python.exe verify_artifacts.py
```

This checks result coverage, matching baseline accuracies, and recorded settings
and source hashes. It does not rerun the experiments.

## Run the experiments

The scripts skip runs that already have saved results. For a fresh full run,
use a separate clone and rename these folders before running the commands below:

| Folder | Rename to |
|---|---|
| `reproduction/results` | `reproduction/recorded_results` |
| `reproduction/label_order_diagnostic` | `reproduction/recorded_label_order_diagnostic` |
| `experiments/results` | `experiments/recorded_results` |
| `experiments/development` | `experiments/recorded_development` |

This preserves the original results for comparison. The scripts will create new
output folders. Run the steps in order, with one training script at a time so
the timing measurements are comparable.

Download the datasets and official split indices:

```powershell
.\.venv\Scripts\python.exe reproduction/download_data.py
.\.venv\Scripts\python.exe reproduction/download_data.py --datasets --drive java_indices python_indices new_datasets
```

Run the 30-split reproduction, check the implementation and summarise the results:

```powershell
.\.venv\Scripts\python.exe reproduction/run_reproduction.py --stop 30 --n-jobs 8
.\.venv\Scripts\python.exe reproduction/validate_pipeline.py
.\.venv\Scripts\python.exe reproduction/summarize.py
```

Run feature-selection validation, evaluation and analysis:

```powershell
.\.venv\Scripts\python.exe experiments/develop.py
.\.venv\Scripts\python.exe experiments/evaluate.py
.\.venv\Scripts\python.exe experiments/analyze.py
.\.venv\Scripts\python.exe experiments/robustness.py
```

Run the label-order diagnostic and check the completed results:

```powershell
.\.venv\Scripts\python.exe reproduction/diagnose_label_order.py
.\.venv\Scripts\python.exe verify_artifacts.py
```

Reproduction tables and plots are written to `reproduction/results/`.
Feature-selection tables and plots are written to `experiments/results/`,
validation results to `experiments/development/`, and label-order comparisons
to `reproduction/label_order_diagnostic/`.

The [reproduction README](reproduction/README.md) includes smaller runs and
instructions for running the original author code. The
[experiments README](experiments/README.md) describes the selection methods,
evaluation settings and output files.

## Original work and license

This project uses [QUANT](https://github.com/angus924/quant) by Angus Dempster,
Daniel F. Schmidt and Geoffrey I. Webb. We target the
[2023 preprint](https://arxiv.org/abs/2308.00928v1).

The original QUANT code and historical implementation retain their
[GNU GPL v3 license](quant/LICENSE.txt). Please cite the original paper and
dataset providers when using this work.
