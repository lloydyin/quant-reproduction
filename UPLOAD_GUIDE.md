# Repository review and upload notes

This directory is a self-contained, English-only repository copy. Run its
commands from this directory. The original working project remains separate.

## Included materials

- `quant/`: unchanged upstream implementation, published results and GPL v3 license.
- `reproduction/`: download, reproduction, analysis and validation scripts;
  all 4,260 primary experiment records; data provenance and version information.
- `experiments/`: feature-budget methods, the frozen protocol and settings,
  504 development records, 3,550 paired evaluation records and robustness checks.
- `verify_artifacts.py`: offline coverage, pairing and source-integrity checks.

Per-split CSVs are retained so readers can audit the reported means, win/tie/loss
counts and statistical results without downloading the full benchmark first.

## Excluded materials

Raw datasets, download archives, feature caches, virtual environments, temporary
build files, internal working notes, course handouts and reference-paper PDFs
are excluded. The manuscript PDF and LaTeX sources are maintained separately in
Overleaf. The `.gitignore` file covers generated data, caches, build output and
the manuscript folder.
Use the included public-data download script when rerunning the models.

## Files to review

- `README.md`: repository overview, installation and reproduction commands.
- `experiments/feature_budget.py`: the feature-selection extension.
- `reproduction/results/comparison.csv`: comparison with the author's results.
- `experiments/results/summary.csv`: the paired extension comparison.

Complete author details and actual contributions in Overleaf. Include the public
repository URL in the report and submit its final PDF through Brightspace.
`COPY_MANIFEST.json` records the prepared file hashes. Later edits will change
the hashes of the corresponding files.

## Local verification and upload

After installing the environment, run from the repository root:

```powershell
.\.venv\Scripts\python.exe verify_artifacts.py
```

This verifies recorded result coverage and consistency without training models
or downloading data.

Use GitHub Desktop, Git or the GitHub website to publish later updates after
reviewing them. In Git terminology, `push` uploads local commits; `pull` retrieves
and integrates remote changes.
