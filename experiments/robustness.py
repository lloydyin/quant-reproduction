"""Post-protocol overlap audit; no parameter or model selection is performed.

The protocol's 24 development datasets use original TRAIN labels. On later
official resamples, some of those cases enter TEST. Report original-split and
non-development-dataset results separately; do not call all five splits a fully
independent holdout. This analysis was added during the frozen evaluation run.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from analyze import METHODS, holm

ROOT = Path(__file__).resolve().parent


def main():
    result = json.loads((ROOT / "results/summary.json").read_text())
    if not result["full_evaluation_complete"]:
        raise SystemExit("Requires all 3550 extension results")
    runs = pd.read_csv(ROOT / "results/all_runs.csv")
    development = json.loads((ROOT / "development/datasets.json").read_text())
    scopes = {
        "original_split_all142": runs[runs["resample"] == 0],
        "nondevelopment118_five_splits": runs[~runs.dataset.isin(development)],
        "development24_five_splits": runs[runs.dataset.isin(development)],
    }
    rows = []
    for scope, subset in scopes.items():
        means = subset.groupby(["dataset", "method"]).mean(numeric_only=True)
        base = means.xs("baseline", level="method")
        group = []
        for method in METHODS:
            part = means.xs(method, level="method").loc[base.index]
            delta = np.round(part.accuracy.to_numpy() - base.accuracy.to_numpy(), 12)
            p = float(wilcoxon(delta).pvalue) if np.any(delta) else 1.0
            group.append(dict(scope=scope, method=method, datasets=len(base),
                mean_accuracy=float(part.accuracy.mean()),
                accuracy_difference_pp=float(100*delta.mean()),
                wins=int((delta>0).sum()), ties=int((delta==0).sum()), losses=int((delta<0).sum()),
                wilcoxon_p=p, mean_feature_fraction=float(part.feature_fraction.mean()),
                total_training_ratio=float(part.training_seconds.sum()/base.training_seconds.sum())))
        group[0]["holm_p"] = 1.0
        for row, adjusted in zip(group[1:], holm([r["wilcoxon_p"] for r in group[1:]])):
            row["holm_p"] = float(adjusted)
        rows.extend(group)
    table = pd.DataFrame(rows)
    table.to_csv(ROOT / "results/robustness.csv", index=False)
    print(table.to_string(index=False))
    note = """# Development-overlap audit

This analysis was added during the frozen paired evaluation, after recognizing
that original TRAIN cases used in internal development can enter TEST in
official resamples 1–4. It is an explicitly additional robustness analysis,
not part of the immutable original protocol and not a basis for retuning.

The 24 development datasets use only original TRAIN labels to choose a single
global mass parameter. Each feature selector is still refitted exclusively on
its current training split. Nevertheless, later resamples on those 24 datasets
are not fully independent of parameter development.

`robustness.csv` reports the original split on all 142 datasets and five splits
on the other 118 datasets, plus the development subset for transparency.
Holm correction is applied separately within each four-comparison scope; these
are exploratory checks, not a new family-wide confirmatory significance claim.
The originally specified 3550-row evaluation is retained without replacement.
"""
    (ROOT / "results/ROBUSTNESS.md").write_text(note, encoding="utf-8")


if __name__ == "__main__":
    main()
