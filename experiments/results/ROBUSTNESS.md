# Development-overlap audit

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
