# QUANT extension experiment protocol

This protocol is written before running extension test-set evaluations.
The original QUANT reproduction (142 datasets x 30 official resamples) is separate.

## Research questions

1. Does the published accuracy reproduce under documented data/code versions?
2. Can the dense quantile feature vector be reduced without a material accuracy loss?
3. Does supervision improve on simply keeping half of the feature columns?
4. Can a training-only pilot forest choose a useful, data-dependent feature budget?

The motivation comes from QUANT Section 3.5: classifier training dominates cost,
and the authors identify feature-space reduction as an opportunity. This project
evaluates a feature-selection extension; it does not claim a new state of the art.

## Proposed method: pilot-guided feature budgeting

Fit a 32-tree ExtraTrees pilot using only the training split. Rank the original
QUANT feature columns by impurity importance. Let k_mass be the smallest prefix
whose cumulative importance reaches a prescribed mass (default 0.95).
Allocate 80% of the selected budget to the highest-ranked features and reserve
20% for evenly spaced columns among the remaining columns, to preserve coverage
of weaker features and reduce reliance on a short pilot.

For the adaptive version, K = clip(ceil(k_mass / 0.8), ceil(F/4), ceil(3F/4)),
where F is the original feature count. Ties in feature scores are broken by
original column index. The final classifier retains the original 200 trees,
entropy criterion and 10% candidate-feature proportion, applied to the selected
feature space. The pilot is discarded after selection.

This adds a supervised training stage and data-dependent feature selection;
it is not only a change to QUANT's depth/divisor or forest tree count.

## Controls and ablations

- baseline: all QUANT features, original 200-tree classifier.
- uniform_half: evenly spaced 50% of columns, no supervision.
- fisher_half: top 50% using training-only univariate ANOVA F statistics.
- pilot_half: pilot ranking plus coverage reserve with a fixed 50% budget.
- adaptive_pilot: the proposed data-dependent budget.

## Development without test-set tuning

Choose 24 datasets by deterministic clustering of log(training size), log(series
length), and log(number of classes), among datasets with at least 100 original
training cases and at least five original training cases per class. Pick the
nearest dataset to each cluster centre. No test accuracy is used for selection.

Use three stratified folds within each selected dataset's ORIGINAL TRAIN split.
Candidate cumulative-importance masses are 0.90, 0.95 and 0.99. All other settings
are fixed. For choosing the mass, first retain candidates whose macro validation
accuracy is within 0.5 percentage points of the highest candidate accuracy;
then choose the lowest mean retained-feature fraction (tie: smaller mass).
This is a development criterion, not a claim of statistical equivalence.
The resulting setting is frozen before evaluating the 142-dataset test results.

## Confirmatory extension evaluation

Use all 142 datasets and official resamples 0-4 (five matched splits each).
Every method receives the identical training and test indices and uses the
same final-classifier seed (resample index). Pilot seed = 10000 + resample index.
Compare methods by per-dataset mean accuracy over these five splits, followed
by an unweighted mean across datasets. Do not compare these five-split means
directly to the paper's 30-split means.

All selectors are fitted exclusively on each training split. During development,
selectors are re-fitted inside each training fold. No feature selector sees
validation or test labels during fitting. Feature caching is sample-wise and
label-independent, as verified by the reproduction pipeline.

## Measurements and analysis

Record test accuracy, retained feature count/fraction, selection time, final
classifier fit time, prediction time, and selected training-matrix size in bytes.
The selection time includes pilot fitting. When a pilot can be reused between
experimental variants, charge its full measured time to EACH variant to avoid
an artificial efficiency advantage. Also record classifier tree-array bytes;
neither this nor feature-matrix bytes is peak process RAM.

Run the paired method benchmark in one process after the original reproduction
finishes, using eight classifier threads. Randomise method evaluation order
deterministically within each dataset/resample. Timing excludes common QUANT
feature extraction and data loading, and therefore measures the downstream
pipeline rather than end-to-end raw-series training. No speed claim based on
cached timings will be presented as reproduction of the paper's single-core time.

Report means, medians, paired win/tie/loss, accuracy differences, and paired
Wilcoxon tests across dataset means. Correct the four variant-versus-baseline
accuracy comparisons using Holm's method. Report efficiency/accuracy trade-offs
and unsuccessful alternatives without suppressing negative results.

## Scope and limitations

Extension evaluation uses five resamples, whereas primary QUANT reproduction
uses 30. The 24 development datasets overlap the 142-dataset evaluation archive,
but development uses only original training labels; no unseen-domain claim is
made. Importance ranking may favour some features, may be unstable on small
training sets, and may discard useful interactions. These are empirical questions,
not assumed advantages. Original feature extraction is still required to fit the
selector, so a smaller selected matrix does not automatically reduce peak RAM
or transformation cost.
