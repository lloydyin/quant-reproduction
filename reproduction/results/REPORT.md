# QUANT reproduction results

Completed 4260/4260 runs across 142/142 datasets; 142 datasets have all 30 resamples.

Status: The 142 x 30 primary accuracy reproduction is complete.

The table uses equal dataset weights and matches the author's resample columns. Unequal completion counts indicate a progress snapshot only. Accuracy is expressed as a fraction; differences are percentage points.

| Scope | Datasets | Reproduced accuracy | Author accuracy | Difference (pp) | Wilcoxon p |
|---|---:|---:|---:|---:|---:|
| All available | 142 | 0.855100 | 0.856774 | -0.1673 | 0.698568 |
| Excluding 4 corrected datasets | 138 | 0.854780 | 0.854635 | +0.0145 | 0.833303 |
| Original 112 | 112 | 0.867108 | 0.867131 | -0.0023 | 0.940992 |
| Additional 30 | 30 | 0.810272 | 0.818104 | -0.7832 | 0.468834 |

## Fifteen largest absolute dataset differences

| Dataset | Resamples | Reproduced | Author | Difference (pp) |
|---|---:|---:|---:|---:|
| Covid3MonthDiscrete | 30 | 0.618033 | 0.805238 | -18.7205 |
| FloodModeling3Discrete | 30 | 0.926812 | 0.964569 | -3.7757 |
| FloodModeling1Discrete | 30 | 0.944059 | 0.965180 | -2.1121 |
| GestureMidAirD3Eq | 30 | 0.478974 | 0.459487 | +1.9487 |
| DodgerLoopDayNmv | 30 | 0.616883 | 0.600433 | +1.6450 |
| PickupGestureWiimoteZEq | 30 | 0.807333 | 0.822000 | -1.4667 |
| FloodModeling2Discrete | 30 | 0.975788 | 0.987267 | -1.1480 |
| PigCVP | 30 | 0.644712 | 0.635897 | +0.8814 |
| Rock | 30 | 0.796667 | 0.805333 | -0.8667 |
| BeetleFly | 30 | 0.886667 | 0.878333 | +0.8333 |
| AllGestureWiimoteYEq | 30 | 0.699667 | 0.706191 | -0.6524 |
| ShakeGestureWiimoteZEq | 30 | 0.849333 | 0.855333 | -0.6000 |
| AllGestureWiimoteXEq | 30 | 0.681095 | 0.675143 | +0.5952 |
| Beef | 30 | 0.780000 | 0.785556 | -0.5556 |
| PLAIDEq | 30 | 0.921539 | 0.916325 | +0.5214 |

## Interpretation and scope

- The author did not fix the ExtraTrees seed; this run uses random_state=resample.
- The local quant/code/quant.py is unchanged. The 2023 implementation and 2025 multivariate update produced identical features on the four checked univariate datasets.
- Features are cached in batches using a sample-wise transform. Checked floating-point differences were at most approximately 3.8e-6, with identical seeded predictions.
- Caching and multithreaded classifiers change the computation schedule. These timings do not reproduce the paper's single-core, freshly transformed pipeline; see environment.json for thread settings.
- The expanded benchmark documents corrections to four datasets; see the README. Do not automatically attribute data-version differences to the model.
- Paired two-sided Wilcoxon tests use dataset-mean accuracy differences rounded to six decimal places. Their p-values do not establish equivalence.
- This study reproduces QUANT's primary accuracy experiment. Competing models and the paper's full sensitivity study were not rerun.

Per-split records: all_runs.csv. Dataset comparisons: comparison.csv. Execution environment: environment.json.
