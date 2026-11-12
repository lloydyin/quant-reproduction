# Feature-budget experiment results

Completed all methods and splits for 142/142 datasets, with 3550/3550 method records. Each method uses the same five official splits.

Accuracy and feature retention are fractions. Accuracy differences are percentage points relative to the matched baseline.

| Method | Mean accuracy | Difference (pp) | Wins/ties/losses | Holm p | Retained feature fraction | Downstream training time ratio |
|---|---:|---:|---|---:|---:|---:|
| QUANT | 0.851134 | +0.0000 | 0/142/0 | 1 | 1.000 | 1.000 |
| Uniform half | 0.848307 | -0.2827 | 41/11/90 | 0.00033896 | 0.500 | 0.589 |
| Fisher half | 0.849850 | -0.1283 | 47/20/75 | 0.0037813 | 0.500 | 0.618 |
| Pilot half | 0.852370 | +0.1237 | 77/14/51 | 0.048126 | 0.500 | 0.790 |
| Adaptive pilot | 0.854189 | +0.3055 | 85/16/41 | 0.00061732 | 0.336 | 0.652 |

Training time includes feature selection, the pilot forest and final classifier fitting. It excludes the common QUANT feature transformation and data loading.
Selected-matrix bytes and tree-array bytes are not peak process RAM. These measurements do not establish end-to-end acceleration or lower peak RAM.
Maximum baseline accuracy difference from the corresponding five primary-reproduction splits: 0.

Settings were frozen after internal training-set validation. Unsuccessful and degraded variants are retained in the report.
