# Historical label-order sensitivity diagnostic

This is separate from the primary 4260 official-index reproduction and the
frozen feature-selection experiment. It does not replace either result set.

The author identifies the historical class-wise Python resampler in
[issue 1](https://github.com/angus924/quant/issues/1#issuecomment-1809307377),
pointing to tsml-eval commit `5dac9ce9955bfb4596735ef117438c4c2a299acc`.
One RNG stream is consumed in unique-label order. Numeric labels represented
as strings can have a different order from their numeric values.

## Audit and experiment

- All 870 current Python splits (30 additional datasets × resamples 1–29)
  match the lexical-label generator in both training membership and order.
- Ten datasets have a different numeric class order; all 290 corresponding
  train memberships change. Named, non-numeric classes keep lexical order.
- Rerun those 290 splits, with unchanged QUANT features, classifier settings,
  classifier label encoding and resample-index seed. Reuse original split zero.
- Correlation and absolute-error comparisons use resamples 1–29 only. The
  `official_mean`, `numeric_mean` and `author_mean` columns use all 30 splits.

Seven datasets reach correlation above 0.8 with the author's per-split accuracies
under numeric ordering, with lower mean absolute error: the three AllGestureWiimote
datasets, MelbournePedestrian, PLAID, PickupGesture and ShakeGesture. Median
correlation across all ten changes from **0.109 to 0.851**.

The three GestureMidAir datasets remain poorly aligned. D1's absolute error
increases; D2 improves but has low correlation; D3's absolute-error change is
negligible at the precision of the author's CSV. Do not count that tiny change
as meaningful improvement. See all rows in `summary.csv`.

This evidence supports a historical label-order mismatch for several datasets,
but does not prove the author's original dtype or recover all original data
and random states. No ordering was chosen separately per dataset to maximize
agreement: the one numeric alternative was tested on every structurally
affected dataset. The primary results still use the official index archives.

Reproduce from the project root, after the paired timing experiment:

```powershell
.\.venv\Scripts\python.exe reproduction/diagnose_label_order.py
```

`configuration.json` records assumptions, source commit and index hash;
`index_audit.csv` records membership/order checks and overlaps;
`<dataset>.csv` contains every diagnostic accuracy.
