# evaluation_results.csv — superseded-data notice

These results are retained as a diagnosed evaluation-methodology artifact per
the paper section on the base-rate fallacy. They are not primary performance
claims. See `results/primary/` for the supported sequential detection findings.

Retained for reproducibility of the original Table III protocol. These numbers do
NOT reflect realistic-prevalence performance — see
`results/methodology-diagnostics/sling_fix_confidence_intervals.json`,
`results/methodology-diagnostics/threshold_sweep_cv_protocol.json`, and
`results/primary/sequential_ato_study*.json` for the corrected methodology and results.

Background: the Table III protocol evaluates on a 20-user sample with ~48%
positive prevalence. At the realistic prevalence of the full corrected holdout
(10,165 messages, 29 positives, 0.29%), the same fixed-threshold configuration
collapses (F1 = 0.0059). Fixed-threshold F1/precision measured on
positive-enriched samples do not generalize to realistic base rates.

The anomaly-first user sampling that produced this artifact has since been
removed from `evaluation.py` (`--max-users` now draws a seeded uniform random
sample); `evaluate.py --max-users 20` no longer regenerates these numbers.
This file is retained unchanged as the historical record.
