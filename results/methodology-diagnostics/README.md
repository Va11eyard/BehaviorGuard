# Methodology diagnostics — dated exhibits

These JSON/CSV files are **snapshots of the scorer as it stood when each was
generated**, not live outputs. They are kept as the evidentiary record for the
paper's methodology chapter. Unlike `results/primary/`, they are not expected to
reproduce byte-for-byte from current `HEAD` unless the generating script pins the
scorer version (see below).

## The scorer change that dates them

On **2026-08-11, between 11:44 and 12:27**, the medium-sensitivity composite
weights were retuned:

| | semantic | linguistic | temporal | linguistic-off, renormalized |
|---|---|---|---|---|
| legacy (coarse grid, F1 at FPR=0) | 0.40 | 0.35 | 0.25 | 0.6154 / 0.3846 |
| current (5-fold CV AUC-PR, `weight_tuning_aucpr.json`) | 0.90 | 0.00 | 0.10 | 0.9 / 0.1 |

At the same time `SystemConfig.linguistic_component_enabled` was added with
default `False`. Both landed in the working tree before commit `2660aa7`, which
is why git alone cannot date them; the 08-11 file mtimes above pin the window.

A second, earlier change affects the **linguistic** axis only: commit `78c3528`
(2026-07-22) replaced constant 0.5/0.6 formality/politeness placeholders with
length/keyword proxies in both profile building and `message_to_current_message`,
and `analyzers/linguistic_ml.py` was rewritten in the same pre-`2660aa7` window.

## Reproducing a legacy-weight exhibit

`SystemConfig.composite_weights` overrides `CompositeScorer.WEIGHTS`. The
scripts in group A below already pass
`turnshift.scorers.composite.LEGACY_MEDIUM_WEIGHTS`, so re-running them
reproduces their committed output without touching the production default.

## Classification

### A. Legacy-weight exhibits — reproduce from HEAD (scripts pinned)

Linguistic-excluded composite at renormalized 0.6154 / 0.3846. Verified: a
weights-restored replay of `sling_exclusion_holdout_eval.py`'s without-linguistic
arm matches the committed JSON exactly (P 0.38095, R 0.27586, F1 0.32,
AUC 0.917070938682198, FPR 0.001282557).

- `sling_exclusion_holdout_eval.json` — without-linguistic arm
- `sling_production_audit_personachat.json` — `metrics_without_sling`
- `sling_bootstrap_stability.json` — bootstrap-width stability for the same audit
- `sling_fix_confidence_intervals.json` — F1 0.3137 [0.143, 0.474]; cited in the paper
- `mahalanobis_bootstrap_comparison.json` — cosine vs Mahalanobis
- `threshold_sweep_cv_protocol.json` — τ sweep, linguistic off
- `kimi_recall_diagnostics.json` — per-positive composite scores, τ sweep
- `behaviorguard_canonical_verify.json` — BG λ=0.5, overrides off, τ=0.60
- `external_positive_holdout_eval.json` — "corrected in-repo templates" arm; cited in the paper

Also legacy-weight, and archived rather than diagnostic:
`../archived-per-message-study/evaluation_results.csv`, `final_ablation_table.md`,
`corrected_vs_original_comparison.md`.

### B. With-s_ling exhibits — do NOT reproduce from HEAD

These score the linguistic component at weight 0.35. Restoring the weights is
**not sufficient**: they also predate the 07-22 linguistic-proxy change. A
weights-restored replay of the with-linguistic arm gives AUC 0.6558 against the
committed 0.6550 (ranking preserved) but F1 0.0081 / recall 0.448 / FPR 31.1%
against the committed 0.0036 / 0.069 / 10.6% — the τ=0.60 operating point does
not survive. Treat these as historical exhibits of the discredited configuration,
which is how the paper already frames them.

- `sling_exclusion_holdout_eval.json`, `sling_production_audit_personachat.json` — with-arms
- `corrected_pipeline_eval.json`
- `corrected_proper_generalization_eval.json`
- `corrected_bg_fair_tuning.json` — records 0.4/0.35/0.25 explicitly, plus an α sweep
- `diagnostic_gate_eval.json`
- `gardening_positive_investigation.json` — per-message composites with `s_ling = 1.0`

### C. Current — consistent with HEAD's default scorer

- `weight_tuning_aucpr.json` (08-11 11:44) — the retune itself
- `stylometric_linguistic_eval.json` (08-11 12:06)
- `baselines_matched_fpr_holdout.json` (08-11 12:27) — AUC-ROC 0.9532 / AUC-PR 0.2633; cited in the paper

### D. Not scorer-dependent

The composite is never invoked, so the retune cannot affect them:
`corrected_injection_validation.json`, `corrected_injection_validation_pc.json`,
`positive_label_surface_audit.json`, `task1_isolation_forest_results.{json,csv}`,
`task5_autoencoder_results.{json,csv}`, `task5_ae_fairness_diagnostic.json`,
`cv_semantic_pr_curve.json` (semantic component alone),
`wildchat_corpus_acquisition.json`, `deployment_math_audit.json` (arithmetic only).

All of `results/primary/` is CUSUM-over-residuals and independent of the
composite, except the `permsg_bg` comparator, which was regenerated under the
current weights so that the primary table reproduces from HEAD.

## Backlog

Stamp a config hash (weights, `linguistic_component_enabled`,
`semantic_scoring_mode`) into every generated diagnostic JSON, plus a check that
flags files whose stamped hash no longer matches current code. Both drifts
documented above were found by manual archaeology; a stamp would have surfaced
them at generation time.

## Known open item

The paper's **F1 = 0.0059 with 90.5% FPR** figure (abstract and three further
places) has no committed generating artifact. The closest is
`kimi_recall_diagnostics.composite_tau_sweep_supplementary` at τ=0.40
(tp 29, fp 9816, tn 320 → F1 0.0059 but FPR 96.8%); the τ=0.60 committed value
is F1 0.0036. Provenance is being re-derived; do not cite these two numbers as
reproduced until that is resolved.
