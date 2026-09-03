# Reproducing Paper Results

This document gives exact steps to reproduce the evaluation results reported in
`paper/behaviorguard.tex`.

---

## Environment

```bash
# Python 3.10+
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

# Exact versions used for the revision experiments
pip install -r requirements-lock.txt
pip install -e ".[dev]"
# or with poetry
poetry install
```

**Hardware used for paper results:** Single CPU (no GPU required for the sidecar).
Embedding inference runs on CPU; evaluation completes in under 10 minutes for the
legacy 20-user sample; full-holdout and sequential ATO studies take longer.

**Random seed:** 42 (set in `evaluation.py`).

**Embedding model pin:** `sentence-transformers/all-MiniLM-L6-v2` at Hub revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41` (see
`src/turnshift/embedding_config.py`).

---

## Datasets

The evaluation uses three conversational benchmarks. Due to size, processed
versions are not committed (except the force-tracked ATO episode files).

Place preprocessed JSON under `datasets/`:

```
datasets/
├── personachat_processed.json                  # original injection (confounded; rebuild input only)
├── personachat_processed_corrected.json        # de-confounded re-injection (DEFAULT for evaluation.py)
├── blended_skill_talk_processed.json
├── blended_skill_talk_processed_corrected.json # default
├── anthropic_hh_processed.json
├── anthropic_hh_processed_corrected.json       # default
├── personachat_ato_episodes.json               # sequential ATO study (tracked)
└── blended_skill_talk_ato_episodes.json        # sequential ATO study (tracked)
```

`evaluation.py` (and `evaluate.py` / `reproduce.py`) load `*_processed_corrected.json`.
The original `*_processed.json` has tail-clustered injections and metadata leakage
(`_ato` session ids, `operation_risk`); it is used only as the input to
`tools/rebuild_injected_datasets.py`. If a corrected file is missing, `evaluation.py`
falls back to the uncorrected one and prints a warning to stderr; those numbers are
not comparable to the paper.

**Important:** anomalies are **not** injected at runtime by `evaluation.py`.
They are baked into the `*_processed*.json` files by
`tools/rebuild_injected_datasets.py` (templates in `tools/anomaly_templates.py`).
External-attack positives are produced by `tools/rebuild_external_injection.py`
into `datasets/*_processed_external.json`.

Raw Hugging Face download helpers live under `tools/`; expected schema is
documented in `datasets/README.md`.

---

## Running the Evaluation

```bash
# Per-message pipeline (archived protocol): corrected data, 80/20 tail of the
# 750-user test split. PersonaChat: 1,521 messages, 6 positives (0.39%).
python evaluation.py

# 20-user uniform random sample (compute shortcut; natural prevalence)
python evaluate.py --max-users 20

# Paper's realistic-prevalence holdout (Sec. "Re-Evaluation at Realistic
# Prevalence"): corrected PersonaChat, 80/20 tail of ALL 5,000 users =
# 10,165 messages / 29 positives / 0.29%. Scores it at tau=0.60:
python scripts/sling_exclusion_holdout_eval.py
#   -> results/methodology-diagnostics/sling_exclusion_holdout_eval.json
# Bootstrap / Clopper-Pearson CIs on the same holdout:
python production_sling_audit_snippet.py            # results/sling_audit_component_scores.npz
python scripts/mahalanobis_bootstrap_comparison.py  # results/mahalanobis_comparison_scores.npz
python scripts/compute_sling_fix_confidence_intervals.py
#   -> results/methodology-diagnostics/sling_fix_confidence_intervals.json
```

The committed diagnostic JSONs are snapshots of the scorer at the time the s_ling
audit was run (with-s_ling F1 0.0036 vs without 0.32 [0.14, 0.47]). Re-running
them with the current scorer reproduces the holdout (10,165 / 29) but not those
metric values: s_ling has since been removed, so both arms coincide, and the
without-s_ling point estimate moves to F1 0.348 / AUC 0.953. This is the same
per-message scorer drift noted for the `permsg_bg` baseline row.

```bash

# Sequential ATO study (primary headline finding)
python scripts/sequential_ato_study.py --dataset personachat
python scripts/sequential_ato_study.py --dataset bst
python scripts/sequential_ato_null_control.py --dataset personachat
python scripts/sequential_ato_null_control.py --dataset bst
```

Output: `full_evaluation_results.json` (legacy pipeline) and
`results/primary/sequential_ato_study*.json` (primary findings).

Note the two holdout definitions: `evaluate_method` scores only the 750 users in
`splits.test`, while the paper's realistic-prevalence numbers use every user's
20% tail (`scripts/personachat_holdout_split.build_holdout_records`). Both are
computed on the corrected data; `evaluate.py` does **not** reproduce the
10,165 / 29 figures.

> The Table III / 20-user F1 figures in earlier drafts are retained only as a
> diagnosed base-rate artifact. Canonical claims use full-holdout and sequential
> detection metrics — see `results/archived-per-message-study/README.md` and
> `results/primary/sequential_ato_study*.json`.

Mean scoring latency (embedding + composite, excluding profile I/O):
13–17 ms per request (CPU, pinned MiniLM). End-to-end latency including profile
load/save is reported separately in the deployment audit.

---

## Running Tests

```bash
pytest tests/ -v --no-cov
```

Expected: **88 tests passing**.

---

## Recent changes (schema & scoring)

- `is_typical_active_time` is now derived from the user profile's
  `temporal_profile.most_active_hours_utc` with a fallback to 09:00–21:00
  only when no hours are learned (cold-start users).
- `normal_override` now averages only over enabled components
  (`semantic`, `linguistic`, `temporal`). When all components are
  disabled, the override is not applied.
- Each `metrics` dict in `full_evaluation_results.json` now includes a
  `per_class_metrics` block with precision, recall, F1, and support
  by both `anomaly_type` and `attack_phase`.
- Embedding model loads are revision-pinned via `embedding_config.py`.
- `evaluate_method(..., max_users=None)` evaluates the full test split;
  `--max-users N` takes a seeded uniform random subsample of N users at the
  split's natural prevalence (compute cap only).

---

## Deviations from Paper / Historical Protocol

- Earlier drafts reported results on 20 users per dataset (`max_users=20`).
  That was a compute shortcut that also positive-enriched the sample by
  filling the cap with anomaly-containing users first. The paper protocol now
  defaults to the full holdout, and the anomaly-first filling has been removed
  from `evaluate_method`: `--max-users 20` now yields a uniform random sample
  and no longer reproduces the diagnosed artifact. The artifact itself is
  preserved in `results/archived-per-message-study/`.
- Latency figures of 13–17 ms are CPU scoring only (no profile I/O). GPU
  inference of the embedding model would reduce latency further but is not
  required.
