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
├── personachat_processed.json
├── personachat_processed_corrected.json   # de-confounded re-injection
├── blended_skill_talk_processed.json
├── blended_skill_talk_processed_corrected.json
├── anthropic_hh_processed.json
├── anthropic_hh_processed_corrected.json
├── personachat_ato_episodes.json          # sequential ATO study (tracked)
└── blended_skill_talk_ato_episodes.json   # sequential ATO study (tracked)
```

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
# Full evaluation (default: all test users / full holdout)
python evaluation.py

# Historical 20-user sample (compute shortcut; superseded by full holdout)
python evaluate.py --max-users 20

# Sequential ATO study (primary headline finding)
python scripts/sequential_ato_study.py --dataset personachat
python scripts/sequential_ato_study.py --dataset bst
python scripts/sequential_ato_null_control.py --dataset personachat
python scripts/sequential_ato_null_control.py --dataset bst
```

Output: `full_evaluation_results.json` (legacy pipeline) and
`results/primary/sequential_ato_study*.json` (primary findings).

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
  `--max-users 20` restores the historical sample for comparison only.

---

## Deviations from Paper / Historical Protocol

- Earlier drafts reported results on 20 users per dataset (`max_users=20`).
  That was a compute shortcut that also positive-enriched the sample. The
  paper protocol now defaults to the full holdout; use `--max-users 20` only
  to reproduce the diagnosed artifact.
- Latency figures of 13–17 ms are CPU scoring only (no profile I/O). GPU
  inference of the embedding model would reduce latency further but is not
  required.
