# Evaluation protocol

Sources: paper §Evaluation Methodology (base-rate fallacy), §Sequential
Detection / Study Design, `src/turnshift/detectors/cusum.py`
(`evaluate_detector`, `placebo_continuation`), and
`scripts/sequential_ato_study.py` / `scripts/sequential_ato_null_control.py`.

## Two protocols, one of which is archived

### Archived: per-message scoring on positive-enriched samples

Earlier drafts evaluated a composite per-message scorer on a 20-user sample
with ~48% positive prevalence and reported F1 of 0.693 / 0.794 / 0.642 on
PersonaChat / Blended Skill Talk / Anthropic HH.

Re-evaluated at the full holdout's realistic prevalence (0.29% on PersonaChat;
10,165 messages, 29 positives), the identical configuration collapses to
**F1 = 0.0059** with a **90.5% false-positive rate**.

Those numbers are retained only as a diagnosed evaluation-methodology artifact
in `results/archived-per-message-study/`. They are **not** primary performance
claims.

`python evaluation.py` / `python reproduce.py` still run this per-message
pipeline. They require local `datasets/*_processed.json` files (gitignored;
see `datasets/README.md`). There is no fast CI slice of `reproduce.py`.

### Primary: sequential CUSUM over embedding residuals

Question: after a mid-conversation author substitution, how many messages does
it take to detect, at a fixed false-alarm budget?

Command:

```bash
python scripts/sequential_ato_study.py --dataset personachat
python scripts/sequential_ato_study.py --dataset bst
python scripts/sequential_ato_null_control.py --dataset personachat
python scripts/sequential_ato_null_control.py --dataset bst
```

Outputs: `results/primary/sequential_ato_study.json` and
`results/primary/sequential_ato_study_bst.json`.

`--recompute` rebuilds embedding trajectories (needs the sentence-transformers
model). Without it, evaluation uses committed `results/primary/sequential_ato_scores*.npz`.

**Warning:** re-running the study script overwrites the JSON **without** the
`null_control_cusum_embed` key. The placebo block is written only by
`sequential_ato_null_control.py`. Use `--out` for smoke tests.

## Split and episode construction

- Per user, history is split; the profile (EMA centroid, residual mean/sd) is
  fit on **train** messages only.
- Test streams are either fully benign or contain a mid-stream **donor
  substitution** of length 5–10 messages (`tools/build_ato_episode_dataset.py`).
- Default donor selection is random. A mimicry variant uses nearest-centroid
  donors (`results/primary/sequential_ato_study_mimicry.json`).

## Alarm-time framework (`evaluate_detector`)

All detectors share one framework:

- Statistic trajectory \(S_i\) per stream (CUSUM or per-message residual).
- **False alarm:** alarm on a fully benign stream, or on the pre-episode
  prefix of an episode stream. Rate is per 1,000 benign messages.
- Operating threshold \(h\) is a quantile of **benign-stream maxima** chosen
  so expected FAs match a target budget (0.5, 1, 2, 5, 10 per 1,000).
- **Detection:** first crossing of \(h\) **inside** the episode window.
  Streams that already alarmed before the episode starts are counted as
  pre-episode alarms and **excluded** from detection rate (they are not
  credited as detections).
- **Delay:** 1-based index of the first alarm within the episode window.
- **Episode AUC:** ROC-AUC of whole-stream max statistic vs episode label.
  Bootstrap 95% CI: \(n=2000\), seed 42.

Canonical CUSUM: \(S_i = \max(0, S_{i-1} + r_i - \kappa)\) with \(\kappa=0.5\)
on standardized cosine-distance residuals to the EMA centroid
(\(\lambda=0.5\) in the sequential study).

## Placebo (length) control

Episode streams are longer than benign streams, so some detections could be
length artifacts. Control (`placebo_continuation`):

- Continue CUSUM from the **true pre-episode state**.
- Feed \(k\) residuals **bootstrap-sampled from that user's own standardized
  train residuals** (same author, same length) instead of the donor.
- \(n_{\mathrm{sim}}=500\), seed 42.
- Crossing rate at the FA = 1/1,000 threshold is the length-induced null
  detection rate (0.7% PersonaChat / 0.1% BST in the committed JSON).

## Seeds and embedding pin

- Seed 42 (`evaluation.py`, sequential study, placebo).
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2` at Hub revision
  `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`
  (`src/turnshift/embedding_config.py`).
