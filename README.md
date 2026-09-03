# TurnShift

Sequential behavioral-shift detection for conversational AI: CUSUM over
per-user embedding residuals.

**Formerly published as BehaviorGuard.**

[![CI](https://github.com/Va11eyard/BehaviorGuard/actions/workflows/ci.yml/badge.svg)](https://github.com/Va11eyard/BehaviorGuard/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Citation](https://img.shields.io/badge/citation-CITATION.cff-blue)](CITATION.cff)

TurnShift maintains a per-user embedding centroid (EMA) and a CUSUM statistic
on cosine-distance residuals. On **proxy author-substitution experiments**
(mid-conversation donor swap; not validated real-world account-takeover
incidents), CUSUM detects the shift with episode AUC **0.974** on PersonaChat
and **0.900** on Blended Skill Talk. At a false-alarm budget of 1 per 1,000
benign messages, detection is **63.0%** / **35.0%** with median delay **4** /
**5** messages. A length-matched placebo that continues CUSUM from the same
pre-episode state but feeds bootstrap-resampled *same-author* residuals
detects at **0.7%** / **0.1%**, which is the evidence that the signal is
authorship change rather than stream length. Per-message detectors, including
the composite scorer from an earlier protocol, detect at most **1.3%** of the
same episodes at the same budget.

## Quickstart

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -e .            # rule-based evaluator, CUSUM detector, headline table from cached scores (no torch)
pip install -e ".[ml]"      # + sentence-transformers/torch: TurnShiftEvaluatorML, profile encoding, --recompute
pip install -e ".[dev]"     # [ml] + pytest, hypothesis, ruff, matplotlib
pip install -e ".[eval]"    # + Hugging Face `datasets` for tools/ corpus builders

# Headline table from committed score caches (no embedding recompute)
python scripts/sequential_ato_study.py --dataset personachat
python scripts/sequential_ato_study.py --dataset bst
```

Placebo control (needs the pinned MiniLM model and episode JSON):

```bash
python scripts/sequential_ato_null_control.py --dataset personachat
python scripts/sequential_ato_null_control.py --dataset bst
```

or in one pass with `python scripts/sequential_ato_study.py --dataset personachat --with-null-control`.
Use `--out path.json` so a local rerun does not overwrite committed
`results/primary/*.json`. Without `--out`, the study script will not silently
drop an existing `null_control_cusum_embed` block (see `docs/evaluation-protocol.md`).

## Results (primary)

Numbers below are read from `results/primary/sequential_ato_study.json` and
`results/primary/sequential_ato_study_bst.json` (`cusum_embed`, FA = 1/1,000).
AUC 0.9736 is the value stored in JSON; the paper rounds it to 0.974.

| Dataset | Episode AUC | Detection @ FA=1/1000 | Median delay | Placebo @ FA=1/1000 |
|---|---|---|---|---|
| PersonaChat | 0.974 | 63.0% | 4 | 0.7% |
| Blended Skill Talk | 0.900 | 35.0% | 5 | 0.1% |

Per-message `permsg_bg` / `permsg_combined` at the same budget: 0.0% / 0.3%
(PersonaChat), 0.3% / 1.3% (BST).

## What this does NOT show

These are the limitations already stated in the paper. They are not caveats
to be read last.

- **Cold start.** 65% of PersonaChat test users have fewer than 10 prior
  training messages; calibrated centroids are poorly anchored for that
  majority.
- **Mimicry.** Nearest-centroid donor selection drops PersonaChat detection
  from 63% to 35% (AUC 0.870).
- **Online / poisoning.** Updating the profile during the episode loses about
  8 detection points vs a frozen profile (71.7% → 63.7%). A dual-λ divergence
  CUSUM intended as a poisoning defense failed (AUC ≈ 0.55).
- **Persona-conditioning.** PersonaChat likely inflates detectability. Anthropic
  HH organic streams are a lower bound (AUC 0.795 / 20.0% detection).
- **Adaptive evasion is unevaluated.** Paraphrase, threshold probing, and
  knowledgeable profile poisoning are out of scope.
- **Text-only.** No voice, keystroke timing, or device fingerprinting.
- **Not a per-message F1 claim.** An earlier protocol on positive-enriched
  samples (~48% prevalence) reported F1 0.693 / 0.794 / 0.642. At realistic
  prevalence (0.29% on PersonaChat) the same config is F1 = 0.0059 with a
  90.5% false-positive rate. That study is archived in
  [`results/archived-per-message-study/`](results/archived-per-message-study/README.md).
  `evaluate.py` runs the per-message pipeline on the corrected data's 750-user
  test split; the 10,165-message / 29-positive realistic-prevalence holdout is
  reproduced by `scripts/sling_exclusion_holdout_eval.py` (see
  [`docs/reproducibility.md`](docs/reproducibility.md)).

## Documentation

- [Threat model](docs/threat-model.md)
- [Evaluation protocol](docs/evaluation-protocol.md)
- [Limitations](docs/limitations.md)
- [Reproducibility](docs/reproducibility.md)
- [Architecture](docs/architecture.md)
- [Paper](paper/README.md)

## Library (per-message scorer)

The composite per-message API is still in the package. It is a **baseline**
in the sequential study, not the headline result.

```python
from turnshift import TurnShiftEvaluatorML, ProfileManager, MessageRecord
from turnshift.models import EvaluationInput, CurrentMessage, SystemConfig

pm = ProfileManager(decay=0.95)
profile = pm.build_from_history(user_id="user_001", messages=messages, account_age_days=90)
evaluator = TurnShiftEvaluatorML()
result = evaluator.evaluate(
    EvaluationInput(user_profile=profile, current_message=current_message, system_config=SystemConfig())
)
```

CLI after install:

```bash
turnshift evaluate --profile profile.json --message "..." --sensitivity high
turnshift version
```

## Citation

See [`CITATION.cff`](CITATION.cff). The software name is TurnShift; the paper
title remains *BehaviorGuard: Context-Aware Anomaly Detection in Conversational
AI via Behavioral User Profiling*.

```
Mynzhassar, Dinmukhammed (2026). BehaviorGuard: Context-Aware Anomaly Detection
in Conversational AI via Behavioral User Profiling.
```

## License

MIT
