# diagnostic_harness.py integration

## Purpose

Run s_ling sub-feature saturation audit and λ-sweep (before/after Mahalanobis + s_ling fix)
using **real** `SemanticAnalyzerML`, `LinguisticAnalyzerML`, and `TemporalAnalyzerML` calls.

## Prerequisites

```powershell
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:PYTHONPATH = "E:\PyCharm 2025.2.1.1\BehaviorGuard\src"
```

## Step 1 — Interface discovery

```powershell
python scripts/inspect_interfaces.py > results/interfaces.txt
```

Review `results/interfaces.txt` for `[SKIP]` lines before wiring.

## Step 4 — Wiring verification (required before trusting Part 5)

```powershell
python scripts/verify_harness_wiring.py
```

Must report `PASS` with s_ling values matching to 4 decimal places.

## Composite / evaluate() parity (required before Part 5)

```powershell
python scripts/verify_composite_wiring.py
```

Confirms `run_lambda_sweep()` uses `TurnShiftEvaluatorML.evaluate()` /
`CompositeScorer` (not a hand-written α·s_sem + β·s_ling + γ·s_temp).

## Step 5 — evaluation.py integration

Set environment flag to run diagnostic instead of full eval:

```powershell
$env:BG_DIAGNOSTIC_HARNESS = "1"
python evaluation.py
```

Or import directly:

```python
from scripts.diagnostic_harness import run_full_diagnostic
report = run_full_diagnostic()
```

## Split protocol

- `split_conversations_8020()` splits each user's timeline at 80%.
- **Train (first 80%)**: organic messages only → `ProfileManager.build_from_history`.
- **Test (last 20%)**: scored with real analyzers; labels from `should_flag`.
- No test messages leak into profile statistics.

## Output

Results written to `results/diagnostic_harness_output.json` when run standalone.
