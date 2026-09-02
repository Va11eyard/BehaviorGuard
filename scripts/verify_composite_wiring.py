#!/usr/bin/env python3
"""Verify harness composite matches TurnShiftEvaluatorML.evaluate()."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import evaluation as ev  # noqa: E402
from turnshift import TurnShiftEvaluatorML  # noqa: E402
from turnshift.models import EvaluationInput  # noqa: E402
from scripts.diagnostic_harness import (  # noqa: E402
    DIAG_CONFIG,
    _build_profile,
    _prev_in_session,
    _production_anomaly_score,
    _production_component_and_composite,
    split_conversations_8020,
)

DATASET = ROOT / "datasets/personachat_processed_corrected.json"
TARGET_USER = "pc_user_00001"
LAMBDA_DECAY = 0.5
TOLERANCE = 1e-4


def main() -> int:
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    users = {u["user_id"]: u for u in data["users"]}

    sp = next(
        s
        for s in split_conversations_8020(data, [TARGET_USER])
        if s.user_id == TARGET_USER
    )
    profile = _build_profile(users[sp.user_id], sp.train_msgs, LAMBDA_DECAY)
    assert profile is not None

    test_idx = 0
    msg = sp.test_msgs[test_idx]
    prev = _prev_in_session(sp.test_msgs, test_idx)
    cur = ev.message_to_current_message(msg, prev, user_profile=profile)
    config = DIAG_CONFIG.model_copy(deep=True)

    evaluator = TurnShiftEvaluatorML()
    eval_result = evaluator.evaluate(
        EvaluationInput(user_profile=profile, current_message=cur, system_config=config)
    )

    s_sem, s_ling, s_temp, composite_raw = _production_component_and_composite(
        evaluator, cur, profile, config
    )
    composite_harness = _production_anomaly_score(evaluator, cur, profile, config)

    print(f"message_user={TARGET_USER}  lambda(profile decay)={LAMBDA_DECAY}")
    print()
    print("--- Q1: EvaluationResult fields ---")
    print("S_raw / S_ema: NOT exposed on EvaluationResult.")
    print("Available: anomaly_score, component_scores, risk_level, metadata.")
    print(f"metadata.detection_mechanism = {eval_result.metadata.get('detection_mechanism')}")
    print()
    print("--- Component scores (full precision vs evaluate rounded) ---")
    print(f"evaluate():  sem={eval_result.component_scores.semantic:.6f} "
          f"ling={eval_result.component_scores.linguistic:.6f} "
          f"temp={eval_result.component_scores.temporal:.6f}")
    print(f"production:  sem={s_sem:.6f} ling={s_ling:.6f} temp={s_temp:.6f}")
    print()
    print("--- Q2: Composite ---")
    print(f"evaluate().anomaly_score              = {eval_result.anomaly_score:.6f}")
    print(f"harness CompositeScorer (full prec.)  = {composite_raw:.6f}")
    print(f"harness evaluate() path (3dp rounded) = {composite_harness:.6f}")
    print(f"diff raw vs evaluate (pre-round)        = {abs(composite_raw - eval_result.anomaly_score):.6f}")
    print(f"diff harness evaluate vs evaluate       = {abs(composite_harness - eval_result.anomaly_score):.6f}")
    print()
    print("--- EMA ---")
    print("λ in run_lambda_sweep = ProfileManager profile-decay, not composite S_ema.")
    print("Production composite = CompositeScorer weighted sum (medium: 0.4/0.35/0.25),")
    print("  no post-composite EMA; overrides disabled in DIAG_CONFIG.")

    ok = (
        abs(composite_harness - eval_result.anomaly_score) <= TOLERANCE
        and abs(composite_raw - eval_result.anomaly_score) <= 0.0005
    )
    if ok:
        print(f"\nPASS: harness composite matches evaluate()")
        return 0
    print("\nFAIL: composite diverges from evaluate() — do not run Part 5")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
