#!/usr/bin/env python3
"""
Full held-out evaluate() comparison: enable_linguistic_scoring on vs off.

Same corrected PersonaChat 80/20 per-user test split as the s_ling audit.
Binary classification at tau=0.60 (evaluation.py canonical threshold).
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import evaluation as ev  # noqa: E402
from behaviorguard import BehaviorGuardEvaluatorML  # noqa: E402
from behaviorguard.models import EvaluationInput, SystemConfig  # noqa: E402
from production_sling_audit_snippet import PRODUCTION_CLASSIFICATION_THRESHOLD  # noqa: E402

DATASET = ROOT / "datasets" / "personachat_processed_corrected.json"
LAMBDA_DECAY = 0.50
THRESHOLD = PRODUCTION_CLASSIFICATION_THRESHOLD

AUDIT_PREDICTIONS = {
    "with_sling": {"auc": 0.6551315896905533, "f1": 0.0036068530207394047,
                   "precision": 0.001851851851851852, "recall": 0.06896551724137931},
    "without_sling": {"auc": 0.9171951119941213, "f1": 0.3137254901960784,
                      "precision": 0.36363636363636365, "recall": 0.27586206896551724},
}


def _messages_by_user(test_data: dict) -> dict[str, list]:
    by: dict[str, list] = defaultdict(list)
    for m in test_data["messages"]:
        by[m["user_id"]].append(m)
    for uid in by:
        by[uid].sort(key=lambda x: x["timestamp"])
    return by


def _prev_in_session(test_msgs: list, i: int):
    if i == 0:
        return None
    p = test_msgs[i - 1]
    if p.get("session_id", "session_0") == test_msgs[i].get("session_id", "session_0"):
        return p
    return None


def run_holdout_eval(enable_linguistic: bool) -> dict:
    test_data = json.loads(DATASET.read_text(encoding="utf-8"))
    evaluator = BehaviorGuardEvaluatorML()
    config = SystemConfig(
        sensitivity_level="medium",
        deployment_context="enterprise",
        enable_semantic_scoring=True,
        enable_linguistic_scoring=enable_linguistic,
        enable_temporal_scoring=True,
        overrides_enabled=False,
    )
    builder = ev._build_profile_with_pm(LAMBDA_DECAY)
    by_user = _messages_by_user(test_data)
    users = {u["user_id"]: u for u in test_data["users"]}

    y_true: list[bool] = []
    y_scores: list[float] = []

    for uid, msgs in sorted(by_user.items()):
        split_idx = int(len(msgs) * 0.8)
        profile = builder(users[uid], msgs[:split_idx])
        if profile is None:
            continue
        test_msgs = msgs[split_idx:]
        for i, msg in enumerate(test_msgs):
            prev = _prev_in_session(test_msgs, i)
            cur = ev.message_to_current_message(msg, prev, user_profile=profile)
            result = evaluator.evaluate(
                EvaluationInput(
                    user_profile=profile,
                    current_message=cur,
                    system_config=config,
                )
            )
            y_true.append(bool(msg.get("should_flag", False)))
            y_scores.append(float(result.anomaly_score))

    y_pred = [s > THRESHOLD for s in y_scores]
    metrics = ev.compute_metrics(y_true, y_pred, y_scores)
    return {
        "enable_linguistic_scoring": enable_linguistic,
        "n_messages": len(y_true),
        "n_anomalous": int(sum(y_true)),
        "threshold": THRESHOLD,
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "auc": metrics["roc_auc"],
        "fpr": metrics["fpr"],
    }


def _compare(label: str, actual: dict, predicted: dict) -> dict:
    diffs = {}
    for key in ("auc", "f1", "precision", "recall"):
        diffs[key] = actual[key] - predicted[key]
    return {
        "label": label,
        "actual": actual,
        "audit_prediction": predicted,
        "delta_actual_minus_audit": diffs,
    }


def main() -> None:
    print(f"Corrected PersonaChat held-out eval (lambda={LAMBDA_DECAY}, tau={THRESHOLD})")
    print("=" * 72)

    with_ling = run_holdout_eval(enable_linguistic=True)
    without_ling = run_holdout_eval(enable_linguistic=False)

    cmp_with = _compare("with_sling", with_ling, AUDIT_PREDICTIONS["with_sling"])
    cmp_without = _compare("without_sling", without_ling, AUDIT_PREDICTIONS["without_sling"])

    report = {
        "dataset": str(DATASET),
        "lambda_decay": LAMBDA_DECAY,
        "threshold": THRESHOLD,
        "with_linguistic_scoring": with_ling,
        "without_linguistic_scoring": without_ling,
        "audit_comparison": [cmp_with, cmp_without],
        "next_priority": (
            "Even with s_ling excluded, recall remains low (~28% at tau=0.60). "
            "Mahalanobis semantic scoring is the next priority — not resolved by s_ling removal."
        ),
    }

    for cmp in report["audit_comparison"]:
        print(f"\n{cmp['label']} (enable_linguistic_scoring="
              f"{cmp['actual']['enable_linguistic_scoring']}):")
        for key in ("auc", "f1", "precision", "recall"):
            act = cmp["actual"][key]
            pred = cmp["audit_prediction"][key]
            delta = cmp["delta_actual_minus_audit"][key]
            print(f"  {key:9s}  actual={act:.6f}  audit={pred:.6f}  delta={delta:+.6f}")

    print(f"\n{report['next_priority']}")

    out = ROOT / "results" / "sling_exclusion_holdout_eval.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
