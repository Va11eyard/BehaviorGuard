#!/usr/bin/env python3
"""
F1-max threshold sweep on corrected PersonaChat with leakage-safe holdout halves.

Protocol
--------
Step 1: Keep 80/20 per-user chronological profile split (s_ling audit). Subdivide
        only the 20% held-out tail into validation-half vs final-test-half (50/50
        stable hash per user). Users with test positives and organic-only users
        share the same hash rule.
Step 2: Build each user's profile ONCE from train_msgs (first 80%); reuse for both halves.
Step 3: Score all held-out messages via evaluate().
Step 4: F1-max tau sweep on validation-half only.
Step 5: Report metrics on final-test-half at selected tau + bootstrap 95% CI on F1.

Usage:
    set HF_HUB_OFFLINE=1; set TRANSFORMERS_OFFLINE=1
    python scripts/threshold_sweep_protocol.py
    python scripts/threshold_sweep_protocol.py --audit-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import f1_score, roc_auc_score

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import evaluation as ev  # noqa: E402
from behaviorguard import BehaviorGuardEvaluatorML  # noqa: E402
from behaviorguard.models import EvaluationInput, SystemConfig  # noqa: E402
from scripts.personachat_holdout_split import (  # noqa: E402
    DEFAULT_DATASET,
    load_dataset,
    partition_holdout,
    print_split_audit,
)

LAMBDA_DECAY = 0.50
TAU_LOW, TAU_HIGH, TAU_STEP = 0.01, 0.99, 0.01
N_BOOTSTRAP = 5000
SEED = 42

# Post-s_ling-fix production diagnostic path (linguistic weight excluded)
EVAL_CONFIG = SystemConfig(
    sensitivity_level="medium",
    deployment_context="enterprise",
    overrides_enabled=False,
    enable_linguistic_scoring=False,
    linguistic_component_enabled=False,
    enable_semantic_scoring=True,
    enable_temporal_scoring=True,
)


def _prev_in_session(msgs: list[dict], i: int) -> dict | None:
    if i == 0:
        return None
    p = msgs[i - 1]
    if p.get("session_id", "session_0") == msgs[i].get("session_id", "session_0"):
        return p
    return None


def build_profiles_once(
    users_lookup: dict[str, dict],
    partitions,
    lambda_decay: float,
) -> dict[str, Any]:
    """Step 2: one profile per user from train_msgs only."""
    builder = ev._build_profile_with_pm(lambda_decay)
    profiles: dict[str, Any] = {}
    skipped: list[str] = []
    for part in partitions:
        profile = builder(users_lookup[part.user_id], part.train_msgs)
        if profile is None:
            skipped.append(part.user_id)
            continue
        profiles[part.user_id] = profile
    return {"profiles": profiles, "skipped_users": skipped}


def score_partition(
    partitions,
    profiles: dict[str, Any],
    *,
    half: str,
    evaluator: BehaviorGuardEvaluatorML,
    config: SystemConfig,
) -> list[dict]:
    """Score validation or final-test messages using pre-built profiles."""
    rows: list[dict] = []
    for part in partitions:
        profile = profiles.get(part.user_id)
        if profile is None:
            continue
        msgs = part.validation_msgs if half == "validation" else part.final_test_msgs
        for i, msg in enumerate(msgs):
            prev = _prev_in_session(msgs, i)
            cur = ev.message_to_current_message(msg, prev, user_profile=profile)
            result = evaluator.evaluate(
                EvaluationInput(
                    user_profile=profile,
                    current_message=cur,
                    system_config=config,
                )
            )
            rows.append(
                {
                    "user_id": part.user_id,
                    "y_true": bool(msg.get("should_flag", False)),
                    "score": float(result.anomaly_score),
                }
            )
    return rows


def f1_max_threshold(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    low: float = TAU_LOW,
    high: float = TAU_HIGH,
    step: float = TAU_STEP,
) -> tuple[float, dict]:
    best_t, best_m, best_f1 = low, {}, -1.0
    for t in np.arange(low, high + step / 2, step):
        t = round(float(t), 2)
        y_pred = y_scores > t
        m = ev.compute_metrics(y_true.tolist(), y_pred.tolist(), y_scores.tolist())
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_t = t
            best_m = m
    return best_t, best_m


def bootstrap_f1_ci(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = SEED,
) -> dict:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    point = float(f1_score(labels, (scores >= threshold).astype(int), zero_division=0.0))
    rng = np.random.RandomState(seed)
    n = len(labels)
    boot: list[float] = []
    skipped = 0
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        lbl = labels[idx]
        if len(np.unique(lbl)) < 2:
            skipped += 1
            continue
        pr = (scores[idx] >= threshold).astype(int)
        boot.append(float(f1_score(lbl, pr, zero_division=0.0)))
    arr = np.array(boot)
    return {
        "point_estimate": point,
        "ci_low": float(np.percentile(arr, 2.5)),
        "ci_high": float(np.percentile(arr, 97.5)),
        "bootstrap_requested": n_bootstrap,
        "bootstrap_skipped_single_class": skipped,
        "bootstrap_effective_n": int(len(arr)),
        "bootstrap_seed": seed,
    }


def clopper_pearson(k: int, n: int) -> tuple[float, float, float]:
    if n == 0:
        return 0.0, 0.0, 0.0
    low, high = binomtest(k, n).proportion_ci(confidence_level=0.95, method="exact")
    return k / n, float(low), float(high)


def metrics_with_cis(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    preds = (scores >= threshold).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    m = ev.compute_metrics(labels.tolist(), preds.tolist(), scores.tolist())
    rec_p, rec_lo, rec_hi = clopper_pearson(tp, tp + fn)
    prec_p, prec_lo, prec_hi = clopper_pearson(tp, tp + fp)
    f1_boot = bootstrap_f1_ci(labels, scores, threshold)
    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": round(m["precision"], 4),
        "recall": round(m["recall"], 4),
        "f1": round(m["f1"], 4),
        "fpr": round(m["fpr"], 4),
        "auc": round(float(roc_auc_score(labels, scores)), 4),
        "f1_bootstrap_ci": f1_boot,
        "recall_clopper_pearson": {"point": rec_p, "ci_low": rec_lo, "ci_high": rec_hi},
        "precision_clopper_pearson": {"point": prec_p, "ci_low": prec_lo, "ci_high": prec_hi},
    }


def run_protocol(
    test_data: dict,
    *,
    lambda_decay: float = LAMBDA_DECAY,
    config: SystemConfig | None = None,
    score: bool = True,
) -> dict[str, Any]:
    split = partition_holdout(test_data)
    users_lookup = {u["user_id"]: u for u in test_data["users"]}

    out: dict[str, Any] = {
        "protocol": {
            "step1": "80/20 profile split unchanged; subdivide held-out 20% only",
            "step2": "profiles built once from train_msgs per user",
            "step3": "evaluate() on validation + final-test holdout messages",
            "step4": f"F1-max tau sweep on validation-half [{TAU_LOW},{TAU_HIGH}] step {TAU_STEP}",
            "step5": "final-test-half metrics + bootstrap F1 CI at validation-selected tau",
            "lambda_decay": lambda_decay,
            "eval_config": {
                "enable_linguistic_scoring": False,
                "linguistic_component_enabled": False,
                "overrides_enabled": False,
            },
        },
        "split_audit": split.audit,
    }

    if not score:
        return out

    profile_info = build_profiles_once(users_lookup, split.users, lambda_decay)
    profiles = profile_info["profiles"]
    out["profile_build"] = {
        "n_profiles_built": len(profiles),
        "n_users_skipped_cold_start": len(profile_info["skipped_users"]),
        "note": "Profiles built from first 80% only; same profile reused for both holdout halves",
    }

    evaluator = BehaviorGuardEvaluatorML()
    cfg = config or EVAL_CONFIG
    val_rows = score_partition(split.users, profiles, half="validation", evaluator=evaluator, config=cfg)
    final_rows = score_partition(split.users, profiles, half="final_test", evaluator=evaluator, config=cfg)

    val_y = np.array([r["y_true"] for r in val_rows], dtype=bool)
    val_s = np.array([r["score"] for r in val_rows], dtype=float)
    final_y = np.array([r["y_true"] for r in final_rows], dtype=bool)
    final_s = np.array([r["score"] for r in final_rows], dtype=float)

    selected_tau, val_metrics_at_best = f1_max_threshold(val_y, val_s)
    final_report = metrics_with_cis(final_y, final_s, selected_tau)

    out["validation_half"] = {
        "n_messages": len(val_rows),
        "n_positives": int(val_y.sum()),
        "selected_tau_f1_max": round(float(selected_tau), 4),
        "metrics_at_selected_tau_on_validation": {
            k: round(val_metrics_at_best[k], 4)
            for k in ("precision", "recall", "f1", "fpr", "roc_auc")
        },
    }
    out["final_test_half"] = {
        "n_messages": len(final_rows),
        "n_positives": int(final_y.sum()),
        **final_report,
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="PersonaChat threshold sweep protocol")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--audit-only", action="store_true", help="Print split audit only (Step 1)")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "threshold_sweep_protocol.json")
    args = parser.parse_args()

    test_data = load_dataset(args.dataset)
    if args.audit_only:
        split = partition_holdout(test_data)
        print_split_audit(split.audit)
        return

    print("Running threshold sweep protocol (corrected holdout split)...", flush=True)
    report = run_protocol(test_data)
    print_split_audit(report["split_audit"])
    print(
        f"\nValidation half: {report['validation_half']['n_positives']} positives, "
        f"tau*={report['validation_half']['selected_tau_f1_max']}",
        flush=True,
    )
    ft = report["final_test_half"]
    print(
        f"Final-test half: {ft['n_positives']} positives @ tau={ft['threshold']}: "
        f"F1={ft['f1']} [{ft['f1_bootstrap_ci']['ci_low']:.3f}, {ft['f1_bootstrap_ci']['ci_high']:.3f}]",
        flush=True,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
