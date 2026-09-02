#!/usr/bin/env python3
"""
Bootstrap AUC comparison: cosine vs Mahalanobis semantic scoring.

Same corrected PersonaChat 80/20 test partition as s_ling audit.
Pre-committed reporting: directional unless bootstrap 95% CI on AUC diff excludes zero.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import evaluation as ev  # noqa: E402
from turnshift import TurnShiftEvaluatorML  # noqa: E402
from turnshift.models import EvaluationInput, SystemConfig  # noqa: E402
from production_sling_audit_snippet import PRODUCTION_CLASSIFICATION_THRESHOLD  # noqa: E402

DATASET = ROOT / "datasets" / "personachat_processed_corrected.json"
SCORES_CACHE = ROOT / "results" / "mahalanobis_comparison_scores.npz"
LAMBDA_DECAY = 0.50
THRESHOLD = PRODUCTION_CLASSIFICATION_THRESHOLD
N_BOOTSTRAP = 5000
SEED = 42

BASE_CONFIG = dict(
    sensitivity_level="medium",
    deployment_context="enterprise",
    enable_semantic_scoring=True,
    enable_linguistic_scoring=True,
    linguistic_component_enabled=False,
    enable_temporal_scoring=True,
    overrides_enabled=False,
)


def clopper_pearson(k: int, n: int) -> tuple[float, float, float]:
    if n == 0:
        return 0.0, 0.0, 0.0
    low, high = binomtest(k, n).proportion_ci(confidence_level=0.95, method="exact")
    return k / n, float(low), float(high)


def f1_cp(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    return clopper_pearson(2 * tp, 2 * tp + fp + fn)


def _messages_by_user(test_data: dict) -> dict[str, list]:
    by: dict[str, list] = defaultdict(list)
    for m in test_data["messages"]:
        by[m["user_id"]].append(m)
    for uid in by:
        by[uid].sort(key=lambda x: x["timestamp"])
    return by


def collect_mahalanobis_only() -> tuple[np.ndarray, np.ndarray]:
    """Collect Mahalanobis scores only (cosine must already be cached)."""
    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    test_data = json.loads(DATASET.read_text(encoding="utf-8"))
    evaluator = TurnShiftEvaluatorML()
    config = SystemConfig(**BASE_CONFIG, semantic_scoring_mode="mahalanobis")
    builder = ev._build_profile_with_pm(LAMBDA_DECAY)
    by_user: dict[str, list] = defaultdict(list)
    for m in test_data["messages"]:
        by_user[m["user_id"]].append(m)
    for uid in by_user:
        by_user[uid].sort(key=lambda x: x["timestamp"])
    users = {u["user_id"]: u for u in test_data["users"]}

    labels: list[int] = []
    scores_maha: list[float] = []
    n_done = 0

    for uid, msgs in sorted(by_user.items()):
        split_idx = int(len(msgs) * 0.8)
        profile = builder(users[uid], msgs[:split_idx])
        if profile is None:
            continue
        test_msgs = msgs[split_idx:]
        for i, msg in enumerate(test_msgs):
            prev = None
            if i > 0:
                p = test_msgs[i - 1]
                if p.get("session_id", "session_0") == msg.get("session_id", "session_0"):
                    prev = p
            cur = ev.message_to_current_message(msg, prev, user_profile=profile)
            result = evaluator.evaluate(
                EvaluationInput(user_profile=profile, current_message=cur, system_config=config)
            )
            labels.append(1 if msg.get("should_flag", False) else 0)
            scores_maha.append(float(result.anomaly_score))
            n_done += 1
            if n_done % 500 == 0:
                print(f"    ... {n_done} messages scored", flush=True)

    return np.array(labels, dtype=int), np.array(scores_maha, dtype=float)


def collect_scores_both_modes() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Single evaluator pass: collect cosine and Mahalanobis composite scores."""
    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    test_data = json.loads(DATASET.read_text(encoding="utf-8"))
    evaluator = TurnShiftEvaluatorML()
    config_cos = SystemConfig(**BASE_CONFIG, semantic_scoring_mode="cosine")
    config_maha = SystemConfig(**BASE_CONFIG, semantic_scoring_mode="mahalanobis")
    builder = ev._build_profile_with_pm(LAMBDA_DECAY)
    by_user: dict[str, list] = defaultdict(list)
    for m in test_data["messages"]:
        by_user[m["user_id"]].append(m)
    for uid in by_user:
        by_user[uid].sort(key=lambda x: x["timestamp"])
    users = {u["user_id"]: u for u in test_data["users"]}

    labels: list[int] = []
    scores_cos: list[float] = []
    scores_maha: list[float] = []

    for uid, msgs in sorted(by_user.items()):
        split_idx = int(len(msgs) * 0.8)
        profile = builder(users[uid], msgs[:split_idx])
        if profile is None:
            continue
        test_msgs = msgs[split_idx:]
        for i, msg in enumerate(test_msgs):
            prev = None
            if i > 0:
                p = test_msgs[i - 1]
                if p.get("session_id", "session_0") == msg.get("session_id", "session_0"):
                    prev = p
            cur = ev.message_to_current_message(msg, prev, user_profile=profile)
            y = 1 if msg.get("should_flag", False) else 0
            r_cos = evaluator.evaluate(
                EvaluationInput(user_profile=profile, current_message=cur, system_config=config_cos)
            )
            r_maha = evaluator.evaluate(
                EvaluationInput(user_profile=profile, current_message=cur, system_config=config_maha)
            )
            labels.append(y)
            scores_cos.append(float(r_cos.anomaly_score))
            scores_maha.append(float(r_maha.anomaly_score))

    return (
        np.array(labels, dtype=int),
        np.array(scores_cos, dtype=float),
        np.array(scores_maha, dtype=float),
    )


def collect_scores(semantic_mode: str) -> tuple[np.ndarray, np.ndarray]:
    test_data = json.loads(DATASET.read_text(encoding="utf-8"))
    evaluator = TurnShiftEvaluatorML()
    config = SystemConfig(
        **BASE_CONFIG,
        semantic_scoring_mode=semantic_mode,
    )
    builder = ev._build_profile_with_pm(LAMBDA_DECAY)
    by_user: dict[str, list] = defaultdict(list)
    for m in test_data["messages"]:
        by_user[m["user_id"]].append(m)
    for uid in by_user:
        by_user[uid].sort(key=lambda x: x["timestamp"])
    users = {u["user_id"]: u for u in test_data["users"]}

    labels: list[int] = []
    scores: list[float] = []

    for uid, msgs in sorted(by_user.items()):
        split_idx = int(len(msgs) * 0.8)
        profile = builder(users[uid], msgs[:split_idx])
        if profile is None:
            continue
        test_msgs = msgs[split_idx:]
        for i, msg in enumerate(test_msgs):
            prev = None
            if i > 0:
                p = test_msgs[i - 1]
                if p.get("session_id", "session_0") == msg.get("session_id", "session_0"):
                    prev = p
            cur = ev.message_to_current_message(msg, prev, user_profile=profile)
            result = evaluator.evaluate(
                EvaluationInput(
                    user_profile=profile,
                    current_message=cur,
                    system_config=config,
                )
            )
            labels.append(1 if msg.get("should_flag", False) else 0)
            scores.append(float(result.anomaly_score))

    return np.array(labels, dtype=int), np.array(scores, dtype=float)


def load_or_collect_scores(semantic_mode: str) -> tuple[np.ndarray, np.ndarray]:
    """Load cached scores for a mode, or collect and cache."""
    if SCORES_CACHE.exists():
        data = np.load(SCORES_CACHE)
        labels = data["labels"].astype(int)
        if semantic_mode in data.files:
            print(f"  Loaded {semantic_mode} from cache ({SCORES_CACHE.name})")
            return labels, data[semantic_mode].astype(float)
        if semantic_mode == "mahalanobis" and "cosine" in data.files:
            print("  Collecting Mahalanobis only (cosine already cached)...")
            labels, scores_maha = collect_mahalanobis_only()
            SCORES_CACHE.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                SCORES_CACHE,
                labels=labels,
                cosine=data["cosine"],
                mahalanobis=scores_maha,
            )
            print(f"  Cached mahalanobis to {SCORES_CACHE.name}")
            return labels, scores_maha

    print("  Cache miss — collecting cosine + Mahalanobis in one pass...")
    labels, scores_cos, scores_maha = collect_scores_both_modes()
    SCORES_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(SCORES_CACHE, labels=labels, cosine=scores_cos, mahalanobis=scores_maha)
    print(f"  Cached both modes to {SCORES_CACHE.name}")
    if semantic_mode == "cosine":
        return labels, scores_cos
    return labels, scores_maha


def metrics_at_threshold(labels: np.ndarray, scores: np.ndarray) -> dict:
    preds = (scores >= THRESHOLD).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    f1_p, f1_lo, f1_hi = f1_cp(tp, fp, fn)
    rec_p, rec_lo, rec_hi = clopper_pearson(tp, tp + fn)
    prec_p, prec_lo, prec_hi = clopper_pearson(tp, tp + fp)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "auc": float(roc_auc_score(labels, scores)),
        "f1": {"point": f1_p, "ci_low": f1_lo, "ci_high": f1_hi},
        "recall": {"point": rec_p, "ci_low": rec_lo, "ci_high": rec_hi},
        "precision": {"point": prec_p, "ci_low": prec_lo, "ci_high": prec_hi},
        "f1_sklearn": float(f1_score(labels, preds, zero_division=0.0)),
    }


def bootstrap_auc_diff(
    labels: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = SEED,
) -> dict:
    """Bootstrap CI for AUC(scores_b) - AUC(scores_a); same logic as s_ling audit."""
    rng = np.random.RandomState(seed)
    n = len(labels)
    diffs: list[float] = []
    skipped_single_class = 0
    skipped_error = 0
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        lbl = labels[idx]
        if len(np.unique(lbl)) < 2:
            skipped_single_class += 1
            continue
        try:
            auc_a = roc_auc_score(lbl, scores_a[idx])
            auc_b = roc_auc_score(lbl, scores_b[idx])
            diffs.append(auc_b - auc_a)
        except ValueError:
            skipped_error += 1
    arr = np.array(diffs)
    low = float(np.percentile(arr, 2.5))
    high = float(np.percentile(arr, 97.5))
    point = float(np.mean(scores_b) * 0 + roc_auc_score(labels, scores_b) - roc_auc_score(labels, scores_a))
    return {
        "auc_difference_point": point,
        "ci_low": low,
        "ci_high": high,
        "ci_width": high - low,
        "ci_contains_zero": low <= 0 <= high,
        "bootstrap_requested": n_bootstrap,
        "bootstrap_skipped_single_class": skipped_single_class,
        "bootstrap_skipped_auc_error": skipped_error,
        "bootstrap_effective_n": len(diffs),
        "interpretation": (
            "confirmed_improvement" if not (low <= 0 <= high) and point > 0
            else "directional_only" if point > 0
            else "no_improvement_or_worse"
        ),
    }


def verify_linguistic_flag() -> dict:
    """Confirm linguistic_component_enabled=False matches enable_linguistic_scoring=False."""
    from turnshift.scorers.composite import CompositeScorer
    from turnshift.models import ComponentScores, SystemConfig

    scorer = CompositeScorer()
    cs = ComponentScores(semantic=0.8, linguistic=0.9, temporal=0.4)
    # Inline minimal profile/message via composite test helpers
    sys.path.insert(0, str(ROOT / "tests"))
    from test_composite_scorer import build_current_message, build_user_profile  # noqa: E402

    profile = build_user_profile(has_sensitive_ops=True)
    msg = build_current_message()
    cfg_flag = SystemConfig(
        sensitivity_level="medium",
        deployment_context="enterprise",
        enable_linguistic_scoring=True,
        linguistic_component_enabled=False,
        overrides_enabled=False,
    )
    cfg_old = SystemConfig(
        sensitivity_level="medium",
        deployment_context="enterprise",
        enable_linguistic_scoring=False,
        linguistic_component_enabled=True,
        overrides_enabled=False,
    )
    cfg_default = SystemConfig(sensitivity_level="medium", deployment_context="enterprise")
    s_flag = scorer.compute_score(cs, cfg_flag, msg, profile).anomaly_score
    s_old = scorer.compute_score(cs, cfg_old, msg, profile).anomaly_score
    s_default = scorer.compute_score(cs, cfg_default, msg, profile).anomaly_score
    expected_excluded = 0.8 * (0.4 / 0.65) + 0.4 * (0.25 / 0.65)
    expected_default = 0.4 * 0.8 + 0.35 * 0.9 + 0.25 * 0.4
    return {
        "linguistic_component_enabled_false_score": s_flag,
        "enable_linguistic_scoring_false_score": s_old,
        "default_score": s_default,
        "expected_renormalized": expected_excluded,
        "expected_default_with_linguistic": expected_default,
        "exclusion_paths_match": abs(s_flag - s_old) < 1e-9,
        "exclusion_matches_expected": abs(s_flag - expected_excluded) < 1e-9,
        "default_preserves_linguistic_weight": abs(s_default - expected_default) < 1e-9,
        "default_linguistic_component_enabled": cfg_default.linguistic_component_enabled,
    }


def main() -> None:
    print("Verifying linguistic_component_enabled flag...")
    flag_check = verify_linguistic_flag()
    print(json.dumps(flag_check, indent=2))

    print("\nCollecting cosine scores...")
    labels, scores_cosine = load_or_collect_scores("cosine")
    print(f"  n={len(labels)}, positives={int(labels.sum())}")

    print("Collecting Mahalanobis scores...")
    labels2, scores_maha = load_or_collect_scores("mahalanobis")
    assert np.array_equal(labels, labels2)

    m_cos = metrics_at_threshold(labels, scores_cosine)
    m_maha = metrics_at_threshold(labels, scores_maha)
    boot = bootstrap_auc_diff(labels, scores_cosine, scores_maha)

    report = {
        "reporting_standard": (
            "Mahalanobis vs cosine: directional unless bootstrap 95% CI on AUC "
            "difference excludes zero. F1/recall/precision use exact Clopper-Pearson CIs."
        ),
        "config": {**BASE_CONFIG, "semantic_scoring_mode_compared": ["cosine", "mahalanobis"],
                   "lambda_decay": LAMBDA_DECAY, "threshold": THRESHOLD},
        "linguistic_component_enabled_flag_check": flag_check,
        "cosine": m_cos,
        "mahalanobis": m_maha,
        "bootstrap_auc_mahalanobis_minus_cosine": boot,
    }

    out = ROOT / "results" / "mahalanobis_bootstrap_comparison.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nCosine    AUC={m_cos['auc']:.6f}  F1={m_cos['f1']['point']:.3f} [{m_cos['f1']['ci_low']:.3f},{m_cos['f1']['ci_high']:.3f}]")
    print(f"Mahalanob AUC={m_maha['auc']:.6f}  F1={m_maha['f1']['point']:.3f} [{m_maha['f1']['ci_low']:.3f},{m_maha['f1']['ci_high']:.3f}]")
    print(f"AUC diff (Maha-Cos): {boot['auc_difference_point']:+.6f}  CI [{boot['ci_low']:+.6f}, {boot['ci_high']:+.6f}]  "
          f"-> {boot['interpretation']}")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
