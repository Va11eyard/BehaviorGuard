#!/usr/bin/env python3
"""
Diagnostic harness: s_ling sub-feature saturation audit + λ-sweep (before/after fix).

All component scores call real BehaviorGuard analyzers — no TF-IDF or manual
approximations. If wiring regresses, functions raise rather than silently fallback.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import evaluation as ev  # noqa: E402
from behaviorguard.analyzers.linguistic_ml import LinguisticAnalyzerML  # noqa: E402
from behaviorguard.analyzers.semantic_ml import SemanticAnalyzerML  # noqa: E402
from behaviorguard.analyzers.temporal_ml import TemporalAnalyzerML  # noqa: E402
from behaviorguard import BehaviorGuardEvaluatorML  # noqa: E402
from behaviorguard.models import (  # noqa: E402
    ComponentScores,
    CurrentMessage,
    EvaluationInput,
    SystemConfig,
    UserProfile,
)

DEFAULT_DATASET = ROOT / "datasets/personachat_processed_corrected.json"
SEED = 42

# Diagnostic config: Mahalanobis semantic + fixed s_ling (post-fix path)
DIAG_CONFIG = SystemConfig(
    sensitivity_level="medium",
    deployment_context="enterprise",
    overrides_enabled=False,
    semantic_scoring_mode="mahalanobis",
    mahalanobis_shrinkage=0.1,
)

# Baseline config: cosine semantic (pre-fix proxy for comparison sweep)
BASELINE_CONFIG = SystemConfig(
    sensitivity_level="medium",
    deployment_context="enterprise",
    overrides_enabled=False,
    semantic_scoring_mode="cosine",
)


@dataclass
class MessageSplit:
    """Per-user 80/20 split: train builds profile, test is scored."""

    user_id: str
    train_msgs: list[dict]
    test_msgs: list[dict]


def split_conversations_8020(
    test_data: dict,
    user_ids: list[str] | None = None,
) -> list[MessageSplit]:
    """80% train (profile) / 20% test (score) per user — no cross-split leakage."""
    by_user: dict[str, list] = defaultdict(list)
    for m in test_data["messages"]:
        by_user[m["user_id"]].append(m)
    for uid in by_user:
        by_user[uid].sort(key=lambda x: x["timestamp"])

    pool = user_ids or sorted(by_user.keys())
    splits: list[MessageSplit] = []
    for uid in pool:
        msgs = by_user[uid]
        if len(msgs) < 5:
            continue
        split_idx = int(len(msgs) * 0.8)
        train = [m for m in msgs[:split_idx] if not m.get("is_anomaly", False)]
        if len(train) < 3:
            continue
        splits.append(
            MessageSplit(user_id=uid, train_msgs=train, test_msgs=msgs[split_idx:])
        )
    return splits


def _prev_in_session(msgs: list[dict], i: int) -> dict | None:
    if i == 0:
        return None
    p = msgs[i - 1]
    if p.get("session_id", "session_0") == msgs[i].get("session_id", "session_0"):
        return p
    return None


def _build_profile(user: dict, train_msgs: list[dict], lambda_decay: float):
    builder = ev._build_profile_with_pm(lambda_decay)
    return builder(user, train_msgs)


def _compute_s_sem_real(
    semantic_analyzer: SemanticAnalyzerML,
    current_message: CurrentMessage,
    user_profile: UserProfile,
    system_config: SystemConfig,
    test_message_index: int = 0,
) -> float:
    """Score via SemanticAnalyzerML.analyze() — no fallback."""
    running_interactions = user_profile.total_interactions + test_message_index
    result = semantic_analyzer.analyze(
        current_message,
        user_profile.semantic_profile,
        system_config=system_config,
        total_interactions=running_interactions,
    )
    return float(result.score)


def _compute_s_ling_real(
    linguistic_analyzer: LinguisticAnalyzerML,
    current_message: CurrentMessage,
    user_profile: UserProfile,
) -> float:
    """Score via LinguisticAnalyzerML.analyze() — no fallback."""
    result = linguistic_analyzer.analyze(
        current_message, user_profile.linguistic_profile
    )
    return float(result.score)


def _compute_s_temp_real(
    temporal_analyzer: TemporalAnalyzerML,
    current_message: CurrentMessage,
    user_profile: UserProfile,
) -> float:
    """Score via TemporalAnalyzerML.analyze() — no fallback."""
    result = temporal_analyzer.analyze(
        current_message, user_profile.temporal_profile
    )
    return float(result.score)


def _production_component_and_composite(
    evaluator: BehaviorGuardEvaluatorML,
    current_message: CurrentMessage,
    user_profile: UserProfile,
    system_config: SystemConfig,
) -> tuple[float, float, float, float]:
    """
    Production scoring path: same analyzer + CompositeScorer calls as
    BehaviorGuardEvaluatorML.evaluate() before output rounding.

    λ (ProfileManager decay) is applied at profile build time, not here.
    total_interactions matches evaluator_ml (static profile count).
    """
    sr = evaluator.semantic_analyzer.analyze(
        current_message,
        user_profile.semantic_profile,
        system_config=system_config,
        total_interactions=user_profile.total_interactions,
    )
    lr = evaluator.linguistic_analyzer.analyze(
        current_message, user_profile.linguistic_profile
    )
    tr = evaluator.temporal_analyzer.analyze(
        current_message, user_profile.temporal_profile
    )
    cs = ComponentScores(
        semantic=sr.score if system_config.enable_semantic_scoring else 0.0,
        linguistic=lr.score if system_config.enable_linguistic_scoring else 0.0,
        temporal=tr.score if system_config.enable_temporal_scoring else 0.0,
    )
    composite = evaluator.composite_scorer.compute_score(
        cs, system_config, current_message, user_profile
    )
    return cs.semantic, cs.linguistic, cs.temporal, composite.anomaly_score


def _production_anomaly_score(
    evaluator: BehaviorGuardEvaluatorML,
    current_message: CurrentMessage,
    user_profile: UserProfile,
    system_config: SystemConfig,
) -> float:
    """Full evaluate() parity including 3-decimal output rounding."""
    result = evaluator.evaluate(
        EvaluationInput(
            user_profile=user_profile,
            current_message=current_message,
            system_config=system_config,
        )
    )
    return float(result.anomaly_score)


def _extract_linguistic_subfeatures(
    current_message: CurrentMessage,
    linguistic_analyzer: LinguisticAnalyzerML,
    user_profile: UserProfile,
) -> dict[str, float]:
    """
    Extract per-sub-feature values for saturation audit.

    LinguisticAnalyzerML has no public per-feature API; _compute_feature_deviations
    is private. We use CurrentMessage.linguistic_features (same inputs as analyze())
    plus z-scores from the analyzer's private helper for deviation audit.
    """
    lf = current_message.linguistic_features
    base = {
        "message_length_tokens": float(lf.message_length_tokens),
        "message_length_chars": float(lf.message_length_chars),
        "lexical_diversity": float(lf.lexical_diversity),
        "formality_score": float(lf.formality_score),
        "politeness_score": float(lf.politeness_score),
    }
    # Optional z-scores via confirmed private method (same feature vector as analyze)
    mean_v, std_v = linguistic_analyzer._extract_profile_statistics(  # noqa: SLF001
        user_profile.linguistic_profile
    )
    cur_v = linguistic_analyzer._extract_feature_vector(lf)  # noqa: SLF001
    z = linguistic_analyzer._compute_feature_deviations(cur_v, mean_v, std_v)  # noqa: SLF001
    return {**base, **{f"z_{k}": v for k, v in z.items()}}


def audit_ling_subfeature_saturation(
    test_data: dict,
    user_ids: list[str],
    lambda_decay: float = 0.5,
    semantic_analyzer: SemanticAnalyzerML | None = None,
    linguistic_analyzer: LinguisticAnalyzerML | None = None,
) -> dict[str, Any]:
    """Organic vs anomalous sub-feature stats on test window (train-only profiles)."""
    semantic_analyzer = semantic_analyzer or SemanticAnalyzerML()
    linguistic_analyzer = linguistic_analyzer or LinguisticAnalyzerML()
    users = {u["user_id"]: u for u in test_data["users"]}
    splits = [s for s in split_conversations_8020(test_data, user_ids) if s.user_id in user_ids]

    organic: dict[str, list[float]] = defaultdict(list)
    anomalous: dict[str, list[float]] = defaultdict(list)
    feature_keys = [
        "message_length_tokens",
        "message_length_chars",
        "lexical_diversity",
        "formality_score",
        "politeness_score",
    ]

    for sp in splits:
        profile = _build_profile(users[sp.user_id], sp.train_msgs, lambda_decay)
        if profile is None:
            continue
        for i, msg in enumerate(sp.test_msgs):
            prev = _prev_in_session(sp.test_msgs, i)
            cur = ev.message_to_current_message(msg, prev, user_profile=profile)
            feats = _extract_linguistic_subfeatures(cur, linguistic_analyzer, profile)
            bucket = anomalous if msg.get("should_flag") else organic
            for k in feature_keys:
                bucket[k].append(feats[k])

    def _stats(vals: list[float]) -> dict:
        if not vals:
            return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
        return {
            "n": len(vals),
            "mean": round(mean(vals), 4),
            "std": round(pstdev(vals), 4) if len(vals) > 1 else 0.0,
            "min": round(min(vals), 4),
            "max": round(max(vals), 4),
        }

    report: dict[str, Any] = {}
    for k in feature_keys:
        o, a = organic[k], anomalous[k]
        o_s, a_s = _stats(o), _stats(a)
        o_rng = (o_s["max"] or 0) - (o_s["min"] or 0) if o_s["n"] else 0.0
        a_rng = (a_s["max"] or 0) - (a_s["min"] or 0) if a_s["n"] else 0.0
        report[k] = {
            "organic": o_s,
            "anomalous": a_s,
            "organic_range": round(o_rng, 4),
            "anomalous_range": round(a_rng, 4),
            "likely_dead_or_saturated": o_rng < 0.05 and a_rng < 0.05,
        }
    return report


def collect_component_rows(
    test_data: dict,
    user_ids: list[str],
    lambda_decay: float,
    system_config: SystemConfig,
    evaluator: BehaviorGuardEvaluatorML | None = None,
) -> list[dict]:
    """Collect s_sem/s_ling/s_temp + production composite via evaluate() path."""
    evaluator = evaluator or BehaviorGuardEvaluatorML()
    users = {u["user_id"]: u for u in test_data["users"]}
    rows: list[dict] = []

    for sp in split_conversations_8020(test_data, user_ids):
        profile = _build_profile(users[sp.user_id], sp.train_msgs, lambda_decay)
        if profile is None:
            continue
        for i, msg in enumerate(sp.test_msgs):
            prev = _prev_in_session(sp.test_msgs, i)
            cur = ev.message_to_current_message(msg, prev, user_profile=profile)
            s_sem, s_ling, s_temp, composite_raw = _production_component_and_composite(
                evaluator, cur, profile, system_config
            )
            composite_eval = _production_anomaly_score(
                evaluator, cur, profile, system_config
            )
            rows.append(
                {
                    "user_id": sp.user_id,
                    "y_true": bool(msg.get("should_flag", False)),
                    "s_sem": s_sem,
                    "s_ling": s_ling,
                    "s_temp": s_temp,
                    "composite_raw": composite_raw,
                    "composite": composite_eval,
                }
            )
    return rows


def run_lambda_sweep(
    test_data: dict,
    user_ids: list[str],
    system_config: SystemConfig,
    evaluator: BehaviorGuardEvaluatorML | None = None,
) -> dict[str, Any]:
    """λ sweep: profile decay varies per λ; composite from production evaluate() path."""
    evaluator = evaluator or BehaviorGuardEvaluatorML()

    lambdas = [round(v, 1) for v in np.arange(0.0, 1.0001, 0.1)]
    per_lambda: dict[str, dict] = {}
    best_f1 = -1.0
    best_lambda = 0.5
    best_row: dict = {}

    for lam in lambdas:
        rows = collect_component_rows(
            test_data, user_ids, lam, system_config, evaluator=evaluator
        )
        y = np.array([r["y_true"] for r in rows], dtype=bool)
        scores = np.array([r["composite"] for r in rows], dtype=float)
        t, m = _f1_max_threshold(y, scores)
        entry = {
            "lambda": lam,
            "precision": round(m["precision"], 4),
            "recall": round(m["recall"], 4),
            "f1": round(m["f1"], 4),
            "fpr": round(m["fpr"], 4),
            "auc": round(m["roc_auc"], 4),
            "threshold": round(float(t), 4),
            "n_messages": len(rows),
        }
        per_lambda[f"{lam:.1f}"] = entry
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_lambda = lam
            best_row = entry

    return {
        "per_lambda": per_lambda,
        "best_lambda": best_lambda,
        "best_at_f1max": best_row,
        "config": system_config.model_dump(),
    }


def _f1_max_threshold(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    low: float = 0.01,
    high: float = 0.99,
    step: float = 0.01,
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


def run_full_diagnostic(
    dataset_path: Path | None = None,
    max_users: int = 35,
) -> dict[str, Any]:
    """
    Full Part 1–4 diagnostic (no Part 5 summary until wiring verified externally).

    Uses 80/20 per-user split; profiles built from train organic messages only.
    """
    path = dataset_path or DEFAULT_DATASET
    test_data = json.loads(path.read_text(encoding="utf-8"))

    by_user: dict[str, list] = defaultdict(list)
    for m in test_data["messages"]:
        by_user[m["user_id"]].append(m)
    anomaly_users = sorted(
        uid for uid, msgs in by_user.items() if any(m.get("should_flag") for m in msgs)
    )
    benign_users = sorted(uid for uid in by_user if uid not in anomaly_users)
    rng = np.random.default_rng(SEED)
    benign_arr = np.array(benign_users)
    rng.shuffle(benign_arr)
    n_benign = min(len(anomaly_users), len(benign_arr), max_users)
    user_ids = sorted(anomaly_users[:max_users]) + sorted(benign_arr[:n_benign].tolist())

    semantic = SemanticAnalyzerML()
    linguistic = LinguisticAnalyzerML()
    evaluator = BehaviorGuardEvaluatorML()

    ling_audit = audit_ling_subfeature_saturation(
        test_data, user_ids, lambda_decay=0.5,
        semantic_analyzer=semantic, linguistic_analyzer=linguistic,
    )
    sweep_before = run_lambda_sweep(
        test_data, user_ids, BASELINE_CONFIG, evaluator=evaluator
    )
    sweep_after = run_lambda_sweep(
        test_data, user_ids, DIAG_CONFIG, evaluator=evaluator
    )

    lam0_before = sweep_before["per_lambda"].get("0.0", {})
    lam0_after = sweep_after["per_lambda"].get("0.0", {})

    return {
        "dataset": str(path),
        "n_users": len(user_ids),
        "split_protocol": "80% train profile (organic only) / 20% test score",
        "ling_subfeature_audit": ling_audit,
        "lambda_sweep_before_fix": sweep_before,
        "lambda_sweep_after_fix": sweep_after,
        "part5_lambda0_comparison": {
            "before_fix_f1_at_lambda0": lam0_before.get("f1"),
            "after_fix_f1_at_lambda0": lam0_after.get("f1"),
            "delta_f1": round(
                (lam0_after.get("f1") or 0) - (lam0_before.get("f1") or 0), 4
            ),
        },
    }


if __name__ == "__main__":
    out_path = ROOT / "results" / "diagnostic_harness_output.json"
    result = run_full_diagnostic()
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out_path}")
    p5 = result["part5_lambda0_comparison"]
    print(
        f"Part 5 λ=0: before F1={p5['before_fix_f1_at_lambda0']} "
        f"after F1={p5['after_fix_f1_at_lambda0']} Δ={p5['delta_f1']}"
    )
