#!/usr/bin/env python3
"""
Fair corrected-data TurnShift tuning vs IF/AE (F1-max protocol).

Tasks:
  1. BG F1-max threshold sweep at λ=0.50, default weights (0.4, 0.35, 0.25)
  2. λ sweep [0,1] step 0.1 with F1-max τ at each λ
  3. (α,β,γ) coarse simplex grid, optimize F1 at FPR=0 operating point
  4. Best combined config vs corrected IF/AE (F1-max)
  5. AUC at each configuration (ranking changes with λ/weights, not with τ)

Usage:
    set HF_HUB_OFFLINE=1; set TRANSFORMERS_OFFLINE=1
    python scripts/corrected_bg_fair_tuning.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
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
from turnshift import TurnShiftEvaluatorML  # noqa: E402
from turnshift.models import EvaluationInput, SystemConfig  # noqa: E402
from scripts.task1_isolation_forest_rerun import _evaluate_optimal_threshold  # noqa: E402
from scripts.task5_autoencoder_rerun import (  # noqa: E402
    SWEEP_HIGH,
    SWEEP_LOW,
    SWEEP_STEP,
    _fit_scores,
    _max_f1_threshold as ae_max_f1_threshold,
    _prepare_baseline_eval,
)

CORRECTED_PATHS = {
    "personachat": ROOT / "datasets/personachat_processed_corrected.json",
    "blended_skill_talk": ROOT / "datasets/blended_skill_talk_processed_corrected.json",
    "anthropic_hh": ROOT / "datasets/anthropic_hh_processed_corrected.json",
}

DISPLAY = {
    "personachat": "PersonaChat",
    "blended_skill_talk": "BST",
    "anthropic_hh": "AnthropicHH",
}

MAX_USERS = 20
DEFAULT_WEIGHTS = (0.4, 0.35, 0.25)
CANONICAL_LAMBDA = 0.50
BG_SWEEP_LOW = 0.01
BG_SWEEP_HIGH = 0.99
BG_SWEEP_STEP = 0.01

BG_CONFIG = SystemConfig(
    sensitivity_level="medium",
    deployment_context="enterprise",
    overrides_enabled=False,
)


def _load_corrected() -> dict[str, dict]:
    return {
        k: json.loads(p.read_text(encoding="utf-8"))
        for k, p in CORRECTED_PATHS.items()
    }


def _prepare_test_profiles(
    test_data: dict,
    max_users: int,
    profile_builder,
) -> dict[str, dict]:
    test_user_ids = set(test_data["splits"]["test"]["user_ids"])
    test_users = [u for u in test_data["users"] if u["user_id"] in test_user_ids]
    test_messages_by_user: dict[str, list] = defaultdict(list)
    for m in test_data["messages"]:
        if m["user_id"] in test_user_ids:
            test_messages_by_user[m["user_id"]].append(m)

    with_a, without = [], []
    for user in test_users:
        msgs = test_messages_by_user[user["user_id"]]
        (with_a if any(m.get("should_flag") for m in msgs) else without).append(user)

    sampled = with_a[:max_users]
    sampled.extend(without[: max(0, max_users - len(sampled))])

    profiles: dict[str, dict] = {}
    for user in sampled:
        msgs = test_messages_by_user[user["user_id"]]
        split_idx = int(len(msgs) * 0.8)
        profile = profile_builder(user, msgs[:split_idx])
        if profile:
            profiles[user["user_id"]] = {
                "profile": profile,
                "test_msgs": msgs[split_idx:],
            }
    return profiles


def _prev_in_session(test_msgs: list, i: int):
    if i == 0:
        return None
    p = test_msgs[i - 1]
    if p.get("session_id", "session_0") == test_msgs[i].get("session_id", "session_0"):
        return p
    return None


def collect_component_scores(
    test_data: dict,
    lambda_decay: float,
    max_users: int = MAX_USERS,
) -> list[dict]:
    """Score test window; return per-message component scores and labels."""
    builder = ev._build_profile_with_pm(lambda_decay)
    profiles = _prepare_test_profiles(test_data, max_users, builder)
    evaluator = TurnShiftEvaluatorML()
    rows: list[dict] = []

    for uid, ud in profiles.items():
        profile = ud["profile"]
        test_msgs = ud["test_msgs"]
        for i, msg in enumerate(test_msgs):
            prev = _prev_in_session(test_msgs, i)
            cur = ev.message_to_current_message(msg, prev, user_profile=profile)
            result = evaluator.evaluate(
                EvaluationInput(
                    user_profile=profile,
                    current_message=cur,
                    system_config=BG_CONFIG,
                )
            )
            cs = result.component_scores
            rows.append(
                {
                    "user_id": uid,
                    "y_true": bool(msg.get("should_flag", False)),
                    "s_sem": float(cs.semantic),
                    "s_ling": float(cs.linguistic),
                    "s_temp": float(cs.temporal),
                    "composite_default": float(result.anomaly_score),
                }
            )
    return rows


def _weighted_scores(
    rows: list[dict],
    alpha: float,
    beta: float,
    gamma: float,
) -> np.ndarray:
    a, b, g = alpha, beta, gamma
    return np.array(
        [a * r["s_sem"] + b * r["s_ling"] + g * r["s_temp"] for r in rows],
        dtype=float,
    )


def _y_true(rows: list[dict]) -> np.ndarray:
    return np.array([r["y_true"] for r in rows], dtype=bool)


def _f1_max_threshold(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    low: float = BG_SWEEP_LOW,
    high: float = BG_SWEEP_HIGH,
    step: float = BG_SWEEP_STEP,
) -> tuple[float, dict]:
    best_t = low
    best_m: dict = {}
    best_f1 = -1.0
    for t in np.arange(low, high + step / 2, step):
        t = round(float(t), 2)
        y_pred = y_scores > t
        m = ev.compute_metrics(y_true.tolist(), y_pred.tolist(), y_scores.tolist())
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_t = t
            best_m = m
    return best_t, best_m


def _metrics_at_threshold(y_true: np.ndarray, y_scores: np.ndarray, t: float) -> dict:
    y_pred = y_scores > t
    return ev.compute_metrics(y_true.tolist(), y_pred.tolist(), y_scores.tolist())


def _metrics_at_fpr_zero(y_true: np.ndarray, y_scores: np.ndarray) -> tuple[float, dict]:
    """Operating point: τ just above max benign score (FPR=0)."""
    benign = y_scores[~y_true]
    if len(benign) == 0:
        t = 1.0
    else:
        t = float(np.max(benign)) + 1e-6
    return t, _metrics_at_threshold(y_true, y_scores, t)


def _row(m: dict, t: float) -> dict[str, float]:
    return {
        "precision": round(m["precision"], 4),
        "recall": round(m["recall"], 4),
        "f1": round(m["f1"], 4),
        "fpr": round(m["fpr"], 4),
        "auc": round(m["roc_auc"], 4),
        "threshold": round(float(t), 4),
    }


def _weight_grid(step: float = 0.1) -> list[tuple[float, float, float]]:
    grid: list[tuple[float, float, float]] = []
    vals = [round(v, 1) for v in np.arange(step, 1.0, step)]
    for alpha in vals:
        for beta in vals:
            gamma = round(1.0 - alpha - beta, 1)
            if gamma >= step - 1e-9 and gamma <= 1.0 - step + 1e-9:
                grid.append((alpha, beta, gamma))
    return grid


def task1_f1max_default_lambda(rows: list[dict]) -> dict:
    y = _y_true(rows)
    scores = _weighted_scores(rows, *DEFAULT_WEIGHTS)
    t, m = _f1_max_threshold(y, scores)
    return _row(m, t)


def task2_lambda_sweep(test_data: dict) -> tuple[dict[str, Any], dict[float, list[dict]]]:
    lambdas = [round(v, 1) for v in np.arange(0.0, 1.0001, 0.1)]
    per_lambda: dict[str, dict] = {}
    rows_cache: dict[float, list[dict]] = {}
    best_f1 = -1.0
    best_lambda = CANONICAL_LAMBDA
    best_row: dict = {}

    for lam in lambdas:
        rows = collect_component_scores(test_data, lam)
        rows_cache[lam] = rows
        y = _y_true(rows)
        scores = _weighted_scores(rows, *DEFAULT_WEIGHTS)
        t, m = _f1_max_threshold(y, scores)
        entry = {**_row(m, t), "lambda": lam}
        per_lambda[f"{lam:.1f}"] = entry
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_lambda = lam
            best_row = entry

    at_canonical = per_lambda.get(f"{CANONICAL_LAMBDA:.1f}", {})
    return {
        "per_lambda": per_lambda,
        "best_lambda": best_lambda,
        "best_at_f1max": best_row,
        "canonical_lambda_0.50_f1max": at_canonical,
    }, rows_cache


def task3_weight_grid(rows: list[dict]) -> dict[str, Any]:
    y = _y_true(rows)
    grid = _weight_grid(0.1)
    results: list[dict] = []
    best_f1 = -1.0
    best_weights = DEFAULT_WEIGHTS
    best_entry: dict = {}

    default_f1_at_fpr0 = None
    for alpha, beta, gamma in grid:
        scores = _weighted_scores(rows, alpha, beta, gamma)
        t, m = _metrics_at_fpr_zero(y, scores)
        entry = {
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            **_row(m, t),
        }
        results.append(entry)
        if (alpha, beta, gamma) == DEFAULT_WEIGHTS:
            default_f1_at_fpr0 = entry["f1"]
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_weights = (alpha, beta, gamma)
            best_entry = entry

    return {
        "grid_step": 0.1,
        "objective": "max F1 at FPR=0 (tau = max(benign_score) + epsilon)",
        "default_weights_f1_at_fpr0": default_f1_at_fpr0,
        "best_weights": {
            "alpha": best_weights[0],
            "beta": best_weights[1],
            "gamma": best_weights[2],
        },
        "best_at_fpr0": best_entry,
        "all_grid": results,
    }


def task4_best_combined(
    lambda_best: float,
    weight_best: tuple[float, float, float],
    rows_cache: dict[float, list[dict]],
    rows_canonical: list[dict],
) -> dict:
    """Best λ + best weights + F1-max τ (fair vs IF/AE)."""
    rows_l = rows_cache[lambda_best]
    y_l = _y_true(rows_l)
    scores_lw = _weighted_scores(rows_l, *weight_best)
    t_lw, m_lw = _f1_max_threshold(y_l, scores_lw)

    y_c = _y_true(rows_canonical)
    scores_def = _weighted_scores(rows_canonical, *DEFAULT_WEIGHTS)
    t_def, m_def = _f1_max_threshold(y_c, scores_def)

    candidates = [
        ("lambda_best_weights_default_f1max", rows_l, DEFAULT_WEIGHTS, lambda_best),
        ("lambda_best_weight_best_f1max", rows_l, weight_best, lambda_best),
        ("lambda_0.50_weight_best_f1max", rows_canonical, weight_best, CANONICAL_LAMBDA),
        ("lambda_0.50_default_f1max", rows_canonical, DEFAULT_WEIGHTS, CANONICAL_LAMBDA),
    ]

    best_name = ""
    best_m: dict = {}
    best_t = 0.0
    best_lam = CANONICAL_LAMBDA
    best_w = DEFAULT_WEIGHTS
    best_f1 = -1.0

    for name, rows, w, lam in candidates:
        y = _y_true(rows)
        sc = _weighted_scores(rows, *w)
        t, m = _f1_max_threshold(y, sc)
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_name = name
            best_m = m
            best_t = t
            best_lam = lam
            best_w = w

    return {
        "selected_config": best_name,
        "lambda": best_lam,
        "alpha": best_w[0],
        "beta": best_w[1],
        "gamma": best_w[2],
        **_row(best_m, best_t),
        "alternatives": {
            "lambda_best_weight_best_f1max": _row(m_lw, t_lw),
            "lambda_0.50_default_f1max": _row(m_def, t_def),
        },
    }


def eval_baselines_f1max(test_data: dict) -> dict[str, dict]:
    if_m = _evaluate_optimal_threshold(test_data, MAX_USERS, "max_f1_threshold")
    train_f, test_f, y_true, _ = _prepare_baseline_eval(test_data, MAX_USERS)
    ae_scores = _fit_scores(train_f, test_f)
    high = max(float(np.max(ae_scores)), SWEEP_HIGH)
    high = float(np.ceil(high * 100) / 100)
    t_ae, m_ae, _ = ae_max_f1_threshold(y_true, ae_scores, high=high)
    return {
        "isolation_forest": {
            "precision": round(if_m["precision"], 4),
            "recall": round(if_m["recall"], 4),
            "f1": round(if_m["f1"], 4),
            "fpr": round(if_m["fpr"], 4),
            "auc": round(if_m["roc_auc"], 4),
            "threshold": round(float(if_m["threshold_used"]), 4),
        },
        "autoencoder": {
            "precision": round(m_ae["precision"], 4),
            "recall": round(m_ae["recall"], 4),
            "f1": round(m_ae["f1"], 4),
            "fpr": round(m_ae["fpr"], 4),
            "auc": round(m_ae["roc_auc"], 4),
            "threshold": round(float(t_ae), 4),
        },
    }


def main() -> None:
    datasets = _load_corrected()
    out: dict[str, Any] = {
        "protocol": {
            "max_users": MAX_USERS,
            "default_weights": DEFAULT_WEIGHTS,
            "bg_threshold_sweep": f"[{BG_SWEEP_LOW},{BG_SWEEP_HIGH}] step {BG_SWEEP_STEP}",
            "lambda_sweep": "0.0..1.0 step 0.1, F1-max tau each",
            "weight_grid": "simplex step 0.1, max F1 at FPR=0",
            "overrides_enabled": False,
        },
        "datasets": {},
    }

    # Cache canonical rows for task1/3
    canonical_rows: dict[str, list[dict]] = {}

    for dk, td in datasets.items():
        dname = DISPLAY[dk]
        print(f"\n{'=' * 60}\n{dname}\n{'=' * 60}")

        print("  Collecting scores λ=0.50...")
        rows = collect_component_scores(td, CANONICAL_LAMBDA)
        canonical_rows[dk] = rows
        y = _y_true(rows)
        scores_def = _weighted_scores(rows, *DEFAULT_WEIGHTS)

        # Fixed τ=0.60 reference (AUC confirmation)
        m_fixed = _metrics_at_threshold(y, scores_def, 0.60)
        fixed_row = _row(m_fixed, 0.60)

        print("  Task 1: F1-max at λ=0.50...")
        t1 = task1_f1max_default_lambda(rows)

        print("  Task 2: λ sweep...")
        t2, rows_cache = task2_lambda_sweep(td)

        print("  Task 3: weight grid at λ=0.50...")
        t3 = task3_weight_grid(rows)

        lam_best = t2["best_lambda"]
        w_best = (
            t3["best_weights"]["alpha"],
            t3["best_weights"]["beta"],
            t3["best_weights"]["gamma"],
        )

        print("  Task 4: best combined...")
        t4 = task4_best_combined(lam_best, w_best, rows_cache, rows)

        print("  Baselines IF/AE F1-max...")
        baselines = eval_baselines_f1max(td)

        # AUC sensitivity: default weights, varying lambda only
        auc_by_lambda = {}
        for key, entry in t2["per_lambda"].items():
            auc_by_lambda[key] = entry["auc"]

        ds_out = {
            "fixed_tau_0.60_default_lambda_weights": fixed_row,
            "task1_f1max_lambda_0.50": t1,
            "task2_lambda_sweep": t2,
            "task3_weight_grid": {
                "objective": t3["objective"],
                "default_weights_f1_at_fpr0": t3["default_weights_f1_at_fpr0"],
                "best_weights": t3["best_weights"],
                "best_at_fpr0": t3["best_at_fpr0"],
            },
            "task4_best_combined_f1max": t4,
            "baselines_f1max": baselines,
            "auc_by_lambda_default_weights": auc_by_lambda,
        }
        out["datasets"][dk] = ds_out

        print(f"  T1 F1-max: P={t1['precision']} R={t1['recall']} F1={t1['f1']} τ*={t1['threshold']}")
        print(f"  T2 λ*={lam_best} F1={t2['best_at_f1max']['f1']} AUC={t2['best_at_f1max']['auc']}")
        print(f"  T3 w*=({w_best[0]},{w_best[1]},{w_best[2]}) F1@FPR=0={t3['best_at_fpr0']['f1']}")
        print(f"  T4 best F1={t4['f1']} vs IF={baselines['isolation_forest']['f1']} AE={baselines['autoencoder']['f1']}")
        print(f"  Fixed τ=0.60 AUC={fixed_row['auc']} (unchanged by τ; scores fixed at λ=0.5)")

    out_path = ROOT / "results" / "corrected_bg_fair_tuning.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
