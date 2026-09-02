#!/usr/bin/env python3
"""
5-fold user-level CV threshold sweep (primary protocol when val positives < 12).

Profiles: 80/20 chronological split unchanged; one profile per user from train.
Scoring: evaluate() on full holdout tail; CV partitions users only.

For each fold k:
  - Validation: 20 holdout-positive users + organic users in buckets != k
  - Test: 5 holdout-positive users in fold k + organic users in bucket k
  - tau* = argmax F1 on validation messages; report test metrics at tau*

Runs independently for cosine and Mahalanobis semantic scoring.

Usage:
    set HF_HUB_OFFLINE=1; set TRANSFORMERS_OFFLINE=1
    python scripts/threshold_sweep_cv_protocol.py --audit-folds
    python scripts/threshold_sweep_cv_protocol.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

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
from scripts.personachat_holdout_split import (  # noqa: E402
    DEFAULT_DATASET,
    N_CV_FOLDS,
    POSITIVE_CV_SEED,
    build_cv_fold_plan,
    build_holdout_records,
    load_dataset,
    print_cv_fold_audit,
)
from scripts.threshold_sweep_protocol import (  # noqa: E402
    LAMBDA_DECAY,
    N_BOOTSTRAP,
    SEED,
    TAU_HIGH,
    TAU_LOW,
    TAU_STEP,
    build_profiles_once,
    f1_max_threshold,
)

SCORES_CACHE = ROOT / "results" / "threshold_sweep_cv_scores.npz"
MAHA_COMPARE_CACHE = ROOT / "results" / "mahalanobis_comparison_scores.npz"
OUT_PATH = ROOT / "results" / "methodology-diagnostics" / "threshold_sweep_cv_protocol.json"

BASE_CONFIG = dict(
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


def collect_holdout_scores_both_modes(
    records,
    users_lookup: dict[str, dict],
    lambda_decay: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Score all holdout messages; cosine + Mahalanobis in one profile pass per user."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    profile_info = build_profiles_once(users_lookup, records, lambda_decay)
    profiles = profile_info["profiles"]
    evaluator = TurnShiftEvaluatorML()
    config_cos = SystemConfig(**BASE_CONFIG, semantic_scoring_mode="cosine")
    config_maha = SystemConfig(**BASE_CONFIG, semantic_scoring_mode="mahalanobis")

    user_ids: list[str] = []
    labels: list[int] = []
    scores_cos: list[float] = []
    scores_maha: list[float] = []
    n_done = 0

    for rec in records:
        profile = profiles.get(rec.user_id)
        if profile is None:
            continue
        for i, msg in enumerate(rec.holdout_msgs):
            prev = _prev_in_session(rec.holdout_msgs, i)
            cur = ev.message_to_current_message(msg, prev, user_profile=profile)
            y = 1 if msg.get("should_flag", False) else 0
            r_cos = evaluator.evaluate(
                EvaluationInput(user_profile=profile, current_message=cur, system_config=config_cos)
            )
            r_maha = evaluator.evaluate(
                EvaluationInput(user_profile=profile, current_message=cur, system_config=config_maha)
            )
            user_ids.append(rec.user_id)
            labels.append(y)
            scores_cos.append(float(r_cos.anomaly_score))
            scores_maha.append(float(r_maha.anomaly_score))
            n_done += 1
            if n_done % 500 == 0:
                print(f"    ... {n_done} holdout messages scored", flush=True)

    return (
        np.array(user_ids, dtype=object),
        np.array(labels, dtype=int),
        np.array(scores_cos, dtype=float),
        np.array(scores_maha, dtype=float),
    )


def build_user_id_and_label_arrays(records) -> tuple[np.ndarray, np.ndarray]:
    user_ids: list[str] = []
    labels: list[int] = []
    for rec in records:
        for msg in rec.holdout_msgs:
            user_ids.append(rec.user_id)
            labels.append(1 if msg.get("should_flag", False) else 0)
    return np.array(user_ids, dtype=object), np.array(labels, dtype=int)


def load_or_collect_scores(
    records,
    users_lookup: dict[str, dict],
    lambda_decay: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    user_ids, labels = build_user_id_and_label_arrays(records)

    if SCORES_CACHE.exists():
        data = np.load(SCORES_CACHE, allow_pickle=True)
        if len(data["labels"]) == len(labels) and np.array_equal(data["labels"], labels):
            print(f"  Loaded scores from {SCORES_CACHE.name}", flush=True)
            return (
                data["user_ids"],
                data["labels"].astype(int),
                data["cosine"].astype(float),
                data["mahalanobis"].astype(float),
            )

    if MAHA_COMPARE_CACHE.exists():
        data = np.load(MAHA_COMPARE_CACHE, allow_pickle=True)
        if len(data["labels"]) == len(labels) and np.array_equal(data["labels"], labels):
            print(
                f"  Reusing cosine/Mahalanobis from {MAHA_COMPARE_CACHE.name}",
                flush=True,
            )
            SCORES_CACHE.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                SCORES_CACHE,
                user_ids=user_ids,
                labels=labels,
                cosine=data["cosine"],
                mahalanobis=data["mahalanobis"],
            )
            return user_ids, labels, data["cosine"].astype(float), data["mahalanobis"].astype(float)

    print("  Collecting cosine + Mahalanobis holdout scores...", flush=True)
    uids, labels2, cos, maha = collect_holdout_scores_both_modes(
        records, users_lookup, lambda_decay
    )
    assert np.array_equal(labels, labels2)
    SCORES_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(SCORES_CACHE, user_ids=uids, labels=labels, cosine=cos, mahalanobis=maha)
    print(f"  Cached to {SCORES_CACHE.name}", flush=True)
    return uids, labels, cos, maha


def _subset_mask(user_ids: np.ndarray, allowed: set[str]) -> np.ndarray:
    return np.array([uid in allowed for uid in user_ids], dtype=bool)


def _confusion(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _metrics_from_confusion(c: dict[str, int]) -> dict[str, float]:
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "fpr": round(fpr, 4),
    }


def run_cv_for_mode(
    mode: str,
    scores: np.ndarray,
    labels: np.ndarray,
    user_ids: np.ndarray,
    fold_plan,
) -> dict[str, Any]:
    fold_results: list[dict[str, Any]] = []
    pooled_test_labels: list[int] = []
    pooled_test_preds: list[int] = []
    pooled_test_scores: list[float] = []
    tau_per_fold: list[float] = []
    f1_per_fold: list[float] = []

    for fold in fold_plan.folds:
        val_mask = _subset_mask(user_ids, fold.validation_user_ids)
        test_mask = _subset_mask(user_ids, fold.test_user_ids)
        val_y = labels[val_mask].astype(bool)
        val_s = scores[val_mask]
        test_y = labels[test_mask].astype(bool)
        test_s = scores[test_mask]

        tau_star, val_m = f1_max_threshold(val_y, val_s)
        test_pred = (test_s > tau_star).astype(int)
        conf = _confusion(test_y.astype(int), test_pred)
        test_m = _metrics_from_confusion(conf)

        fold_results.append(
            {
                "fold": fold.fold_index,
                "tau_star": round(float(tau_star), 4),
                "validation": {
                    "n_users": len(fold.validation_user_ids),
                    "n_messages": int(val_mask.sum()),
                    "n_positives": int(val_y.sum()),
                    "f1_at_tau_star": round(val_m["f1"], 4),
                },
                "test": {
                    "n_users": len(fold.test_user_ids),
                    "n_messages": int(test_mask.sum()),
                    "n_positives": int(test_y.sum()),
                    **test_m,
                    **conf,
                },
            }
        )
        tau_per_fold.append(float(tau_star))
        f1_per_fold.append(test_m["f1"])
        pooled_test_labels.extend(test_y.astype(int).tolist())
        pooled_test_preds.extend(test_pred.tolist())
        pooled_test_scores.extend(test_s.tolist())

    pooled_y = np.array(pooled_test_labels, dtype=int)
    pooled_pred = np.array(pooled_test_preds, dtype=int)
    pooled_s = np.array(pooled_test_scores, dtype=float)
    pooled_conf = _confusion(pooled_y, pooled_pred)
    pooled_metrics = _metrics_from_confusion(pooled_conf)
    boot_preds = pooled_pred
    rng = np.random.RandomState(SEED)
    n = len(pooled_y)
    boot_f1: list[float] = []
    skipped = 0
    for _ in range(N_BOOTSTRAP):
        idx = rng.choice(n, size=n, replace=True)
        lbl = pooled_y[idx]
        if len(np.unique(lbl)) < 2:
            skipped += 1
            continue
        boot_f1.append(
            float(f1_score(lbl, boot_preds[idx], zero_division=0.0))
        )
    arr = np.array(boot_f1)
    pooled_f1_boot = {
        "point_estimate": pooled_metrics["f1"],
        "ci_low": float(np.percentile(arr, 2.5)),
        "ci_high": float(np.percentile(arr, 97.5)),
        "ci_method": "bootstrap_message_level_pooled_test_at_fold_tau_star",
        "bootstrap_requested": N_BOOTSTRAP,
        "bootstrap_skipped_single_class": skipped,
        "bootstrap_effective_n": int(len(arr)),
        "bootstrap_seed": SEED,
    }

    tau_arr = np.array(tau_per_fold)
    f1_arr = np.array(f1_per_fold)
    return {
        "semantic_mode": mode,
        "fold_results": fold_results,
        "tau_star_per_fold": [round(t, 4) for t in tau_per_fold],
        "tau_star_mean": round(float(tau_arr.mean()), 4),
        "tau_star_std": round(float(tau_arr.std(ddof=0)), 4),
        "tau_star_min": round(float(tau_arr.min()), 4),
        "tau_star_max": round(float(tau_arr.max()), 4),
        "test_f1_per_fold": f1_per_fold,
        "test_f1_mean": round(float(f1_arr.mean()), 4),
        "test_f1_std": round(float(f1_arr.std(ddof=0)), 4),
        "pooled_test": {**pooled_conf, **pooled_metrics, "f1_bootstrap_ci": pooled_f1_boot},
    }


def run_protocol(test_data: dict) -> dict[str, Any]:
    records = build_holdout_records(test_data)
    fold_plan = build_cv_fold_plan(records)
    users_lookup = {u["user_id"]: u for u in test_data["users"]}

    user_ids, labels, cos, maha = load_or_collect_scores(
        records, users_lookup, LAMBDA_DECAY
    )

    cosine_cv = run_cv_for_mode("cosine", cos, labels, user_ids, fold_plan)
    maha_cv = run_cv_for_mode("mahalanobis", maha, labels, user_ids, fold_plan)

    f1_diff = maha_cv["test_f1_mean"] - cosine_cv["test_f1_mean"]
    pooled_f1_diff = maha_cv["pooled_test"]["f1"] - cosine_cv["pooled_test"]["f1"]

    return {
        "protocol": {
            "method": "5_fold_user_level_cv",
            "reason": "single_split validation positives (11) < 12",
            "n_folds": N_CV_FOLDS,
            "positive_cv_seed": POSITIVE_CV_SEED,
            "tau_sweep": f"[{TAU_LOW},{TAU_HIGH}] step {TAU_STEP}",
            "lambda_decay": LAMBDA_DECAY,
            "profile_split": "80/20 chronological per user (unchanged)",
            "eval_config": BASE_CONFIG,
        },
        "fold_plan_audit": fold_plan.audit,
        "cosine": cosine_cv,
        "mahalanobis": maha_cv,
        "comparison": {
            "mean_test_f1_mahalanobis_minus_cosine": round(f1_diff, 4),
            "pooled_test_f1_mahalanobis_minus_cosine": round(pooled_f1_diff, 4),
            "mahalanobis_f1_advantage_holds_mean_across_folds": f1_diff > 0,
            "mahalanobis_f1_advantage_holds_pooled": pooled_f1_diff > 0,
            "tau_stability": {
                "cosine_range": [
                    cosine_cv["tau_star_min"],
                    cosine_cv["tau_star_max"],
                ],
                "mahalanobis_range": [
                    maha_cv["tau_star_min"],
                    maha_cv["tau_star_max"],
                ],
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="5-fold CV threshold sweep protocol")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--audit-folds", action="store_true")
    parser.add_argument("--output", type=Path, default=OUT_PATH)
    args = parser.parse_args()

    test_data = load_dataset(args.dataset)
    if args.audit_folds:
        records = build_holdout_records(test_data)
        print_cv_fold_audit(build_cv_fold_plan(records))
        return

    print("Running 5-fold CV threshold sweep (cosine + Mahalanobis)...", flush=True)
    report = run_protocol(test_data)

    for mode in ("cosine", "mahalanobis"):
        r = report[mode]
        print(
            f"\n{mode}: tau* per fold = {r['tau_star_per_fold']} "
            f"(mean={r['tau_star_mean']:.2f} std={r['tau_star_std']:.2f})",
            flush=True,
        )
        print(
            f"  test F1 per fold = {r['test_f1_per_fold']} "
            f"mean={r['test_f1_mean']:.4f} std={r['test_f1_std']:.4f}",
            flush=True,
        )
        p = r["pooled_test"]
        ci = p["f1_bootstrap_ci"]
        print(
            f"  pooled test F1={p['f1']} [{ci['ci_low']:.3f}, {ci['ci_high']:.3f}] "
            f"TP={p['tp']} FP={p['fp']} FN={p['fn']}",
            flush=True,
        )

    cmp = report["comparison"]
    print(
        f"\nMahalanobis - cosine: mean fold F1 {cmp['mean_test_f1_mahalanobis_minus_cosine']:+.4f}, "
        f"pooled F1 {cmp['pooled_test_f1_mahalanobis_minus_cosine']:+.4f}",
        flush=True,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
