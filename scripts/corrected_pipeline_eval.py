#!/usr/bin/env python3
"""
Full evaluation on corrected (de-confounded) injected datasets.

Runs TurnShift (canonical), Isolation Forest (F1-max), Autoencoder (F1-max),
compares against original confounded results, and writes comparison tables.

Usage:
    set HF_HUB_OFFLINE=1
    set TRANSFORMERS_OFFLINE=1
    python scripts/corrected_pipeline_eval.py
"""

from __future__ import annotations

import argparse
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
from scripts.audit_positive_surface_form import _manual_category  # noqa: E402
from scripts.task1_isolation_forest_rerun import (  # noqa: E402
    _evaluate_optimal_threshold,
    _prepare_isolation_forest_eval,
)
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

ORIGINAL_RESULTS = {
    "behaviorguard": ROOT / "results/methodology-diagnostics/behaviorguard_canonical_verify.json",
    "isolation_forest": ROOT / "results/methodology-diagnostics/task1_isolation_forest_results.json",
    "autoencoder": ROOT / "results/methodology-diagnostics/task5_autoencoder_results.json",
}

BG_THRESHOLD = 0.60
MAX_USERS = 20


def _load_corrected() -> dict[str, dict]:
    out = {}
    for key, path in CORRECTED_PATHS.items():
        with open(path, encoding="utf-8") as fh:
            out[key] = json.load(fh)
    return out


def _collect_aligned_test_msgs(test_data: dict, max_users: int) -> list[dict]:
    test_user_ids = set(test_data["splits"]["test"]["user_ids"])
    test_users = [u for u in test_data["users"] if u["user_id"] in test_user_ids]
    test_messages_by_user: dict[str, list] = defaultdict(list)
    for m in test_data["messages"]:
        if m["user_id"] in test_user_ids:
            test_messages_by_user[m["user_id"]].append(m)

    users_with, users_without = [], []
    for user in test_users:
        msgs = test_messages_by_user[user["user_id"]]
        if any(m.get("should_flag") for m in msgs):
            users_with.append(user)
        else:
            users_without.append(user)
    sampled = users_with[:max_users]
    sampled.extend(users_without[: max(0, max_users - len(sampled))])

    aligned: list[dict] = []
    for user in sampled:
        msgs = test_messages_by_user[user["user_id"]]
        split_idx = int(len(msgs) * 0.8)
        aligned.extend(msgs[split_idx:])
    return aligned


def _metrics_dict(m: dict) -> dict[str, float]:
    return {
        "precision": round(m["precision"], 4),
        "recall": round(m["recall"], 4),
        "f1": round(m["f1"], 4),
        "fpr": round(m["fpr"], 4),
        "auc": round(m["roc_auc"], 4),
    }


def _stratified_ab(
    aligned_msgs: list[dict],
    scores: list[float],
    threshold: float,
) -> dict[str, Any]:
    """Recall on category-(a) vs (b) test positives."""
    by_cat: dict[str, list[bool]] = defaultdict(list)
    for msg, score in zip(aligned_msgs, scores):
        if not msg.get("should_flag", False):
            continue
        cat, _ = _manual_category(msg)
        by_cat[cat].append(score > threshold)

    out = {}
    for cat in ("a_overt", "b_benign_surface"):
        preds = by_cat.get(cat, [])
        if not preds:
            out[cat] = {"support": 0, "recall": None}
            continue
        recall = sum(preds) / len(preds)
        out[cat] = {"support": len(preds), "recall": round(recall, 4)}
    return out


def eval_behaviorguard(test_data: dict, dataset_key: str) -> dict[str, Any]:
    pm_builder = ev._build_profile_with_pm(ev.CANONICAL_LAMBDA)
    metrics, predictions = ev.evaluate_method(
        "behaviorguard",
        dataset_key,
        test_data,
        max_users=MAX_USERS,
        overrides_enabled=False,
        profile_builder=pm_builder,
    )
    aligned = _collect_aligned_test_msgs(test_data, MAX_USERS)
    scores = [p["predicted_score"] for p in predictions]
    return {
        "method": "behaviorguard",
        "threshold": BG_THRESHOLD,
        **_metrics_dict(metrics),
        "stratified_ab": _stratified_ab(aligned, scores, BG_THRESHOLD),
        "n_test": metrics.get("num_predictions"),
    }


def eval_isolation_forest(test_data: dict, dataset_key: str) -> dict[str, Any]:
    metrics = _evaluate_optimal_threshold(test_data, MAX_USERS, "max_f1_threshold")
    t_star = float(metrics["threshold_used"])
    train_f, test_f, y_true, _ = _prepare_isolation_forest_eval(test_data, MAX_USERS)
    from turnshift.baselines.isolation_forest_baseline import IsolationForestBaseline

    iso = IsolationForestBaseline(contamination="auto", random_state=ev.SEED)
    iso.fit(train_f)
    scores = iso.predict(test_f)["anomaly_scores"].tolist()
    aligned = _collect_aligned_test_msgs(test_data, MAX_USERS)
    return {
        "method": "isolation_forest",
        "threshold": round(t_star, 4),
        "precision": round(metrics["precision"], 4),
        "recall": round(metrics["recall"], 4),
        "f1": round(metrics["f1"], 4),
        "fpr": round(metrics["fpr"], 4),
        "auc": round(metrics["roc_auc"], 4),
        "stratified_ab": _stratified_ab(aligned, scores, t_star),
        "n_test": len(y_true),
    }


def eval_autoencoder(test_data: dict, dataset_key: str) -> dict[str, Any]:
    train_f, test_f, y_true, _ = _prepare_baseline_eval(test_data, MAX_USERS)
    scores = _fit_scores(train_f, test_f)
    score_high = max(float(np.max(scores)), SWEEP_HIGH)
    score_high = float(np.ceil(score_high * 100) / 100)
    t_star, m_best, _ = ae_max_f1_threshold(
        y_true, scores, low=SWEEP_LOW, high=score_high, step=SWEEP_STEP
    )
    aligned = _collect_aligned_test_msgs(test_data, MAX_USERS)
    return {
        "method": "autoencoder",
        "threshold": round(t_star, 4),
        "precision": round(m_best["precision"], 4),
        "recall": round(m_best["recall"], 4),
        "f1": round(m_best["f1"], 4),
        "fpr": round(m_best["fpr"], 4),
        "auc": round(m_best["roc_auc"], 4),
        "stratified_ab": _stratified_ab(aligned, scores.tolist(), t_star),
        "n_test": len(y_true),
    }


def _load_original_baseline(method: str, dataset_key: str) -> dict[str, Any] | None:
    path = ORIGINAL_RESULTS[method]
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if method == "behaviorguard":
        for row in data.get("results", []):
            if row.get("dataset_key") == dataset_key:
                return {
                    "precision": round(row["precision"], 4),
                    "recall": round(row["recall"], 4),
                    "f1": round(row["f1"], 4),
                    "fpr": round(row["fpr"], 4),
                    "auc": round(row["auc"], 4),
                }
    elif method == "isolation_forest":
        for row in data.get("results", []):
            if (
                row.get("dataset_key") == dataset_key
                and row.get("contamination") == "max_f1_threshold"
            ):
                return {
                    "precision": row["precision"],
                    "recall": row["recall"],
                    "f1": row["f1"],
                    "fpr": row["fpr"],
                    "auc": row["auc"],
                    "threshold": row.get("threshold"),
                }
    elif method == "autoencoder":
        for row in data.get("summary", []):
            if row.get("dataset_key") == dataset_key:
                return {
                    "precision": row["precision"],
                    "recall": row["recall"],
                    "f1": row["f1"],
                    "fpr": row["fpr"],
                    "auc": row["auc"],
                    "threshold": row.get("threshold"),
                }
    return None


def _delta(orig: dict | None, corr: dict) -> dict[str, float | None]:
    if not orig:
        return {}
    return {k: round(corr[k] - orig[k], 4) for k in ("precision", "recall", "f1", "fpr", "auc")}


def _substantial_drop(delta_f1: float | None, threshold: float = 0.15) -> bool:
    return delta_f1 is not None and delta_f1 < -threshold


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results")
    parser.add_argument(
        "--methods",
        default="turnshift,isolation_forest,autoencoder",
    )
    args = parser.parse_args()
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    corrected = _load_corrected()
    corrected_results: list[dict] = []
    comparison: list[dict] = []

    eval_fns = {
        "behaviorguard": eval_behaviorguard,
        "isolation_forest": eval_isolation_forest,
        "autoencoder": eval_autoencoder,
    }

    for dataset_key, test_data in corrected.items():
        dname = DISPLAY[dataset_key]
        print(f"\n{'=' * 72}\nCORRECTED EVAL: {dname}\n{'=' * 72}")
        for method in methods:
            print(f"\n--- {method} ---")
            row = eval_fns[method](test_data, dataset_key)
            row["dataset"] = dname
            row["dataset_key"] = dataset_key
            row["pipeline"] = "corrected_v1"
            corrected_results.append(row)
            print(
                f"  P={row['precision']:.4f} R={row['recall']:.4f} "
                f"F1={row['f1']:.4f} FPR={row['fpr']:.4f} AUC={row['auc']:.4f} "
                f"t={row['threshold']}"
            )
            orig = _load_original_baseline(method, dataset_key)
            delta = _delta(orig, row)
            comp = {
                "dataset": dname,
                "dataset_key": dataset_key,
                "method": method,
                "original": orig,
                "corrected": {
                    k: row[k]
                    for k in ("precision", "recall", "f1", "fpr", "auc", "threshold")
                },
                "delta_corrected_minus_original": delta,
                "substantial_f1_drop": _substantial_drop(
                    delta.get("f1") if delta else None
                ),
                "stratified_ab_corrected": row.get("stratified_ab"),
            }
            comparison.append(comp)
            if orig:
                print(
                    f"  vs original F1: {orig['f1']:.4f} -> {row['f1']:.4f} "
                    f"(delta {delta.get('f1', 0):+.4f})"
                )

    payload = {
        "pipeline": "corrected_v1",
        "max_users": MAX_USERS,
        "methods": methods,
        "corrected_results": corrected_results,
        "comparison": comparison,
    }
    json_path = out_dir / "corrected_pipeline_eval.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)

    # Markdown comparison table
    lines = [
        "# Original (confounded) vs Corrected pipeline evaluation",
        "",
        "| Dataset | Method | Orig F1 | Corr F1 | Δ F1 | Orig R | Corr R | Substantial drop? |",
        "|---------|--------|---------|---------|------|--------|--------|-------------------|",
    ]
    for c in comparison:
        o, r = c.get("original") or {}, c["corrected"]
        d = c.get("delta_corrected_minus_original") or {}
        lines.append(
            f"| {c['dataset']} | {c['method']} | "
            f"{o.get('f1', 'n/a')} | {r['f1']} | {d.get('f1', 'n/a')} | "
            f"{o.get('recall', 'n/a')} | {r['recall']} | "
            f"{'YES' if c['substantial_f1_drop'] else 'no'} |"
        )
    md_path = out_dir / "corrected_vs_original_comparison.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nWrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
