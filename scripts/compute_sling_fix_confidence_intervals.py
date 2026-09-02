#!/usr/bin/env python3
"""
Confidence intervals for s_ling FIX_IT holdout metrics.

- F1: bootstrap 95% CI (message-level resampling, same protocol as AUC bootstrap)
- Recall / precision: exact Clopper-Pearson (legitimate binomial proportions)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
THRESHOLD = 0.60
N_BOOTSTRAP = 5000
SEED = 42

COMPONENT_CACHE = ROOT / "results" / "sling_audit_component_scores.npz"
COSINE_CACHE = ROOT / "results" / "mahalanobis_comparison_scores.npz"

# Confusion counts for Clopper-Pearson recall/precision (holdout eval)
WITHOUT_SLING = {"tp": 8, "fp": 13, "fn": 21, "tn": 10123, "n_pos": 29}
WITH_SLING = {"tp": 2, "fp": 1076, "fn": 27, "tn": 10059, "n_pos": 29}


def clopper_pearson(k: int, n: int, confidence: float = 0.95) -> tuple[float, float, float]:
    """Exact Clopper-Pearson CI for binomial proportion k/n."""
    if n == 0:
        return 0.0, 0.0, 0.0
    point = k / n
    low, high = binomtest(k, n).proportion_ci(confidence_level=confidence, method="exact")
    return point, float(low), float(high)


def bootstrap_f1_ci(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float = THRESHOLD,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = SEED,
) -> dict:
    """
    Message-level bootstrap CI on F1 at fixed threshold.

    Resamples all test messages with replacement (same protocol as AUC bootstrap
    in production_sling_audit_snippet.py / mahalanobis_bootstrap_comparison.py).
    """
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    preds = (scores >= threshold).astype(int)
    point = float(f1_score(labels, preds, zero_division=0.0))

    rng = np.random.RandomState(seed)
    n = len(labels)
    boot_f1: list[float] = []
    skipped_single_class = 0
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        lbl = labels[idx]
        if len(np.unique(lbl)) < 2:
            skipped_single_class += 1
            continue
        pr = (scores[idx] >= threshold).astype(int)
        boot_f1.append(float(f1_score(lbl, pr, zero_division=0.0)))

    arr = np.array(boot_f1)
    return {
        "point_estimate": point,
        "ci_low": float(np.percentile(arr, 2.5)),
        "ci_high": float(np.percentile(arr, 97.5)),
        "ci_method": "bootstrap_message_level_95",
        "bootstrap_requested": n_bootstrap,
        "bootstrap_skipped_single_class": skipped_single_class,
        "bootstrap_effective_n": int(len(arr)),
        "bootstrap_seed": seed,
    }


def load_score_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load per-message labels and composite scores with/without s_ling."""
    comp = np.load(COMPONENT_CACHE)
    labels = comp["labels"].astype(int)
    s_sem = comp["semantic"].astype(float)
    s_ling = comp["linguistic"].astype(float)
    s_temp = comp["temporal"].astype(float)

    scores_with = 0.40 * s_sem + 0.35 * s_ling + 0.25 * s_temp

    # Without s_ling: prefer held-out evaluate() scores (linguistic excluded)
    if COSINE_CACHE.exists():
        cos = np.load(COSINE_CACHE)
        if len(cos["labels"]) == len(labels) and np.array_equal(cos["labels"], labels):
            scores_without = cos["cosine"].astype(float)
        else:
            a_renorm = 0.40 / 0.65
            g_renorm = 0.25 / 0.65
            scores_without = a_renorm * s_sem + g_renorm * s_temp
    else:
        a_renorm = 0.40 / 0.65
        g_renorm = 0.25 / 0.65
        scores_without = a_renorm * s_sem + g_renorm * s_temp

    return labels, scores_with, scores_without


def metric_record(name: str, point: float, low: float, high: float, ci_method: str, **extra) -> dict:
    return {
        "metric": name,
        "point_estimate": point,
        "ci_low": low,
        "ci_high": high,
        "ci_method": ci_method,
        **extra,
    }


def main() -> None:
    labels, scores_with, scores_without = load_score_arrays()
    assert int(labels.sum()) == WITHOUT_SLING["n_pos"]

    f1_wos = bootstrap_f1_ci(labels, scores_without)
    f1_ws = bootstrap_f1_ci(labels, scores_with)

    recall_wos = clopper_pearson(WITHOUT_SLING["tp"], WITHOUT_SLING["n_pos"])
    prec_wos = clopper_pearson(WITHOUT_SLING["tp"], WITHOUT_SLING["tp"] + WITHOUT_SLING["fp"])

    results = {
        "methods": {
            "f1": "bootstrap message-level resampling (n=5000, seed=42)",
            "recall_precision": "scipy.stats.binomtest(...).proportion_ci(method='exact')",
        },
        "confidence_level": 0.95,
        "evaluation_context": {
            "dataset": "personachat_processed_corrected.json",
            "split": "80/20 per-user index",
            "n_test_messages": int(len(labels)),
            "n_test_positives": int(labels.sum()),
            "n_test_negatives": int(len(labels) - labels.sum()),
            "threshold": THRESHOLD,
            "score_sources": {
                "with_sling": "component composite 0.40/0.35/0.25 from sling_audit_component_scores.npz",
                "without_sling": "evaluate() composite from mahalanobis_comparison_scores.npz (cosine, linguistic excluded)",
            },
        },
        "metrics": {
            "f1_without_sling": {
                "metric": "f1_without_sling",
                **f1_wos,
                "note": "linguistic component excluded",
            },
            "f1_with_sling": {
                "metric": "f1_with_sling",
                **f1_ws,
                "note": "default composite with linguistic weight 0.35",
            },
            "recall_without_sling": metric_record(
                "recall_without_sling",
                recall_wos[0],
                recall_wos[1],
                recall_wos[2],
                "clopper_pearson_exact_95",
                tp=WITHOUT_SLING["tp"],
                n_positives=WITHOUT_SLING["n_pos"],
            ),
            "precision_without_sling": metric_record(
                "precision_without_sling",
                prec_wos[0],
                prec_wos[1],
                prec_wos[2],
                "clopper_pearson_exact_95",
                tp=WITHOUT_SLING["tp"],
                n_predicted_positive=WITHOUT_SLING["tp"] + WITHOUT_SLING["fp"],
            ),
        },
        "reporting_standard": (
            "F1: bootstrap 95% CI via message-level resampling. "
            "Recall/precision: exact 95% Clopper-Pearson. "
            "n=29 test positives is underpowered for generalization claims."
        ),
    }

    out = ROOT / "results" / "sling_fix_confidence_intervals.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
