#!/usr/bin/env python3
"""
Kimi recall-limitation diagnostics on corrected PersonaChat holdout.

1. Anomaly-type counts (ATO/SE/PI) before per-type recall; flag n < 15.
2. s_sem distance histogram (detected vs missed) — primary evidence; Pattern A/B/C.
3. Verified F1 improvement power analysis vs Kimi's asserted 40%/60%/80% table.

Usage:
    python scripts/kimi_recall_diagnostics.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest, mannwhitneyu, norm
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "datasets" / "personachat_processed_corrected.json"
COMPONENT_CACHE = ROOT / "results" / "sling_audit_component_scores.npz"
COSINE_CACHE = ROOT / "results" / "mahalanobis_comparison_scores.npz"
OUTPUT = ROOT / "results" / "methodology-diagnostics" / "kimi_recall_diagnostics.json"

THRESHOLD = 0.60
PER_TYPE_MIN_N = 15
BASELINE_F1 = 0.364  # CV pooled cosine, s_ling excluded (user-specified power baseline)
N_BOOTSTRAP_POWER = 5000
SEED = 42

TYPE_MAP = {
    "account_takeover": "ATO",
    "social_engineering": "SE",
    "prompt_injection": "PI",
}

# Kimi histogram framework (recall root-cause diagnostic):
#   A — threshold trade-off: missed positives overlap detected scores near τ; τ adjustment
#       recovers a material share of FNs without implying embedding failure.
#   B — representational limitation: missed positives occupy a lower, largely non-overlapping
#       s_sem region; τ alone cannot close the recall gap.
#   C — no semantic signal: both groups concentrated at low s_sem with negligible separation.
PATTERN_DEFINITIONS = {
    "A_threshold_tradeoff": (
        "Missed positives substantially overlap detected s_sem and/or cluster just below "
        "the composite operating threshold; lowering τ is expected to recover misses."
    ),
    "B_representational_limitation": (
        "Missed positives occupy a lower, largely non-overlapping s_sem region relative to "
        "detected positives; recall gap is primarily representational, not τ placement."
    ),
    "C_no_semantic_signal": (
        "Detected and missed positives both concentrate at low s_sem with negligible "
        "separation (flat/no signal)."
    ),
}

KIMI_ASSERTED_POWER = {
    0.05: 0.40,
    0.10: 0.60,
    0.15: 0.80,
}


def load_test_messages(dataset_path: Path) -> list[dict]:
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    by_user: dict[str, list[dict]] = {}
    for m in data["messages"]:
        by_user.setdefault(m["user_id"], []).append(m)
    for uid in by_user:
        by_user[uid].sort(key=lambda x: x["timestamp"])
    test_msgs: list[dict] = []
    for uid in sorted(by_user.keys()):
        msgs = by_user[uid]
        split = int(len(msgs) * 0.8)
        test_msgs.extend(msgs[split:])
    return test_msgs


def cohens_d(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    """
    Cohen's d using canonical n-weighted pooled standard deviation.

    pooled = sqrt(((n_a - 1) * s_a^2 + (n_b - 1) * s_b^2) / (n_a + n_b - 2))
    d = (mean_a - mean_b) / pooled
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return {
            "cohens_d": float("nan"),
            "method": "canonical_n_weighted_pooled_sd",
            "detected": {"n": na, "mean": float(a.mean()) if na else None, "std_ddof1": None},
            "missed": {"n": nb, "mean": float(b.mean()) if nb else None, "std_ddof1": None},
            "pooled_std": None,
            "mean_diff_detected_minus_missed": None,
        }

    sa = float(a.std(ddof=1))
    sb = float(b.std(ddof=1))
    pooled = math.sqrt(((na - 1) * sa**2 + (nb - 1) * sb**2) / (na + nb - 2))
    mean_diff = float(a.mean() - b.mean())
    d = mean_diff / pooled if pooled else 0.0

    return {
        "cohens_d": round(d, 2),
        "method": "canonical_n_weighted_pooled_sd",
        "detected": {
            "n": na,
            "mean": round(float(a.mean()), 3),
            "std_ddof1": round(sa, 3),
            "min": round(float(a.min()), 3),
            "max": round(float(a.max()), 3),
        },
        "missed": {
            "n": nb,
            "mean": round(float(b.mean()), 3),
            "std_ddof1": round(sb, 3),
            "min": round(float(b.min()), 3),
            "max": round(float(b.max()), 3),
        },
        "pooled_std": round(pooled, 3),
        "mean_diff_detected_minus_missed": round(mean_diff, 3),
    }


def histogram_bins(values: np.ndarray, bin_edges: np.ndarray) -> dict[str, list[int]]:
    counts, _ = np.histogram(values, bins=bin_edges)
    return {"edges": [round(float(x), 4) for x in bin_edges], "counts": [int(c) for c in counts]}


def classify_histogram_pattern(
    s_sem_detected: np.ndarray,
    s_sem_missed: np.ndarray,
    composite_missed: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Classify Kimi A/B/C from s_sem separation and composite near-miss mass."""
    if len(s_sem_detected) == 0 or len(s_sem_missed) == 0:
        return {"pattern": "indeterminate", "reason": "empty detected or missed group"}

    effect = cohens_d(s_sem_detected, s_sem_missed)
    d = effect["cohens_d"]
    if isinstance(d, float) and math.isnan(d):
        d = 0.0
    min_det = float(s_sem_detected.min())
    max_mis = float(s_sem_missed.max())
    overlap_count = int((s_sem_missed >= min_det).sum())
    overlap_frac = overlap_count / len(s_sem_missed)

    # Composite "near miss" band just below τ (primary operating score)
    near_miss_frac = float(((composite_missed >= threshold - 0.08) & (composite_missed < threshold)).mean())
    recoverable_at_055 = int((composite_missed >= 0.55).sum())

    both_low = float(s_sem_detected.mean()) < 0.25 and float(s_sem_missed.mean()) < 0.25
    flat = abs(d) < 0.3 and overlap_frac > 0.5

    if both_low and flat:
        pattern = "C"
    elif max_mis < min_det and overlap_frac < 0.15 and d > 0.5:
        # Primary: s_sem separation — representational
        pattern = "B"
    elif overlap_frac >= 0.35 or near_miss_frac >= 0.40:
        pattern = "A"
    elif near_miss_frac >= 0.25:
        pattern = "A"
    else:
        pattern = "B"

    return {
        "pattern": pattern,
        "pattern_label": {
            "A": "threshold_tradeoff",
            "B": "representational_limitation",
            "C": "no_semantic_signal",
        }[pattern],
        "cohens_d_s_sem_detected_minus_missed": effect["cohens_d"],
        "cohens_d": effect,
        "s_sem_detected_min_mean_max": [
            effect["detected"]["min"],
            effect["detected"]["mean"],
            effect["detected"]["max"],
        ],
        "s_sem_missed_min_mean_max": [
            effect["missed"]["min"],
            effect["missed"]["mean"],
            effect["missed"]["max"],
        ],
        "s_sem_overlap_fraction_missed_gte_min_detected": round(overlap_frac, 4),
        "s_sem_overlap_count": overlap_count,
        "composite_missed_near_tau_fraction": round(near_miss_frac, 4),
        "composite_missed_recoverable_if_tau_0_55": recoverable_at_055,
        "mann_whitney_u_pvalue_two_sided": round(
            float(mannwhitneyu(s_sem_detected, s_sem_missed, alternative="two-sided").pvalue),
            6,
        ),
        "definitions": PATTERN_DEFINITIONS,
    }


def recall_for_target_f1(f1_target: float, precision: float) -> float | None:
    """Solve F1 = 2PR/(P+R) for R given fixed precision."""
    if precision <= f1_target:
        return None
    # f1_target * (P + R) = 2PR  =>  R = f1_target*P / (2P - f1_target)
    denom = 2 * precision - f1_target
    if denom <= 0:
        return None
    return f1_target * precision / denom


def binomial_power_two_proportion(
    n: int,
    p0: float,
    p1: float,
    alpha: float = 0.05,
) -> float:
    """
    Approximate two-sided power for H0: p=p0 vs H1: p=p1 with n trials (Cochran-Armitage / normal).
    Uses pooled variance under H0 for critical value.
    """
    if n == 0 or p0 == p1:
        return alpha
    z_alpha = norm.ppf(1 - alpha / 2)
    se0 = math.sqrt(p0 * (1 - p0) / n)
    if se0 == 0:
        return 1.0 if p1 != p0 else alpha
    z_crit = z_alpha * se0
    # Power: P(|p_hat - p0| > z_crit) under p1
    se1 = math.sqrt(p1 * (1 - p1) / n)
    z1 = (p0 + z_crit - p1) / se1
    z2 = (p0 - z_crit - p1) / se1
    return float(1 - norm.cdf(z1) + norm.cdf(z2))


def simulate_f1_improvement_power(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    baseline_f1: float,
    delta_f1: float,
    n_sim: int,
    seed: int,
) -> dict[str, Any]:
    """
    Simulation power on positive class (n=29): resample positives with replacement;
    under H1 each FN is promoted to TP independently with probability q calibrated
    so expected F1 ≈ baseline + delta (FP held at observed count). One-sided test:
    observed recall > baseline recall (McNemar-style on positives).
    """
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    preds = (scores >= threshold).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    n_pos = tp + fn
    target_f1 = baseline_f1 + delta_f1

    def f1_from_tp(tp_k: int) -> float:
        fn_k = n_pos - tp_k
        prec = tp_k / (tp_k + fp) if (tp_k + fp) else 0.0
        rec = tp_k / n_pos if n_pos else 0.0
        if prec + rec == 0:
            return 0.0
        return 2 * prec * rec / (prec + rec)

    best_q, best_err = 0.0, float("inf")
    for q_int in range(0, 1001):
        q = q_int / 1000.0
        exp_tp = tp + q * fn
        f1_exp = f1_from_tp(int(round(exp_tp)))
        err = abs(f1_exp - target_f1)
        if err < best_err:
            best_err = err
            best_q = q

    p0 = tp / n_pos
    p1 = min(1.0, (tp + best_q * fn) / n_pos)
    rng = np.random.RandomState(seed)
    rejections = 0
    effective = 0
    for _ in range(n_sim):
        boot_tp = int(rng.binomial(n_pos, p1))
        if boot_tp <= tp:
            effective += 1
            continue
        # One-sided binomial test: is boot_tp significantly > tp under p0?
        pval = binomtest(boot_tp, n_pos, p0, alternative="greater").pvalue
        effective += 1
        if pval < 0.05:
            rejections += 1

    power = rejections / effective if effective else 0.0
    return {
        "target_f1": round(target_f1, 4),
        "promotion_probability_on_fn_positives": round(best_q, 4),
        "expected_additional_tps": round(best_q * fn, 2),
        "implied_target_recall_under_h1": round(p1, 4),
        "method": "positive_class_bootstrap_binomial_test_n_sim",
        "n_sim": n_sim,
        "power_one_sided_alpha_0_05": round(power, 4),
    }


def recall_power_analysis(n_pos: int, tp_baseline: int, precision: float, baseline_f1: float) -> list[dict]:
    rows = []
    for delta in (0.05, 0.10, 0.15):
        target_f1 = baseline_f1 + delta
        target_r = recall_for_target_f1(target_f1, precision)
        if target_r is None:
            rows.append({"delta_f1": delta, "error": "infeasible at fixed precision"})
            continue
        target_tp = int(round(target_r * n_pos))
        target_tp = max(tp_baseline + 1, min(n_pos, target_tp))
        p0 = tp_baseline / n_pos
        p1 = target_tp / n_pos
        power = binomial_power_two_proportion(n_pos, p0, p1)
        rows.append(
            {
                "delta_f1": delta,
                "target_f1": round(target_f1, 4),
                "assumed_fixed_precision": round(precision, 4),
                "implied_target_recall": round(target_tp / n_pos, 4),
                "implied_target_tp": target_tp,
                "baseline_tp": tp_baseline,
                "method": "two_proportion_normal_approximation_on_recall_n_pos=29",
                "alpha": 0.05,
                "two_sided": True,
                "verified_power": round(power, 4),
                "kimi_asserted_power": KIMI_ASSERTED_POWER[delta],
                "absolute_error_vs_kimi": round(power - KIMI_ASSERTED_POWER[delta], 4),
            }
        )
    return rows


def main() -> None:
    test_msgs = load_test_messages(DEFAULT_DATASET)
    comp = np.load(COMPONENT_CACHE)
    cos = np.load(COSINE_CACHE)
    labels = comp["labels"].astype(int)
    s_sem_all = comp["semantic"].astype(float)
    composite = cos["cosine"].astype(float)
    assert len(test_msgs) == len(labels) == 10165

    preds = (composite >= THRESHOLD).astype(int)
    pos_indices = [i for i, m in enumerate(test_msgs) if m.get("should_flag")]

    # --- Type breakdown (counts BEFORE recall) ---
    type_counts_raw: dict[str, int] = {"ATO": 0, "SE": 0, "PI": 0}
    for i in pos_indices:
        raw = test_msgs[i].get("anomaly_type", "unknown")
        short = TYPE_MAP.get(raw, raw)
        type_counts_raw[short] = type_counts_raw.get(short, 0) + 1

    per_type_recall: dict[str, Any] = {}
    for t in ("ATO", "SE", "PI"):
        n_t = type_counts_raw[t]
        idx_t = [
            i
            for i in pos_indices
            if TYPE_MAP.get(test_msgs[i].get("anomaly_type"), test_msgs[i].get("anomaly_type")) == t
        ]
        tp_t = int(sum(preds[i] for i in idx_t))
        per_type_recall[t] = {
            "n_positives": n_t,
            "tp": tp_t,
            "fn": n_t - tp_t,
            "recall": round(tp_t / n_t, 4) if n_t else None,
            "underpowered_per_type_recall": n_t < PER_TYPE_MIN_N,
            "interpretation": (
                "suggestive_only_not_confirmatory"
                if n_t < PER_TYPE_MIN_N
                else "sample_adequate_for_exploratory_recall"
            ),
        }

    any_type_under_15 = any(type_counts_raw[t] < PER_TYPE_MIN_N for t in ("ATO", "SE", "PI"))

    # --- s_sem histogram diagnostic ---
    det_idx = [i for i in pos_indices if preds[i] == 1]
    mis_idx = [i for i in pos_indices if preds[i] == 0]
    s_det = s_sem_all[det_idx]
    s_mis = s_sem_all[mis_idx]
    c_mis = composite[mis_idx]

    bin_edges = np.linspace(0.0, 1.0, 21)
    pattern = classify_histogram_pattern(s_det, s_mis, c_mis, THRESHOLD)

    positive_records = []
    for i in pos_indices:
        m = test_msgs[i]
        positive_records.append(
            {
                "message_index": i,
                "user_id": m["user_id"],
                "anomaly_type": TYPE_MAP.get(m.get("anomaly_type"), m.get("anomaly_type")),
                "s_sem": round(float(s_sem_all[i]), 4),
                "composite_score": round(float(composite[i]), 4),
                "detected_at_tau": bool(preds[i]),
            }
        )

    # Tau sweep on composite for missed recovery (supplementary)
    tau_grid = np.arange(0.40, 0.601, 0.01)
    sweep = []
    for t in tau_grid:
        pr = (composite >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(labels, pr, labels=[0, 1]).ravel()
        sweep.append(
            {
                "tau": round(float(t), 2),
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "tn": int(tn),
                "f1": round(float(f1_score(labels, pr, zero_division=0)), 4),
            }
        )

    # --- Power analysis ---
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    prec_holdout = precision_score(labels, preds, zero_division=0)
    f1_holdout = f1_score(labels, preds, zero_division=0)

    recall_power_cv = recall_power_analysis(29, 8, 0.533, BASELINE_F1)
    recall_power_holdout = recall_power_analysis(29, tp, prec_holdout, f1_holdout)

    sim_power = []
    for delta in (0.05, 0.10, 0.15):
        sim_power.append(
            {
                "delta_f1": delta,
                **simulate_f1_improvement_power(
                    labels, composite, THRESHOLD, BASELINE_F1, delta, N_BOOTSTRAP_POWER, SEED + int(delta * 100)
                ),
                "kimi_asserted_power": KIMI_ASSERTED_POWER[delta],
            }
        )

    paper_framing = (
        "threshold_tradeoff"
        if pattern["pattern"] == "A"
        else "representational_limitation_deferred_to_v2"
        if pattern["pattern"] == "B"
        else "representational_limitation_deferred_to_v2"
    )
    if pattern["pattern"] == "A" and pattern.get("composite_missed_near_tau_fraction", 0) >= 0.25:
        paper_framing = "threshold_tradeoff_with_composite_near_misses"

    out = {
        "evaluation_context": {
            "dataset": str(DEFAULT_DATASET.name),
            "split": "80/20 per-user chronological holdout",
            "n_test_messages": 10165,
            "n_test_positives": 29,
            "threshold": THRESHOLD,
            "scoring": "composite cosine evaluate(), s_ling excluded",
            "baseline_f1_for_power": BASELINE_F1,
            "holdout_f1_at_tau": round(float(f1_holdout), 4),
            "confusion_at_tau": {"tp": tp, "fp": fp, "fn": 29 - tp},
        },
        "anomaly_type_breakdown": {
            "counts_before_recall": type_counts_raw,
            "total": 29,
            "any_type_under_15_examples": any_type_under_15,
            "types_under_15": [t for t in ("ATO", "SE", "PI") if type_counts_raw[t] < PER_TYPE_MIN_N],
            "flag": "per_type_recall_suggestive_only_not_confirmatory",
            "per_type_recall_secondary_only": per_type_recall,
        },
        "distance_histogram_primary": {
            "score": "s_sem (semantic component, continuous)",
            "n_detected": len(det_idx),
            "n_missed": len(mis_idx),
            "histogram_detected": histogram_bins(s_det, bin_edges),
            "histogram_missed": histogram_bins(s_mis, bin_edges),
            "per_positive_records": positive_records,
            "pattern_classification": pattern,
            "composite_tau_sweep_supplementary": sweep,
        },
        "power_analysis": {
            "kimi_asserted_table": KIMI_ASSERTED_POWER,
            "method_1_recall_two_proportion": {
                "description": (
                    "Map ΔF1 to implied recall at fixed CV precision P=0.533; "
                    "two-sided normal-approximation power on n=29 positives (TP/29)."
                ),
                "baseline_f1": BASELINE_F1,
                "rows": recall_power_cv,
            },
            "method_1b_recall_holdout_precision": {
                "description": "Same recall power using holdout precision at τ=0.60.",
                "baseline_f1": round(float(f1_holdout), 4),
                "rows": recall_power_holdout,
            },
            "method_2_simulation": {
                "description": (
                    "Positive-class bootstrap + one-sided exact binomial test (n_sim=5000): "
                    "promote FN positives with probability q calibrated to target F1≈baseline+Δ."
                ),
                "rows": sim_power,
            },
            "verification_summary": [
                {
                    "delta_f1": d,
                    "kimi_asserted": KIMI_ASSERTED_POWER[d],
                    "verified_recall_power": next(r["verified_power"] for r in recall_power_cv if r["delta_f1"] == d),
                    "verified_simulation_power": next(
                        r["power_one_sided_alpha_0_05"] for r in sim_power if r["delta_f1"] == d
                    ),
                }
                for d in (0.05, 0.10, 0.15)
            ],
        },
        "paper_framing_recommendation": {
            "primary_evidence": "s_sem distance histogram (Pattern {})".format(pattern["pattern"]),
            "secondary_evidence": "anomaly-type breakdown (underpowered per-type)",
            "recommended_wording": paper_framing,
            "rationale": pattern,
        },
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"\nSaved to {OUTPUT}")


if __name__ == "__main__":
    main()
