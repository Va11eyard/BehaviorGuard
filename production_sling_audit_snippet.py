#!/usr/bin/env python3
"""
Production-path s_ling audit snippet — paste into evaluation.py

Purpose:
  Determine whether s_ling (linguistic component score) in CompositeScorer's
  fixed-weight composite (0.4 semantic / 0.35 linguistic / 0.25 temporal) is:
    (a) harmless constant offset → skip fix
    (b) random noise → fix by zeroing weight
    (c) real signal → keep, contradicts saturation concern

Method:
  1. Statistics: mean/std/correlation of s_ling for organic vs anomalous
  2. Direct comparison: composite AUC/F1 WITH s_ling (0.35) vs WITHOUT
     s_ling (weight 0, semantic/temporal renormalized to sum to 1.0)
  3. Bootstrap 95% CI on AUC difference

Output: three-branch decision printed to stdout.

INSERTION POINT in evaluation.py:
  After running evaluate() on all test messages and collecting
  EvaluationResult objects. Find where component_scores are accessible.
"""

import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

# Canonical binary classification threshold in evaluation.py (Table III / τ₂).
# evaluate_method() sets predicted_label = score > 0.60; NOT 0.5.
PRODUCTION_CLASSIFICATION_THRESHOLD = 0.60

# =============================================================================
# INSERT THIS BLOCK INTO YOUR evaluate() LOOP
# =============================================================================

def run_sling_production_audit(
    component_scores: list[dict],  # List of {"semantic": float, "linguistic": float, "temporal": float}
    labels: list[int],             # 0 = organic, 1 = anomalous
    threshold: float = PRODUCTION_CLASSIFICATION_THRESHOLD,
    n_bootstrap: int = 1000,       # Bootstrap samples for AUC difference CI
    seed: int = 42,
    quiet: bool = False,
) -> dict:
    """
    Run the production-path s_ling audit.

    Args:
        component_scores: List of component score dicts from CompositeScorer
        labels: Binary labels (0=organic, 1=anomalous)
        threshold: Threshold for binary classification (F1 computation).
            Use PRODUCTION_CLASSIFICATION_THRESHOLD (0.60) to match evaluation.py.
        n_bootstrap: Number of bootstrap resamples for AUC difference CI
        seed: Random seed for bootstrap

    Returns:
        dict with all audit results and a decision string
    """
    rng = np.random.RandomState(seed)
    labels_arr = np.array(labels, dtype=int)
    organic_mask = labels_arr == 0
    anomalous_mask = labels_arr == 1

    s_sem = np.array([c["semantic"] for c in component_scores], dtype=float)
    s_ling = np.array([c["linguistic"] for c in component_scores], dtype=float)
    s_temp = np.array([c["temporal"] for c in component_scores], dtype=float)

    n_total = len(labels_arr)
    n_organic = int(np.sum(organic_mask))
    n_anomalous = int(np.sum(anomalous_mask))

    # =====================================================================
    # 1. DESCRIPTIVE STATISTICS
    # =====================================================================

    ling_org = s_ling[organic_mask]
    ling_anom = s_ling[anomalous_mask]

    stats = {
        "organic_mean": float(np.mean(ling_org)),
        "organic_std": float(np.std(ling_org, ddof=1)),
        "anomalous_mean": float(np.mean(ling_anom)),
        "anomalous_std": float(np.std(ling_anom, ddof=1)),
        "mean_diff": float(abs(np.mean(ling_org) - np.mean(ling_anom))),
        "pooled_std": float(np.std(s_ling, ddof=1)),
        "correlation_with_label": float(
            np.corrcoef(s_ling, labels_arr)[0, 1] if np.std(s_ling) > 1e-10 else 0.0
        ),
    }

    # =====================================================================
    # 2. COMPOSITE SCORES: WITH vs WITHOUT s_ling
    # =====================================================================

    # Normal production composite (fixed weights)
    composite_with = 0.40 * s_sem + 0.35 * s_ling + 0.25 * s_temp

    # s_ling excluded: semantic/temporal renormalized to sum to 1.0
    # 0.40 / (0.40 + 0.25) = 0.6154
    # 0.25 / (0.40 + 0.25) = 0.3846
    a_renorm = 0.40 / 0.65
    g_renorm = 0.25 / 0.65
    composite_without = a_renorm * s_sem + 0.0 * s_ling + g_renorm * s_temp

    def _score_composite(scores, lbls, thresh):
        """Compute AUC, F1, precision, recall for a composite score vector."""
        auc = roc_auc_score(lbls, scores)
        preds = (scores >= thresh).astype(int)
        f1 = f1_score(lbls, preds, zero_division=0.0)
        prec = precision_score(lbls, preds, zero_division=0.0)
        rec = recall_score(lbls, preds, zero_division=0.0)
        return {"auc": auc, "f1": f1, "precision": prec, "recall": rec}

    metrics_with = _score_composite(composite_with, labels_arr, threshold)
    metrics_without = _score_composite(composite_without, labels_arr, threshold)

    auc_diff = metrics_with["auc"] - metrics_without["auc"]
    f1_diff = metrics_with["f1"] - metrics_without["f1"]

    # =====================================================================
    # 3. BOOTSTRAP 95% CI ON AUC DIFFERENCE
    # =====================================================================

    boot_diffs = []
    n_skipped_single_class = 0
    n_skipped_auc_error = 0
    for _ in range(n_bootstrap):
        idx = rng.choice(n_total, size=n_total, replace=True)
        s_sem_b, s_ling_b, s_temp_b, lbls_b = s_sem[idx], s_ling[idx], s_temp[idx], labels_arr[idx]

        # Skip resamples with only one class (AUC undefined)
        if len(np.unique(lbls_b)) < 2:
            n_skipped_single_class += 1
            continue

        comp_with_b = 0.40 * s_sem_b + 0.35 * s_ling_b + 0.25 * s_temp_b
        comp_without_b = a_renorm * s_sem_b + g_renorm * s_temp_b

        try:
            auc_with_b = roc_auc_score(lbls_b, comp_with_b)
            auc_without_b = roc_auc_score(lbls_b, comp_without_b)
            boot_diffs.append(auc_with_b - auc_without_b)
        except ValueError:
            n_skipped_auc_error += 1
            continue

    boot_diffs = np.array(boot_diffs)
    bootstrap_effective_n = int(len(boot_diffs))
    ci_low = float(np.percentile(boot_diffs, 2.5))
    ci_high = float(np.percentile(boot_diffs, 97.5))
    ci_width = ci_high - ci_low
    ci_contains_zero = ci_low <= 0 <= ci_high

    # =====================================================================
    # 4. THREE-BRANCH DECISION
    # =====================================================================

    decision = None
    decision_explanation = None

    if ci_contains_zero:
        decision = "SKIP_FIX"
        decision_explanation = (
            "AUC difference includes zero at 95% confidence. "
            "s_ling is statistically harmless — constant offset or noise "
            "that does not affect ranking performance. "
            "Move to Mahalanobis distance implementation."
        )
    elif auc_diff > 0 and not ci_contains_zero:
        decision = "CONTRADICTS_SATURATION"
        decision_explanation = (
            "AUC is meaningfully HIGHER with s_ling included. "
            "s_ling contributes real discriminative signal — saturation concern "
            "is contradicted by data. Keep s_ling at 0.35 weight."
        )
    elif auc_diff < 0 and not ci_contains_zero:
        decision = "FIX_IT"
        decision_explanation = (
            "AUC is meaningfully HIGHER with s_ling EXCLUDED. "
            "s_ling degrades composite separability. "
            "Worth implementing exclusion (set weight 0, renormalize)."
        )

    # =====================================================================
    # 5. FORMATTED OUTPUT
    # =====================================================================

    if not quiet:
        print("\n" + "=" * 78)
        print("PRODUCTION-PATH s_ling AUDIT")
        print("=" * 78)
        print(f"\nSample: {n_total} messages ({n_organic} organic, {n_anomalous} anomalous)")
        print(f"Threshold: {threshold}")
        print(f"Bootstrap: {n_bootstrap} resamples (seed={seed})")

        print("\n--- 1. s_ling DESCRIPTIVE STATISTICS ---")
        print(f"  Organic:    mean={stats['organic_mean']:.6f}, std={stats['organic_std']:.6f}")
        print(f"  Anomalous:  mean={stats['anomalous_mean']:.6f}, std={stats['anomalous_std']:.6f}")
        print(f"  Mean diff:  {stats['mean_diff']:.6f}")
        print(f"  Correlation with label: {stats['correlation_with_label']:+.6f}")
        if stats['correlation_with_label'] > 0.1:
            print("  >>> s_ling correlates positively with anomaly label — real signal")
        elif stats['correlation_with_label'] < -0.1:
            print("  >>> s_ling correlates negatively with anomaly label — unexpected, investigate")
        else:
            print("  >>> s_ling is label-uncorrelated — constant offset or random noise")

        print("\n--- 2. COMPOSITE PERFORMANCE: WITH vs WITHOUT s_ling ---")
        print(f"\n  WITH s_ling (0.40/0.35/0.25):")
        print(f"    AUC = {metrics_with['auc']:.6f}")
        print(f"    F1  = {metrics_with['f1']:.6f}  (P={metrics_with['precision']:.4f}, R={metrics_with['recall']:.4f})")

        print(f"\n  WITHOUT s_ling ({a_renorm:.4f}/{0.0:.4f}/{g_renorm:.4f}):")
        print(f"    AUC = {metrics_without['auc']:.6f}")
        print(f"    F1  = {metrics_without['f1']:.6f}  (P={metrics_without['precision']:.4f}, R={metrics_without['recall']:.4f})")

        print(f"\n  AUC difference (with - without): {auc_diff:+.6f}")
        print(f"  F1  difference (with - without): {f1_diff:+.6f}")
        print(f"  95% CI on AUC difference: [{ci_low:+.6f}, {ci_high:+.6f}]")
        print(f"  CI width: {ci_width:.6f}")
        print(f"  Bootstrap resamples requested: {n_bootstrap}")
        print(f"  Skipped (single-class draw): {n_skipped_single_class}")
        print(f"  Skipped (AUC error): {n_skipped_auc_error}")
        print(f"  Effective resamples (both classes): {bootstrap_effective_n}")
        print(f"  CI contains zero: {'YES (not significant)' if ci_contains_zero else 'NO (significant)'}")

        print("\n--- 3. DECISION ---")
        print(f"  [{decision}]")
        print(f"  {decision_explanation}")
        print("=" * 78)

    return {
        "statistics": stats,
        "metrics_with_sling": metrics_with,
        "metrics_without_sling": metrics_without,
        "auc_difference": auc_diff,
        "f1_difference": f1_diff,
        "bootstrap_ci": {"low": ci_low, "high": ci_high, "width": ci_width},
        "bootstrap_requested": n_bootstrap,
        "bootstrap_skipped_single_class": n_skipped_single_class,
        "bootstrap_skipped_auc_error": n_skipped_auc_error,
        "bootstrap_effective_n": bootstrap_effective_n,
        "ci_contains_zero": ci_contains_zero,
        "decision": decision,
        "decision_explanation": decision_explanation,
        "n_messages": n_total,
        "n_organic": n_organic,
        "n_anomalous": n_anomalous,
    }


# =============================================================================
# INSERTION INSTRUCTIONS FOR evaluation.py
# =============================================================================

"""
FIND the section in your evaluation.py where EvaluationResult objects are
collected after running evaluate() on test messages. You need access to:
  - results: list of EvaluationResult (or equivalent)
  - labels: list[int] of ground-truth labels

INSERT the following (adapt variable names as needed):

--- CUT HERE ---

# === s_ling PRODUCTION AUDIT (insert after results collection) ===
from production_sling_audit_snippet import run_sling_production_audit

# Extract component scores from your results
audit_component_scores = []
for r in results:  # adapt: your result variable name
    audit_component_scores.append({
        "semantic": r.component_scores["semantic"],      # adapt key names if different
        "linguistic": r.component_scores["linguistic"],
        "temporal": r.component_scores["temporal"],
    })

audit_labels = [r.label for r in results]  # adapt: ground truth labels

sling_audit = run_sling_production_audit(
    component_scores=audit_component_scores,
    labels=audit_labels,
    threshold=PRODUCTION_CLASSIFICATION_THRESHOLD,  # 0.60 in evaluation.py
    n_bootstrap=1000,
    seed=42,
)

# Save if desired
with open("sling_audit_results.json", "w") as f:
    json.dump(sling_audit, f, indent=2)

# === END s_ling PRODUCTION AUDIT ===

--- CUT HERE ---

IMPORTANT: The snippet reads component_scores["semantic"], etc. If your
EvaluationResult uses different keys (e.g., "s_sem", "s_ling", "s_temp"),
change the key names in the audit_component_scores construction block.

The threshold parameter should match evaluation.py binary classification:
PRODUCTION_CLASSIFICATION_THRESHOLD (0.60), i.e. predicted_label = score > 0.60.
This is τ₂ from the paper; NOT 0.5.
"""


def _build_component_score_list(
    s_sem: np.ndarray,
    s_ling: np.ndarray,
    s_temp: np.ndarray,
) -> list[dict]:
    return [
        {"semantic": float(s), "linguistic": float(l), "temporal": float(t)}
        for s, l, t in zip(s_sem, s_ling, s_temp)
    ]


def _run_self_test_case(
    name: str,
    s_sem: np.ndarray,
    s_ling: np.ndarray,
    s_temp: np.ndarray,
    labels: list[int],
    expected_decision: str | None,
    forbidden_decision: str | None = None,
    *,
    verbose: bool = False,
    n_bootstrap: int = 800,
) -> dict:
    """Run one synthetic scenario; optional assert on decision branch."""
    component_scores = _build_component_score_list(s_sem, s_ling, s_temp)
    if verbose:
        result = run_sling_production_audit(
            component_scores=component_scores,
            labels=labels,
            n_bootstrap=n_bootstrap,
        )
    else:
        with redirect_stdout(io.StringIO()):
            result = run_sling_production_audit(
                component_scores=component_scores,
                labels=labels,
                n_bootstrap=n_bootstrap,
                quiet=True,
            )

    decision = result["decision"]
    status = "PASS"
    if expected_decision is not None and decision != expected_decision:
        status = "FAIL"
    if forbidden_decision is not None and decision == forbidden_decision:
        status = "FAIL"

    print(f"  [{status}] {name}: decision={decision}", end="")
    if expected_decision:
        print(f" (expected {expected_decision})", end="")
    if forbidden_decision:
        print(f" (must not be {forbidden_decision})", end="")
    print()

    if status == "FAIL":
        raise AssertionError(
            f"{name}: got {decision}, expected={expected_decision}, forbidden={forbidden_decision}"
        )
    return result


def collect_personachat_production_scores(
    dataset_path: Path | None = None,
    lambda_decay: float = 0.50,
) -> tuple[list[dict], list[int]]:
    """
    Collect component scores from evaluate() on corrected PersonaChat test split.

    Uses the same 80/20 per-user protocol as corrected_proper_generalization_eval.py.
    """
    root = Path(__file__).resolve().parent
    if dataset_path is None:
        dataset_path = root / "datasets" / "personachat_processed_corrected.json"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Corrected PersonaChat dataset not found: {dataset_path}")

    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "src"))

    import evaluation as ev  # noqa: E402
    from turnshift import TurnShiftEvaluatorML  # noqa: E402
    from turnshift.models import EvaluationInput, SystemConfig  # noqa: E402

    test_data = json.loads(dataset_path.read_text(encoding="utf-8"))
    evaluator = TurnShiftEvaluatorML()
    config = SystemConfig(
        sensitivity_level="medium",
        deployment_context="enterprise",
        overrides_enabled=False,
    )
    builder = ev._build_profile_with_pm(lambda_decay)

    by_user: dict[str, list] = {}
    for m in test_data["messages"]:
        by_user.setdefault(m["user_id"], []).append(m)
    for uid in by_user:
        by_user[uid].sort(key=lambda x: x["timestamp"])
    users = {u["user_id"]: u for u in test_data["users"]}

    component_scores: list[dict] = []
    labels: list[int] = []

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
            cs = result.component_scores
            component_scores.append(
                {
                    "semantic": float(cs.semantic),
                    "linguistic": float(cs.linguistic),
                    "temporal": float(cs.temporal),
                }
            )
            labels.append(1 if msg.get("should_flag", False) else 0)

    return component_scores, labels


SCORES_CACHE = Path(__file__).resolve().parent / "results" / "sling_audit_component_scores.npz"


def save_component_scores_cache(
    component_scores: list[dict],
    labels: list[int],
    path: Path | None = None,
) -> Path:
    path = path or SCORES_CACHE
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        semantic=np.array([c["semantic"] for c in component_scores], dtype=float),
        linguistic=np.array([c["linguistic"] for c in component_scores], dtype=float),
        temporal=np.array([c["temporal"] for c in component_scores], dtype=float),
        labels=np.array(labels, dtype=int),
    )
    return path


def load_component_scores_cache(path: Path | None = None) -> tuple[list[dict], list[int]]:
    path = path or SCORES_CACHE
    data = np.load(path)
    component_scores = [
        {"semantic": float(s), "linguistic": float(l), "temporal": float(t)}
        for s, l, t in zip(data["semantic"], data["linguistic"], data["temporal"])
    ]
    labels = data["labels"].astype(int).tolist()
    return component_scores, labels


def run_bootstrap_stability_check(
    component_scores: list[dict],
    labels: list[int],
    n_bootstrap_values: list[int] | None = None,
) -> dict:
    """Compare bootstrap CI width across resample counts on fixed component scores."""
    if n_bootstrap_values is None:
        n_bootstrap_values = [1000, 5000]
    results = {}
    for n in n_bootstrap_values:
        audit = run_sling_production_audit(
            component_scores=component_scores,
            labels=labels,
            n_bootstrap=n,
            quiet=True,
        )
        results[str(n)] = {
            "bootstrap_requested": audit["bootstrap_requested"],
            "bootstrap_skipped_single_class": audit["bootstrap_skipped_single_class"],
            "bootstrap_skipped_auc_error": audit["bootstrap_skipped_auc_error"],
            "bootstrap_effective_n": audit["bootstrap_effective_n"],
            "ci_low": audit["bootstrap_ci"]["low"],
            "ci_high": audit["bootstrap_ci"]["high"],
            "ci_width": audit["bootstrap_ci"]["width"],
        }
    return results


def run_personachat_production_audit(
    dataset_path: Path | None = None,
    lambda_decay: float = 0.50,
    n_bootstrap: int = 1000,
    output_path: Path | None = None,
) -> dict:
    """Run production-path s_ling audit on corrected PersonaChat evaluate() scores."""
    print(f"Collecting component scores from evaluate() (lambda={lambda_decay})...")
    component_scores, labels = collect_personachat_production_scores(
        dataset_path=dataset_path,
        lambda_decay=lambda_decay,
    )
    print(f"Collected {len(labels)} test messages "
          f"({sum(l == 0 for l in labels)} organic, {sum(l == 1 for l in labels)} anomalous)")

    cache_path = save_component_scores_cache(component_scores, labels)

    audit = run_sling_production_audit(
        component_scores=component_scores,
        labels=labels,
        threshold=PRODUCTION_CLASSIFICATION_THRESHOLD,
        n_bootstrap=n_bootstrap,
    )

    if output_path is None:
        output_path = Path(__file__).resolve().parent / "results" / "methodology-diagnostics" / "sling_production_audit_personachat.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"\nSaved audit JSON to {output_path}")
    print(f"Saved component-score cache to {cache_path}")
    return audit


# =============================================================================
# STANDALONE TEST (for validation only — uses synthetic data)
# =============================================================================

if __name__ == "__main__":
    run_real = "--real" in sys.argv
    run_bootstrap = "--bootstrap-stability" in sys.argv

    if run_bootstrap:
        if not SCORES_CACHE.exists():
            print(f"Cache missing at {SCORES_CACHE}; collecting scores first...")
            collect_personachat_production_scores()
            # collect via full audit path to also save cache
            run_personachat_production_audit(n_bootstrap=1000)
        component_scores, labels = load_component_scores_cache()
        print(f"Bootstrap stability check on {len(labels)} messages "
              f"({sum(l == 0 for l in labels)} organic, {sum(l == 1 for l in labels)} anomalous)")
        stability = run_bootstrap_stability_check(component_scores, labels, [1000, 5000])
        for n, stats in stability.items():
            print(f"\n  n_bootstrap={n}:")
            print(f"    skipped (single-class): {stats['bootstrap_skipped_single_class']}")
            print(f"    skipped (AUC error):    {stats['bootstrap_skipped_auc_error']}")
            print(f"    effective resamples:    {stats['bootstrap_effective_n']}")
            print(f"    CI: [{stats['ci_low']:+.6f}, {stats['ci_high']:+.6f}]  width={stats['ci_width']:.6f}")
        out = Path(__file__).resolve().parent / "results" / "methodology-diagnostics" / "sling_bootstrap_stability.json"
        out.write_text(json.dumps(stability, indent=2), encoding="utf-8")
        print(f"\nSaved to {out}")
    elif not run_real:
        print("Self-test mode: synthetic component scores")
        print(f"Production F1 threshold: {PRODUCTION_CLASSIFICATION_THRESHOLD} (evaluation.py)")
        print("Pass --real to audit corrected PersonaChat via evaluate()\n")

        rng = np.random.RandomState(42)
        n_org, n_anom = 150, 50
        labels_base = [0] * n_org + [1] * n_anom

        # Case 1: constant offset (saturated s_ling, same for both classes)
        s_sem_1 = np.concatenate([rng.normal(0.2, 0.1, n_org), rng.normal(0.6, 0.15, n_anom)])
        s_ling_1 = np.concatenate([rng.normal(0.95, 0.02, n_org), rng.normal(0.95, 0.02, n_anom)])
        s_temp_1 = np.concatenate([rng.normal(0.3, 0.1, n_org), rng.normal(0.4, 0.1, n_anom)])

        print("--- Synthetic self-tests ---")
        _run_self_test_case(
            "constant_offset",
            s_sem_1,
            s_ling_1,
            s_temp_1,
            labels_base,
            expected_decision="SKIP_FIX",
            verbose=True,
        )

        # Case 2: label-uncorrelated noise (uniform random s_ling)
        s_sem_2 = np.concatenate([rng.normal(0.2, 0.1, n_org), rng.normal(0.6, 0.15, n_anom)])
        s_ling_2 = rng.uniform(0.0, 1.0, n_org + n_anom)
        s_temp_2 = np.concatenate([rng.normal(0.3, 0.1, n_org), rng.normal(0.4, 0.1, n_anom)])

        _run_self_test_case(
            "label_uncorrelated_noise",
            s_sem_2,
            s_ling_2,
            s_temp_2,
            labels_base,
            expected_decision=None,
            forbidden_decision="CONTRADICTS_SATURATION",
        )

        # Case 3: genuine label correlation (s_ling is the primary signal)
        s_sem_3 = np.concatenate([rng.normal(0.35, 0.05, n_org), rng.normal(0.35, 0.05, n_anom)])
        s_ling_3 = np.concatenate([rng.normal(0.2, 0.08, n_org), rng.normal(0.7, 0.08, n_anom)])
        s_temp_3 = np.concatenate([rng.normal(0.35, 0.05, n_org), rng.normal(0.35, 0.05, n_anom)])

        _run_self_test_case(
            "genuine_label_correlation",
            s_sem_3,
            s_ling_3,
            s_temp_3,
            labels_base,
            expected_decision="CONTRADICTS_SATURATION",
        )

        print("\nAll synthetic self-tests passed.")
    else:
        run_personachat_production_audit()
