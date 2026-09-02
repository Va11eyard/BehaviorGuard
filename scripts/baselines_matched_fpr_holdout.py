#!/usr/bin/env python3
"""
Fair full-holdout baseline comparison at matched FPR with significance tests.

Reports AUC-ROC, AUC-PR, and detection rate at FPR in {1%, 5%} for:
  BehaviorGuard (linguistic off), IsolationForest, Autoencoder, rule-based.

Significance: paired bootstrap difference-in-AUC (percentile CI) + DeLong test
when available; Holm-Bonferroni across BG-vs-baseline comparisons.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

SEED = 42
N_BOOT = 2000
DATASET = ROOT / "datasets" / "personachat_processed_corrected.json"
TARGET_FPRS = [0.01, 0.05]


def holm_bonferroni(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        adj = (m - rank) * pvals[idx]
        running = max(running, adj)
        adjusted[idx] = min(1.0, running)
    return adjusted


def delong_pvalue(y_true: np.ndarray, s1: np.ndarray, s2: np.ndarray) -> float:
    """Approximate DeLong p-value for AUC(s1)-AUC(s2); falls back to NaN."""
    try:
        from scipy import stats

        # Structural components (Sun & Xu / DeLong)
        pos = s1[y_true == 1]
        neg = s1[y_true == 0]
        pos2 = s2[y_true == 1]
        neg2 = s2[y_true == 0]
        if len(pos) < 2 or len(neg) < 2:
            return float("nan")

        def v10(scores_pos, scores_neg):
            return np.array([(scores_pos[i] > scores_neg).mean() for i in range(len(scores_pos))])

        def v01(scores_pos, scores_neg):
            return np.array([(scores_pos > scores_neg[j]).mean() for j in range(len(scores_neg))])

        v10_1, v10_2 = v10(pos, neg), v10(pos2, neg2)
        v01_1, v01_2 = v01(pos, neg), v01(pos2, neg2)
        auc1 = roc_auc_score(y_true, s1)
        auc2 = roc_auc_score(y_true, s2)
        s10 = np.cov(np.vstack([v10_1, v10_2]))
        s01 = np.cov(np.vstack([v01_1, v01_2]))
        var = s10[0, 0] / len(pos) + s01[0, 0] / len(neg)
        var2 = s10[1, 1] / len(pos) + s01[1, 1] / len(neg)
        cov = s10[0, 1] / len(pos) + s01[0, 1] / len(neg)
        se = np.sqrt(max(var + var2 - 2 * cov, 1e-12))
        z = (auc1 - auc2) / se
        return float(2 * (1 - stats.norm.cdf(abs(z))))
    except Exception:
        return float("nan")


def bootstrap_auc_diff(y: np.ndarray, a: np.ndarray, b: np.ndarray) -> dict:
    rng = np.random.default_rng(SEED)
    diffs = []
    n = len(y)
    for _ in range(N_BOOT):
        idx = rng.choice(n, n, replace=True)
        yt = y[idx]
        if len(np.unique(yt)) < 2:
            continue
        diffs.append(roc_auc_score(yt, a[idx]) - roc_auc_score(yt, b[idx]))
    arr = np.asarray(diffs)
    return {
        "diff_mean": float(np.mean(arr)),
        "ci95_percentile": [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))],
        "p_two_sided_approx": float(2 * min(np.mean(arr <= 0), np.mean(arr >= 0))),
    }


def recall_at_fpr(y: np.ndarray, scores: np.ndarray, target: float) -> dict:
    fpr, tpr, thr = roc_curve(y, scores)
    # largest threshold with fpr <= target
    ok = np.where(fpr <= target)[0]
    if len(ok) == 0:
        return {"recall": 0.0, "threshold": float(thr[0]) if len(thr) else 1.0, "actual_fpr": float(fpr[0])}
    i = ok[-1]
    return {"recall": float(tpr[i]), "threshold": float(thr[i]) if i < len(thr) else 0.0, "actual_fpr": float(fpr[i])}


def score_all() -> dict[str, np.ndarray]:
    import evaluation as ev
    from behaviorguard.baselines.autoencoder_baseline import AutoencoderBaseline
    from behaviorguard.baselines.isolation_forest_baseline import IsolationForestBaseline
    from behaviorguard.models import EvaluationInput, SystemConfig

    data = json.loads(DATASET.read_text(encoding="utf-8"))
    builder = ev._build_profile_with_pm(0.50)
    by = defaultdict(list)
    for m in data["messages"]:
        by[m["user_id"]].append(m)
    for uid in by:
        by[uid].sort(key=lambda x: x["timestamp"])
    users = {u["user_id"]: u for u in data["users"]}

    # Train IF/AE on benign train features
    train_feats = []
    for uid, msgs in by.items():
        split = int(len(msgs) * 0.8)
        for msg in msgs[:split]:
            if msg.get("is_anomaly"):
                continue
            train_feats.append(ev.extract_features_for_baseline(msg) if hasattr(ev, "extract_features_for_baseline") else None)

    # Fallback feature extraction
    def feats(msg, profile=None):
        text = msg["message_text"]
        words = text.split()
        return np.array(
            [
                len(words),
                len(text),
                len(set(w.lower() for w in words)) / max(len(words), 1),
                text.count("?"),
                float("!" in text),
            ],
            dtype=np.float64,
        )

    train_X = []
    for uid, msgs in by.items():
        split = int(len(msgs) * 0.8)
        for msg in msgs[:split]:
            if msg.get("is_anomaly"):
                continue
            train_X.append(feats(msg))
    train_X = np.vstack(train_X)
    print(f"Training baselines on {len(train_X)} benign feature vectors...", flush=True)
    iso = IsolationForestBaseline(contamination=0.1, random_state=SEED)
    iso.fit(train_X)
    ae = AutoencoderBaseline(
        input_dim=train_X.shape[1],
        hidden_dims=[32, 16],
        latent_dim=8,
        epochs=25,
        random_seed=SEED,
    )
    ae.fit(train_X, verbose=False)
    print("Scoring full holdout...", flush=True)

    bg_config = SystemConfig(
        sensitivity_level="medium",
        deployment_context="enterprise",
        overrides_enabled=False,
        enable_linguistic_scoring=False,
        linguistic_component_enabled=False,
        enable_semantic_scoring=True,
        enable_temporal_scoring=True,
    )

    y, bg, iforest, autoenc, rule = [], [], [], [], []
    for uid, msgs in sorted(by.items()):
        split = int(len(msgs) * 0.8)
        profile = builder(users[uid], msgs[:split])
        if profile is None:
            continue
        test = msgs[split:]
        for i, msg in enumerate(test):
            prev = test[i - 1] if i > 0 and test[i - 1].get("session_id") == msg.get("session_id") else None
            cur = ev.message_to_current_message(msg, prev, user_profile=profile)
            r = ev.evaluator.evaluate(
                EvaluationInput(user_profile=profile, current_message=cur, system_config=bg_config)
            )
            x = feats(msg).reshape(1, -1)
            iso_out = iso.predict(x)
            iso_s = float(iso_out["anomaly_scores"][0])
            ae_out = ae.predict(x)
            ae_s = float(ae_out["anomaly_scores"][0])
            # simple rule score
            low = msg["message_text"].lower()
            rule_s = float(
                any(
                    k in low
                    for k in ("ignore previous", "admin", "password", "jailbreak", "delete all", "export all")
                )
            )
            y.append(int(bool(msg.get("should_flag") or msg.get("is_anomaly"))))
            bg.append(float(r.anomaly_score))
            iforest.append(iso_s)
            autoenc.append(ae_s)
            rule.append(rule_s)

            if len(y) % 2000 == 0:
                print(f"  scored {len(y)} test messages...", flush=True)

    return {
        "y_true": np.asarray(y),
        "behaviorguard": np.asarray(bg),
        "isolation_forest": np.asarray(iforest),
        "autoencoder": np.asarray(autoenc),
        "rule_based": np.asarray(rule),
    }


def main() -> None:
    scores = score_all()
    y = scores["y_true"]
    methods = ["behaviorguard", "isolation_forest", "autoencoder", "rule_based"]
    table = {}
    for m in methods:
        s = scores[m]
        table[m] = {
            "auc_roc": float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else float("nan"),
            "auc_pr": float(average_precision_score(y, s)),
            "at_fpr": {str(t): recall_at_fpr(y, s, t) for t in TARGET_FPRS},
        }

    comparisons = []
    pvals = []
    for m in methods[1:]:
        boot = bootstrap_auc_diff(y, scores["behaviorguard"], scores[m])
        p_delong = delong_pvalue(y, scores["behaviorguard"], scores[m])
        comparisons.append(
            {
                "pair": f"behaviorguard_vs_{m}",
                "bootstrap_auc_diff": boot,
                "delong_p": p_delong,
            }
        )
        pvals.append(boot["p_two_sided_approx"])

    adjusted = holm_bonferroni(pvals)
    for c, adj in zip(comparisons, adjusted):
        c["holm_adjusted_p"] = adj

    report = {
        "dataset": DATASET.name,
        "n_messages": int(len(y)),
        "n_positives": int(y.sum()),
        "prevalence": float(y.mean()),
        "seed": SEED,
        "bootstrap_method": "percentile",
        "n_bootstrap": N_BOOT,
        "methods": table,
        "significance": comparisons,
        "note": "Matched-FPR operating points replace shared score-threshold comparison.",
    }
    out = ROOT / "results" / "baselines_matched_fpr_holdout.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
