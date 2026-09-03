#!/usr/bin/env python3
"""
Full-holdout per-message evaluation on externally-authored positives.

Compares the corrected (author-written templates) PersonaChat holdout against
the external-injection variant (AdvBench / Do-Not-Answer / in-the-wild jailbreaks)
under the best-supported per-message config: linguistic off, overrides off,
cosine semantic, tau=0.60, ProfileManager lambda=0.50.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import evaluation as ev  # noqa: E402
from turnshift.models import EvaluationInput, SystemConfig  # noqa: E402
from turnshift.scorers.composite import LEGACY_MEDIUM_WEIGHTS  # noqa: E402

SEED = 42
LAMBDA_DECAY = 0.50
THRESHOLD = 0.60
N_BOOT = 2000


def _by_user(messages: list[dict]) -> dict[str, list]:
    by: dict[str, list] = defaultdict(list)
    for m in messages:
        by[m["user_id"]].append(m)
    for uid in by:
        by[uid].sort(key=lambda x: x["timestamp"])
    return by


def _prev(msgs: list, i: int):
    if i == 0:
        return None
    p = msgs[i - 1]
    return p if p.get("session_id") == msgs[i].get("session_id") else None


def score_holdout(dataset_path: Path) -> dict:
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    builder = ev._build_profile_with_pm(LAMBDA_DECAY)
    # Pinned to the pre-2026-08-11 legacy weights so the committed snapshot reproduces.
    config = SystemConfig(
        sensitivity_level="medium",
        deployment_context="enterprise",
        overrides_enabled=False,
        enable_linguistic_scoring=False,
        linguistic_component_enabled=False,
        enable_semantic_scoring=True,
        enable_temporal_scoring=True,
        semantic_scoring_mode="cosine",
        composite_weights=LEGACY_MEDIUM_WEIGHTS,
    )
    users = {u["user_id"]: u for u in data["users"]}
    by = _by_user(data["messages"])

    y_true: list[int] = []
    y_scores: list[float] = []
    t0 = time.perf_counter()
    n_users = 0
    for uid, msgs in sorted(by.items()):
        split = int(len(msgs) * 0.8)
        profile = builder(users[uid], msgs[:split])
        if profile is None:
            continue
        n_users += 1
        test = msgs[split:]
        for i, msg in enumerate(test):
            cur = ev.message_to_current_message(msg, _prev(test, i), user_profile=profile)
            r = ev.evaluator.evaluate(
                EvaluationInput(user_profile=profile, current_message=cur, system_config=config)
            )
            y_true.append(int(bool(msg.get("should_flag") or msg.get("is_anomaly"))))
            y_scores.append(float(r.anomaly_score))
        if n_users % 500 == 0:
            print(f"  ... {n_users} users ({time.perf_counter() - t0:.0f}s)", flush=True)

    yt = np.asarray(y_true)
    ys = np.asarray(y_scores)
    yp = (ys >= THRESHOLD).astype(int)
    tp = int(((yp == 1) & (yt == 1)).sum())
    fp = int(((yp == 1) & (yt == 0)).sum())
    fn = int(((yp == 0) & (yt == 1)).sum())
    tn = int(((yp == 0) & (yt == 0)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    auc = float(roc_auc_score(yt, ys)) if len(np.unique(yt)) > 1 else float("nan")
    ap = float(average_precision_score(yt, ys)) if yt.sum() else float("nan")

    rng = np.random.default_rng(SEED)
    boot_f1, boot_auc = [], []
    n = len(yt)
    for _ in range(N_BOOT):
        idx = rng.choice(n, n, replace=True)
        ytb, ysb = yt[idx], ys[idx]
        if len(np.unique(ytb)) < 2:
            continue
        ypb = (ysb >= THRESHOLD).astype(int)
        tpb = int(((ypb == 1) & (ytb == 1)).sum())
        fpb = int(((ypb == 1) & (ytb == 0)).sum())
        fnb = int(((ypb == 0) & (ytb == 1)).sum())
        pb = tpb / max(tpb + fpb, 1)
        rb = tpb / max(tpb + fnb, 1)
        boot_f1.append(2 * pb * rb / max(pb + rb, 1e-12))
        boot_auc.append(roc_auc_score(ytb, ysb))

    return {
        "dataset": dataset_path.name,
        "n_users_scored": n_users,
        "n_test_messages": int(n),
        "n_positives": int(yt.sum()),
        "prevalence": round(float(yt.mean()), 6),
        "threshold": THRESHOLD,
        "config": "linguistic_off_overrides_off_cosine_lambda0.5",
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "f1_ci95_percentile": [
            round(float(np.percentile(boot_f1, 2.5)), 4),
            round(float(np.percentile(boot_f1, 97.5)), 4),
        ],
        "auc_roc": round(auc, 4),
        "auc_roc_ci95_percentile": [
            round(float(np.percentile(boot_auc, 2.5)), 4),
            round(float(np.percentile(boot_auc, 97.5)), 4),
        ],
        "auc_pr": round(ap, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "fpr": round(fp / max(fp + tn, 1), 4),
        "bootstrap_method": "percentile",
        "n_bootstrap": N_BOOT,
        "seed": SEED,
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        default="personachat",
        help="Comma-separated keys: personachat,blended_skill_talk,anthropic_hh",
    )
    args = parser.parse_args()

    report = {"protocol": "full_holdout_80_20_external_vs_corrected", "results": {}}
    for key in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        for tag, suffix in (("corrected", "_processed_corrected.json"), ("external", "_processed_external.json")):
            path = ROOT / "datasets" / f"{key}{suffix}"
            if not path.exists():
                print(f"[SKIP] {path}")
                continue
            print(f"\n=== {key} / {tag} ===", flush=True)
            report["results"][f"{key}_{tag}"] = score_holdout(path)

    out = ROOT / "results" / "methodology-diagnostics" / "external_positive_holdout_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
