#!/usr/bin/env python3
"""
Retune composite weights (alpha, beta, gamma) by AUC-PR under user-level 5-fold CV.

Replaces the deprecated F1-at-FPR=0 objective. Uses cached component scores when
available; otherwise scores the full corrected PersonaChat holdout once and caches.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

SEED = 42
N_FOLDS = 5
DATASET = ROOT / "datasets" / "personachat_processed_corrected.json"
CACHE = ROOT / "results" / "component_scores_holdout_personachat.npz"


def _round_w(a: float, b: float, c: float) -> tuple[float, float, float]:
    return (round(float(a), 4), round(float(b), 4), round(float(c), 4))


def _weight_grid(step: float = 0.1):
    vals = np.arange(0.0, 1.0 + 1e-9, step)
    seen: set[tuple[float, float, float]] = set()
    for a in vals:
        for b in vals:
            c = 1.0 - a - b
            if c < -1e-9 or c > 1.0 + 1e-9:
                continue
            w = _round_w(a, b, max(0.0, min(1.0, float(c))))
            if abs(sum(w) - 1.0) > 1e-6 or w in seen:
                continue
            seen.add(w)
            yield w
    # Always include the legacy paper default even if it is off the coarse grid.
    legacy = _round_w(0.4, 0.35, 0.25)
    if legacy not in seen:
        yield legacy


def build_or_load_component_scores(force: bool = False) -> dict:
    if CACHE.exists() and not force:
        d = np.load(CACHE, allow_pickle=True)
        return {k: d[k] for k in d.files}

    import evaluation as ev
    from turnshift.models import EvaluationInput, SystemConfig

    data = json.loads(DATASET.read_text(encoding="utf-8"))
    builder = ev._build_profile_with_pm(0.50)
    by = defaultdict(list)
    for m in data["messages"]:
        by[m["user_id"]].append(m)
    for uid in by:
        by[uid].sort(key=lambda x: x["timestamp"])
    users = {u["user_id"]: u for u in data["users"]}

    # Score with all components enabled; extract from metadata when present,
    # else re-run analyzers via evaluator component scores.
    sem, ling, temp, y, uids = [], [], [], [], []
    config = SystemConfig(
        sensitivity_level="medium",
        deployment_context="enterprise",
        overrides_enabled=False,
        enable_linguistic_scoring=True,
        linguistic_component_enabled=True,
        enable_semantic_scoring=True,
        enable_temporal_scoring=True,
    )
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
                EvaluationInput(user_profile=profile, current_message=cur, system_config=config)
            )
            cs = r.component_scores
            sem.append(float(cs.semantic))
            ling.append(float(cs.linguistic))
            temp.append(float(cs.temporal))
            y.append(int(bool(msg.get("should_flag") or msg.get("is_anomaly"))))
            uids.append(uid)

    cache = {
        "semantic": np.array(sem),
        "linguistic": np.array(ling),
        "temporal": np.array(temp),
        "y_true": np.array(y),
        "user_ids": np.array(uids, dtype=object),
    }
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, **cache)
    return cache


def composite(s, l, t, w):
    return w[0] * s + w[1] * l + w[2] * t


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-rescore", action="store_true")
    parser.add_argument("--step", type=float, default=0.1)
    args = parser.parse_args()

    cache = build_or_load_component_scores(force=args.force_rescore)
    s, l, t = cache["semantic"], cache["linguistic"], cache["temporal"]
    y = cache["y_true"].astype(int)
    uids = cache["user_ids"]
    unique_users = np.array(sorted(set(uids.tolist())))
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    grid = list(_weight_grid(args.step))
    fold_scores = {w: [] for w in grid}

    for fold_i, (tr_u, te_u) in enumerate(kf.split(unique_users)):
        te_set = set(unique_users[te_u].tolist())
        mask = np.array([u in te_set for u in uids])
        yt, st, lt, tt = y[mask], s[mask], l[mask], t[mask]
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        for w in grid:
            scores = composite(st, lt, tt, w)
            fold_scores[w].append(average_precision_score(yt, scores))

    ranked = sorted(
        (
            {
                "weights": {"semantic": w[0], "linguistic": w[1], "temporal": w[2]},
                "mean_auc_pr": float(np.mean(v)),
                "std_auc_pr": float(np.std(v)),
                "n_folds": len(v),
            }
            for w, v in fold_scores.items()
            if v
        ),
        key=lambda r: r["mean_auc_pr"],
        reverse=True,
    )
    best = ranked[0]
    # Full-holdout metrics at best and at legacy default
    legacy = _round_w(0.4, 0.35, 0.25)
    ling_off = _round_w(0.4 / 0.65, 0.0, 0.25 / 0.65)

    def full_metrics(w):
        sc = composite(s, l, t, w)
        return {
            "auc_pr": float(average_precision_score(y, sc)),
            "auc_roc": float(roc_auc_score(y, sc)),
        }

    legacy_cv = fold_scores.get(legacy) or []
    report = {
        "protocol": "user_level_5fold_cv_auc_pr",
        "seed": SEED,
        "n_messages": int(len(y)),
        "n_positives": int(y.sum()),
        "best_weights": best,
        "legacy_default_0.4_0.35_0.25": {
            "weights": {"semantic": 0.4, "linguistic": 0.35, "temporal": 0.25},
            "full_holdout": full_metrics(legacy),
            "cv_mean_auc_pr": float(np.mean(legacy_cv)) if legacy_cv else None,
        },
        "linguistic_excluded_renormalized": {
            "weights": {
                "semantic": ling_off[0],
                "linguistic": ling_off[1],
                "temporal": ling_off[2],
            },
            "full_holdout": full_metrics(ling_off),
        },
        "best_full_holdout": full_metrics(
            (
                best["weights"]["semantic"],
                best["weights"]["linguistic"],
                best["weights"]["temporal"],
            )
        ),
        "top5": ranked[:5],
        "objective_note": "Replaces F1-at-FPR=0 tuning criticized by reviewer; AUC-PR is prevalence-aware.",
    }
    out = ROOT / "results" / "weight_tuning_aucpr.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
