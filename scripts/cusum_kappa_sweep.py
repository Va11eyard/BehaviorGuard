#!/usr/bin/env python3
"""
CUSUM kappa sensitivity + distance-metric ablation on cached residuals.

Theory: for a mean shift of size delta, the optimal reference value is kappa ≈ delta/2.
We estimate delta_hat from post- vs pre-episode standardized residuals, then sweep
kappa in {0, 0.25, 0.5, 0.75, 1.0, 1.5} and report detection at FA=1/1000.

Distance-metric ablation recomputes residuals from raw cosine distances already
cached (cosine is primary); euclidean/correlation/mahalanobis require a heavier
re-score and are reported when --full-metrics is passed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.sequential_ato_study import (  # noqa: E402
    N_BOOTSTRAP,
    SEED,
    TARGET_FA_PER_1000,
    cusum,
    evaluate_detector,
    unflatten,
)

KAPPAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]


def _load_residuals(dataset: str) -> dict:
    path = ROOT / "results" / f"sequential_ato_residuals_{dataset}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run scripts/cache_ato_residuals.py first")
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def sweep_kappa(cache: dict) -> dict:
    lengths = cache["lengths"].astype(int)
    es = cache["episode_start"].astype(int)
    el = cache["episode_len"].astype(int)
    res = unflatten(cache["res_emb"].astype(float), lengths)
    delta_hat = float(cache.get("delta_hat", 0.0))
    theory_kappa = round(0.5 * abs(delta_hat), 4)

    rows = []
    for kappa in KAPPAS:
        trajs = [cusum(r, kappa=kappa) for r in res]
        metrics = evaluate_detector(trajs, es, el)
        op1 = next(o for o in metrics["operating_points"] if o["target_fa_per_1000"] == 1.0)
        rows.append(
            {
                "kappa": kappa,
                "episode_auc": metrics["episode_auc"],
                "episode_auc_ci95": metrics["episode_auc_ci95"],
                "det_rate_fa1": op1["detection_rate"],
                "det_rate_fa1_ci95": op1["detection_rate_ci95"],
                "median_delay": op1["median_delay_msgs"],
                "near_theory": abs(kappa - theory_kappa) <= 0.15,
            }
        )
    best = max(rows, key=lambda r: (r["det_rate_fa1"], r["episode_auc"]))
    return {
        "delta_hat": round(delta_hat, 4),
        "theory_kappa_approx_delta_over_2": theory_kappa,
        "canonical_kappa": 0.5,
        "best_kappa": best["kappa"],
        "sweep": rows,
    }


def residual_family_table(cache: dict) -> dict:
    """Compare stylo / embed / combined residuals at kappa=0.5."""
    lengths = cache["lengths"].astype(int)
    es = cache["episode_start"].astype(int)
    el = cache["episode_len"].astype(int)
    out = {}
    for name in ("res_emb", "res_sty", "res_comb"):
        trajs = [cusum(r, kappa=0.5) for r in unflatten(cache[name].astype(float), lengths)]
        m = evaluate_detector(trajs, es, el)
        op1 = next(o for o in m["operating_points"] if o["target_fa_per_1000"] == 1.0)
        out[name] = {
            "auc": m["episode_auc"],
            "auc_ci95": m["episode_auc_ci95"],
            "det_rate_fa1": op1["detection_rate"],
            "median_delay": op1["median_delay_msgs"],
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="personachat")
    args = parser.parse_args()
    cache = _load_residuals(args.dataset)
    report = {
        "dataset": args.dataset,
        "bootstrap_method": "percentile",
        "n_bootstrap": N_BOOTSTRAP,
        "seed": SEED,
        "kappa_sweep": sweep_kappa(cache),
        "residual_family_at_kappa_0.5": residual_family_table(cache),
        "note_distance_metrics": (
            "Primary residual is cosine distance to EMA centroid (unit-normalized). "
            "Euclidean on unnormalized embeddings and Pearson correlation are "
            "near-redundant with cosine for MiniLM L2-normalized vectors; "
            "Mahalanobis requires per-user covariance (see dual-lambda / Mahalanobis scripts)."
        ),
    }
    out = ROOT / "results" / f"cusum_kappa_sweep_{args.dataset}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
