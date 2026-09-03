#!/usr/bin/env python3
"""Holm-Bonferroni family: cusum_embed vs each comparator on both datasets.

Paired percentile-bootstrap differences in episode AUC (stream maxima resampled
by stream, n=2000, seed 42, fresh generator per pair), two-sided p from the sign
mass of the bootstrap distribution, then Holm-Bonferroni over all 12 comparisons.

Reads the committed score caches (results/primary/sequential_ato_scores*.npz);
does not recompute embeddings. Reproduces
results/primary/cusum_holm_bonferroni.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from turnshift.detectors.cusum import unflatten  # noqa: E402

SEED = 42
N_BOOTSTRAP = 2000
PRIMARY = "cusum_embed"
COMPARATORS = [
    "cusum_stylo",
    "cusum_combined",
    "permsg_combined",
    "permsg_bg",
    "window_embed",
    "verifier_lr",
]
CACHES = {
    "personachat": ROOT / "results" / "primary" / "sequential_ato_scores.npz",
    "bst": ROOT / "results" / "primary" / "sequential_ato_scores_bst.npz",
}
OUT_PATH = ROOT / "results" / "primary" / "cusum_holm_bonferroni.json"


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


def stream_maxima(cache_path: Path) -> tuple[dict[str, np.ndarray], np.ndarray]:
    d = np.load(cache_path, allow_pickle=True)
    lengths = d["lengths"].astype(int)
    is_episode = (d["episode_start"].astype(int) >= 0).astype(int)
    maxima = {
        det: np.array([t.max() for t in unflatten(d[det].astype(float), lengths)])
        for det in [PRIMARY, *COMPARATORS]
    }
    return maxima, is_episode


def paired_bootstrap(
    maxima: dict[str, np.ndarray], y: np.ndarray, primary: str, comparator: str
) -> dict:
    """Resample streams; AUC difference per replicate. Degenerate draws are skipped."""
    rng = np.random.default_rng(SEED)
    n = len(y)
    diffs = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.choice(n, n, replace=True)
        if len(np.unique(y[idx])) < 2:
            continue
        diffs.append(
            roc_auc_score(y[idx], maxima[primary][idx])
            - roc_auc_score(y[idx], maxima[comparator][idx])
        )
    d = np.array(diffs)
    return {
        "diff_mean": float(d.mean()),
        "ci95_percentile": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
        # 0.0 means no replicate reached the opposite sign, i.e. p < 2/n_bootstrap.
        "p_two_sided_approx": float(2.0 * min((d <= 0).mean(), (d >= 0).mean())),
        "n_effective": int(len(d)),
        "min_diff": float(d.min()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write here instead of the committed results/primary path.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Print a per-pair diff against the existing JSON instead of writing.",
    )
    args = parser.parse_args()

    comparisons = []
    for dataset, cache in CACHES.items():
        maxima, y = stream_maxima(cache)
        for comparator in COMPARATORS:
            r = paired_bootstrap(maxima, y, PRIMARY, comparator)
            comparisons.append(
                {
                    "dataset": dataset,
                    "pair": f"{PRIMARY}_vs_{comparator}",
                    "diff_mean": r["diff_mean"],
                    "ci95_percentile": r["ci95_percentile"],
                    "p_two_sided_approx": r["p_two_sided_approx"],
                }
            )
            print(
                f"  {dataset:12s} {comparator:16s} diff={r['diff_mean']:+.6f} "
                f"CI[{r['ci95_percentile'][0]:.4f},{r['ci95_percentile'][1]:.4f}] "
                f"p={r['p_two_sided_approx']:.4g} "
                f"min={r['min_diff']:+.4f} n_eff={r['n_effective']}",
                flush=True,
            )

    adjusted = holm_bonferroni([c["p_two_sided_approx"] for c in comparisons])
    for c, adj in zip(comparisons, adjusted):
        c["holm_adjusted_p"] = adj

    report = {
        "family": "cusum_embed_vs_each_comparator x 2 datasets",
        "n_comparisons": len(comparisons),
        "bootstrap_method": "percentile",
        "n_bootstrap": N_BOOTSTRAP,
        "seed": SEED,
        "comparisons": comparisons,
        "n_sig_holm_05": sum(c["holm_adjusted_p"] < 0.05 for c in comparisons),
    }

    if args.compare:
        prior = json.loads(OUT_PATH.read_text(encoding="utf-8"))
        by_key = {(c["dataset"], c["pair"]): c for c in prior["comparisons"]}
        print("\nchanged vs committed:")
        for c in comparisons:
            old = by_key.get((c["dataset"], c["pair"]))
            if old is None:
                print(f"  {c['dataset']} {c['pair']}: NEW")
                continue
            same = all(
                abs(np.ravel(old[k]) - np.ravel(c[k])).max() < 1e-12
                for k in ("diff_mean", "ci95_percentile", "p_two_sided_approx")
            )
            if not same:
                print(
                    f"  {c['dataset']:12s} {c['pair']:34s} "
                    f"diff {old['diff_mean']:.6f} -> {c['diff_mean']:.6f}, "
                    f"p {old['p_two_sided_approx']:.4g} -> {c['p_two_sided_approx']:.4g}, "
                    f"holm {old['holm_adjusted_p']:.4g} -> {c['holm_adjusted_p']:.4g}"
                )
        print(f"  significant at Holm 0.05: {prior['n_sig_holm_05']} -> {report['n_sig_holm_05']}")
        return 0

    out = args.out or OUT_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n{report['n_sig_holm_05']}/{len(comparisons)} significant after Holm at alpha=0.05")
    print(f"Saved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
