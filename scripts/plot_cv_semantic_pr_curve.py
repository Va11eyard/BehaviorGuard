#!/usr/bin/env python3
"""
Precision-recall curves with low-FPR ROC inset: cosine vs Mahalanobis.

Uses the same holdout scores as threshold_sweep_cv_protocol / final_ablation_table.md
(rows 3--4). Main panel: PR curves with F1-optimal τ* markers.
Inset: ROC for FPR ∈ [0, 0.05] (Mahalanobis ranking advantage region).

Outputs paper/figures/cv_semantic_pr_curve.{pdf,png}
        paper/figures/cv_semantic_pr_curve.caption.tex
        results/cv_semantic_pr_curve.json

Usage:
    python scripts/plot_cv_semantic_pr_curve.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

ROOT = Path(__file__).resolve().parents[1]
SCORES_CACHE = ROOT / "results" / "threshold_sweep_cv_scores.npz"
PROTOCOL_JSON = ROOT / "results" / "threshold_sweep_cv_protocol.json"
OUT_DIR = ROOT / "paper" / "figures"
OUT_PNG = OUT_DIR / "cv_semantic_pr_curve.png"
OUT_PDF = OUT_DIR / "cv_semantic_pr_curve.pdf"
OUT_CAPTION = OUT_DIR / "cv_semantic_pr_curve.caption.tex"
META_JSON = ROOT / "results" / "cv_semantic_pr_curve.json"

FIGURE_CAPTION = (
    "Precision--recall (main panel) and low-FPR ROC inset for cosine vs.\\ "
    "Mahalanobis semantic scoring on PersonaChat 5-fold CV pooled test "
    "(linguistic excluded; $N{=}10{,}165$ messages, 29 positives). "
    "Inset shows the low-FPR region of the ROC curve, where Mahalanobis's "
    "ranking advantage concentrates (mean $\\Delta$TPR $+0.282$ at "
    "FPR $\\leq 0.05$); this does not translate to better precision--recall "
    "performance at deployable operating points (main panel)."
)

# Validation-selected τ* (uniform across folds; final_ablation_table.md rows 3--4)
TAU_COSINE = 0.61
TAU_MAHALANOBIS = 0.36
ROC_INSET_FPR_MAX = 0.05

# Match evaluation.py λ-sensitivity plot + calibrate_theta.py export settings
FIGSIZE = (6.0, 4.0)
DPI = 150


def _operating_point(labels: np.ndarray, scores: np.ndarray, tau: float) -> dict:
    preds = (scores > tau).astype(int)
    prec = float(precision_score(labels, preds, zero_division=0.0))
    rec = float(recall_score(labels, preds, zero_division=0.0))
    f1 = float(f1_score(labels, preds, zero_division=0.0))
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    return {
        "tau": tau,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def _pr_curve_points(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    precision, recall, _ = precision_recall_curve(labels, scores)
    return recall, precision


def _roc_low_fpr_stats(
    labels: np.ndarray,
    scores_cos: np.ndarray,
    scores_maha: np.ndarray,
    fpr_max: float,
) -> dict:
    fpr_c, tpr_c, _ = roc_curve(labels, scores_cos)
    fpr_m, tpr_m, _ = roc_curve(labels, scores_maha)
    grid = np.linspace(0.0, fpr_max, 200)
    tpr_c_i = np.interp(grid, fpr_c, tpr_c)
    tpr_m_i = np.interp(grid, fpr_m, tpr_m)
    delta = tpr_m_i - tpr_c_i
    return {
        "fpr_max": fpr_max,
        "mean_delta_tpr_maha_minus_cosine": round(float(delta.mean()), 3),
        "max_delta_tpr_maha_minus_cosine": round(float(delta.max()), 3),
        "auc_cosine": round(float(roc_auc_score(labels, scores_cos)), 4),
        "auc_mahalanobis": round(float(roc_auc_score(labels, scores_maha)), 4),
        "auc_delta": round(float(roc_auc_score(labels, scores_maha) - roc_auc_score(labels, scores_cos)), 4),
        "fpr_grid": grid,
        "tpr_cosine": tpr_c_i,
        "tpr_mahalanobis": tpr_m_i,
    }


def plot_pr_curve(
    labels: np.ndarray,
    scores_cos: np.ndarray,
    scores_maha: np.ndarray,
    out_png: Path,
    out_pdf: Path,
) -> dict:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    except ImportError as exc:
        raise SystemExit("matplotlib required for PR curve figure") from exc

    roc_stats = _roc_low_fpr_stats(labels, scores_cos, scores_maha, ROC_INSET_FPR_MAX)
    mean_delta = roc_stats["mean_delta_tpr_maha_minus_cosine"]

    op_cos = _operating_point(labels, scores_cos, TAU_COSINE)
    op_maha = _operating_point(labels, scores_maha, TAU_MAHALANOBIS)
    rec_cos, prec_cos = _pr_curve_points(labels, scores_cos)
    rec_maha, prec_maha = _pr_curve_points(labels, scores_maha)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(
        rec_cos,
        prec_cos,
        color="#1f77b4",
        linewidth=1.8,
        label="Cosine $s_{\\mathrm{sem}}$",
    )
    ax.plot(
        rec_maha,
        prec_maha,
        color="#d62728",
        linewidth=1.8,
        linestyle="--",
        label="Mahalanobis $s_{\\mathrm{sem}}$",
    )

    ax.scatter(
        [op_cos["recall"]],
        [op_cos["precision"]],
        color="#1f77b4",
        s=64,
        marker="o",
        zorder=5,
        edgecolors="white",
        linewidths=0.8,
    )
    ax.scatter(
        [op_maha["recall"]],
        [op_maha["precision"]],
        color="#d62728",
        s=64,
        marker="s",
        zorder=5,
        edgecolors="white",
        linewidths=0.8,
    )

    ax.annotate(
        f"Cosine $\\tau^*$={TAU_COSINE:.2f}\nF1={op_cos['f1']:.3f}",
        xy=(op_cos["recall"], op_cos["precision"]),
        xytext=(12, 10),
        textcoords="offset points",
        fontsize=9,
        color="#1f77b4",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#cccccc", alpha=0.9),
    )
    ax.annotate(
        f"Mahal. $\\tau^*$={TAU_MAHALANOBIS:.2f}\nF1={op_maha['f1']:.3f}",
        xy=(op_maha["recall"], op_maha["precision"]),
        xytext=(12, -28),
        textcoords="offset points",
        fontsize=9,
        color="#d62728",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#cccccc", alpha=0.9),
    )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("PersonaChat 5-fold CV pooled test (linguistic excluded)")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.95)

    # Low-FPR ROC inset (Mahalanobis AUC advantage region)
    ax_ins = inset_axes(ax, width="44%", height="44%", loc="lower right", borderpad=1.4)
    ax_ins.plot(
        roc_stats["fpr_grid"],
        roc_stats["tpr_cosine"],
        color="#1f77b4",
        linewidth=1.4,
        label="Cosine",
    )
    ax_ins.plot(
        roc_stats["fpr_grid"],
        roc_stats["tpr_mahalanobis"],
        color="#d62728",
        linewidth=1.4,
        linestyle="--",
        label="Mahal.",
    )
    ax_ins.set_xlim(0.0, ROC_INSET_FPR_MAX)
    ax_ins.set_ylim(0.0, 1.0)
    ax_ins.set_xlabel("FPR", fontsize=8)
    ax_ins.set_ylabel("TPR", fontsize=8)
    ax_ins.tick_params(labelsize=7)
    ax_ins.grid(True, alpha=0.3)
    ax_ins.set_title(
        f"ROC inset (FPR $\\leq {ROC_INSET_FPR_MAX:.2f}$)\n"
        f"mean $\\Delta$TPR $={mean_delta:+.3f}$",
        fontsize=7.5,
    )
    ax_ins.legend(loc="lower right", fontsize=6.5, framealpha=0.9)

    fig.subplots_adjust(left=0.12, right=0.96, top=0.92, bottom=0.14)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    # Drop large arrays from metadata; keep scalar ROC summary
    roc_summary = {k: v for k, v in roc_stats.items() if k not in ("fpr_grid", "tpr_cosine", "tpr_mahalanobis")}

    return {
        "n_messages": int(len(labels)),
        "n_positives": int(labels.sum()),
        "cosine": {"operating_point": op_cos, "curve_points": int(len(rec_cos))},
        "mahalanobis": {"operating_point": op_maha, "curve_points": int(len(rec_maha))},
        "roc_inset": roc_summary,
        "figure_caption": FIGURE_CAPTION,
        "outputs": {"png": str(out_png), "pdf": str(out_pdf), "caption_tex": str(OUT_CAPTION)},
    }


def main() -> None:
    if not SCORES_CACHE.exists():
        raise SystemExit(f"Missing scores cache: {SCORES_CACHE}")

    data = np.load(SCORES_CACHE, allow_pickle=True)
    labels = data["labels"].astype(int)
    scores_cos = data["cosine"].astype(float)
    scores_maha = data["mahalanobis"].astype(float)

    meta = plot_pr_curve(labels, scores_cos, scores_maha, OUT_PNG, OUT_PDF)
    OUT_CAPTION.write_text(
        f"% Paste-ready figure caption for cv_semantic_pr_curve\n{FIGURE_CAPTION}\n",
        encoding="utf-8",
    )
    meta["protocol"] = "5-fold user CV pooled test (threshold_sweep_cv_protocol)"
    meta["score_source"] = str(SCORES_CACHE)
    if PROTOCOL_JSON.exists():
        proto = json.loads(PROTOCOL_JSON.read_text(encoding="utf-8"))
        meta["tau_star_per_fold"] = {
            "cosine": proto["cosine"]["tau_star_per_fold"],
            "mahalanobis": proto["mahalanobis"]["tau_star_per_fold"],
        }
        meta["pooled_test_table"] = {
            "cosine": proto["cosine"]["pooled_test"],
            "mahalanobis": proto["mahalanobis"]["pooled_test"],
        }

    META_JSON.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print(f"\nSaved {OUT_PNG}")
    print(f"Saved {OUT_PDF}")
    print(f"Saved {OUT_CAPTION}")
    print(f"Metadata: {META_JSON}")


if __name__ == "__main__":
    main()
