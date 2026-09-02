#!/usr/bin/env python3
"""
Verify TurnShift canonical Table III row reproduces from evaluate_method().

Canonical config: EMA λ=0.50 (ProfileManager), overrides disabled, τ=0.60.

Usage:
    python scripts/verify_turnshift_canonical.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import evaluation as ev  # noqa: E402

DATASET_KEYS = ["personachat", "blended_skill_talk", "anthropic_hh"]
DATASET_NAMES = {
    "personachat": "PersonaChat",
    "blended_skill_talk": "BST",
    "anthropic_hh": "AnthropicHH",
}

PAPER_TABLE_III_BG = {
    "personachat": {
        "precision": 0.531,
        "recall": 1.000,
        "f1": 0.693,
        "fpr": 0.821,
        "auc": 0.891,
    },
    "blended_skill_talk": {
        "precision": 0.658,
        "recall": 1.000,
        "f1": 0.794,
        "fpr": 0.491,
        "auc": 0.897,
    },
    "anthropic_hh": {
        "precision": 0.491,
        "recall": 0.930,
        "f1": 0.642,
        "fpr": 0.859,
        "auc": 0.736,
    },
}

ROUND3 = 3


def _round3(x: float) -> float:
    return round(float(x), ROUND3)


def _match_status(got: float, expected: float, tol: float = 0.0005) -> str:
    if abs(got - expected) <= tol:
        return "MATCH"
    if _round3(got) == _round3(expected):
        return "MATCH(rounded)"
    return "MISMATCH"


def main() -> None:
    pm_builder = ev._build_profile_with_pm(ev.CANONICAL_LAMBDA)
    rows = []

    print("=" * 80)
    print("TurnShift canonical reproduction (λ=0.50, overrides OFF, τ=0.60)")
    print("=" * 80)

    for dk in DATASET_KEYS:
        metrics, _ = ev.evaluate_method(
            "behaviorguard",
            dk,
            ev.datasets[dk],
            overrides_enabled=False,
            profile_builder=pm_builder,
        )
        paper = PAPER_TABLE_III_BG[dk]
        row = {
            "dataset": DATASET_NAMES[dk],
            "dataset_key": dk,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "fpr": metrics["fpr"],
            "auc": metrics["roc_auc"],
            "paper_f1": paper["f1"],
            "paper_recall": paper["recall"],
            "paper_auc": paper["auc"],
            "f1_status": _match_status(metrics["f1"], paper["f1"]),
            "recall_status": _match_status(metrics["recall"], paper["recall"]),
            "auc_status": _match_status(metrics["roc_auc"], paper["auc"]),
        }
        rows.append(row)

        print(f"\n{DATASET_NAMES[dk]}:")
        print(
            f"  Got:    P={metrics['precision']:.4f} R={metrics['recall']:.4f} "
            f"F1={metrics['f1']:.4f} FPR={metrics['fpr']:.4f} AUC={metrics['roc_auc']:.4f}"
        )
        print(
            f"  Paper:  P={paper['precision']:.3f} R={paper['recall']:.3f} "
            f"F1={paper['f1']:.3f} FPR={paper['fpr']:.3f} AUC={paper['auc']:.3f}"
        )
        print(
            f"  Status: F1={row['f1_status']} Recall={row['recall_status']} "
            f"AUC={row['auc_status']}"
        )

    out = ROOT / "results" / "behaviorguard_canonical_verify.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "config": {
                    "method": "behaviorguard",
                    "lambda": ev.CANONICAL_LAMBDA,
                    "overrides_enabled": False,
                    "threshold": 0.60,
                    "seed": ev.SEED,
                },
                "paper_table_iii": PAPER_TABLE_III_BG,
                "results": rows,
            },
            fh,
            indent=2,
        )
    print(f"\nSaved: {out}")
    print("=" * 80)


if __name__ == "__main__":
    main()
