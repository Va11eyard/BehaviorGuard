#!/usr/bin/env python3
"""
F1-maximizing threshold sweep for the Autoencoder baseline.

Mirrors evaluate_method() data prep and training config (hidden [256,128,64],
latent 32, 50 epochs, min-max score normalization, seed 42). Sweeps anomaly
scores on the test set from 0.01 to 0.99 in 0.01 steps.

Usage:
    python scripts/task5_autoencoder_rerun.py --output-dir results
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import evaluation as ev  # noqa: E402
from turnshift.baselines.autoencoder_baseline import AutoencoderBaseline

RANDOM_SEED = ev.SEED
DECISION_THRESHOLD = 0.60
SWEEP_LOW = 0.01
SWEEP_HIGH = 0.99
SWEEP_STEP = 0.01

DATASET_DISPLAY_NAMES = {
    "personachat": "PersonaChat",
    "blended_skill_talk": "BST",
    "anthropic_hh": "AnthropicHH",
}

PAPER_TABLE_III = {
    "personachat": {"precision": 0.455, "recall": 0.769, "f1": 0.571, "fpr": 0.857},
    "blended_skill_talk": {"precision": 0.506, "recall": 0.788, "f1": 0.617, "fpr": 0.727},
    "anthropic_hh": {"precision": 0.382, "recall": 0.456, "f1": 0.416, "fpr": 0.656},
}

ALL_DATASETS = list(DATASET_DISPLAY_NAMES.keys())


def _prepare_baseline_eval(
    test_data: dict,
    max_users: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Same prep as evaluate_method() for isolation_forest / autoencoder."""
    test_user_ids = set(test_data["splits"]["test"]["user_ids"])
    test_users = [u for u in test_data["users"] if u["user_id"] in test_user_ids]
    test_messages_by_user: dict[str, list] = defaultdict(list)
    for m in test_data["messages"]:
        if m["user_id"] in test_user_ids:
            test_messages_by_user[m["user_id"]].append(m)

    users_with_anomalies = []
    users_without_anomalies = []
    for user in test_users:
        user_msgs = test_messages_by_user[user["user_id"]]
        has_anomaly = any(m.get("should_flag", False) for m in user_msgs)
        if has_anomaly:
            users_with_anomalies.append(user)
        else:
            users_without_anomalies.append(user)

    sampled_test_users = users_with_anomalies[:max_users]
    remaining = max_users - len(sampled_test_users)
    if remaining > 0:
        sampled_test_users.extend(users_without_anomalies[:remaining])

    test_user_profiles: dict[str, dict] = {}
    for user in sampled_test_users:
        user_msgs = test_messages_by_user[user["user_id"]]
        split_idx = int(len(user_msgs) * 0.8)
        train_msgs = user_msgs[:split_idx]
        profile = ev.build_user_profile(user, train_msgs)
        if profile:
            test_user_profiles[user["user_id"]] = {
                "profile": profile,
                "test_messages": user_msgs[split_idx:],
            }

    train_features_list: list[np.ndarray] = []
    test_features_list: list[np.ndarray] = []
    y_true_list: list[bool] = []

    train_user_ids = set(test_data["splits"]["train"]["user_ids"])
    train_users = [u for u in test_data["users"] if u["user_id"] in train_user_ids][:20]

    for _user_id, user_data in test_user_profiles.items():
        profile = user_data["profile"]
        for train_user in train_users:
            train_user_msgs = [
                m
                for m in test_data["messages"]
                if m["user_id"] == train_user["user_id"]
                and not m.get("is_anomaly", False)
            ]
            for msg in train_user_msgs[:10]:
                train_features_list.append(
                    ev.extract_features_for_baselines(msg, profile)
                )

    for user_data in test_user_profiles.values():
        profile = user_data["profile"]
        for msg in user_data["test_messages"]:
            test_features_list.append(ev.extract_features_for_baselines(msg, profile))
            y_true_list.append(bool(msg.get("should_flag", False)))

    return (
        np.array(train_features_list),
        np.array(test_features_list),
        np.array(y_true_list, dtype=bool),
        len(train_features_list),
    )


def _build_autoencoder_eval_config(input_dim: int) -> AutoencoderBaseline:
    """Match evaluation.py evaluate_method() autoencoder branch."""
    return AutoencoderBaseline(
        input_dim=input_dim,
        random_seed=RANDOM_SEED,
    )


def _fit_scores(
    train_features: np.ndarray,
    test_features: np.ndarray,
) -> np.ndarray:
    ae = _build_autoencoder_eval_config(train_features.shape[1])
    ae.fit(train_features, verbose=False)
    return ae.predict(test_features)["anomaly_scores"]


def _max_f1_threshold(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    low: float = SWEEP_LOW,
    high: float = SWEEP_HIGH,
    step: float = SWEEP_STEP,
) -> tuple[float, dict[str, float], list[dict[str, Any]]]:
    """Sweep thresholds; return t*, metrics at t*, and full sweep log."""
    best_t = low
    best_f1 = -1.0
    best_metrics: dict[str, float] = {}
    sweep_rows: list[dict[str, Any]] = []

    for t in np.arange(low, high + step / 2, step):
        t = round(float(t), 2)
        y_pred = y_scores > t
        m = ev.compute_metrics(y_true.tolist(), y_pred.tolist(), y_scores.tolist())
        row = {
            "threshold": t,
            "precision": round(m["precision"], 4),
            "recall": round(m["recall"], 4),
            "f1": round(m["f1"], 4),
            "fpr": round(m["fpr"], 4),
        }
        sweep_rows.append(row)
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_t = t
            best_metrics = m

    return best_t, best_metrics, sweep_rows


def _anomaly_rate(metrics: dict) -> float:
    tp = metrics.get("true_positives", 0)
    fn = metrics.get("false_negatives", 0)
    tn = metrics.get("true_negatives", 0)
    fp = metrics.get("false_positives", 0)
    total = tp + fn + tn + fp
    return round((tp + fn) / total, 4) if total > 0 else 0.0


def _check_training_reproducibility(
    train_features: np.ndarray,
    test_features: np.ndarray,
) -> dict[str, Any]:
    """Train twice with committed config; compare test scores."""
    scores_a = _fit_scores(train_features, test_features)
    scores_b = _fit_scores(train_features, test_features)
    diff = np.abs(scores_a - scores_b)
    return {
        "max_abs_score_diff": float(np.max(diff)),
        "mean_abs_score_diff": float(np.mean(diff)),
        "scores_identical": bool(np.array_equal(scores_a, scores_b)),
    }


def run_sweep(
    dataset_names: list[str],
    output_dir: Path,
    max_users: int,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    repro_by_dataset: dict[str, dict] = {}
    all_sweeps: dict[str, list] = {}

    for dataset_key in dataset_names:
        if dataset_key not in ev.datasets:
            raise KeyError(f"Unknown dataset: {dataset_key}")

        test_data = ev.datasets[dataset_key]
        train_features, test_features, y_true, n_train = _prepare_baseline_eval(
            test_data, max_users
        )

        print(f"\n=== {DATASET_DISPLAY_NAMES[dataset_key]} ===")
        repro = _check_training_reproducibility(train_features, test_features)
        repro_by_dataset[dataset_key] = repro
        print(
            f"  Repro check (2x train, eval config): "
            f"identical={repro['scores_identical']}, "
            f"max_diff={repro['max_abs_score_diff']:.6f}"
        )

        y_scores = _fit_scores(train_features, test_features)
        score_high = max(float(np.max(y_scores)), SWEEP_HIGH)
        score_high = float(np.ceil(score_high * 100) / 100)

        # Reference: committed pipeline threshold 0.60
        m_fixed = ev.compute_metrics(
            y_true.tolist(),
            (y_scores > DECISION_THRESHOLD).tolist(),
            y_scores.tolist(),
        )
        print(
            f"  fixed_tau_0.60: P={m_fixed['precision']:.4f} "
            f"R={m_fixed['recall']:.4f} F1={m_fixed['f1']:.4f} "
            f"FPR={m_fixed['fpr']:.4f} AUC={m_fixed['roc_auc']:.4f}"
        )

        t_star, m_best, sweep_rows = _max_f1_threshold(
            y_true, y_scores, high=score_high
        )
        all_sweeps[dataset_key] = sweep_rows

        paper = PAPER_TABLE_III[dataset_key]
        row = {
            "dataset": DATASET_DISPLAY_NAMES[dataset_key],
            "dataset_key": dataset_key,
            "condition": "max_f1_threshold",
            "precision": round(m_best["precision"], 4),
            "recall": round(m_best["recall"], 4),
            "f1": round(m_best["f1"], 4),
            "fpr": round(m_best["fpr"], 4),
            "auc": round(m_best["roc_auc"], 4),
            "n_train": n_train,
            "n_test": len(y_true),
            "anomaly_rate": _anomaly_rate(m_best),
            "threshold": t_star,
            "paper_f1": paper["f1"],
            "paper_recall": paper["recall"],
            "score_min": round(float(np.min(y_scores)), 4),
            "score_max": round(float(np.max(y_scores)), 4),
            "score_mean": round(float(np.mean(y_scores)), 4),
        }
        summary_rows.append(row)

        print(
            f"  max_f1 t*={t_star}: P={row['precision']:.4f} "
            f"R={row['recall']:.4f} F1={row['f1']:.4f} "
            f"FPR={row['fpr']:.4f} AUC={row['auc']:.4f}"
        )
        print(
            f"  paper Table III: F1={paper['f1']:.3f} R={paper['recall']:.3f} "
            f"(delta F1={row['f1'] - paper['f1']:+.3f})"
        )

    out_json = output_dir / "task5_autoencoder_results.json"
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "random_seed_numpy": RANDOM_SEED,
                "training_config": {
                    "hidden_dims": [256, 128, 64],
                    "latent_dim": 32,
                    "epochs": 50,
                    "batch_size": 32,
                    "random_seed": RANDOM_SEED,
                    "score_normalization": "min-max on training errors (no upper clip)",
                    "source": "evaluation.py evaluate_method()",
                },
                "paper_table_iii": PAPER_TABLE_III,
                "reproducibility": repro_by_dataset,
                "config_notes": {
                    "paper_claims": "50 epochs, Adam lr=1e-3, encoder 256-128-64-32",
                    "committed_eval": "matches paper defaults via AutoencoderBaseline",
                    "torch_manual_seed": f"{RANDOM_SEED} in fit() before weight init",
                    "dataloader_shuffle": "True with fixed torch.Generator",
                },
                "summary": summary_rows,
                "full_sweeps": all_sweeps,
            },
            fh,
            indent=2,
            ensure_ascii=False,
        )

    csv_cols = [
        "dataset",
        "precision",
        "recall",
        "f1",
        "fpr",
        "auc",
        "threshold",
        "n_train",
        "n_test",
        "anomaly_rate",
        "paper_f1",
        "paper_recall",
    ]
    df = pd.DataFrame(summary_rows)[csv_cols]
    csv_path = output_dir / "task5_autoencoder_results.csv"
    df.to_csv(csv_path, index=False)

    print(f"\nSaved: {out_json}")
    print(f"Saved: {csv_path}")
    print("\n" + "=" * 80)
    print("SUMMARY: Autoencoder max-F1 threshold (3 datasets)")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)

    return summary_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Autoencoder F1-max threshold sweep per dataset"
    )
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--datasets", default="all")
    parser.add_argument("--max-users", type=int, default=20)
    args = parser.parse_args()

    if args.datasets.strip().lower() == "all":
        names = ALL_DATASETS
    else:
        names = [d.strip() for d in args.datasets.split(",") if d.strip()]

    run_sweep(names, Path(args.output_dir), args.max_users)


if __name__ == "__main__":
    main()
