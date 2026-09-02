#!/usr/bin/env python3
"""
Re-evaluate Isolation Forest baseline under contamination configurations,
native-threshold control, and optimal operating thresholds (Youden / max-F1).

Fixed-threshold modes use evaluation.py (sigmoid score_samples + threshold 0.60).
Optimal-threshold modes fit contamination='auto' and pick t* on test sigmoid scores.

Usage:
    python scripts/task1_isolation_forest_rerun.py --output-dir results
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
from behaviorguard.baselines.isolation_forest_baseline import IsolationForestBaseline

RANDOM_SEED = ev.SEED
DECISION_THRESHOLD = 0.60
FIXED_CONTAMINATION_CONFIGS: list[float | str] = [0.1, 0.5, "auto"]
NATIVE_CONTAMINATION = "auto"

DATASET_DISPLAY_NAMES = {
    "personachat": "PersonaChat",
    "blended_skill_talk": "BST",
    "anthropic_hh": "AnthropicHH",
}

ALL_DATASETS = list(DATASET_DISPLAY_NAMES.keys())


PAPER_TABLE_III = {
    "personachat": {"f1": 0.316, "recall": 0.231},
    "blended_skill_talk": {"f1": 0.175, "recall": 0.135},
    "anthropic_hh": {"f1": 0.364, "recall": 0.281},
}


def _anomaly_rate(metrics: dict) -> float:
    tp = metrics.get("true_positives", 0)
    fn = metrics.get("false_negatives", 0)
    tn = metrics.get("true_negatives", 0)
    fp = metrics.get("false_positives", 0)
    total = tp + fn + tn + fp
    return round((tp + fn) / total, 4) if total > 0 else 0.0


def _row_from_metrics(
    dataset_key: str,
    condition: str,
    metrics: dict,
    threshold: float | str,
) -> dict[str, Any]:
    return {
        "dataset": DATASET_DISPLAY_NAMES.get(dataset_key, dataset_key),
        "dataset_key": dataset_key,
        "contamination": condition,
        "precision": round(metrics["precision"], 4),
        "recall": round(metrics["recall"], 4),
        "f1": round(metrics["f1"], 4),
        "fpr": round(metrics["fpr"], 4),
        "auc": round(metrics["roc_auc"], 4),
        "n_train": metrics.get("n_train_features"),
        "n_test": metrics.get("num_predictions"),
        "anomaly_rate": _anomaly_rate(metrics),
        "threshold": threshold,
    }


def _prepare_isolation_forest_eval(
    test_data: dict,
    max_users: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Mirror evaluate_method() data prep for isolation_forest without scoring.

    Returns train_features, test_features, y_true, n_train_features.
    """
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

    train_features = np.array(train_features_list)
    test_features = np.array(test_features_list)
    y_true = np.array(y_true_list, dtype=bool)
    return train_features, test_features, y_true, len(train_features)


def _fit_sigmoid_scores(
    test_data: dict,
    max_users: int,
    contamination: float | str = "auto",
) -> tuple[np.ndarray, np.ndarray, int]:
    """Fit IsolationForest and return (y_true, sigmoid scores, n_train)."""
    train_features, test_features, y_true, n_train = _prepare_isolation_forest_eval(
        test_data, max_users
    )
    iso_forest = IsolationForestBaseline(
        contamination=contamination, random_state=RANDOM_SEED
    )
    iso_forest.fit(train_features)
    y_scores = iso_forest.predict(test_features)["anomaly_scores"]
    return y_true, y_scores, n_train


def _metrics_at_threshold(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float,
    n_train: int,
) -> dict[str, Any]:
    y_pred = y_scores > threshold
    metrics = ev.compute_metrics(
        y_true.tolist(), y_pred.tolist(), y_scores.tolist()
    )
    metrics["n_train_features"] = n_train
    metrics["num_predictions"] = len(y_true)
    metrics["threshold_used"] = float(threshold)
    return metrics


def _youden_threshold(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(y_true.astype(int), y_scores)
    if len(thresholds) == 0:
        return DECISION_THRESHOLD
    youden_idx = int(np.argmax(tpr - fpr))
    return float(thresholds[youden_idx])


def _max_f1_threshold(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    low: float = 0.50,
    high: float = 0.99,
    step: float = 0.01,
) -> tuple[float, float]:
    best_t = low
    best_f1 = -1.0
    for t in np.arange(low, high + step / 2, step):
        y_pred = y_scores > t
        m = ev.compute_metrics(y_true.tolist(), y_pred.tolist(), y_scores.tolist())
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_t = float(t)
    return best_t, best_f1


def _evaluate_optimal_threshold(
    test_data: dict,
    max_users: int,
    method: str,
) -> dict[str, Any]:
    """Fit contamination=auto; pick t* via Youden or max-F1 sweep on test scores."""
    y_true, y_scores, n_train = _fit_sigmoid_scores(test_data, max_users, "auto")
    if method == "optimal_threshold":
        t_star = _youden_threshold(y_true, y_scores)
    elif method == "max_f1_threshold":
        t_star, _ = _max_f1_threshold(y_true, y_scores)
    else:
        raise ValueError(f"Unknown optimal method: {method}")
    return _metrics_at_threshold(y_true, y_scores, t_star, n_train)


def _evaluate_sklearn_native(
    test_data: dict,
    max_users: int,
    contamination: float | str,
) -> dict[str, Any]:
    """Classify via sklearn predict() at the given contamination."""
    train_features, test_features, y_true, n_train = _prepare_isolation_forest_eval(
        test_data, max_users
    )
    iso_forest = IsolationForestBaseline(
        contamination=contamination, random_state=RANDOM_SEED
    )
    iso_forest.fit(train_features)
    y_pred = iso_forest.predict_sklearn_labels(test_features) == -1
    y_scores = iso_forest.predict(test_features)["anomaly_scores"]
    metrics = ev.compute_metrics(y_true.tolist(), y_pred.tolist(), y_scores.tolist())
    metrics["n_train_features"] = n_train
    metrics["num_predictions"] = len(y_true)
    return metrics


def _evaluate_native_threshold(
    dataset_key: str,
    test_data: dict,
    max_users: int,
) -> dict[str, Any]:
    """Fit contamination=auto and classify via sklearn native predict()."""
    return _evaluate_sklearn_native(test_data, max_users, NATIVE_CONTAMINATION)


def _print_row(row: dict[str, Any]) -> None:
    print(
        f"  Precision={row['precision']:.4f}  Recall={row['recall']:.4f}  "
        f"F1={row['f1']:.4f}  FPR={row['fpr']:.4f}  AUC={row['auc']:.4f}  "
        f"threshold={row['threshold']}"
    )


def _close_to_paper(dataset_key: str, row: dict[str, Any], tol: float = 0.05) -> bool:
    target = PAPER_TABLE_III.get(dataset_key)
    if not target:
        return False
    return (
        abs(row["f1"] - target["f1"]) <= tol
        and abs(row["recall"] - target["recall"]) <= tol
    )


def _condition_label(contamination: float | str) -> str:
    if contamination == "auto":
        return "auto_fixed_threshold"
    return str(contamination)


def run_sweep(
    dataset_names: list[str],
    output_dir: Path,
    max_users: int,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results: list[dict[str, Any]] = []
    add_native_01 = False

    for dataset_key in dataset_names:
        if dataset_key not in ev.datasets:
            raise KeyError(
                f"Unknown dataset '{dataset_key}'. "
                f"Available: {', '.join(ev.datasets.keys())}"
            )
        test_data = ev.datasets[dataset_key]
        optimal_match = False

        for contamination in FIXED_CONTAMINATION_CONFIGS:
            condition = _condition_label(contamination)
            print(
                f"\n--- {DATASET_DISPLAY_NAMES[dataset_key]} | "
                f"{condition} (fixed threshold {DECISION_THRESHOLD}) ---"
            )
            metrics, _predictions = ev.evaluate_method(
                "isolation_forest",
                dataset_key,
                test_data,
                max_users=max_users,
                contamination=contamination,
            )
            row = _row_from_metrics(
                dataset_key, condition, metrics, DECISION_THRESHOLD
            )
            all_results.append(row)
            _print_row(row)

        for method in ("optimal_threshold", "max_f1_threshold"):
            print(
                f"\n--- {DATASET_DISPLAY_NAMES[dataset_key]} | "
                f"{method} (contamination=auto, t* on test sigmoid scores) ---"
            )
            metrics = _evaluate_optimal_threshold(test_data, max_users, method)
            t_star = metrics["threshold_used"]
            row = _row_from_metrics(dataset_key, method, metrics, round(t_star, 4))
            all_results.append(row)
            _print_row(row)
            if _close_to_paper(dataset_key, row):
                optimal_match = True
                print("  [paper match] within ±0.05 of Table III F1/recall")

        if not optimal_match:
            add_native_01 = True

    if add_native_01:
        print(
            "\n--- Adding native_0.1_threshold "
            "(sklearn predict, contamination=0.1) — optimal did not match paper ---"
        )
        for dataset_key in dataset_names:
            test_data = ev.datasets[dataset_key]
            print(
                f"\n--- {DATASET_DISPLAY_NAMES[dataset_key]} | "
                f"native_0.1_threshold (sklearn predict) ---"
            )
            metrics = _evaluate_sklearn_native(test_data, max_users, 0.1)
            row = _row_from_metrics(
                dataset_key,
                "native_0.1_threshold",
                metrics,
                "sklearn_native_c0.1",
            )
            all_results.append(row)
            _print_row(row)

    json_path = output_dir / "task1_isolation_forest_results.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "random_seed": RANDOM_SEED,
                "fixed_threshold": DECISION_THRESHOLD,
                "fixed_contamination_configs": [
                    _condition_label(c) for c in FIXED_CONTAMINATION_CONFIGS
                ],
                "optimal_conditions": ["optimal_threshold", "max_f1_threshold"],
                "paper_table_iii": PAPER_TABLE_III,
                "added_native_0.1": add_native_01,
                "results": all_results,
            },
            fh,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\nResults saved to JSON: {json_path}")

    csv_cols = [
        "dataset",
        "contamination",
        "precision",
        "recall",
        "f1",
        "fpr",
        "auc",
        "n_train",
        "n_test",
        "anomaly_rate",
        "threshold",
    ]
    df = pd.DataFrame(all_results)[csv_cols]
    csv_path = output_dir / "task1_isolation_forest_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Results saved to CSV:  {csv_path}")

    print("\n" + "=" * 80)
    print("SUMMARY: Isolation Forest Re-evaluation")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80 + "\n")

    return all_results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-run Isolation Forest baseline under multiple contamination configs"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to write result files (default: results)",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="all",
        help="Comma-separated dataset keys or 'all' "
        "(personachat,blended_skill_talk,anthropic_hh)",
    )
    parser.add_argument(
        "--max-users",
        type=int,
        default=20,
        help="Max test users per dataset (default: 20)",
    )
    args = parser.parse_args()

    if args.datasets.strip().lower() == "all":
        dataset_names = ALL_DATASETS
    else:
        dataset_names = [d.strip() for d in args.datasets.split(",") if d.strip()]

    run_sweep(dataset_names, Path(args.output_dir), args.max_users)


if __name__ == "__main__":
    main()
