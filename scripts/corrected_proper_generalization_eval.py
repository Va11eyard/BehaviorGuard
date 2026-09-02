#!/usr/bin/env python3
"""
Proper generalization eval on corrected datasets (no test-set hyperparameter leakage).

Usage:
    set HF_HUB_OFFLINE=1; set TRANSFORMERS_OFFLINE=1
    python scripts/corrected_proper_generalization_eval.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import evaluation as ev  # noqa: E402
from turnshift import TurnShiftEvaluatorML  # noqa: E402
from turnshift.baselines.autoencoder_baseline import AutoencoderBaseline  # noqa: E402
from turnshift.baselines.isolation_forest_baseline import IsolationForestBaseline  # noqa: E402
from turnshift.models import EvaluationInput, SystemConfig  # noqa: E402

CORRECTED_PATHS = {
    "personachat": ROOT / "datasets/personachat_processed_corrected.json",
    "blended_skill_talk": ROOT / "datasets/blended_skill_talk_processed_corrected.json",
    "anthropic_hh": ROOT / "datasets/anthropic_hh_processed_corrected.json",
}

DISPLAY = {
    "personachat": "PersonaChat",
    "blended_skill_talk": "BST",
    "anthropic_hh": "AnthropicHH",
}

SEED = 42
DEFAULT_WEIGHTS = (0.4, 0.35, 0.25)
LAMBDAS = [round(v, 1) for v in np.arange(0.0, 1.0001, 0.1)]
K_FOLD = 5
K_FOLD_THRESHOLD = 50
BG_T_LOW, BG_T_HIGH, BG_T_STEP = 0.01, 0.99, 0.01

BG_CONFIG = SystemConfig(
    sensitivity_level="medium",
    deployment_context="enterprise",
    overrides_enabled=False,
)

LEAKED_SOURCES = {
    "behaviorguard_insample": ROOT / "results" / "corrected_bg_fair_tuning.json",
    "baselines_insample": ROOT / "results" / "corrected_pipeline_eval.json",
}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _weight_grid(step: float = 0.1) -> list[tuple[float, float, float]]:
    grid: list[tuple[float, float, float]] = []
    vals = [round(v, 1) for v in np.arange(step, 1.0, step)]
    for alpha in vals:
        for beta in vals:
            gamma = round(1.0 - alpha - beta, 1)
            if step - 1e-9 <= gamma <= 1.0 - step + 1e-9:
                grid.append((alpha, beta, gamma))
    return grid


def _messages_by_user(test_data: dict) -> dict[str, list]:
    by: dict[str, list] = defaultdict(list)
    for m in test_data["messages"]:
        by[m["user_id"]].append(m)
    for uid in by:
        by[uid].sort(key=lambda x: x["timestamp"])
    return by


def build_eval_pool(
    test_data: dict,
    use_test_split_only: bool,
) -> tuple[list[str], list[str]]:
    by_user = _messages_by_user(test_data)
    scope = set(test_data["splits"]["test"]["user_ids"]) if use_test_split_only else None

    anomaly, benign = [], []
    for uid, msgs in by_user.items():
        if scope is not None and uid not in scope:
            continue
        if any(m.get("should_flag") for m in msgs):
            anomaly.append(uid)
        else:
            benign.append(uid)

    rng = np.random.default_rng(SEED)
    benign_arr = np.array(sorted(benign))
    rng.shuffle(benign_arr)
    n_benign = min(len(anomaly), len(benign_arr))
    return sorted(anomaly), sorted(benign_arr[:n_benign].tolist())


def _user_lookup(test_data: dict) -> dict[str, dict]:
    return {u["user_id"]: u for u in test_data["users"]}


def _prev_in_session(test_msgs: list, i: int):
    if i == 0:
        return None
    p = test_msgs[i - 1]
    if p.get("session_id", "session_0") == test_msgs[i].get("session_id", "session_0"):
        return p
    return None


def _build_profile(user: dict, msgs: list, lambda_decay: float):
    builder = ev._build_profile_with_pm(lambda_decay)
    return builder(user, msgs)


def collect_bg_rows(
    test_data: dict,
    user_ids: list[str],
    lambda_decay: float,
    evaluator: TurnShiftEvaluatorML | None = None,
    system_config: SystemConfig | None = None,
) -> list[dict]:
    if evaluator is None:
        evaluator = TurnShiftEvaluatorML()
    config = system_config or BG_CONFIG
    by_user = _messages_by_user(test_data)
    users = _user_lookup(test_data)
    rows: list[dict] = []

    for uid in user_ids:
        msgs = by_user[uid]
        split_idx = int(len(msgs) * 0.8)
        profile = _build_profile(users[uid], msgs[:split_idx], lambda_decay)
        if profile is None:
            continue
        for i, msg in enumerate(msgs[split_idx:]):
            prev = _prev_in_session(msgs[split_idx:], i)
            cur = ev.message_to_current_message(msg, prev, user_profile=profile)
            result = evaluator.evaluate(
                EvaluationInput(
                    user_profile=profile,
                    current_message=cur,
                    system_config=config,
                )
            )
            cs = result.component_scores
            rows.append(
                {
                    "user_id": uid,
                    "y_true": bool(msg.get("should_flag", False)),
                    "s_sem": float(cs.semantic),
                    "s_ling": float(cs.linguistic),
                    "s_temp": float(cs.temporal),
                }
            )
    return rows


def _composite(rows: list[dict], w: tuple[float, float, float]) -> np.ndarray:
    a, b, g = w
    return np.array(
        [a * r["s_sem"] + b * r["s_ling"] + g * r["s_temp"] for r in rows],
        dtype=float,
    )


def _y_true(rows: list[dict]) -> np.ndarray:
    return np.array([r["y_true"] for r in rows], dtype=bool)


def _f1_max_threshold(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    low: float,
    high: float,
    step: float,
) -> tuple[float, dict]:
    best_t, best_m, best_f1 = low, {}, -1.0
    for t in np.arange(low, high + step / 2, step):
        t = round(float(t), 2)
        y_pred = y_scores > t
        m = ev.compute_metrics(y_true.tolist(), y_pred.tolist(), y_scores.tolist())
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_t = t
            best_m = m
    return best_t, best_m


def _metrics_at_t(y_true: np.ndarray, y_scores: np.ndarray, t: float) -> dict:
    y_pred = y_scores > t
    return ev.compute_metrics(y_true.tolist(), y_pred.tolist(), y_scores.tolist())


def _row(m: dict, t: float) -> dict[str, float]:
    return {
        "precision": round(m["precision"], 4),
        "recall": round(m["recall"], 4),
        "f1": round(m["f1"], 4),
        "fpr": round(m["fpr"], 4),
        "auc": round(m["roc_auc"], 4),
        "threshold": round(float(t), 4),
    }


def tune_bg_with_cache(
    cache: dict[float, list[dict]],
    tune_ids: set[str],
) -> dict[str, Any]:
    tune_rows_by_lam = {
        lam: [r for r in rows if r["user_id"] in tune_ids]
        for lam, rows in cache.items()
    }

    best_lambda = 0.5
    best_lam_f1 = -1.0
    for lam in LAMBDAS:
        rows = tune_rows_by_lam[lam]
        if not rows:
            continue
        y = _y_true(rows)
        scores = _composite(rows, DEFAULT_WEIGHTS)
        _, m = _f1_max_threshold(y, scores, BG_T_LOW, BG_T_HIGH, BG_T_STEP)
        if m["f1"] > best_lam_f1:
            best_lam_f1 = m["f1"]
            best_lambda = lam

    tune_rows = tune_rows_by_lam[best_lambda]
    y = _y_true(tune_rows)
    best_w = DEFAULT_WEIGHTS
    best_w_f1 = -1.0
    for w in _weight_grid(0.1):
        scores = _composite(tune_rows, w)
        _, m = _f1_max_threshold(y, scores, BG_T_LOW, BG_T_HIGH, BG_T_STEP)
        if m["f1"] > best_w_f1:
            best_w_f1 = m["f1"]
            best_w = w

    return {
        "lambda": best_lambda,
        "alpha": best_w[0],
        "beta": best_w[1],
        "gamma": best_w[2],
        "tune_f1_insample": round(best_w_f1, 4),
    }


def select_bg_threshold(
    cache: dict[float, list[dict]],
    val_ids: set[str],
    hp: dict[str, Any],
) -> float:
    lam = hp["lambda"]
    w = (hp["alpha"], hp["beta"], hp["gamma"])
    rows = [r for r in cache[lam] if r["user_id"] in val_ids]
    y = _y_true(rows)
    scores = _composite(rows, w)
    t, _ = _f1_max_threshold(y, scores, BG_T_LOW, BG_T_HIGH, BG_T_STEP)
    return t


def eval_bg_test(
    cache: dict[float, list[dict]],
    test_ids: set[str],
    hp: dict[str, Any],
    threshold: float,
) -> dict[str, float]:
    lam = hp["lambda"]
    w = (hp["alpha"], hp["beta"], hp["gamma"])
    rows = [r for r in cache[lam] if r["user_id"] in test_ids]
    y = _y_true(rows)
    scores = _composite(rows, w)
    m = _metrics_at_t(y, scores, threshold)
    return _row(m, threshold)


def _baseline_train_features(
    test_data: dict,
    tune_user_ids: list[str],
) -> np.ndarray:
    by_user = _messages_by_user(test_data)
    users = _user_lookup(test_data)
    feats: list[np.ndarray] = []
    for uid in tune_user_ids:
        msgs = by_user[uid]
        split_idx = int(len(msgs) * 0.8)
        profile = ev.build_user_profile(users[uid], msgs[:split_idx])
        if profile is None:
            continue
        organic = [m for m in msgs[:split_idx] if not m.get("is_anomaly", False)]
        for msg in organic[:10]:
            feats.append(ev.extract_features_for_baselines(msg, profile))
    return np.array(feats) if feats else np.zeros((0, 1))


def _baseline_eval_rows(
    test_data: dict,
    user_ids: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    by_user = _messages_by_user(test_data)
    users = _user_lookup(test_data)
    feats, labels = [], []
    for uid in user_ids:
        msgs = by_user[uid]
        split_idx = int(len(msgs) * 0.8)
        profile = ev.build_user_profile(users[uid], msgs[:split_idx])
        if profile is None:
            continue
        for msg in msgs[split_idx:]:
            feats.append(ev.extract_features_for_baselines(msg, profile))
            labels.append(bool(msg.get("should_flag", False)))
    return np.array(feats), np.array(labels, dtype=bool)


def eval_baseline_fold(
    test_data: dict,
    tune_ids: list[str],
    val_ids: list[str],
    test_ids: list[str],
    method: str,
) -> dict[str, Any]:
    train_f = _baseline_train_features(test_data, tune_ids)
    val_f, val_y = _baseline_eval_rows(test_data, val_ids)
    test_f, test_y = _baseline_eval_rows(test_data, test_ids)

    if len(train_f) == 0 or len(val_f) == 0 or len(test_f) == 0:
        return {"error": "insufficient features", "method": method}

    if method == "isolation_forest":
        model = IsolationForestBaseline(contamination="auto", random_state=SEED)
        model.fit(train_f)
        val_scores = model.predict(val_f)["anomaly_scores"]
        test_scores = model.predict(test_f)["anomaly_scores"]
        high = BG_T_HIGH
    else:
        model = AutoencoderBaseline(input_dim=train_f.shape[1], random_seed=SEED)
        model.fit(train_f, verbose=False)
        val_scores = model.predict(val_f)["anomaly_scores"]
        test_scores = model.predict(test_f)["anomaly_scores"]
        high = float(np.ceil(max(float(np.max(val_scores)), float(np.max(test_scores))) * 100) / 100)
        high = max(high, 0.99)

    val_t, _ = _f1_max_threshold(val_y, val_scores, BG_T_LOW, high, BG_T_STEP)
    test_m = _metrics_at_t(test_y, test_scores, val_t)
    return {
        "method": method,
        "val_threshold": round(float(val_t), 4),
        "test": _row(test_m, val_t),
        "n_train_features": len(train_f),
        "n_test_messages": len(test_y),
    }


def split_holdout(user_ids: list[str]) -> tuple[list[str], list[str], list[str]]:
    rng = np.random.default_rng(SEED)
    ids = list(user_ids)
    rng.shuffle(ids)
    n = len(ids)
    n_tune = int(round(n * 0.4))
    n_val = int(round(n * 0.2))
    tune = ids[:n_tune]
    val = ids[n_tune : n_tune + n_val]
    test = ids[n_tune + n_val :]
    return tune, val, test


def kfold_splits(user_ids: list[str], k: int) -> list[tuple[list[str], list[str], list[str]]]:
    rng = np.random.default_rng(SEED)
    ids = np.array(user_ids)
    rng.shuffle(ids)
    folds = np.array_split(ids, k)
    splits = []
    for i in range(k):
        test = folds[i].tolist()
        remain = np.concatenate([folds[j] for j in range(k) if j != i])
        n_rem = len(remain)
        n_tune = int(round(n_rem * 0.5))
        n_val = int(round(n_rem * 0.25))
        tune = remain[:n_tune].tolist()
        val = remain[n_tune : n_tune + n_val].tolist()
        splits.append((tune, val, test))
    return splits


def run_one_split(
    test_data: dict,
    all_user_ids: list[str],
    tune: list[str],
    val: list[str],
    test: list[str],
    bg_cache: dict[float, list[dict]] | None = None,
    system_config: SystemConfig | None = None,
    skip_baselines: bool = False,
) -> dict[str, Any]:
    tune_s, val_s, test_s = set(tune), set(val), set(test)

    if bg_cache is None:
        bg_cache = {}
        evaluator = TurnShiftEvaluatorML()
        config = system_config or BG_CONFIG
        for lam in LAMBDAS:
            bg_cache[lam] = collect_bg_rows(
                test_data, all_user_ids, lam, evaluator=evaluator, system_config=config
            )

    hp = tune_bg_with_cache(bg_cache, tune_s)
    bg_t = select_bg_threshold(bg_cache, val_s, hp)
    bg_test = eval_bg_test(bg_cache, test_s, hp, bg_t)

    if skip_baselines:
        if_res = {"test": None}
        ae_res = {"test": None}
    else:
        if_res = eval_baseline_fold(test_data, tune, val, test, "isolation_forest")
        ae_res = eval_baseline_fold(test_data, tune, val, test, "autoencoder")

    return {
        "n_tune_users": len(tune),
        "n_val_users": len(val),
        "n_test_users": len(test),
        "bg_hyperparams": hp,
        "bg_val_threshold": bg_t,
        "behaviorguard": bg_test,
        "isolation_forest": if_res.get("test"),
        "autoencoder": ae_res.get("test"),
    }


def _mean_std(rows: list[dict], key: str) -> dict[str, float]:
    vals = [r[key] for r in rows if r.get(key) is not None]
    if not vals:
        return {"mean": None, "std": None}
    return {"mean": round(float(np.mean(vals)), 4), "std": round(float(np.std(vals)), 4)}


def aggregate_folds(fold_results: list[dict]) -> dict[str, Any]:
    out: dict[str, Any] = {"n_folds": len(fold_results), "folds": fold_results}
    for method in ("behaviorguard", "isolation_forest", "autoencoder"):
        tests = []
        for fr in fold_results:
            if method == "behaviorguard":
                tests.append(fr.get("behaviorguard", {}))
            else:
                tests.append(fr.get(method) or {})
        agg = {}
        for metric in ("precision", "recall", "f1", "fpr", "auc"):
            agg[metric] = _mean_std(tests, metric)
        out[f"{method}_test_aggregate"] = agg
    return out


def load_leaked_numbers() -> dict[str, Any]:
    leaked: dict[str, Any] = {
        "label": "hyperparameter search upper bound, not a generalization estimate",
    }
    bg_path = LEAKED_SOURCES["behaviorguard_insample"]
    base_path = LEAKED_SOURCES["baselines_insample"]
    if bg_path and bg_path.exists():
        bg = _load_json(bg_path)
        for dk, ds in bg.get("datasets", {}).items():
            leaked.setdefault(dk, {})["turnshift_t4_best_combined"] = ds.get(
                "task4_best_combined_f1max"
            )
            leaked[dk]["behaviorguard_f1max_lambda_0.50"] = ds.get("task1_f1max_lambda_0.50")
    if base_path and base_path.exists():
        base = _load_json(base_path)
        for row in base.get("corrected_results", []):
            dk = row.get("dataset_key")
            method = row.get("method")
            if dk and method:
                leaked.setdefault(dk, {})[f"{method}_insample_f1max"] = {
                    k: row[k]
                    for k in ("precision", "recall", "f1", "fpr", "auc", "threshold")
                    if k in row
                }
    return leaked


def run_dataset(
    dk: str,
    test_data: dict,
    system_config: SystemConfig | None = None,
    skip_baselines: bool = False,
) -> dict[str, Any]:
    use_test_only = dk != "personachat"
    anomaly_ids, benign_ids = build_eval_pool(test_data, use_test_split_only=use_test_only)
    all_users = anomaly_ids + benign_ids

    strategy = "k_fold" if len(anomaly_ids) < K_FOLD_THRESHOLD else "holdout_40_20_40"
    print(
        f"  pool: {len(anomaly_ids)} anomaly + {len(benign_ids)} benign users; "
        f"strategy={strategy}"
    )

    # Precompute BG scores for all λ (dominant cost)
    print("  Precomputing BG component scores (11 λ values)...")
    bg_cache: dict[float, list[dict]] = {}
    evaluator = TurnShiftEvaluatorML()
    config = system_config or BG_CONFIG
    for lam in LAMBDAS:
        print(f"    λ={lam:.1f}...", flush=True)
        bg_cache[lam] = collect_bg_rows(
            test_data, all_users, lam, evaluator=evaluator, system_config=config
        )

    if strategy == "k_fold":
        splits = kfold_splits(all_users, K_FOLD)
        fold_results = []
        for fi, (t, v, te) in enumerate(splits):
            print(f"  fold {fi + 1}/{len(splits)}...", flush=True)
            fold_results.append(
                run_one_split(
                    test_data,
                    all_users,
                    t,
                    v,
                    te,
                    bg_cache=bg_cache,
                    system_config=config,
                    skip_baselines=skip_baselines,
                )
            )
        proper = aggregate_folds(fold_results)
    else:
        tune, val, test = split_holdout(all_users)
        single = run_one_split(
            test_data,
            all_users,
            tune,
            val,
            test,
            bg_cache=bg_cache,
            system_config=config,
            skip_baselines=skip_baselines,
        )
        proper = {
            "strategy": "holdout_40_20_40",
            "split_sizes": {
                "tune": len(tune),
                "val": len(val),
                "test": len(test),
            },
            "result": single,
        }

    return {
        "dataset": DISPLAY[dk],
        "dataset_key": dk,
        "n_anomaly_users": len(anomaly_ids),
        "n_benign_users": len(benign_ids),
        "eval_scope": "test_split_only" if use_test_only else "all_users",
        "strategy": strategy,
        "proper_generalization": proper,
    }


def main() -> None:
    leaked = load_leaked_numbers()
    results: dict[str, Any] = {
        "protocol": {
            "seed": SEED,
            "tune_fraction": 0.4,
            "val_fraction": 0.2,
            "test_fraction": 0.4,
            "k_fold": K_FOLD,
            "k_fold_if_anomaly_users_lt": K_FOLD_THRESHOLD,
            "bg_tune": "λ grid 0..1 step 0.1 + weight simplex step 0.1, F1-max on tune",
            "threshold_selection": "F1-max on val split only (BG, IF, AE)",
            "within_user_split": "80% profile / 20% scored window (unchanged)",
        },
        "leaked_insample_upper_bound": leaked,
        "datasets": {},
    }

    for dk, path in CORRECTED_PATHS.items():
        print(f"\n{'=' * 60}\n{DISPLAY[dk]}\n{'=' * 60}")
        test_data = _load_json(path)
        ds_out = run_dataset(dk, test_data)
        results["datasets"][dk] = ds_out

        proper = ds_out["proper_generalization"]
        if ds_out["strategy"] == "k_fold":
            for method in ("behaviorguard", "isolation_forest", "autoencoder"):
                agg = proper[f"{method}_test_aggregate"]
                print(
                    f"  {method} test: F1={agg['f1']['mean']}±{agg['f1']['std']} "
                    f"AUC={agg['auc']['mean']}±{agg['auc']['std']}"
                )
        else:
            r = proper["result"]
            for method in ("behaviorguard", "isolation_forest", "autoencoder"):
                m = r["behaviorguard"] if method == "behaviorguard" else r.get(method)
                if m:
                    print(f"  {method} test: F1={m['f1']} AUC={m['auc']}")

    out_path = ROOT / "results" / "corrected_proper_generalization_eval.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
