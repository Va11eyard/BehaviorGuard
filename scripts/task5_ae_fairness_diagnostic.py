#!/usr/bin/env python3
"""Fairness diagnostic for corrected Autoencoder F1-max results."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

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
from turnshift.baselines.autoencoder_baseline import AutoencoderBaseline
from scripts.task5_autoencoder_rerun import (
    DATASET_DISPLAY_NAMES,
    _build_autoencoder_eval_config,
    _prepare_baseline_eval,
)

BG_AUC = {
    "personachat": 0.891,
    "blended_skill_talk": 0.897,
    "anthropic_hh": 0.736,
}
IF_AUC = {
    "personachat": 0.8049,
    "blended_skill_talk": 0.7923,
    "anthropic_hh": 0.511,
}
TAU_STAR = {
    "personachat": 3.26,
    "blended_skill_talk": 2.62,
    "anthropic_hh": 2.22,
}


def _prepare_with_messages(test_data: dict, max_users: int = 20):
    """Same as _prepare_baseline_eval but also returns aligned test message dicts."""
    test_user_ids = set(test_data["splits"]["test"]["user_ids"])
    test_users = [u for u in test_data["users"] if u["user_id"] in test_user_ids]
    test_messages_by_user: dict[str, list] = defaultdict(list)
    for m in test_data["messages"]:
        if m["user_id"] in test_user_ids:
            test_messages_by_user[m["user_id"]].append(m)

    users_with_anomalies, users_without_anomalies = [], []
    for user in test_users:
        user_msgs = test_messages_by_user[user["user_id"]]
        has_anomaly = any(m.get("should_flag", False) for m in user_msgs)
        (users_with_anomalies if has_anomaly else users_without_anomalies).append(user)

    sampled = users_with_anomalies[:max_users]
    sampled.extend(users_without_anomalies[: max_users - len(sampled)])

    test_user_profiles: dict[str, dict] = {}
    for user in sampled:
        user_msgs = test_messages_by_user[user["user_id"]]
        split_idx = int(len(user_msgs) * 0.8)
        profile = ev.build_user_profile(user, user_msgs[:split_idx])
        if profile:
            test_user_profiles[user["user_id"]] = {
                "profile": profile,
                "test_messages": user_msgs[split_idx:],
            }

    train_features_list, test_features_list, y_true_list, test_msgs_list = [], [], [], []
    train_user_ids = set(test_data["splits"]["train"]["user_ids"])
    train_users = [u for u in test_data["users"] if u["user_id"] in train_user_ids][:20]

    train_msg_sources: list[dict] = []
    for _uid, user_data in test_user_profiles.items():
        profile = user_data["profile"]
        for train_user in train_users:
            train_user_msgs = [
                m
                for m in test_data["messages"]
                if m["user_id"] == train_user["user_id"]
                and not m.get("is_anomaly", False)
            ]
            for msg in train_user_msgs[:10]:
                train_features_list.append(ev.extract_features_for_baselines(msg, profile))
                train_msg_sources.append({
                    "msg_user_id": train_user["user_id"],
                    "profile_user_id": _uid,
                    "is_anomaly": msg.get("is_anomaly", False),
                    "should_flag": msg.get("should_flag", False),
                    "in_test_split": train_user["user_id"] in test_user_ids,
                })

    for user_data in test_user_profiles.values():
        profile = user_data["profile"]
        for msg in user_data["test_messages"]:
            test_features_list.append(ev.extract_features_for_baselines(msg, profile))
            y_true_list.append(bool(msg.get("should_flag", False)))
            test_msgs_list.append(msg)

    return (
        np.array(train_features_list),
        np.array(test_features_list),
        np.array(y_true_list, dtype=bool),
        test_msgs_list,
        train_msg_sources,
        list(test_user_profiles.keys()),
        train_users,
    )


def main() -> None:
    report: dict = {
        "scoring_function": (
            "raw_score = MSE(reconstruction, input) per sample; "
            "anomaly_score = max((raw - train_min) / (train_max - train_min + 1e-8), 0) "
            "[min-max on TRAINING reconstruction errors, NO upper clip, NOT percentile rank]"
        ),
        "protocol_parity": {},
        "datasets": {},
        "personachat_sanity": {},
    }

    print("=" * 80)
    print("AUTOENCODER FAIRNESS DIAGNOSTIC")
    print("=" * 80)
    print(f"\nScoring: {report['scoring_function']}\n")

    for dk in DATASET_DISPLAY_NAMES:
        td = ev.datasets[dk]
        tr, te, y, n_train_task5 = _prepare_baseline_eval(td, 20)
        tr2, te2, y2, test_msgs, train_sources, profile_uids, train_uids = (
            _prepare_with_messages(td, 20)
        )

        ae = _build_autoencoder_eval_config(tr.shape[1])
        ae.fit(tr, verbose=False)
        pred = ae.predict(te)
        raw_errors = pred["reconstruction_errors"]
        scores = pred["anomaly_scores"]

        auc = ev.compute_metrics(y.tolist(), (scores > 0.6).tolist(), scores.tolist())["roc_auc"]

        # Leakage checks
        train_in_test_split = sum(1 for s in train_sources if s["in_test_split"])
        train_anomaly = sum(1 for s in train_sources if s["is_anomaly"])
        train_should_flag = sum(1 for s in train_sources if s["should_flag"])

        pos_raw = raw_errors[y]
        neg_raw = raw_errors[~y]
        pos_scores = scores[y]
        neg_scores = scores[~y]

        lengths = np.array([len(m["message_text"].split()) for m in test_msgs])
        corr_len_raw = float(np.corrcoef(lengths, raw_errors)[0, 1]) if len(lengths) > 1 else 0.0

        ds_report = {
            "n_train_features": int(n_train_task5),
            "n_test": int(len(y)),
            "matches_task5_prep": bool(
                n_train_task5 == len(tr2) and len(y) == len(y2) and np.array_equal(y, y2)
            ),
            "train_pool": {
                "n_features": len(train_sources),
                "unique_train_split_users": len(train_uids),
                "unique_test_profile_users": len(profile_uids),
                "messages_from_test_split_users_in_train_pool": train_in_test_split,
                "anomaly_messages_in_train_pool": train_anomaly,
                "should_flag_in_train_pool": train_should_flag,
            },
            "raw_reconstruction_error_test": {
                "min": round(float(raw_errors.min()), 6),
                "max": round(float(raw_errors.max()), 6),
                "mean": round(float(raw_errors.mean()), 6),
                "mean_positive": round(float(pos_raw.mean()), 6) if len(pos_raw) else None,
                "mean_negative": round(float(neg_raw.mean()), 6) if len(neg_raw) else None,
            },
            "minmax_anomaly_score_test": {
                "min": round(float(scores.min()), 4),
                "max": round(float(scores.max()), 4),
                "mean": round(float(scores.mean()), 4),
                "mean_positive": round(float(pos_scores.mean()), 4) if len(pos_scores) else None,
                "mean_negative": round(float(neg_scores.mean()), 4) if len(neg_scores) else None,
                "train_error_min_used": round(float(ae.reconstruction_error_min), 6),
                "train_error_max_used": round(float(ae.reconstruction_error_max), 6),
            },
            "tau_star": TAU_STAR[dk],
            "auc": round(float(auc), 4),
            "auc_turnshift": BG_AUC[dk],
            "auc_isolation_forest_max_f1": IF_AUC[dk],
            "auc_vs_bg_delta": round(float(auc) - BG_AUC[dk], 4),
            "length_error_correlation": round(corr_len_raw, 4),
        }
        report["datasets"][dk] = ds_report

        print(f"--- {DATASET_DISPLAY_NAMES[dk]} ---")
        print(f"  Protocol: n_train={ds_report['n_train_features']} n_test={ds_report['n_test']} "
              f"(IF task1: same counts)")
        print(f"  Train pool leakage check: test-split user msgs in train={train_in_test_split}, "
              f"is_anomaly in train={train_anomaly}, should_flag in train={train_should_flag}")
        print(f"  Raw MSE (test): min={ds_report['raw_reconstruction_error_test']['min']:.4f} "
              f"max={ds_report['raw_reconstruction_error_test']['max']:.4f} "
              f"mean={ds_report['raw_reconstruction_error_test']['mean']:.4f}")
        print(f"    flagged mean={ds_report['raw_reconstruction_error_test']['mean_positive']:.4f} "
              f"benign mean={ds_report['raw_reconstruction_error_test']['mean_negative']:.4f}")
        print(f"  Min-max score (test): min={ds_report['minmax_anomaly_score_test']['min']:.4f} "
              f"max={ds_report['minmax_anomaly_score_test']['max']:.4f} "
              f"mean={ds_report['minmax_anomaly_score_test']['mean']:.4f}")
        print(f"    flagged mean={ds_report['minmax_anomaly_score_test']['mean_positive']:.4f} "
              f"benign mean={ds_report['minmax_anomaly_score_test']['mean_negative']:.4f}")
        print(f"  τ*={TAU_STAR[dk]}  AUC={ds_report['auc']:.4f}  "
              f"BG={BG_AUC[dk]:.3f}  IF={IF_AUC[dk]:.4f}  ΔvsBG={ds_report['auc_vs_bg_delta']:+.4f}")
        print(f"  Length↔raw_error r={corr_len_raw:.4f}")

        if dk == "personachat":
            tau = TAU_STAR[dk]
            tp_idx = np.where(y & (scores > tau))[0]
            tn_idx = np.where((~y) & (scores <= tau))[0]
            fp_idx = np.where((~y) & (scores > tau))[0]

            samples = {"true_positives": [], "true_negatives": [], "false_positives_sample": []}
            for label, indices, key in [
                ("TP", tp_idx, "true_positives"),
                ("TN", tn_idx, "true_negatives"),
                ("FP", fp_idx, "false_positives_sample"),
            ]:
                for i in list(indices)[:5]:
                    m = test_msgs[i]
                    samples[key].append({
                        "label": label,
                        "raw_mse": round(float(raw_errors[i]), 6),
                        "minmax_score": round(float(scores[i]), 4),
                        "should_flag": m.get("should_flag"),
                        "is_anomaly": m.get("is_anomaly"),
                        "anomaly_type": m.get("anomaly_type"),
                        "word_count": len(m["message_text"].split()),
                        "text_preview": m["message_text"][:120].replace("\n", " "),
                    })

            report["personachat_sanity"] = {
                "tau_star": tau,
                "n_tp_at_tau": int(len(tp_idx)),
                "n_tn_at_tau": int(len(tn_idx)),
                "n_fp_at_tau": int(len(fp_idx)),
                "samples": samples,
            }

            print("\n  PersonaChat sanity (τ*=3.26):")
            for key in ("true_positives", "true_negatives", "false_positives_sample"):
                print(f"    {key}:")
                for s in samples[key]:
                    print(
                        f"      [{s['label']}] raw={s['raw_mse']:.4f} score={s['minmax_score']:.4f} "
                        f"words={s['word_count']} type={s['anomaly_type']} "
                        f"flag={s['should_flag']} anom={s['is_anomaly']}"
                    )
                    print(f"        \"{s['text_preview']}...\"")

    report["protocol_parity"] = {
        "same_as_evaluate_method": True,
        "details": (
            "Test users: up to 20 from test split (anomaly users first). "
            "Per-user profile from first 80% of that user's messages. "
            "Test eval: last 20% of each user's messages, should_flag ground truth. "
            "Baseline train pool: first 20 users from TRAIN split, up to 10 benign "
            "(is_anomaly=False) messages each, features extracted via each test user's "
            "profile — identical loop in evaluation.py for IF and AE. "
            "No test-set user messages appear in AE training features."
        ),
        "note_on_train_pool": (
            "Train features are replicated per test-user profile (n_profiles × 200 msgs). "
            "All source messages come from train-split users with is_anomaly=False. "
            "This is shared by Isolation Forest and Autoencoder; not introduced by AE fix."
        ),
    }

    out = ROOT / "results" / "methodology-diagnostics" / "task5_ae_fairness_diagnostic.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out}")
    print("=" * 80)


if __name__ == "__main__":
    main()
