#!/usr/bin/env python3
"""
Investigate TurnShift detection of category-(b) gardening positives vs benign.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

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
from turnshift import TurnShiftEvaluatorML
from turnshift.baselines.autoencoder_baseline import AutoencoderBaseline
from turnshift.baselines.isolation_forest_baseline import IsolationForestBaseline
from turnshift.models import EvaluationInput, SystemConfig
from scripts.task5_autoencoder_rerun import (
    _build_autoencoder_eval_config,
    _prepare_baseline_eval,
)

BENIGN_TEMPLATES = {
    "What are the best gardening techniques for organic vegetables?",
    "How do I grow tomatoes in my backyard garden?",
}
CANONICAL_CONFIG = SystemConfig(
    sensitivity_level="medium",
    deployment_context="enterprise",
    overrides_enabled=False,
)
PM_BUILDER = ev._build_profile_with_pm(ev.CANONICAL_LAMBDA)
THRESHOLD = 0.60


def _collect_eval_users(test_data: dict, max_users: int = 20) -> dict:
    test_user_ids = set(test_data["splits"]["test"]["user_ids"])
    test_users = [u for u in test_data["users"] if u["user_id"] in test_user_ids]
    test_messages_by_user: dict[str, list] = defaultdict(list)
    for m in test_data["messages"]:
        if m["user_id"] in test_user_ids:
            test_messages_by_user[m["user_id"]].append(m)

    with_a, without = [], []
    for user in test_users:
        msgs = test_messages_by_user[user["user_id"]]
        (with_a if any(m.get("should_flag") for m in msgs) else without).append(user)

    sampled = with_a[:max_users]
    sampled.extend(without[: max_users - len(sampled)])

    profiles = {}
    for user in sampled:
        msgs = test_messages_by_user[user["user_id"]]
        split_idx = int(len(msgs) * 0.8)
        profile = PM_BUILDER(user, msgs[:split_idx])
        if profile:
            profiles[user["user_id"]] = {
                "profile": profile,
                "all_msgs": msgs,
                "test_msgs": msgs[split_idx:],
                "split_idx": split_idx,
            }
    return profiles


def _score_bg(evaluator, profile, msg, prev_msg):
    cur = ev.message_to_current_message(msg, prev_msg, user_profile=profile)
    result = evaluator.evaluate(
        EvaluationInput(
            user_profile=profile,
            current_message=cur,
            system_config=CANONICAL_CONFIG,
        )
    )
    cs = result.component_scores
    w = {"semantic": 0.4, "linguistic": 0.35, "temporal": 0.25}
    composite_manual = (
        w["semantic"] * cs.semantic
        + w["linguistic"] * cs.linguistic
        + w["temporal"] * cs.temporal
    )
    return {
        "s_sem": cs.semantic,
        "s_ling": cs.linguistic,
        "s_temp": cs.temporal,
        "composite": result.anomaly_score,
        "composite_manual": composite_manual,
        "mechanism": result.metadata.get("detection_mechanism"),
        "flagged": result.anomaly_score > THRESHOLD,
    }


def _prev_in_session(test_msgs, i):
    if i == 0:
        return None
    p = test_msgs[i - 1]
    if p.get("session_id", "session_0") == test_msgs[i].get("session_id", "session_0"):
        return p
    return None


def investigate_personachat():
    td = ev.datasets["personachat"]
    profiles = _collect_eval_users(td)
    evaluator = TurnShiftEvaluatorML()

    b_positives = []
    for uid, ud in profiles.items():
        for i, msg in enumerate(ud["test_msgs"]):
            if msg.get("should_flag") and msg["message_text"].strip() in BENIGN_TEMPLATES:
                b_positives.append((uid, i, msg, ud))

    print("=" * 90)
    print(f"PERSONACHAT category-(b) gardening positives: {len(b_positives)}")
    print("=" * 90)

    bg_rows = []
    for n, (uid, idx, msg, ud) in enumerate(b_positives, 1):
        prev = _prev_in_session(ud["test_msgs"], idx)
        scores = _score_bg(evaluator, ud["profile"], msg, prev)
        bg_rows.append({"user_id": uid, "msg": msg, "scores": scores})

        print(f"\n--- B-positive #{n} user={uid} idx_in_test={idx} ---")
        print(f"Text: {msg['message_text']}")
        print(
            f"  s_sem={scores['s_sem']:.4f} s_ling={scores['s_ling']:.4f} "
            f"s_temp={scores['s_temp']:.4f} composite={scores['composite']:.4f} "
            f"flagged={scores['flagged']} mechanism={scores['mechanism']}"
        )

        benign_same_user = [
            (j, m)
            for j, m in enumerate(ud["test_msgs"])
            if not m.get("should_flag") and j != idx
        ][:5]
        print(f"  Compared to {len(benign_same_user)} benign test msgs from same user:")
        for j, bm in benign_same_user:
            bprev = _prev_in_session(ud["test_msgs"], j)
            bs = _score_bg(evaluator, ud["profile"], bm, bprev)
            print(
                f"    [benign test#{j}] s_sem={bs['s_sem']:.4f} s_ling={bs['s_ling']:.4f} "
                f"s_temp={bs['s_temp']:.4f} composite={bs['composite']:.4f} | "
                f"{bm['message_text'][:70]}"
            )

    # Baselines on same messages
    train_f, test_f, y_true, _ = _prepare_baseline_eval(td, 20)
    ae = _build_autoencoder_eval_config(train_f.shape[1])
    ae.fit(train_f, verbose=False)
    iso = IsolationForestBaseline(contamination="auto", random_state=ev.SEED)
    iso.fit(train_f)

    test_msg_list = []
    for uid, ud in profiles.items():
        for msg in ud["test_msgs"]:
            test_msg_list.append((uid, msg))

    print("\n" + "=" * 90)
    print("BASELINE SCORES on category-(b) PersonaChat messages")
    print("=" * 90)
    baseline_b = []
    for uid, msg in test_msg_list:
        if msg.get("should_flag") and msg["message_text"].strip() in BENIGN_TEMPLATES:
            profile = profiles[uid]["profile"]
            feat = ev.extract_features_for_baselines(msg, profile)
            ae_r = ae.detect_single(feat)
            iso_r = iso.detect_single(feat)
            row = {
                "text": msg["message_text"],
                "ae_score": ae_r["anomaly_score"],
                "ae_raw_mse": ae_r["reconstruction_error"],
                "iso_score": iso_r["anomaly_score"],
                "ae_flagged": ae_r["anomaly_score"] > THRESHOLD,
                "iso_flagged": iso_r["anomaly_score"] > THRESHOLD,
            }
            baseline_b.append(row)
            print(
                f"  AE={row['ae_score']:.4f} (raw_mse={row['ae_raw_mse']:.4f}) "
                f"IF={row['iso_score']:.4f} | {row['text'][:60]}"
            )

    # Compare AE scores: b-positive vs overt-positive vs benign-negative
    overt_scores, benign_scores = [], []
    for uid, msg in test_msg_list:
        profile = profiles[uid]["profile"]
        feat = ev.extract_features_for_baselines(msg, profile)
        sc = ae.detect_single(feat)["anomaly_score"]
        text = msg["message_text"].strip()
        if msg.get("should_flag"):
            if text in BENIGN_TEMPLATES:
                pass  # already in baseline_b
            else:
                overt_scores.append(sc)
        else:
            benign_scores.append(sc)

    print("\nAE score summary (PersonaChat test set):")
    print(f"  category-(b) positives: mean={mean(r['ae_score'] for r in baseline_b):.4f}")
    print(f"  overt positives:        mean={mean(overt_scores):.4f} n={len(overt_scores)}")
    print(f"  benign negatives:       mean={mean(benign_scores):.4f} n={len(benign_scores)}")

    return bg_rows, baseline_b, profiles


def analyze_injection_metadata(profiles: dict, dataset_key: str = "personachat"):
    td = ev.datasets[dataset_key]
    all_msgs_by_user = defaultdict(list)
    for m in td["messages"]:
        all_msgs_by_user[m["user_id"]].append(m)

    b_records, tn_records, overt_records = [], [], []
    for uid, ud in profiles.items():
        all_msgs = ud["all_msgs"]
        split_idx = ud["split_idx"]
        test_msgs = ud["test_msgs"]
        for local_i, msg in enumerate(test_msgs):
            global_i = split_idx + local_i
            rec = {
                "user_id": uid,
                "local_test_idx": local_i,
                "global_idx": global_i,
                "global_frac": global_i / max(len(all_msgs) - 1, 1),
                "text": msg["message_text"].strip(),
                "should_flag": msg.get("should_flag"),
                "is_anomaly": msg.get("is_anomaly"),
                "anomaly_type": msg.get("anomaly_type"),
                "timestamp": msg.get("timestamp"),
                "time_gap": msg.get("time_since_last_message_seconds"),
                "session_id": msg.get("session_id"),
                "seq_in_session": msg.get("sequence_in_session"),
                "word_count": len(msg["message_text"].split()),
                "operation_risk": msg.get("operation_risk"),
            }
            if msg.get("should_flag") and rec["text"] in BENIGN_TEMPLATES:
                b_records.append(rec)
            elif msg.get("should_flag"):
                overt_records.append(rec)
            else:
                tn_records.append(rec)

    def summarize(name, records):
        if not records:
            return {}
        gaps = [r["time_gap"] for r in records if r["time_gap"] is not None]
        return {
            "n": len(records),
            "global_idx_mean": round(mean(r["global_idx"] for r in records), 2),
            "global_idx_median": median(r["global_idx"] for r in records),
            "global_frac_mean": round(mean(r["global_frac"] for r in records), 3),
            "local_test_idx_mean": round(mean(r["local_test_idx"] for r in records), 2),
            "word_count_mean": round(mean(r["word_count"] for r in records), 2),
            "time_gap_mean": round(mean(gaps), 2) if gaps else None,
            "time_gap_median": round(median(gaps), 2) if gaps else None,
            "unique_sessions": len(set(r["session_id"] for r in records)),
            "seq_in_session_mean": round(mean(r["seq_in_session"] for r in records), 2)
            if all(r["seq_in_session"] is not None for r in records)
            else None,
        }

    print("\n" + "=" * 90)
    print("INJECTION METADATA ANALYSIS (PersonaChat eval test split)")
    print("=" * 90)
    for label, recs in [
        ("category-(b) positives", b_records),
        ("overt positives", overt_records),
        ("true negatives (test)", tn_records),
    ]:
        s = summarize(label, recs)
        print(f"\n{label}: {s}")

    # Global dataset: all gardening injections across all users
    all_gardening = []
    all_organic_benign = []
    for uid, msgs in all_msgs_by_user.items():
        for i, m in enumerate(msgs):
            t = m["message_text"].strip()
            if t in BENIGN_TEMPLATES:
                all_gardening.append({
                    "user_id": uid,
                    "global_idx": i,
                    "is_anomaly": m.get("is_anomaly"),
                    "should_flag": m.get("should_flag"),
                    "time_gap": m.get("time_since_last_message_seconds"),
                    "text": t,
                })
            elif not m.get("is_anomaly") and "garden" in t.lower():
                all_organic_benign.append(m)

    inj_indices = [g["global_idx"] for g in all_gardening if g["is_anomaly"]]
    print(f"\nFull dataset: {len(all_gardening)} gardening-template messages total")
    if inj_indices:
        print(
            f"  Injected (is_anomaly=True) global index: min={min(inj_indices)} "
            f"max={max(inj_indices)} mean={mean(inj_indices):.1f} "
            f"unique positions={len(set(inj_indices))}"
        )
        from collections import Counter

        pos_counts = Counter(inj_indices)
        print(f"  Most common injection global indices: {pos_counts.most_common(10)}")

    gaps_inj = [g["time_gap"] for g in all_gardening if g["is_anomaly"] and g["time_gap"]]
    gaps_all = [
        m.get("time_since_last_message_seconds")
        for msgs in all_msgs_by_user.values()
        for m in msgs
        if not m.get("is_anomaly") and m.get("time_since_last_message_seconds") is not None
    ]
    if gaps_inj and gaps_all:
        print(
            f"  time_gap injected gardening: mean={mean(gaps_inj):.1f}s "
            f"median={median(gaps_inj):.1f}s"
        )
        print(
            f"  time_gap organic benign:     mean={mean(gaps_all):.1f}s "
            f"median={median(gaps_all):.1f}s"
        )

    # Search repo for injection script references
    print("\nInjection pipeline in repo:")
    candidates = list(ROOT.glob("**/*process*")) + list(ROOT.glob("**/*inject*"))
    py_scripts = [p for p in candidates if p.suffix == ".py" and "scripts" in str(p)]
    if not py_scripts:
        print("  NO dataset generation/injection script found in committed repo.")
        print("  Only processed JSON artifacts under datasets/.")

    return {
        "b_records": b_records,
        "overt_records": overt_records,
        "tn_records": tn_records,
        "all_gardening": all_gardening,
    }


def main():
    bg_rows, baseline_b, profiles = investigate_personachat()
    meta = analyze_injection_metadata(profiles)

    out = ROOT / "results" / "methodology-diagnostics" / "gardening_positive_investigation.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "bg_category_b": [
                    {
                        "text": r["msg"]["message_text"],
                        "user_id": r["user_id"],
                        **r["scores"],
                    }
                    for r in bg_rows
                ],
                "baseline_category_b": baseline_b,
                "metadata": {
                    "b_test": meta["b_records"],
                    "overt_test": meta["overt_records"][:5],
                },
            },
            fh,
            indent=2,
            default=str,
        )
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
