#!/usr/bin/env python3
"""Validate corrected injection pipeline (Steps 3-5 acceptance checks)."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

ROOT = Path(__file__).resolve().parents[1]

DATASET_FILES = {
    "personachat": "datasets/personachat_processed_corrected.json",
    "blended_skill_talk": "datasets/blended_skill_talk_processed_corrected.json",
    "anthropic_hh": "datasets/anthropic_hh_processed_corrected.json",
}

ORIGINAL_FILES = {
    "personachat": "datasets/personachat_processed.json",
    "blended_skill_talk": "datasets/blended_skill_talk_processed.json",
    "anthropic_hh": "datasets/anthropic_hh_processed.json",
}


def _load(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def validate_dataset(name: str, corrected_path: Path, original_path: Path | None) -> dict:
    data = _load(corrected_path)
    msgs = data["messages"]
    by_user: dict[str, list] = defaultdict(list)
    for m in msgs:
        by_user[m["user_id"]].append(m)

    anom_users = {m["user_id"] for m in msgs if m.get("is_anomaly")}
    test_compositions = []
    full_pct_injected_test = 0
    positions = []

    for uid in sorted(anom_users):
        umsgs = sorted(by_user[uid], key=lambda x: x["timestamp"])
        split = int(len(umsgs) * 0.8)
        train, test = umsgs[:split], umsgs[split:]
        n_test_inj = sum(1 for m in test if m.get("is_anomaly"))
        n_test_org = len(test) - n_test_inj
        pct_inj = 100.0 * n_test_inj / len(test) if test else 0.0
        pct_org = 100.0 * n_test_org / len(test) if test else 0.0
        if test and n_test_org == 0:
            full_pct_injected_test += 1
        test_compositions.append(
            {
                "user_id": uid,
                "n_total": len(umsgs),
                "n_train": split,
                "n_test": len(test),
                "test_injected": n_test_inj,
                "test_organic": n_test_org,
                "pct_test_injected": round(pct_inj, 1),
                "pct_test_organic": round(pct_org, 1),
            }
        )
        positions.extend(i for i, m in enumerate(umsgs) if m.get("is_anomaly"))

    # Metadata leakage checks
    anom_msgs = [m for m in msgs if m.get("is_anomaly")]
    org_msgs = [m for m in msgs if not m.get("is_anomaly")]
    ato_suffix = sum(1 for m in anom_msgs if str(m.get("session_id", "")).endswith("_ato"))
    se_suffix = sum(1 for m in anom_msgs if str(m.get("session_id", "")).endswith("_se"))
    pi_suffix = sum(1 for m in anom_msgs if str(m.get("session_id", "")).endswith("_pi"))
    op_risk_anom = Counter(m.get("operation_risk") for m in anom_msgs)
    op_risk_org = Counter(m.get("operation_risk") for m in org_msgs)
    missing_gap = sum(
        1 for m in msgs if m.get("time_since_last_message_seconds") is None
    )
    unique_texts = len({m["message_text"] for m in anom_msgs})

    # Train window injection rate (should be >0 for most users)
    train_inj_users = 0
    for uid in anom_users:
        umsgs = sorted(by_user[uid], key=lambda x: x["timestamp"])
        split = int(len(umsgs) * 0.8)
        if any(m.get("is_anomaly") for m in umsgs[:split]):
            train_inj_users += 1

    orig_positions = None
    if original_path and original_path.exists():
        orig = _load(original_path)
        ob = defaultdict(list)
        for m in orig["messages"]:
            ob[m["user_id"]].append(m)
        orig_positions = []
        for uid in anom_users:
            umsgs = sorted(ob[uid], key=lambda x: x["timestamp"])
            orig_positions.extend(i for i, m in enumerate(umsgs) if m.get("is_anomaly"))

    summary = {
        "dataset": name,
        "n_messages": len(msgs),
        "n_anomalies": len(anom_msgs),
        "n_anomaly_users": len(anom_users),
        "anomaly_index_min": min(positions) if positions else None,
        "anomaly_index_max": max(positions) if positions else None,
        "anomaly_index_mean": round(mean(positions), 2) if positions else None,
        "orig_anomaly_index_mean": round(mean(orig_positions), 2) if orig_positions else None,
        "users_100pct_injected_test_window": full_pct_injected_test,
        "users_with_train_injections": train_inj_users,
        "test_window_pct_organic_mean": round(
            mean(c["pct_test_organic"] for c in test_compositions), 2
        ),
        "test_window_pct_injected_mean": round(
            mean(c["pct_test_injected"] for c in test_compositions), 2
        ),
        "test_window_pct_organic_min": min(c["pct_test_organic"] for c in test_compositions),
        "leakage": {
            "ato_session_suffix_count": ato_suffix,
            "se_session_suffix_count": se_suffix,
            "pi_session_suffix_count": pi_suffix,
            "operation_risk_anomaly": dict(op_risk_anom),
            "operation_risk_organic": dict(op_risk_org),
            "missing_time_since_last_message": missing_gap,
        },
        "unique_anomaly_texts": unique_texts,
        "per_user_test_composition": test_compositions,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/corrected_injection_validation.json")
    args = parser.parse_args()

    reports = []
    for name, rel in DATASET_FILES.items():
        path = ROOT / rel
        if not path.exists():
            print(f"SKIP {name}: {path} not found (run rebuild first)")
            continue
        orig = ROOT / ORIGINAL_FILES[name]
        print(f"Validating {name}...")
        rep = validate_dataset(name, path, orig)
        reports.append(rep)
        print(
            f"  anomaly users={rep['n_anomaly_users']} "
            f"100% injected test={rep['users_100pct_injected_test_window']} "
            f"mean test organic={rep['test_window_pct_organic_mean']}% "
            f"index mean {rep['orig_anomaly_index_mean']} -> {rep['anomaly_index_mean']}"
        )
        print(
            f"  leakage: ato={rep['leakage']['ato_session_suffix_count']} "
            f"op_risk stored={rep['leakage']['operation_risk_anomaly']} "
            f"unique texts={rep['unique_anomaly_texts']}"
        )

    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(reports, fh, indent=2, ensure_ascii=False)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
