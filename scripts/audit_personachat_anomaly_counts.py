#!/usr/bin/env python3
"""Audit anomaly counts and label fields in corrected PersonaChat."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "personachat_processed_corrected.json"


def conv_key(m: dict) -> str:
    return str(m.get("conversation_id") or m.get("session_id") or m.get("user_id"))


def is_anom_should_flag(m: dict) -> bool:
    return bool(m.get("should_flag", False))


def is_anom_is_anomaly(m: dict) -> bool:
    return bool(m.get("is_anomaly", False))


def split_8020_by_user(messages: list[dict]) -> tuple[list[dict], list[dict]]:
    by_user: dict[str, list] = defaultdict(list)
    for m in messages:
        by_user[m["user_id"]].append(m)
    train, test = [], []
    for uid in by_user:
        msgs = sorted(by_user[uid], key=lambda x: x["timestamp"])
        split_idx = int(len(msgs) * 0.8)
        train.extend(msgs[:split_idx])
        test.extend(msgs[split_idx:])
    return train, test


def main() -> None:
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    messages = data["messages"]

    print("=== SCHEMA ===")
    sample = messages[0]
    print("Keys:", sorted(sample.keys()))
    anom_sample = next(m for m in messages if m.get("should_flag") or m.get("is_anomaly"))
    print("Anomalous sample fields:")
    for k in sorted(anom_sample.keys()):
        print(f"  {k}: {anom_sample[k]!r}")

    sf_true = sum(1 for m in messages if is_anom_should_flag(m))
    ia_true = sum(1 for m in messages if is_anom_is_anomaly(m))
    mismatch = sum(
        1 for m in messages if is_anom_should_flag(m) != is_anom_is_anomaly(m)
    )
    print(f"\nFull dataset messages: {len(messages)}")
    print(f"should_flag=True: {sf_true}")
    print(f"is_anomaly=True:  {ia_true}")
    print(f"Field mismatches:  {mismatch}")

    # Conversations (session-level)
    by_conv = defaultdict(list)
    for m in messages:
        by_conv[(m["user_id"], conv_key(m))].append(m)
    full_convs = len(by_conv)
    full_anom_convs = sum(
        1 for ms in by_conv.values() if any(is_anom_should_flag(x) for x in ms)
    )

    train, test = split_8020_by_user(messages)
    test_by_conv = defaultdict(list)
    for m in test:
        test_by_conv[(m["user_id"], conv_key(m))].append(m)

    test_convs = len(test_by_conv)
    test_anom_convs = sum(
        1 for ms in test_by_conv.values() if any(is_anom_should_flag(x) for x in ms)
    )
    test_anom_msgs_sf = sum(1 for m in test if is_anom_should_flag(m))
    test_anom_msgs_ia = sum(1 for m in test if is_anom_is_anomaly(m))
    train_anom_msgs = sum(1 for m in train if is_anom_should_flag(m))

    print("\n=== FULL DATASET (before split) ===")
    print(f"Conversations (user, session): {full_convs}")
    print(f"Conversations with >=1 anomalous msg: {full_anom_convs} ({100*full_anom_convs/full_convs:.2f}%)")
    print(f"Anomalous messages: {sf_true}")

    print("\n=== TEST SPLIT (20%, per-user index split) ===")
    print(f"Test messages: {len(test)}")
    print(f"Test anomalous messages (should_flag): {test_anom_msgs_sf}")
    print(f"Test anomalous messages (is_anomaly):  {test_anom_msgs_ia}")
    print(f"Train anomalous messages: {train_anom_msgs}")
    print(f"Test conversations: {test_convs}")
    print(f"Test conversations with >=1 anomalous msg: {test_anom_convs} ({100*test_anom_convs/test_convs:.2f}%)")
    print(f"Pct of full anomalies landing in test: {100*test_anom_msgs_sf/max(1,sf_true):.2f}%")

    # Users with anomalies
    by_user = defaultdict(list)
    for m in messages:
        by_user[m["user_id"]].append(m)
    users_with_any_anom = sum(
        1 for ms in by_user.values() if any(is_anom_should_flag(x) for x in ms)
    )
    users_test_anom = 0
    per_user_test_anom = []
    for uid, msgs in by_user.items():
        msgs = sorted(msgs, key=lambda x: x["timestamp"])
        split_idx = int(len(msgs) * 0.8)
        test_part = msgs[split_idx:]
        n = sum(1 for m in test_part if is_anom_should_flag(m))
        if n:
            users_test_anom += 1
            per_user_test_anom.append((uid, n, len(test_part)))

    print(f"\nUsers with any anomaly (full timeline): {users_with_any_anom} / {len(by_user)}")
    print(f"Users with >=1 anomalous TEST message: {users_test_anom}")
    print(f"Top users by test anomalies: {sorted(per_user_test_anom, key=lambda x: -x[1])[:10]}")

    # Position analysis: where do anomalies fall in timeline?
    rel_positions = []
    in_test_window = 0
    for uid, msgs in by_user.items():
        msgs = sorted(msgs, key=lambda x: x["timestamp"])
        n = len(msgs)
        split_idx = int(n * 0.8)
        for i, m in enumerate(msgs):
            if is_anom_should_flag(m):
                rel_positions.append(i / max(1, n - 1))
                if i >= split_idx:
                    in_test_window += 1

    print(f"\n=== SPLIT BIAS CHECK ===")
    print(f"Anomalies in test window (index >= 80%): {in_test_window} / {sf_true}")
    print(f"Mean relative timeline position of anomalies: {sum(rel_positions)/len(rel_positions):.3f}")

    if "metadata" in data:
        print("\n=== DATASET METADATA ===")
        print(json.dumps(data["metadata"], indent=2))

    # collect_personachat label extraction simulation
    labels_from_collector = [
        1 if m.get("should_flag", False) else 0 for m in test
    ]
    labels_alt = [
        1 if m.get("is_anomaly", False) else 0 for m in test
    ]
    print(f"\n=== LABEL EXTRACTION (collect_personachat_production_scores) ===")
    print(f"Using should_flag: sum={sum(labels_from_collector)}")
    print(f"Using is_anomaly:  sum={sum(labels_alt)}")
    print(f"Default-false-only if wrong key: would miss {test_anom_msgs_ia - test_anom_msgs_sf} if is_anomaly used vs should_flag")


if __name__ == "__main__":
    main()
