#!/usr/bin/env python3
"""
Build a third ATO episode dataset from real multi-turn chat traffic.

Fallback ladder (first reachable wins):
  1. allenai/WildChat-1M (hashed hashed_ip / conversation user)
  2. lmsys/lmsys-chat-1m
  3. RyokoAI/ShareGPT52K (conversation-as-user; weaker identity)

Outputs datasets/wildchat_ato_episodes.json in the same schema as the
PersonaChat/BST episode files so sequential_ato_study.py --dataset wildchat works.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SEED = 42
N_EPISODE_USERS = 300
K_MIN, K_MAX = 5, 10
TRAIN_FRACTION = 0.8
MIN_TRAIN_MSGS = 5
MIN_TAIL_MSGS = 1
EPISODE_GAP_SECONDS = 300
MAX_USERS = 5000
OUTPUT = ROOT / "datasets" / "wildchat_ato_episodes.json"
META_OUT = ROOT / "results" / "wildchat_corpus_acquisition.json"


def _try_load():
    from datasets import load_dataset

    attempts = []
    # ShareGPT first for speed; WildChat/LMSYS if already cached
    for name, loader in [
        ("RyokoAI/ShareGPT52K", lambda: load_dataset("RyokoAI/ShareGPT52K", split="train")),
        ("allenai/WildChat-1M", lambda: load_dataset("allenai/WildChat-1M", split="train", streaming=False)),
        ("lmsys/lmsys-chat-1m", lambda: load_dataset("lmsys/lmsys-chat-1m", split="train")),
    ]:
        print(f"TRY {name}", flush=True)
        try:
            ds = loader()
            print(f"  OK n={len(ds)} cols={ds.column_names[:10]}", flush=True)
            return name, ds, attempts
        except Exception as e:
            msg = f"{type(e).__name__}: {str(e)[:240]}"
            print(f"  FAIL {msg}", flush=True)
            attempts.append({"name": name, "error": msg})
    return None, None, attempts


def _extract_user_messages(name: str, row: dict) -> tuple[str | None, list[str]]:
    """Return (user_id, list of user-role utterance texts)."""
    if "ShareGPT" in name:
        conv = row.get("conversations") or row.get("conversation") or []
        uid = str(row.get("id") or row.get("hash") or id(row))
        texts = []
        for turn in conv:
            role = (turn.get("from") or turn.get("role") or "").lower()
            val = turn.get("value") or turn.get("content") or ""
            if role in ("human", "user") and val.strip():
                texts.append(val.strip())
        return uid, texts

    # WildChat / LMSYS: conversation is list of {role, content}
    conv = row.get("conversation") or row.get("messages") or []
    uid = str(row.get("hashed_ip") or row.get("user_id") or row.get("conversation_id") or row.get("id") or "")
    texts = []
    for turn in conv:
        role = (turn.get("role") or "").lower()
        content = turn.get("content") or ""
        if role == "user" and content.strip():
            texts.append(content.strip()[:2000])
    return (uid or None), texts


def build_streams(name: str, ds) -> tuple[list[dict], dict]:
    by_user: dict[str, list[str]] = defaultdict(list)
    n_rows = 0
    for row in ds:
        n_rows += 1
        if n_rows > 200_000:  # hard cap for ShareGPT-scale processing time
            break
        uid, texts = _extract_user_messages(name, row)
        if not uid or len(texts) < 2:
            continue
        by_user[uid].extend(texts)
        if len(by_user) >= MAX_USERS * 3:
            # enough candidates
            pass

    # Keep users with enough messages
    eligible_raw = {u: msgs for u, msgs in by_user.items() if len(msgs) >= 8}
    rng = random.Random(SEED)
    keep_uids = sorted(eligible_raw.keys())
    if len(keep_uids) > MAX_USERS:
        keep_uids = sorted(rng.sample(keep_uids, MAX_USERS))

    base = datetime(2024, 1, 1, 12, 0, 0)
    by_msgs: dict[str, list[dict]] = {}
    for uid in keep_uids:
        texts = eligible_raw[uid][:80]  # cap length
        msgs = []
        for i, t in enumerate(texts):
            msgs.append(
                {
                    "message_text": t,
                    "timestamp": (base + timedelta(minutes=5 * i)).isoformat(),
                    "session_id": f"{uid}_s0",
                }
            )
        by_msgs[uid] = msgs

    eligible = {}
    for uid, msgs in by_msgs.items():
        split = int(len(msgs) * TRAIN_FRACTION)
        train, tail = msgs[:split], msgs[split:]
        if len(train) >= MIN_TRAIN_MSGS and len(tail) >= MIN_TAIL_MSGS:
            eligible[uid] = (train, tail)

    n_ep = min(N_EPISODE_USERS, max(1, len(eligible) // 10))
    episode_users = set(rng.sample(sorted(eligible.keys()), n_ep))
    donor_counts = {uid: len(msgs) for uid, msgs in by_msgs.items()}

    streams = []
    k_values = []
    n_ep_msgs = 0
    n_benign = 0
    for uid in sorted(eligible.keys()):
        train, tail = eligible[uid]
        record = {
            "user_id": uid,
            "train": train,
            "stream": [{**m, "is_episode": False} for m in tail],
            "episode": None,
        }
        n_benign += len(tail)
        if uid in episode_users:
            k = rng.randint(K_MIN, K_MAX)
            donors = [d for d, c in donor_counts.items() if d != uid and c >= k]
            if not donors:
                streams.append(record)
                continue
            donor_id = rng.choice(sorted(donors))
            donor_msgs = by_msgs[donor_id]
            start = rng.randint(0, len(donor_msgs) - k)
            segment = donor_msgs[start : start + k]
            last_ts = datetime.fromisoformat(tail[-1]["timestamp"])
            ep_start = len(record["stream"])
            for j, dm in enumerate(segment):
                ts = last_ts + timedelta(seconds=EPISODE_GAP_SECONDS * (j + 1))
                record["stream"].append(
                    {
                        "message_text": dm["message_text"],
                        "timestamp": ts.isoformat(),
                        "session_id": tail[-1]["session_id"],
                        "is_episode": True,
                    }
                )
            record["episode"] = {"start_idx": ep_start, "length": k, "donor_id": donor_id}
            k_values.append(k)
            n_ep_msgs += k
        streams.append(record)

    audit = {
        "source_corpus": name,
        "n_users_total": len(by_msgs),
        "n_eligible_users": len(eligible),
        "n_streams": len(streams),
        "n_episode_streams": len(k_values),
        "n_benign_streams": len(streams) - len(k_values),
        "n_benign_stream_messages": n_benign,
        "n_episode_messages": n_ep_msgs,
        "episode_length_distribution": {str(k): k_values.count(k) for k in range(K_MIN, K_MAX + 1)},
        "stream_level_prevalence": round(len(k_values) / max(len(streams), 1), 4),
        "identity_note": (
            "ShareGPT uses conversation-as-user (no cross-session identity); "
            "WildChat/LMSYS use hashed user ids when available."
        ),
    }
    return streams, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-sharegpt", action="store_true")
    args = parser.parse_args()

    name, ds, attempts = _try_load()
    meta = {"attempts": attempts, "selected": name}
    if name is None or ds is None:
        META_OUT.parent.mkdir(parents=True, exist_ok=True)
        META_OUT.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        raise SystemExit("No real-traffic corpus could be loaded. See " + str(META_OUT))

    streams, audit = build_streams(name, ds)
    out = {
        "config": {
            "source": name,
            "seed": SEED,
            "n_episode_users": audit["n_episode_streams"],
            "k_range": [K_MIN, K_MAX],
            "train_fraction": TRAIN_FRACTION,
            "injection_style": "author_substitution_contiguous_donor_segment",
        },
        "audit": audit,
        "streams": streams,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    meta["audit"] = audit
    META_OUT.parent.mkdir(parents=True, exist_ok=True)
    META_OUT.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
