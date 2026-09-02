#!/usr/bin/env python3
"""
Rebuild processed datasets with de-confounded synthetic anomaly injection.

Removes tail-clustered injections, metadata leakage (_ato sessions, operation_risk),
and fixed-template memorization by re-injecting from organic-only timelines.

Usage:
    python tools/rebuild_injected_datasets.py
    python tools/rebuild_injected_datasets.py --datasets personachat
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.anomaly_templates import pick_message  # noqa: E402

INJECTION_SEED = 42
SESSION_GAP_SECONDS = 1800  # 30 minutes — matches organic session boundaries

BENIGN_SURFACE_MARKERS = {
    "What are the best gardening techniques for organic vegetables?",
    "How do I grow tomatoes in my backyard garden?",
}

DATASET_CONFIG: dict[str, dict[str, str]] = {
    "personachat": {
        "input": "datasets/personachat_processed.json",
        "output": "datasets/personachat_processed_corrected.json",
        "dataset_source": "personachat",
    },
    "blended_skill_talk": {
        "input": "datasets/blended_skill_talk_processed.json",
        "output": "datasets/blended_skill_talk_processed_corrected.json",
        "dataset_source": "blended_skill_talk",
    },
    "anthropic_hh": {
        "input": "datasets/anthropic_hh_processed.json",
        "output": "datasets/anthropic_hh_processed_corrected.json",
        "dataset_source": "anthropic_hh",
    },
}


def _user_rng(user_id: str, seed: int):
    import random

    return random.Random(seed ^ (hash(user_id) & 0xFFFFFFFF))


def _surface_style(orig: dict) -> str:
    text = orig.get("message_text", "").strip()
    if text in BENIGN_SURFACE_MARKERS:
        return "benign_surface"
    if orig.get("anomaly_type") == "account_takeover" and not _looks_overt(text):
        return "benign_surface"
    return "overt"


def _looks_overt(text: str) -> bool:
    lower = text.lower()
    overt_markers = (
        "admin", "password", "ignore previous", "jailbreak", "bypass",
        "malicious_", "attacker", "hacked", "export", "delete all",
        "developer mode", "system override", "grant ",
    )
    return any(m in lower for m in overt_markers)


def _injection_plan(orig_anomalies: list[dict]) -> list[tuple[str, str]]:
    """Return list of (anomaly_type, surface_style) preserving original counts."""
    plan: list[tuple[str, str]] = []
    for msg in sorted(orig_anomalies, key=lambda m: m["timestamp"]):
        cat = msg.get("anomaly_type") or "account_takeover"
        plan.append((cat, _surface_style(msg)))
    return plan


def _pick_positions(n_organic: int, k: int, rng) -> list[int]:
    """Distinct insertion indices in [0, n_organic] (inclusive end)."""
    if k == 0:
        return []
    if k > n_organic + 1:
        raise ValueError(f"Cannot insert {k} messages into timeline of {n_organic} organic")
    return sorted(rng.sample(range(n_organic + 1), k))


def _timestamp_for_insert(
    organic: list[dict],
    pos: int,
    rng,
) -> str:
    from datetime import datetime as dt

    if not organic:
        base = dt.now()
        return (base - timedelta(seconds=rng.randint(60, 3600))).isoformat()

    if pos == 0:
        first = dt.fromisoformat(organic[0]["timestamp"])
        return (first - timedelta(seconds=rng.randint(30, 180))).isoformat()

    if pos >= len(organic):
        last = dt.fromisoformat(organic[-1]["timestamp"])
        return (last + timedelta(seconds=rng.randint(30, 180))).isoformat()

    prev = dt.fromisoformat(organic[pos - 1]["timestamp"])
    nxt = dt.fromisoformat(organic[pos]["timestamp"])
    gap = (nxt - prev).total_seconds()
    if gap <= 1:
        offset = 0.5
    elif gap <= 120:
        offset = gap / 2
    else:
        offset = rng.uniform(30, min(gap - 30, 600))
    return (prev + timedelta(seconds=offset)).isoformat()


def _build_injected_message(
    user_id: str,
    text: str,
    anomaly_type: str,
    timestamp: str,
    dataset_source: str,
    inj_idx: int,
) -> dict:
    return {
        "message_id": f"{user_id}_inj_{inj_idx:04d}",
        "user_id": user_id,
        "message_text": text,
        "timestamp": timestamp,
        "is_anomaly": True,
        "should_flag": True,
        "anomaly_type": anomaly_type,
        "dataset_source": dataset_source,
    }


def _assign_sessions_and_gaps(user_id: str, messages: list[dict]) -> None:
    session_idx = 0
    session_id = f"{user_id}_s{session_idx}"
    seq = 0

    for i, msg in enumerate(messages):
        ts = datetime.fromisoformat(msg["timestamp"])
        if i > 0:
            prev_ts = datetime.fromisoformat(messages[i - 1]["timestamp"])
            prev_session = messages[i - 1]["session_id"]
            gap = (ts - prev_ts).total_seconds()
            if gap > SESSION_GAP_SECONDS:
                session_idx += 1
                session_id = f"{user_id}_s{session_idx}"
                seq = 0
                msg["time_since_last_message_seconds"] = 30.0
            elif session_id == prev_session:
                msg["time_since_last_message_seconds"] = max(gap, 0.0)
            else:
                msg["time_since_last_message_seconds"] = 30.0
        else:
            msg["time_since_last_message_seconds"] = 30.0

        seq += 1
        msg["session_id"] = session_id
        msg["sequence_in_session"] = seq


def _strip_leaky_metadata(msg: dict) -> dict:
    out = copy.deepcopy(msg)
    out.pop("operation_risk", None)
    return out


def rebuild_user_timeline(
    user_id: str,
    all_msgs: list[dict],
    dataset_source: str,
    seed: int,
) -> list[dict]:
    organic = sorted(
        [_strip_leaky_metadata(m) for m in all_msgs if not m.get("is_anomaly")],
        key=lambda m: m["timestamp"],
    )
    orig_anomalies = [m for m in all_msgs if m.get("is_anomaly")]
    plan = _injection_plan(orig_anomalies)

    if not plan:
        _assign_sessions_and_gaps(user_id, organic)
        return organic

    rng = _user_rng(user_id, seed)
    positions = _pick_positions(len(organic), len(plan), rng)
    used_texts: set[str] = set()

    # Insert from highest index to lowest to keep positions stable
    timeline = list(organic)
    for inj_num, (pos, (category, surface)) in enumerate(
        sorted(zip(positions, plan), key=lambda x: x[0], reverse=True)
    ):
        for _attempt in range(30):
            text = pick_message(category, surface, rng)
            if text not in used_texts or _attempt == 29:
                used_texts.add(text)
                break
        ts = _timestamp_for_insert(timeline, pos, rng)
        new_msg = _build_injected_message(
            user_id, text, category, ts, dataset_source, inj_num
        )
        timeline.insert(pos, new_msg)

    timeline.sort(key=lambda m: m["timestamp"])
    _assign_sessions_and_gaps(user_id, timeline)
    return timeline


def rebuild_dataset(data: dict, dataset_source: str, seed: int = INJECTION_SEED) -> dict:
    by_user: dict[str, list[dict]] = defaultdict(list)
    for msg in data["messages"]:
        by_user[msg["user_id"]].append(msg)

    new_messages: list[dict] = []
    for user_id in sorted(by_user.keys()):
        new_messages.extend(
            rebuild_user_timeline(user_id, by_user[user_id], dataset_source, seed)
        )

    new_messages.sort(key=lambda m: (m["user_id"], m["timestamp"]))

    n_anom = sum(1 for m in new_messages if m.get("is_anomaly"))
    out = copy.deepcopy(data)
    out["messages"] = new_messages
    out["metadata"] = {
        **data.get("metadata", {}),
        "injection_protocol": "corrected_v1",
        "injection_seed": seed,
        "num_messages": len(new_messages),
        "num_anomalies": n_anom,
        "processing_timestamp": datetime.now().isoformat(),
        "corrections": [
            "randomized_insertion_positions",
            "neutralized_session_ids",
            "removed_operation_risk_field",
            "populated_time_since_last_message_seconds",
            "diversified_anomaly_templates",
        ],
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild de-confounded injected datasets")
    parser.add_argument(
        "--datasets",
        default="all",
        help="Comma-separated keys or 'all'",
    )
    parser.add_argument("--seed", type=int, default=INJECTION_SEED)
    args = parser.parse_args()

    if args.datasets.strip().lower() == "all":
        keys = list(DATASET_CONFIG.keys())
    else:
        keys = [k.strip() for k in args.datasets.split(",") if k.strip()]

    for key in keys:
        cfg = DATASET_CONFIG[key]
        in_path = ROOT / cfg["input"]
        out_path = ROOT / cfg["output"]
        print(f"\n[{key}] Loading {in_path}...")
        with open(in_path, encoding="utf-8") as fh:
            data = json.load(fh)

        orig_anom = sum(1 for m in data["messages"] if m.get("is_anomaly"))
        rebuilt = rebuild_dataset(data, cfg["dataset_source"], seed=args.seed)
        new_anom = sum(1 for m in rebuilt["messages"] if m.get("is_anomaly"))

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(rebuilt, fh, ensure_ascii=False, indent=2)

        print(
            f"  Wrote {out_path.name}: {len(rebuilt['messages'])} msgs, "
            f"{new_anom} anomalies (was {orig_anom})"
        )


if __name__ == "__main__":
    main()
