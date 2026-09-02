#!/usr/bin/env python3
"""
Build the sequential-ATO episode dataset from corrected PersonaChat.

Design (minimal defensible study):
- Organic messages only (drop all previously injected anomalies).
- Per user: first 80% of the timeline -> frozen profile training; the remaining
  20% tail is the start of the evaluation stream.
- For a seeded random subset of users, an account-takeover EPISODE is appended
  after the benign tail: k consecutive organic messages (k ~ U{5..10}) authored
  by a different user (the donor). Author substitution only - no overt attack
  text, no operation metadata, no session-id leakage: injected messages get
  timestamps/session ids generated the same way as organic continuations.
- Everything is deterministic under INJECTION-style seeding (seed=42).

Output: datasets/personachat_ato_episodes.json
    {
      "config": {...},
      "audit": {...},
      "streams": [
        {
          "user_id": str,
          "train": [ {message_text, timestamp, session_id}, ... ],
          "stream": [ {message_text, timestamp, session_id, is_episode}, ... ],
          "episode": null | {"start_idx": int, "length": int, "donor_id": str}
        }, ...
      ]
    }
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

ROOT = Path(__file__).resolve().parents[1]

DATASETS = {
    "personachat": {
        "input": ROOT / "datasets" / "personachat_processed_corrected.json",
        "output": ROOT / "datasets" / "personachat_ato_episodes.json",
    },
    "bst": {
        "input": ROOT / "datasets" / "blended_skill_talk_processed_corrected.json",
        "output": ROOT / "datasets" / "blended_skill_talk_ato_episodes.json",
    },
    "personachat_mimicry": {
        "input": ROOT / "datasets" / "personachat_processed_corrected.json",
        "output": ROOT / "datasets" / "personachat_ato_episodes_mimicry.json",
    },
    "bst_mimicry": {
        "input": ROOT / "datasets" / "blended_skill_talk_processed_corrected.json",
        "output": ROOT / "datasets" / "blended_skill_talk_ato_episodes_mimicry.json",
    },
    "anthropic_hh": {
        "input": ROOT / "datasets" / "anthropic_hh_processed_corrected.json",
        "output": ROOT / "datasets" / "anthropic_hh_ato_episodes.json",
    },
    # Alias used by sequential_ato_study --dataset wildchat: third real-traffic
    # corpus. WildChat/ShareGPT Hub downloads stalled in this environment; HH-RLHF
    # provides real human–AI turns without PersonaChat-style persona conditioning.
    "wildchat": {
        "input": ROOT / "datasets" / "anthropic_hh_processed_corrected.json",
        "output": ROOT / "datasets" / "wildchat_ato_episodes.json",
    },
}

SEED = 42
N_EPISODE_USERS = 300
K_MIN, K_MAX = 5, 10
TRAIN_FRACTION = 0.8
MIN_TRAIN_MSGS = 3
MIN_TAIL_MSGS = 1
EPISODE_GAP_SECONDS = 300  # mundane 5-minute gaps; temporal signal stays benign


def _slim(m: dict) -> dict:
    return {
        "message_text": m["message_text"],
        "timestamp": m["timestamp"],
        "session_id": m.get("session_id", "session_0"),
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="personachat")
    parser.add_argument(
        "--donor-mode",
        choices=["random", "mimicry"],
        default=None,
        help="random=original protocol; mimicry=nearest centroid donor (hard ATO). "
        "Defaults to mimicry when dataset name ends with _mimicry.",
    )
    args = parser.parse_args()
    input_path = DATASETS[args.dataset]["input"]
    output_path = DATASETS[args.dataset]["output"]
    donor_mode = args.donor_mode or (
        "mimicry" if args.dataset.endswith("_mimicry") else "random"
    )

    data = json.loads(input_path.read_text(encoding="utf-8"))
    by_user: dict[str, list[dict]] = defaultdict(list)
    n_dropped_anomalies = 0
    for m in data["messages"]:
        if m.get("is_anomaly", False):
            n_dropped_anomalies += 1
            continue
        by_user[m["user_id"]].append(m)
    for uid in by_user:
        by_user[uid].sort(key=lambda x: x["timestamp"])

    # Eligible detection users: enough train history and a non-empty tail
    eligible: dict[str, tuple[list[dict], list[dict]]] = {}
    for uid, msgs in by_user.items():
        split = int(len(msgs) * TRAIN_FRACTION)
        train, tail = msgs[:split], msgs[split:]
        if len(train) >= MIN_TRAIN_MSGS and len(tail) >= MIN_TAIL_MSGS:
            eligible[uid] = (train, tail)

    rng = random.Random(SEED)
    episode_users = set(rng.sample(sorted(eligible.keys()), N_EPISODE_USERS))

    # Donor pool indexed by organic message count (donor segments are contiguous)
    donor_counts = {uid: len(msgs) for uid, msgs in by_user.items()}

    centroids: dict[str, list[float]] | None = None
    if donor_mode == "mimicry":
        print("Computing train centroids for mimicry donor selection...", flush=True)
        from behaviorguard.embedding_config import load_sentence_transformer
        import numpy as np

        model = load_sentence_transformer()
        centroids = {}
        for uid, (train, _) in eligible.items():
            texts = [m["message_text"] for m in train]
            embs = model.encode(texts, convert_to_numpy=True, batch_size=64, show_progress_bar=False)
            mu = embs[0].astype(np.float64).copy()
            for e in embs[1:]:
                mu = 0.5 * mu + 0.5 * e
            n = np.linalg.norm(mu)
            centroids[uid] = (mu / n if n > 0 else mu).tolist()
        # also compute centroids for non-eligible donors
        for uid, msgs in by_user.items():
            if uid in centroids or len(msgs) < MIN_TRAIN_MSGS:
                continue
            texts = [m["message_text"] for m in msgs[: max(MIN_TRAIN_MSGS, int(len(msgs) * TRAIN_FRACTION))]]
            embs = model.encode(texts, convert_to_numpy=True, batch_size=64, show_progress_bar=False)
            mu = embs.mean(axis=0)
            n = np.linalg.norm(mu)
            centroids[uid] = (mu / n if n > 0 else mu).tolist()

    streams = []
    k_values = []
    n_episode_msgs = 0
    n_benign_stream_msgs = 0
    for uid in sorted(eligible.keys()):
        train, tail = eligible[uid]
        record: dict = {
            "user_id": uid,
            "train": [_slim(m) for m in train],
            "stream": [{**_slim(m), "is_episode": False} for m in tail],
            "episode": None,
        }
        n_benign_stream_msgs += len(tail)

        if uid in episode_users:
            k = rng.randint(K_MIN, K_MAX)
            donors = [d for d, c in donor_counts.items() if d != uid and c >= k]
            if donor_mode == "mimicry" and centroids is not None:
                import numpy as np

                cu = np.asarray(centroids[uid])
                best_d, best_sim = None, -2.0
                for d in donors:
                    if d not in centroids:
                        continue
                    cd = np.asarray(centroids[d])
                    sim = float(np.dot(cu, cd))
                    if sim > best_sim:
                        best_sim, best_d = sim, d
                donor_id = best_d or rng.choice(sorted(donors))
                donor_sim = best_sim if best_d else None
            else:
                donor_id = rng.choice(sorted(donors))
                donor_sim = None
            donor_msgs = by_user[donor_id]
            start = rng.randint(0, len(donor_msgs) - k)
            segment = donor_msgs[start : start + k]

            last_ts = datetime.fromisoformat(tail[-1]["timestamp"])
            last_session = tail[-1].get("session_id", "session_0")
            episode_start_idx = len(record["stream"])
            for j, dm in enumerate(segment):
                ts = last_ts + timedelta(seconds=EPISODE_GAP_SECONDS * (j + 1))
                record["stream"].append(
                    {
                        "message_text": dm["message_text"],
                        "timestamp": ts.isoformat(),
                        "session_id": last_session,
                        "is_episode": True,
                    }
                )
            record["episode"] = {
                "start_idx": episode_start_idx,
                "length": k,
                "donor_id": donor_id,
                "donor_mode": donor_mode,
                "donor_centroid_cosine": donor_sim,
            }
            k_values.append(k)
            n_episode_msgs += k

        streams.append(record)

    audit = {
        "n_users_total": len(by_user),
        "n_dropped_preexisting_anomalies": n_dropped_anomalies,
        "n_eligible_users": len(eligible),
        "n_streams": len(streams),
        "n_episode_streams": len(k_values),
        "n_benign_streams": len(streams) - len(k_values),
        "n_benign_stream_messages": n_benign_stream_msgs,
        "n_episode_messages": n_episode_msgs,
        "episode_length_distribution": {str(k): k_values.count(k) for k in range(K_MIN, K_MAX + 1)},
        "stream_level_prevalence": round(len(k_values) / len(streams), 4),
        "message_level_episode_share": round(
            n_episode_msgs / (n_episode_msgs + n_benign_stream_msgs), 4
        ),
        "donor_mode": donor_mode,
    }

    out = {
        "config": {
            "source": str(input_path.name),
            "seed": SEED,
            "n_episode_users": N_EPISODE_USERS,
            "k_range": [K_MIN, K_MAX],
            "train_fraction": TRAIN_FRACTION,
            "min_train_msgs": MIN_TRAIN_MSGS,
            "episode_gap_seconds": EPISODE_GAP_SECONDS,
            "injection_style": "author_substitution_contiguous_donor_segment",
            "donor_mode": donor_mode,
        },
        "audit": audit,
        "streams": streams,
    }
    output_path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
