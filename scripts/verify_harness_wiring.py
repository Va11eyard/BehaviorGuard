#!/usr/bin/env python3
"""Pre-Part-5 verification: s_sem, s_ling, s_temp match direct analyzer calls."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import evaluation as ev  # noqa: E402
from turnshift.analyzers.linguistic_ml import LinguisticAnalyzerML  # noqa: E402
from turnshift.analyzers.semantic_ml import SemanticAnalyzerML  # noqa: E402
from turnshift.analyzers.temporal_ml import TemporalAnalyzerML  # noqa: E402
from scripts.diagnostic_harness import (  # noqa: E402
    DIAG_CONFIG,
    _build_profile,
    _compute_s_ling_real,
    _compute_s_sem_real,
    _compute_s_temp_real,
    _prev_in_session,
    split_conversations_8020,
)

DATASET = ROOT / "datasets/personachat_processed_corrected.json"
TARGET_USER = "pc_user_00001"
TOLERANCE = 1e-4


def _running_total_interactions(profile_total: int, test_message_index: int) -> int:
    """Chronological position: train organic count + prior test-window messages."""
    return profile_total + test_message_index


def main() -> int:
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    users = {u["user_id"]: u for u in data["users"]}

    splits = split_conversations_8020(data, user_ids=[TARGET_USER])
    sp = next((s for s in splits if s.user_id == TARGET_USER), None)
    if sp is None:
        print(f"FAIL: no 80/20 split for {TARGET_USER}")
        return 1

    profile = _build_profile(users[sp.user_id], sp.train_msgs, 0.5)
    if profile is None:
        print("FAIL: could not build profile")
        return 1

    test_idx = 0
    msg = sp.test_msgs[test_idx]
    prev = _prev_in_session(sp.test_msgs, test_idx)
    cur = ev.message_to_current_message(msg, prev, user_profile=profile)

    running_n = _running_total_interactions(profile.total_interactions, test_idx)
    system_config = DIAG_CONFIG

    semantic = SemanticAnalyzerML()
    linguistic = LinguisticAnalyzerML()
    temporal = TemporalAnalyzerML()

    # --- s_sem ---
    direct_sem = float(
        semantic.analyze(
            cur,
            profile.semantic_profile,
            system_config=system_config,
            total_interactions=running_n,
        ).score
    )
    harness_sem = _compute_s_sem_real(
        semantic, cur, profile, system_config, test_message_index=test_idx
    )
    diff_sem = abs(direct_sem - harness_sem)

    # --- s_ling ---
    direct_ling = float(
        linguistic.analyze(cur, profile.linguistic_profile).score
    )
    harness_ling = _compute_s_ling_real(linguistic, cur, profile)
    diff_ling = abs(direct_ling - harness_ling)

    # --- s_temp ---
    direct_temp = float(
        temporal.analyze(cur, profile.temporal_profile).score
    )
    harness_temp = _compute_s_temp_real(temporal, cur, profile)
    diff_temp = abs(direct_temp - harness_temp)

    print(f"message_user={sp.user_id}")
    print(f"message_text={cur.text[:80]!r}...")
    print()
    print("--- profile construction ---")
    print("per_user_profile: YES — one ProfileManager.build_from_history() per user_id")
    print(f"  train built from that user's own 80% organic messages only (not pooled)")
    # Sanity: two users must not share identical semantic centroids
    other = next(s for s in split_conversations_8020(data) if s.user_id != TARGET_USER)
    other_profile = _build_profile(users[other.user_id], other.train_msgs, 0.5)
    distinct = (
        other_profile is not None
        and profile.semantic_profile.embedding_mean
        != other_profile.semantic_profile.embedding_mean
    )
    print(f"  distinct_profiles_across_users: {distinct}")
    print(f"train_organic_messages={len(sp.train_msgs)}")
    print(f"profile.total_interactions={profile.total_interactions}")
    print(f"embedding_sample_count={profile.semantic_profile.embedding_sample_count}")
    print(f"test_message_index={test_idx}")
    print(f"total_interactions passed to analyze()={running_n}")
    print(
        "note: running count = profile.total_interactions + test_message_index "
        "(not hardcoded 0)"
    )
    print()
    print("--- component score parity ---")
    print(f"s_sem  direct={direct_sem:.6f}  harness={harness_sem:.6f}  diff={diff_sem:.6f}")
    print(f"s_ling direct={direct_ling:.6f}  harness={harness_ling:.6f}  diff={diff_ling:.6f}")
    print(f"s_temp direct={direct_temp:.6f}  harness={harness_temp:.6f}  diff={diff_temp:.6f}")

    all_ok = diff_sem <= TOLERANCE and diff_ling <= TOLERANCE and diff_temp <= TOLERANCE
    if all_ok:
        print(f"\nPASS: all three diffs <= {TOLERANCE}")
        return 0
    print(f"\nFAIL: one or more diffs > {TOLERANCE} — do not run Part 5")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
