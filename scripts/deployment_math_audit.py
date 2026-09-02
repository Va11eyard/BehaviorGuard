#!/usr/bin/env python3
"""
Correct deployment math: per-profile storage size and end-to-end latency.

Reports:
  - serialized JSON size for cosine-only vs full Mahalanobis profiles
  - scoring latency (embedding + composite)
  - ProfileStore load/save latency
  - budgeted remote-store RTT and session-cache discussion
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from turnshift.embedding_config import EMBEDDING_MODEL_NAME, EMBEDDING_MODEL_REVISION  # noqa: E402
from turnshift.models import (  # noqa: E402
    CurrentMessage,
    EvaluationInput,
    LinguisticFeatures,
    LinguisticProfile,
    OperationalProfile,
    RequestedOperation,
    SemanticProfile,
    SystemConfig,
    TemporalContext,
    TemporalProfile,
    UserProfile,
)
from turnshift.profile_manager import MessageRecord, ProfileManager  # noqa: E402
from turnshift.utils.profile_store import ProfileStore  # noqa: E402


def _size_bytes(obj) -> int:
    return len(json.dumps(obj.model_dump(), separators=(",", ":")).encode("utf-8"))


def main() -> None:
    from turnshift.evaluator_ml import TurnShiftEvaluatorML

    pm = ProfileManager(decay=0.5)
    texts = [
        "I love hiking on weekends with my dog.",
        "My favorite food is pasta with tomato sauce.",
        "I work as a software engineer in the city.",
        "Do you have any book recommendations?",
        "I went to the gym this morning before work.",
        "Looking forward to the concert next Friday.",
        "Can you help me debug this Python function?",
        "The weather has been unusually cold lately.",
        "I am planning a trip to the mountains.",
        "Thanks for the suggestion, that was helpful.",
    ]
    msgs = [
        MessageRecord(
            text=t,
            timestamp=f"2024-01-01T12:{i:02d}:00",
            session_id="s0",
            is_anomaly=False,
        )
        for i, t in enumerate(texts)
    ]
    full_profile = pm.build_from_history("u_demo", msgs, account_age_days=100)

    # Cosine-only profile: drop covariance / mean arrays
    cosine_profile = full_profile.model_copy(deep=True)
    cosine_profile.semantic_profile.embedding_mean = None
    cosine_profile.semantic_profile.embedding_covariance = None

    cold = UserProfile(
        user_id="cold",
        account_age_days=1,
        total_interactions=0,
        semantic_profile=SemanticProfile(
            typical_topics=[],
            primary_domains=[],
            topic_diversity_score=0.0,
            embedding_centroid_summary="cold",
        ),
        linguistic_profile=LinguisticProfile(
            avg_message_length_tokens=10,
            avg_message_length_chars=40,
            lexical_diversity_mean=0.5,
            lexical_diversity_std=0.1,
            formality_score_mean=0.5,
            formality_score_std=0.1,
            politeness_score_mean=0.5,
            politeness_score_std=0.1,
            question_ratio_mean=0.2,
            uses_technical_vocabulary=False,
            uses_code_blocks=False,
            primary_languages=["en"],
            typical_sentence_complexity="simple",
        ),
        temporal_profile=TemporalProfile(
            typical_session_duration_minutes=10,
            typical_inter_message_gap_seconds=30,
            most_active_hours_utc=list(range(9, 18)),
            most_active_days_of_week=["Mon", "Tue", "Wed", "Thu", "Fri"],
            average_messages_per_session=5,
            longest_session_duration_minutes=30,
            typical_session_frequency_per_week=3,
            last_activity_timestamp="2024-01-01T12:00:00",
        ),
        operational_profile=OperationalProfile(
            common_intent_types=[],
            tools_used_historically=[],
            has_requested_sensitive_ops=False,
            typical_risk_level="low",
        ),
    )

    sizes = {
        "full_mahalanobis_compact_bytes": _size_bytes(full_profile),
        "cosine_only_compact_bytes": _size_bytes(cosine_profile),
        "cold_start_compact_bytes": _size_bytes(cold),
    }
    for k in list(sizes.keys()):
        sizes[k.replace("_bytes", "_kb")] = round(sizes[k] / 1024, 2)
        sizes[k.replace("_bytes", "_per_1m_users_gb")] = round(sizes[k] * 1_000_000 / (1024**3), 2)

    # Latency: scoring only
    evaluator = TurnShiftEvaluatorML()
    cur = CurrentMessage(
        text="Can you help me export my account settings?",
        timestamp="2024-01-01T13:00:00",
        session_id="s0",
        message_sequence_in_session=1,
        time_since_last_message_seconds=60,
        requested_operation=RequestedOperation(
            type="read",
            risk_classification="low",
            targets=[],
            requires_auth=False,
        ),
        linguistic_features=LinguisticFeatures(
            message_length_tokens=8,
            message_length_chars=40,
            lexical_diversity=0.8,
            formality_score=0.4,
            politeness_score=0.5,
            contains_code=False,
            contains_urls=False,
            language="en",
        ),
        temporal_context=TemporalContext(
            hour_of_day_utc=13,
            day_of_week="Mon",
            is_typical_active_time=True,
            time_since_last_session_hours=1.0,
        ),
    )
    cfg = SystemConfig(
        sensitivity_level="medium",
        deployment_context="enterprise",
        overrides_enabled=False,
        enable_linguistic_scoring=False,
        linguistic_component_enabled=False,
    )
    # warmup
    evaluator.evaluate(EvaluationInput(user_profile=full_profile, current_message=cur, system_config=cfg))
    times = []
    for _ in range(30):
        t0 = time.perf_counter()
        evaluator.evaluate(EvaluationInput(user_profile=full_profile, current_message=cur, system_config=cfg))
        times.append((time.perf_counter() - t0) * 1000)

    with tempfile.TemporaryDirectory() as td:
        store = ProfileStore(td)
        t0 = time.perf_counter()
        store.save(full_profile)
        save_ms_full = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        store.load("u_demo")
        load_ms_full = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        store.save(cosine_profile)
        save_ms_cos = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        store.load("u_demo")
        load_ms_cos = (time.perf_counter() - t0) * 1000

    report = {
        "embedding_model": EMBEDDING_MODEL_NAME,
        "embedding_revision": EMBEDDING_MODEL_REVISION,
        "storage": sizes,
        "storage_note": (
            "Earlier ~1.7 KB / ~1.7 GB-per-1M claim is valid only for cosine-only "
            "profiles (centroid without 384x384 covariance). Full Mahalanobis "
            "profiles are ~500x larger."
        ),
        "latency_ms": {
            "scoring_only_mean": round(float(np.mean(times)), 2),
            "scoring_only_p50": round(float(np.percentile(times, 50)), 2),
            "scoring_only_p95": round(float(np.percentile(times, 95)), 2),
            "profile_save_full_mahalanobis": round(save_ms_full, 2),
            "profile_load_full_mahalanobis": round(load_ms_full, 2),
            "profile_save_cosine_only": round(save_ms_cos, 2),
            "profile_load_cosine_only": round(load_ms_cos, 2),
            "budgeted_remote_redis_rtt_ms": [1.0, 5.0],
        },
        "end_to_end_estimate_ms": {
            "cosine_local_store": round(float(np.mean(times)) + load_ms_cos + save_ms_cos, 2),
            "cosine_remote_store_mid": round(float(np.mean(times)) + 3.0 + load_ms_cos, 2),
            "mahalanobis_local_store": round(float(np.mean(times)) + load_ms_full + save_ms_full, 2),
        },
        "caching_recommendation": (
            "Keep active-session profiles in process memory / Redis with TTL; "
            "write-through on session end. Prefer cosine-only profiles in production "
            "unless Mahalanobis is shown to pay for its storage cost."
        ),
    }
    out = ROOT / "results" / "deployment_math_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
