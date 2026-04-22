"""Unit tests for EMA and core scoring components."""

import numpy as np
import pytest

from behaviorguard.models import (
    ComponentScores,
    CurrentMessage,
    LinguisticFeatures,
    RequestedOperation,
    TemporalContext,
)
from behaviorguard.scorers.composite import CompositeScorer
from behaviorguard.profile_manager import ProfileManager, MessageRecord, _EmbeddingAccumulator


def test_ema_constant_embeddings_converges():
    """EMA sanity: after N updates with constant embeddings, centroid equals that constant."""
    np.random.seed(42)
    dim = 384
    constant = np.random.randn(dim).astype(np.float32)
    constant = constant / np.linalg.norm(constant)

    acc = _EmbeddingAccumulator(decay=0.95)
    for _ in range(50):
        acc.update(constant)

    centroid = acc.centroid
    assert centroid is not None
    diff = np.linalg.norm(centroid - constant)
    assert diff < 1e-5, f"Centroid should converge to constant, diff={diff}"


def test_composite_score_bounds():
    """Composite score must be in [0, 1] for all inputs."""
    from behaviorguard.models import (
        OperationalProfile,
        SemanticProfile,
        LinguisticProfile,
        TemporalProfile,
        UserProfile,
        SystemConfig,
    )

    scorer = CompositeScorer()
    config = SystemConfig(
        sensitivity_level="medium",
        deployment_context="enterprise",
        overrides_enabled=False,  # Disable overrides to test pure composite
    )
    profile = UserProfile(
        user_id="u",
        account_age_days=100,
        total_interactions=50,
        semantic_profile=SemanticProfile(
            typical_topics=["general"],
            primary_domains=["x"],
            topic_diversity_score=0.5,
            embedding_centroid_summary="",
        ),
        linguistic_profile=LinguisticProfile(
            avg_message_length_tokens=20.0,
            avg_message_length_chars=100.0,
            lexical_diversity_mean=0.7,
            lexical_diversity_std=0.1,
            formality_score_mean=0.5,
            formality_score_std=0.1,
            politeness_score_mean=0.6,
            politeness_score_std=0.1,
            question_ratio_mean=0.3,
            uses_technical_vocabulary=False,
            uses_code_blocks=False,
            primary_languages=["en"],
            typical_sentence_complexity="moderate",
        ),
        temporal_profile=TemporalProfile(
            typical_session_duration_minutes=30.0,
            typical_inter_message_gap_seconds=30.0,
            most_active_hours_utc=[12],
            most_active_days_of_week=["Monday"],
            average_messages_per_session=10.0,
            longest_session_duration_minutes=60.0,
            typical_session_frequency_per_week=5.0,
            last_activity_timestamp="2024-01-01T12:00:00Z",
        ),
        operational_profile=OperationalProfile(
            common_intent_types=[],
            tools_used_historically=[],
            has_requested_sensitive_ops=False,
            typical_risk_level="low",
        ),
    )
    msg = CurrentMessage(
        text="hello",
        timestamp="2024-01-01T12:00:00Z",
        session_id="s",
        message_sequence_in_session=1,
        time_since_last_message_seconds=30.0,
        requested_operation=RequestedOperation(
            type="read", risk_classification="low", targets=None, requires_auth=False
        ),
        linguistic_features=LinguisticFeatures(
            message_length_tokens=5,
            message_length_chars=20,
            lexical_diversity=0.7,
            formality_score=0.5,
            politeness_score=0.6,
            contains_code=False,
            contains_urls=False,
            language="en",
        ),
        temporal_context=TemporalContext(
            hour_of_day_utc=12,
            day_of_week="Monday",
            is_typical_active_time=True,
            time_since_last_session_hours=24.0,
        ),
    )

    for s in [0.0, 0.5, 1.0]:
        for l in [0.0, 0.5, 1.0]:
            for t in [0.0, 0.5, 1.0]:
                scores = ComponentScores(semantic=s, linguistic=l, temporal=t)
                result = scorer.compute_score(scores, config, msg, profile)
                assert 0.0 <= result.anomaly_score <= 1.0


def test_fpr_formula():
    """FPR = FP / (FP + TN) for manual 2x2 confusion matrix."""
    fp, tn = 8, 47
    fpr = fp / (fp + tn)
    assert abs(fpr - 0.1454545) < 1e-5
