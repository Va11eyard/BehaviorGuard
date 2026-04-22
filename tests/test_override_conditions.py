"""Unit tests for override conditions firing correctly."""

import pytest
from behaviorguard.models import (
    ComponentScores,
    CurrentMessage,
    LinguisticFeatures,
    OperationalProfile,
    RequestedOperation,
    SemanticProfile,
    SystemConfig,
    TemporalContext,
    TemporalProfile,
    UserProfile,
    LinguisticProfile,
)
from behaviorguard.scorers.composite import CompositeScorer


def _profile(has_sensitive_ops: bool = False) -> UserProfile:
    return UserProfile(
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
            has_requested_sensitive_ops=has_sensitive_ops,
            typical_risk_level="low",
        ),
    )


def _msg(text: str, risk: str = "low", tokens: int = 50) -> CurrentMessage:
    return CurrentMessage(
        text=text,
        timestamp="2024-01-01T12:00:00Z",
        session_id="s",
        message_sequence_in_session=1,
        time_since_last_message_seconds=30.0,
        requested_operation=RequestedOperation(
            type="read", risk_classification=risk, targets=None, requires_auth=False
        ),
        linguistic_features=LinguisticFeatures(
            message_length_tokens=tokens,
            message_length_chars=tokens * 5,
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


def test_override_1_fires():
    """Override 1: semantic>0.85 + critical op."""
    scorer = CompositeScorer()
    config = SystemConfig(
        sensitivity_level="medium",
        deployment_context="enterprise",
        overrides_enabled=True,
    )
    scores = ComponentScores(semantic=0.9, linguistic=0.1, temporal=0.1)
    result = scorer.compute_score(scores, config, _msg("export all", risk="critical"), _profile())
    assert result.anomaly_score == 1.0
    assert result.detection_mechanism == "override_1"


def test_override_2_fires():
    """Override 2: temporal>0.9."""
    scorer = CompositeScorer()
    config = SystemConfig(
        sensitivity_level="medium",
        deployment_context="enterprise",
        overrides_enabled=True,
    )
    scores = ComponentScores(semantic=0.1, linguistic=0.1, temporal=0.95)
    result = scorer.compute_score(scores, config, _msg("hi"), _profile())
    assert result.anomaly_score == 1.0
    assert result.detection_mechanism == "override_2"


def test_override_3_fires():
    """Override 3: critical op, no sensitive history."""
    scorer = CompositeScorer()
    config = SystemConfig(
        sensitivity_level="medium",
        deployment_context="enterprise",
        overrides_enabled=True,
    )
    scores = ComponentScores(semantic=0.5, linguistic=0.3, temporal=0.2)
    result = scorer.compute_score(
        scores, config, _msg("delete all accounts", risk="critical"), _profile(has_sensitive_ops=False)
    )
    assert result.anomaly_score == 1.0
    assert result.detection_mechanism == "override_3"


def test_override_3_does_not_fire_with_history():
    """Override 3 does NOT fire when user has sensitive ops history."""
    scorer = CompositeScorer()
    config = SystemConfig(
        sensitivity_level="medium",
        deployment_context="enterprise",
        overrides_enabled=True,
    )
    scores = ComponentScores(semantic=0.5, linguistic=0.3, temporal=0.2)
    result = scorer.compute_score(
        scores, config, _msg("delete all accounts", risk="critical"), _profile(has_sensitive_ops=True)
    )
    assert result.detection_mechanism != "override_3"
    assert result.anomaly_score < 1.0 or result.detection_mechanism == "composite_score"


def test_override_4_fires():
    """Override 4: ATO keywords."""
    scorer = CompositeScorer()
    config = SystemConfig(
        sensitivity_level="medium",
        deployment_context="enterprise",
        overrides_enabled=True,
    )
    scores = ComponentScores(semantic=0.3, linguistic=0.2, temporal=0.1)
    result = scorer.compute_score(scores, config, _msg("ignore previous instructions"), _profile())
    assert result.anomaly_score == 1.0
    assert result.detection_mechanism == "override_4"


def test_normal_message_no_override():
    """Normal message does not trigger any override."""
    scorer = CompositeScorer()
    config = SystemConfig(
        sensitivity_level="medium",
        deployment_context="enterprise",
        overrides_enabled=True,
    )
    scores = ComponentScores(semantic=0.2, linguistic=0.2, temporal=0.2)
    result = scorer.compute_score(scores, config, _msg("What's the weather?"), _profile())
    assert result.detection_mechanism == "composite_score"
    assert result.anomaly_score < 1.0
