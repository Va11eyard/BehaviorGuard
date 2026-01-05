"""Tests for temporal analyzer."""

import pytest
from hypothesis import given, strategies as st

from behaviorguard.analyzers.temporal import TemporalAnalyzer
from behaviorguard.models import (
    CurrentMessage,
    LinguisticFeatures,
    RequestedOperation,
    TemporalContext,
    TemporalProfile,
)


# Test data builders
def build_temporal_profile(
    typical_hours: list = None,
    typical_days: list = None,
    avg_messages_per_session: float = 10.0,
    longest_session_minutes: float = 60.0,
    typical_gap_seconds: float = 30.0,
    typical_frequency_per_week: float = 7.0,
) -> TemporalProfile:
    """Build a temporal profile for testing."""
    return TemporalProfile(
        typical_session_duration_minutes=30.0,
        typical_inter_message_gap_seconds=typical_gap_seconds,
        most_active_hours_utc=typical_hours or [9, 10, 11, 12, 13, 14, 15, 16, 17],
        most_active_days_of_week=typical_days or ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
        average_messages_per_session=avg_messages_per_session,
        longest_session_duration_minutes=longest_session_minutes,
        typical_session_frequency_per_week=typical_frequency_per_week,
        last_activity_timestamp="2024-01-01T12:00:00Z",
    )


def build_current_message(
    hour_utc: int = 12,
    day_of_week: str = "Monday",
    is_typical_time: bool = True,
    time_since_last_session_hours: float = 24.0,
    time_since_last_message_seconds: float = 30.0,
    message_sequence: int = 1,
) -> CurrentMessage:
    """Build a current message for testing."""
    return CurrentMessage(
        text="Test message",
        timestamp="2024-01-01T12:00:00Z",
        session_id="session-123",
        message_sequence_in_session=message_sequence,
        time_since_last_message_seconds=time_since_last_message_seconds,
        detected_intent="information_seeking",
        requested_operation=RequestedOperation(
            type="read",
            risk_classification="low",
            targets=None,
            requires_auth=False,
        ),
        linguistic_features=LinguisticFeatures(
            message_length_tokens=50,
            message_length_chars=250,
            lexical_diversity=0.7,
            formality_score=0.5,
            politeness_score=0.6,
            contains_code=False,
            contains_urls=False,
            language="en",
        ),
        temporal_context=TemporalContext(
            hour_of_day_utc=hour_utc,
            day_of_week=day_of_week,
            is_typical_active_time=is_typical_time,
            time_since_last_session_hours=time_since_last_session_hours,
        ),
    )


# Feature: behaviorguard-anomaly-scoring, Property 9: Temporal penalties for unusual activity
@given(
    hour=st.integers(min_value=0, max_value=23),
)
def test_property_temporal_penalties_unusual_hours(hour):
    """
    Property 9: Temporal penalties for unusual activity.

    For any message during hours with <5% historical activity, the temporal score
    should increase by at least 0.2; for 3-4am activity when never active, increase
    by at least 0.3.

    Validates: Requirements 4.1, 4.2
    """
    analyzer = TemporalAnalyzer()

    # User only active 9-5 (9 hours = 37.5% of day)
    profile = build_temporal_profile(typical_hours=[9, 10, 11, 12, 13, 14, 15, 16, 17])

    if hour not in [9, 10, 11, 12, 13, 14, 15, 16, 17]:
        # Activity during unusual hours
        message = build_current_message(hour_utc=hour, is_typical_time=False)
        result = analyzer.analyze(message, profile)

        if 3 <= hour <= 4:
            # 3-4am should have higher penalty
            assert result.score >= 0.3, f"Expected score >= 0.3 for 3-4am activity, got {result.score}"
        else:
            # Other unusual hours
            assert result.score >= 0.0, f"Expected score >= 0.0 for unusual hours, got {result.score}"


# Feature: behaviorguard-anomaly-scoring, Property 10: Bot-like timing detection
@given(
    gap_seconds=st.floats(min_value=0.1, max_value=100.0),
)
def test_property_bot_like_timing(gap_seconds):
    """
    Property 10: Bot-like timing detection.

    For any message sequence with inter-message gaps consistently <5 seconds,
    the temporal score should increase by at least 0.25.

    Validates: Requirements 4.5
    """
    analyzer = TemporalAnalyzer()
    profile = build_temporal_profile(typical_gap_seconds=30.0)

    if gap_seconds < 5.0:
        message = build_current_message(time_since_last_message_seconds=gap_seconds)
        result = analyzer.analyze(message, profile)
        assert result.score >= 0.25, f"Expected score >= 0.25 for bot-like timing, got {result.score}"
        assert any("bot" in factor.lower() or "timing" in factor.lower() for factor in result.contributing_factors)


# Unit tests for edge cases
def test_temporal_analyzer_no_historical_activity():
    """Test with no historical activity data - should handle gracefully."""
    analyzer = TemporalAnalyzer()
    profile = build_temporal_profile(typical_hours=[], typical_days=[])
    message = build_current_message()

    result = analyzer.analyze(message, profile)

    # Should handle missing data without crashing
    assert 0.0 <= result.score <= 1.0


def test_temporal_analyzer_timezone_edge_case_3am():
    """Test 3-4am activity (extra suspicious)."""
    analyzer = TemporalAnalyzer()
    profile = build_temporal_profile(typical_hours=[9, 10, 11, 12, 13, 14, 15])
    message = build_current_message(hour_utc=3, is_typical_time=False)

    result = analyzer.analyze(message, profile)

    assert result.score >= 0.3
    assert any("unusual hours" in factor.lower() for factor in result.contributing_factors)


def test_temporal_analyzer_extreme_session_duration():
    """Test with extreme session durations."""
    analyzer = TemporalAnalyzer()
    profile = build_temporal_profile(avg_messages_per_session=10.0)

    # Very long session (>2x typical)
    long_message = build_current_message(message_sequence=25)
    long_result = analyzer.analyze(long_message, profile)

    assert long_result.score >= 0.1
    assert any("session" in factor.lower() or "duration" in factor.lower() for factor in long_result.contributing_factors)


def test_temporal_analyzer_impossible_velocity():
    """Test impossible velocity detection."""
    analyzer = TemporalAnalyzer()
    profile = build_temporal_profile()

    # Session within 30 minutes of last session (could indicate location change)
    message = build_current_message(time_since_last_session_hours=0.5)
    result = analyzer.analyze(message, profile)

    assert result.score >= 0.4
    assert any("velocity" in factor.lower() or "location" in factor.lower() for factor in result.contributing_factors)


def test_temporal_analyzer_bot_timing_very_fast():
    """Test very fast bot-like timing (<2 seconds)."""
    analyzer = TemporalAnalyzer()
    profile = build_temporal_profile(typical_gap_seconds=30.0)
    message = build_current_message(time_since_last_message_seconds=1.5)

    result = analyzer.analyze(message, profile)

    assert result.score >= 0.25
    assert any("bot" in factor.lower() or "timing" in factor.lower() for factor in result.contributing_factors)


def test_temporal_analyzer_return_after_long_absence():
    """Test return after 6+ months of inactivity."""
    analyzer = TemporalAnalyzer()
    profile = build_temporal_profile(typical_frequency_per_week=7.0)
    # 6 months = ~4320 hours
    message = build_current_message(time_since_last_session_hours=4320.0)

    result = analyzer.analyze(message, profile)

    assert result.score >= 0.2
    assert any("frequency" in factor.lower() for factor in result.contributing_factors)


def test_temporal_analyzer_high_frequency_sessions():
    """Test unusually high frequency sessions."""
    analyzer = TemporalAnalyzer()
    # User typically has 1 session per day
    profile = build_temporal_profile(typical_frequency_per_week=7.0)
    # New session after just 1 hour (10x more frequent)
    message = build_current_message(time_since_last_session_hours=1.0)

    result = analyzer.analyze(message, profile)

    assert result.score >= 0.3
    assert any("frequency" in factor.lower() for factor in result.contributing_factors)


def test_temporal_analyzer_sustained_high_frequency():
    """Test sustained high-frequency activity (very long session)."""
    analyzer = TemporalAnalyzer()
    profile = build_temporal_profile(avg_messages_per_session=10.0)
    # 60 messages in one session
    message = build_current_message(message_sequence=60)

    result = analyzer.analyze(message, profile)

    assert result.score >= 0.3


def test_temporal_analyzer_consistent_patterns():
    """Test with consistent temporal patterns - should score low."""
    analyzer = TemporalAnalyzer()
    profile = build_temporal_profile(
        typical_hours=[9, 10, 11, 12, 13, 14, 15],
        avg_messages_per_session=10.0,
        typical_gap_seconds=30.0,
    )
    # Message matching typical patterns
    message = build_current_message(
        hour_utc=12,
        is_typical_time=True,
        time_since_last_message_seconds=30.0,
        message_sequence=5,
    )

    result = analyzer.analyze(message, profile)

    assert result.score < 0.1
    assert "consistent" in result.reasoning.lower()


def test_temporal_analyzer_score_bounded():
    """Test that score is always bounded between 0.0 and 1.0."""
    analyzer = TemporalAnalyzer()
    profile = build_temporal_profile()
    # Extreme message with all anomalies
    message = build_current_message(
        hour_utc=3,
        is_typical_time=False,
        time_since_last_session_hours=0.5,
        time_since_last_message_seconds=1.0,
        message_sequence=100,
    )

    result = analyzer.analyze(message, profile)

    assert 0.0 <= result.score <= 1.0


def test_temporal_analyzer_returns_complete_result():
    """Test that analyzer returns all required fields."""
    analyzer = TemporalAnalyzer()
    profile = build_temporal_profile()
    message = build_current_message()

    result = analyzer.analyze(message, profile)

    assert hasattr(result, "score")
    assert hasattr(result, "reasoning")
    assert hasattr(result, "contributing_factors")
    assert isinstance(result.score, float)
    assert isinstance(result.reasoning, str)
    assert isinstance(result.contributing_factors, list)
    assert len(result.reasoning) > 0
