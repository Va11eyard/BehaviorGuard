"""Tests for linguistic analyzer."""

import pytest
from hypothesis import given, strategies as st

from behaviorguard.analyzers.linguistic import LinguisticAnalyzer
from behaviorguard.models import (
    CurrentMessage,
    LinguisticFeatures,
    LinguisticProfile,
    RequestedOperation,
    TemporalContext,
)


# Test data builders
def build_linguistic_profile(
    avg_length: float = 50.0,
    lexical_diversity_mean: float = 0.7,
    lexical_diversity_std: float = 0.1,
    formality_mean: float = 0.5,
    formality_std: float = 0.1,
    politeness_mean: float = 0.6,
    politeness_std: float = 0.1,
    primary_languages: list = None,
    uses_technical: bool = False,
    uses_code: bool = False,
) -> LinguisticProfile:
    """Build a linguistic profile for testing."""
    return LinguisticProfile(
        avg_message_length_tokens=avg_length,
        avg_message_length_chars=avg_length * 5,
        lexical_diversity_mean=lexical_diversity_mean,
        lexical_diversity_std=lexical_diversity_std,
        formality_score_mean=formality_mean,
        formality_score_std=formality_std,
        politeness_score_mean=politeness_mean,
        politeness_score_std=politeness_std,
        question_ratio_mean=0.2,
        uses_technical_vocabulary=uses_technical,
        uses_code_blocks=uses_code,
        primary_languages=primary_languages or ["en"],
        typical_sentence_complexity="moderate",
    )


def build_current_message(
    text: str = "Test message",
    length_tokens: int = 50,
    lexical_diversity: float = 0.7,
    formality: float = 0.5,
    politeness: float = 0.6,
    language: str = "en",
    contains_code: bool = False,
    contains_urls: bool = False,
) -> CurrentMessage:
    """Build a current message for testing."""
    return CurrentMessage(
        text=text,
        timestamp="2024-01-01T12:00:00Z",
        session_id="session-123",
        message_sequence_in_session=1,
        time_since_last_message_seconds=30.0,
        detected_intent="information_seeking",
        requested_operation=RequestedOperation(
            type="read",
            risk_classification="low",
            targets=None,
            requires_auth=False,
        ),
        linguistic_features=LinguisticFeatures(
            message_length_tokens=length_tokens,
            message_length_chars=length_tokens * 5,
            lexical_diversity=lexical_diversity,
            formality_score=formality,
            politeness_score=politeness,
            contains_code=contains_code,
            contains_urls=contains_urls,
            language=language,
        ),
        temporal_context=TemporalContext(
            hour_of_day_utc=12,
            day_of_week="Monday",
            is_typical_active_time=True,
            time_since_last_session_hours=24.0,
        ),
    )


# Feature: behaviorguard-anomaly-scoring, Property 7: Linguistic penalties are correctly applied
@given(
    length_deviation=st.floats(min_value=0.0, max_value=5.0),
    diversity_deviation=st.floats(min_value=0.0, max_value=5.0),
)
def test_property_linguistic_penalties(length_deviation, diversity_deviation):
    """
    Property 7: Linguistic penalties are correctly applied.

    For any message where length deviates >3 std devs, the linguistic score should
    increase by at least 0.15; for dramatic lexical diversity change, increase by 0.1-0.3;
    for formality mismatch, increase by 0.1-0.2.

    Validates: Requirements 3.1, 3.2, 3.3
    """
    analyzer = LinguisticAnalyzer()

    # Test length deviation
    if length_deviation > 3.0:
        profile = build_linguistic_profile(avg_length=50.0, lexical_diversity_std=10.0)
        # Message with >3 std dev length deviation
        message = build_current_message(length_tokens=int(50 + length_deviation * 10))
        result = analyzer.analyze(message, profile)
        assert result.score >= 0.15, f"Expected score >= 0.15 for length deviation, got {result.score}"

    # Test diversity deviation
    if diversity_deviation > 3.0:
        profile = build_linguistic_profile(
            lexical_diversity_mean=0.5, lexical_diversity_std=0.1
        )
        # Message with dramatic diversity change
        new_diversity = min(1.0, 0.5 + diversity_deviation * 0.1)
        message = build_current_message(lexical_diversity=new_diversity)
        result = analyzer.analyze(message, profile)
        assert result.score >= 0.1, f"Expected score >= 0.1 for diversity shift, got {result.score}"


# Feature: behaviorguard-anomaly-scoring, Property 8: Language switching increases linguistic score
@given(
    has_language_history=st.booleans(),
)
def test_property_language_switching(has_language_history):
    """
    Property 8: Language switching increases linguistic score.

    For any user profile without code-switching history and message in a different
    language than primary language, the linguistic score should increase by 0.2 to 0.4.

    Validates: Requirements 3.5
    """
    analyzer = LinguisticAnalyzer()

    if not has_language_history:
        # User only speaks English
        profile = build_linguistic_profile(primary_languages=["en"])
        # Message in Spanish
        message = build_current_message(language="es")
        result = analyzer.analyze(message, profile)
        assert result.score >= 0.2, f"Expected score >= 0.2 for language switch, got {result.score}"
        assert any("language" in factor.lower() for factor in result.contributing_factors)


# Unit tests for edge cases
def test_linguistic_analyzer_zero_std_deviation():
    """Test with zero standard deviation - should handle gracefully."""
    analyzer = LinguisticAnalyzer()
    profile = build_linguistic_profile(
        avg_length=50.0,
        lexical_diversity_std=0.0,  # Zero variance
        formality_std=0.0,
        politeness_std=0.0,
    )
    # Message with different length
    message = build_current_message(length_tokens=100)

    result = analyzer.analyze(message, profile)

    # Should handle zero std dev without crashing
    assert 0.0 <= result.score <= 1.0
    assert result.score > 0  # Should detect deviation


def test_linguistic_analyzer_extreme_message_length():
    """Test with extreme message lengths."""
    analyzer = LinguisticAnalyzer()
    profile = build_linguistic_profile(avg_length=50.0, lexical_diversity_std=10.0)

    # Very short message
    short_message = build_current_message(length_tokens=5)
    short_result = analyzer.analyze(short_message, profile)
    assert 0.0 <= short_result.score <= 1.0

    # Very long message
    long_message = build_current_message(length_tokens=500)
    long_result = analyzer.analyze(long_message, profile)
    assert 0.0 <= long_result.score <= 1.0
    assert long_result.score >= 0.15  # Should detect deviation


def test_linguistic_analyzer_multiple_language_switches():
    """Test with multiple language switches."""
    analyzer = LinguisticAnalyzer()
    # User speaks English and Spanish
    profile = build_linguistic_profile(primary_languages=["en", "es"])
    # Message in French (new language)
    message = build_current_message(language="fr")

    result = analyzer.analyze(message, profile)

    # Should detect language switch but with lower penalty
    assert result.score >= 0.2
    assert any("language" in factor.lower() for factor in result.contributing_factors)


def test_linguistic_analyzer_formality_shift():
    """Test formality shift detection."""
    analyzer = LinguisticAnalyzer()
    # User typically casual (low formality)
    profile = build_linguistic_profile(formality_mean=0.2, formality_std=0.1)
    # Message is very formal
    message = build_current_message(formality=0.8)

    result = analyzer.analyze(message, profile)

    assert result.score >= 0.1
    assert any("formality" in factor.lower() for factor in result.contributing_factors)


def test_linguistic_analyzer_politeness_inversion_rude():
    """Test politeness inversion - polite to rude."""
    analyzer = LinguisticAnalyzer()
    # User typically polite
    profile = build_linguistic_profile(politeness_mean=0.8)
    # Message is rude/demanding
    message = build_current_message(politeness=0.2)

    result = analyzer.analyze(message, profile)

    assert result.score >= 0.1
    assert any("politeness" in factor.lower() for factor in result.contributing_factors)


def test_linguistic_analyzer_politeness_inversion_overly_polite():
    """Test politeness inversion - direct to overly polite (social engineering)."""
    analyzer = LinguisticAnalyzer()
    # User typically direct
    profile = build_linguistic_profile(politeness_mean=0.3)
    # Message is overly polite
    message = build_current_message(politeness=0.9)

    result = analyzer.analyze(message, profile)

    assert result.score >= 0.15  # Higher penalty for social engineering indicator
    assert any("politeness" in factor.lower() for factor in result.contributing_factors)


def test_linguistic_analyzer_technical_vocabulary_mismatch():
    """Test technical vocabulary mismatch."""
    analyzer = LinguisticAnalyzer()
    # User not typically technical
    profile = build_linguistic_profile(uses_technical=False, uses_code=False)
    # Message contains code
    message = build_current_message(contains_code=True)

    result = analyzer.analyze(message, profile)

    assert result.score >= 0.2
    assert any("technical" in factor.lower() for factor in result.contributing_factors)


def test_linguistic_analyzer_consistent_patterns():
    """Test with consistent linguistic patterns - should score low."""
    analyzer = LinguisticAnalyzer()
    profile = build_linguistic_profile(
        avg_length=50.0,
        lexical_diversity_mean=0.7,
        formality_mean=0.5,
        politeness_mean=0.6,
        primary_languages=["en"],
    )
    # Message matching profile exactly
    message = build_current_message(
        length_tokens=50,
        lexical_diversity=0.7,
        formality=0.5,
        politeness=0.6,
        language="en",
    )

    result = analyzer.analyze(message, profile)

    assert result.score < 0.1  # Should be very low
    assert "consistent" in result.reasoning.lower()


def test_linguistic_analyzer_score_bounded():
    """Test that score is always bounded between 0.0 and 1.0."""
    analyzer = LinguisticAnalyzer()
    profile = build_linguistic_profile()
    # Extreme message with all anomalies
    message = build_current_message(
        length_tokens=500,
        lexical_diversity=1.0,
        formality=1.0,
        politeness=1.0,
        language="fr",
        contains_code=True,
    )

    result = analyzer.analyze(message, profile)

    assert 0.0 <= result.score <= 1.0


def test_linguistic_analyzer_returns_complete_result():
    """Test that analyzer returns all required fields."""
    analyzer = LinguisticAnalyzer()
    profile = build_linguistic_profile()
    message = build_current_message()

    result = analyzer.analyze(message, profile)

    assert hasattr(result, "score")
    assert hasattr(result, "reasoning")
    assert hasattr(result, "contributing_factors")
    assert isinstance(result.score, float)
    assert isinstance(result.reasoning, str)
    assert isinstance(result.contributing_factors, list)
    assert len(result.reasoning) > 0
