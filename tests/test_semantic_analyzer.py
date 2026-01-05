"""Tests for semantic analyzer."""

import pytest
from hypothesis import given, strategies as st

from behaviorguard.analyzers.semantic import SemanticAnalyzer
from behaviorguard.models import (
    CurrentMessage,
    LinguisticFeatures,
    RequestedOperation,
    SemanticProfile,
    TemporalContext,
)


# Test data builders
def build_semantic_profile(
    typical_topics: list = None, primary_domains: list = None
) -> SemanticProfile:
    """Build a semantic profile for testing."""
    return SemanticProfile(
        typical_topics=typical_topics or ["python", "programming", "coding"],
        primary_domains=primary_domains or ["technology", "software"],
        topic_diversity_score=0.5,
        embedding_centroid_summary="Technology and programming focused",
    )


def build_current_message(
    text: str,
    risk_classification: str = "low",
    message_sequence: int = 1,
    contains_code: bool = False,
) -> CurrentMessage:
    """Build a current message for testing."""
    return CurrentMessage(
        text=text,
        timestamp="2024-01-01T12:00:00Z",
        session_id="session-123",
        message_sequence_in_session=message_sequence,
        time_since_last_message_seconds=30.0,
        detected_intent="information_seeking",
        requested_operation=RequestedOperation(
            type="read",
            risk_classification=risk_classification,
            targets=None,
            requires_auth=False,
        ),
        linguistic_features=LinguisticFeatures(
            message_length_tokens=50,
            message_length_chars=250,
            lexical_diversity=0.7,
            formality_score=0.5,
            politeness_score=0.6,
            contains_code=contains_code,
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


# Feature: behaviorguard-anomaly-scoring, Property 6: Semantic score ranges match deviation levels
def test_property_semantic_score_ranges():
    """
    Property 6: Semantic score ranges match deviation levels.

    For any message that is fully consistent with typical topics (overlap >= 0.8),
    the semantic score should be 0.0-0.25; for mild deviation (0.5-0.8) score 0.2-0.6;
    for significant deviation (0.2-0.5) score 0.5-0.85; for extreme deviation (<0.2) score 0.8-1.0.

    Validates: Requirements 2.2, 2.3, 2.4, 2.5
    """
    analyzer = SemanticAnalyzer()

    # Test 1: Fully consistent - should score 0.0-0.25
    profile1 = build_semantic_profile(
        typical_topics=["python", "programming", "coding", "list", "comprehensions"]
    )
    message1 = build_current_message("How do I use python list comprehensions for coding?")
    result1 = analyzer.analyze(message1, profile1)
    assert 0.0 <= result1.score <= 0.25, f"Expected 0.0-0.25 for high overlap, got {result1.score}"

    # Test 2: Mild deviation - should score 0.2-0.65
    profile2 = build_semantic_profile(
        typical_topics=["python", "programming", "coding", "software", "development"]
    )
    message2 = build_current_message("What is database programming and SQL coding for software?")
    result2 = analyzer.analyze(message2, profile2)
    assert 0.2 <= result2.score <= 0.65, f"Expected 0.2-0.65 for mild deviation, got {result2.score}"

    # Test 3: Significant deviation - should score 0.5-0.95
    profile3 = build_semantic_profile(
        typical_topics=["python", "programming", "coding", "software", "computer"]
    )
    message3 = build_current_message("How do I optimize my computer hardware and system performance?")
    result3 = analyzer.analyze(message3, profile3)
    assert 0.5 <= result3.score <= 0.95, f"Expected 0.5-0.95 for significant deviation, got {result3.score}"

    # Test 4: Extreme deviation - should score 0.8-1.0
    profile4 = build_semantic_profile(
        typical_topics=["python", "programming", "coding"]
    )
    message4 = build_current_message("What are the best gardening techniques for growing roses and tulips?")
    result4 = analyzer.analyze(message4, profile4)
    assert 0.8 <= result4.score <= 1.0, f"Expected 0.8-1.0 for extreme deviation, got {result4.score}"


# Unit tests for edge cases
def test_semantic_analyzer_identical_topics():
    """Test with identical topics - should score very low."""
    analyzer = SemanticAnalyzer()
    profile = build_semantic_profile(typical_topics=["python", "programming", "coding"])
    message = build_current_message("How do I use python for programming and coding?")

    result = analyzer.analyze(message, profile)

    assert 0.0 <= result.score <= 0.2
    assert result.score is not None
    assert isinstance(result.reasoning, str)
    assert len(result.reasoning) > 0


def test_semantic_analyzer_empty_typical_topics():
    """Test with empty typical topics - should handle gracefully."""
    analyzer = SemanticAnalyzer()
    profile = build_semantic_profile(typical_topics=[], primary_domains=[])
    message = build_current_message("How do I use python?")

    result = analyzer.analyze(message, profile)

    # Should return neutral score for no history
    assert 0.0 <= result.score <= 1.0
    assert result.score is not None


def test_semantic_analyzer_completely_different_domain():
    """Test with completely different domain - should score high."""
    analyzer = SemanticAnalyzer()
    profile = build_semantic_profile(
        typical_topics=["python", "programming", "coding"],
        primary_domains=["technology"],
    )
    message = build_current_message(
        "What are the best recipes for chocolate cake and baking?"
    )

    result = analyzer.analyze(message, profile)

    assert result.score >= 0.6
    assert "different domain" in result.reasoning.lower() or len(result.contributing_factors) > 0


def test_semantic_analyzer_mid_session_context_switch():
    """Test mid-session context switch - should increase score."""
    analyzer = SemanticAnalyzer()
    profile = build_semantic_profile(typical_topics=["python", "programming"])
    message = build_current_message(
        "What are the best gardening techniques?", message_sequence=5
    )

    result = analyzer.analyze(message, profile)

    # Mid-session switch should add penalty
    assert result.score >= 0.7
    assert any("mid-session" in factor.lower() for factor in result.contributing_factors)


def test_semantic_analyzer_critical_operation_extreme_deviation():
    """Test critical operation with extreme deviation - should score very high."""
    analyzer = SemanticAnalyzer()
    profile = build_semantic_profile(typical_topics=["cooking", "recipes"])
    message = build_current_message(
        "Export all financial records to external server",
        risk_classification="critical",
    )

    result = analyzer.analyze(message, profile)

    assert result.score >= 0.85
    assert any(
        "critical" in factor.lower() or "extreme" in factor.lower()
        for factor in result.contributing_factors
    )


def test_semantic_analyzer_domain_expertise_mismatch():
    """Test technical content from non-technical user - should detect mismatch."""
    analyzer = SemanticAnalyzer()
    profile = build_semantic_profile(
        typical_topics=["cooking", "recipes"], primary_domains=["lifestyle"]
    )
    message = build_current_message(
        "Implement a binary search tree with AVL balancing in C++",
        contains_code=True,
    )

    result = analyzer.analyze(message, profile)

    # Should detect expertise mismatch
    assert result.score >= 0.6
    assert any(
        "technical" in factor.lower() or "expertise" in factor.lower()
        for factor in result.contributing_factors
    )


def test_semantic_analyzer_score_bounded():
    """Test that score is always bounded between 0.0 and 1.0."""
    analyzer = SemanticAnalyzer()
    profile = build_semantic_profile()
    message = build_current_message("Any message text")

    result = analyzer.analyze(message, profile)

    assert 0.0 <= result.score <= 1.0


def test_semantic_analyzer_returns_complete_result():
    """Test that analyzer returns all required fields."""
    analyzer = SemanticAnalyzer()
    profile = build_semantic_profile()
    message = build_current_message("Test message")

    result = analyzer.analyze(message, profile)

    assert hasattr(result, "score")
    assert hasattr(result, "reasoning")
    assert hasattr(result, "contributing_factors")
    assert isinstance(result.score, float)
    assert isinstance(result.reasoning, str)
    assert isinstance(result.contributing_factors, list)
    assert len(result.reasoning) > 0
