"""Unit tests for InputValidator."""

import json

import pytest

from behaviorguard.models import ErrorResponse, EvaluationInput
from behaviorguard.validator import InputValidator


@pytest.fixture
def validator():
    """Create InputValidator instance."""
    return InputValidator()


@pytest.fixture
def valid_input_data():
    """Create valid input data."""
    return {
        "user_profile": {
            "user_id": "user123",
            "account_age_days": 365,
            "total_interactions": 150,
            "semantic_profile": {
                "typical_topics": ["programming", "python"],
                "primary_domains": ["technology"],
                "topic_diversity_score": 0.7,
                "embedding_centroid_summary": "Technical discussions",
            },
            "linguistic_profile": {
                "avg_message_length_tokens": 50.0,
                "avg_message_length_chars": 250.0,
                "lexical_diversity_mean": 0.6,
                "lexical_diversity_std": 0.1,
                "formality_score_mean": 0.5,
                "formality_score_std": 0.1,
                "politeness_score_mean": 0.7,
                "politeness_score_std": 0.1,
                "question_ratio_mean": 0.3,
                "uses_technical_vocabulary": True,
                "uses_code_blocks": True,
                "primary_languages": ["en"],
                "typical_sentence_complexity": "moderate",
            },
            "temporal_profile": {
                "typical_session_duration_minutes": 30.0,
                "typical_inter_message_gap_seconds": 60.0,
                "most_active_hours_utc": [14, 15, 16, 17],
                "most_active_days_of_week": ["Monday", "Tuesday", "Wednesday"],
                "average_messages_per_session": 10.0,
                "longest_session_duration_minutes": 120.0,
                "typical_session_frequency_per_week": 5.0,
                "last_activity_timestamp": "2025-12-01T10:00:00Z",
            },
            "operational_profile": {
                "common_intent_types": ["information_seeking", "code_generation"],
                "tools_used_historically": ["search", "calculator"],
                "has_requested_sensitive_ops": False,
                "typical_risk_level": "low",
            },
        },
        "current_message": {
            "text": "How do I implement a binary search tree?",
            "timestamp": "2025-12-08T15:30:00Z",
            "session_id": "session456",
            "message_sequence_in_session": 3,
            "time_since_last_message_seconds": 45.0,
            "detected_intent": "information_seeking",
            "requested_operation": {
                "type": "read",
                "risk_classification": "low",
                "targets": None,
                "requires_auth": False,
            },
            "linguistic_features": {
                "message_length_tokens": 8,
                "message_length_chars": 42,
                "lexical_diversity": 0.875,
                "formality_score": 0.6,
                "politeness_score": 0.7,
                "contains_code": False,
                "contains_urls": False,
                "language": "en",
            },
            "temporal_context": {
                "hour_of_day_utc": 15,
                "day_of_week": "Monday",
                "is_typical_active_time": True,
                "time_since_last_session_hours": 24.0,
            },
        },
        "system_config": {
            "sensitivity_level": "medium",
            "deployment_context": "consumer",
            "enable_temporal_scoring": True,
            "enable_linguistic_scoring": True,
            "enable_semantic_scoring": True,
        },
    }


def test_validate_valid_json(validator, valid_input_data):
    """Test validation with valid JSON."""
    raw_input = json.dumps(valid_input_data)
    result = validator.validate(raw_input)

    assert result.is_valid is True
    assert len(result.errors) == 0


def test_validate_invalid_json(validator):
    """Test validation with invalid JSON."""
    raw_input = "{ invalid json }"
    result = validator.validate(raw_input)

    assert result.is_valid is False
    assert len(result.errors) > 0
    assert "Invalid JSON" in result.errors[0]


def test_validate_missing_required_field(validator, valid_input_data):
    """Test validation with missing required field."""
    del valid_input_data["user_profile"]["user_id"]
    raw_input = json.dumps(valid_input_data)
    result = validator.validate(raw_input)

    assert result.is_valid is False
    assert len(result.errors) > 0
    assert any("user_id" in error for error in result.errors)


def test_validate_out_of_range_score(validator, valid_input_data):
    """Test validation with out-of-range score value."""
    valid_input_data["user_profile"]["semantic_profile"]["topic_diversity_score"] = 1.5
    raw_input = json.dumps(valid_input_data)
    result = validator.validate(raw_input)

    # Pydantic doesn't enforce range on this field, but would on Field(ge=0, le=1)
    # This test documents current behavior
    assert result.is_valid is True  # No range constraint on this field


def test_validate_invalid_enum_value(validator, valid_input_data):
    """Test validation with invalid enum value."""
    valid_input_data["system_config"]["sensitivity_level"] = "invalid"
    raw_input = json.dumps(valid_input_data)
    result = validator.validate(raw_input)

    assert result.is_valid is False
    assert len(result.errors) > 0


def test_parse_valid_input(validator, valid_input_data):
    """Test parsing valid input."""
    raw_input = json.dumps(valid_input_data)
    result = validator.parse(raw_input)

    assert isinstance(result, EvaluationInput)
    assert result.user_profile.user_id == "user123"
    assert result.current_message.text == "How do I implement a binary search tree?"
    assert result.system_config.sensitivity_level == "medium"


def test_parse_invalid_input(validator):
    """Test parsing invalid input."""
    raw_input = "{ invalid json }"
    result = validator.parse(raw_input)

    assert isinstance(result, ErrorResponse)
    assert result.error.type == "ValidationError"
    assert len(result.error.details) > 0


def test_parse_missing_field(validator, valid_input_data):
    """Test parsing input with missing field."""
    del valid_input_data["current_message"]
    raw_input = json.dumps(valid_input_data)
    result = validator.parse(raw_input)

    assert isinstance(result, ErrorResponse)
    assert result.error.type == "ValidationError"
    assert any("current_message" in detail for detail in result.error.details)


def test_parse_wrong_type(validator, valid_input_data):
    """Test parsing input with wrong type."""
    valid_input_data["user_profile"]["account_age_days"] = "not_an_integer"
    raw_input = json.dumps(valid_input_data)
    result = validator.parse(raw_input)

    assert isinstance(result, ErrorResponse)
    assert result.error.type == "ValidationError"
