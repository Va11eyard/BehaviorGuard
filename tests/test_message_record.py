"""MessageRecord Pydantic JSON round-trip and build_from_history sensitive-ops flag."""

import json
from unittest.mock import MagicMock, patch

import numpy as np

from behaviorguard.profile_manager import MessageRecord, ProfileManager


def test_message_record_model_validate_round_trip_minimal() -> None:
    data = {
        "text": "hello",
        "timestamp": "2025-01-01T12:00:00",
    }
    r1 = MessageRecord.model_validate(data)
    dumped = r1.model_dump()
    s = json.dumps(dumped)
    r2 = MessageRecord.model_validate(json.loads(s))
    assert r1 == r2
    assert r1.operation_risk is None
    assert r1.session_id == "default"
    assert r1.is_anomaly is False


def test_message_record_model_validate_with_operation_risk() -> None:
    data = {
        "text": "delete all accounts",
        "timestamp": "2025-01-01T12:00:00",
        "session_id": "s1",
        "is_anomaly": False,
        "operation_risk": "critical",
    }
    r = MessageRecord.model_validate(data)
    assert r.operation_risk == "critical"
    s = json.dumps(r.model_dump())
    r2 = MessageRecord.model_validate(json.loads(s))
    assert r2.model_dump() == r.model_dump()


def test_message_record_invalid_risk_becomes_none() -> None:
    r = MessageRecord.model_validate(
        {
            "text": "x",
            "timestamp": "2025-01-01T00:00:00",
            "operation_risk": "bogus",
        }
    )
    assert r.operation_risk is None


def _fake_st_encode(texts, **kwargs):
    if isinstance(texts, str):
        n = 1
    else:
        n = len(texts)
    return np.random.randn(n, 384).astype(np.float32)


@patch("sentence_transformers.SentenceTransformer")
def test_build_from_history_has_requested_sensitive_from_normal_history(
    mock_st: MagicMock,
) -> None:
    """Normal history with high/critical → has_requested_sensitive_ops True."""
    m = MagicMock()
    m.encode = _fake_st_encode
    mock_st.return_value = m
    pm = ProfileManager(decay=0.95, embedding_model="all-MiniLM-L6-v2")
    msgs = [
        MessageRecord(
            text="just chatting about the weather here",
            timestamp="2025-01-01T10:00:00",
            session_id="a",
            is_anomaly=False,
            operation_risk="low",
        ),
        MessageRecord(
            text="another harmless message to build profile stats",
            timestamp="2025-01-01T10:01:00",
            session_id="a",
            is_anomaly=False,
            operation_risk="low",
        ),
        MessageRecord(
            text="export all customer records please",
            timestamp="2025-01-01T10:02:00",
            session_id="a",
            is_anomaly=False,
            operation_risk="high",
        ),
    ]
    p = pm.build_from_history("u1", msgs, account_age_days=30)
    assert p.operational_profile.has_requested_sensitive_ops is True


@patch("sentence_transformers.SentenceTransformer")
def test_build_from_history_no_sensitive_ops_when_only_low(mock_st: MagicMock) -> None:
    m = MagicMock()
    m.encode = _fake_st_encode
    mock_st.return_value = m
    pm = ProfileManager(decay=0.95, embedding_model="all-MiniLM-L6-v2")
    msgs = [
        MessageRecord(
            text="one two three four five six seven eight",
            timestamp="2025-01-01T10:00:00",
            is_anomaly=False,
            operation_risk="low",
        ),
        MessageRecord(
            text="two two three four five six seven eight nine",
            timestamp="2025-01-01T10:01:00",
            is_anomaly=False,
            operation_risk="low",
        ),
        MessageRecord(
            text="three two three four five six seven eight nine ten",
            timestamp="2025-01-01T10:02:00",
            is_anomaly=False,
            operation_risk="medium",
        ),
    ]
    p = pm.build_from_history("u2", msgs, account_age_days=30)
    assert p.operational_profile.has_requested_sensitive_ops is False
