"""build_user_profile() in evaluation.py — has_requested_sensitive_ops from training operation_risk."""

import sys
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_REAL_OPEN = open


def _open_dataset_stub(path, *a, **k):
    p = str(path)
    if "personachat_processed" in p or "blended_skill_talk_processed" in p or "anthropic_hh_processed" in p:
        return StringIO('{"metadata":{},"users":[],"messages":[]}')
    return _REAL_OPEN(path, *a, **k)


def _fake_sentence_transformer(*a, **k):
    m = MagicMock()

    def enc(texts, **kwargs):
        if isinstance(texts, str):
            n = 1
        else:
            n = len(texts)
        return np.random.randn(n, 384).astype(np.float32)

    m.encode = enc
    return m


@patch("sentence_transformers.SentenceTransformer", _fake_sentence_transformer)
@patch("builtins.open", _open_dataset_stub)
def test_build_user_profile_has_requested_sensitive_ops_from_operation_risk() -> None:
    if "evaluation" in sys.modules:
        del sys.modules["evaluation"]
    import evaluation

    profile = evaluation.build_user_profile(
        {"user_id": "u1", "account_age_days": 100},
        [
            {
                "message_text": "one two three four five six",
                "timestamp": "2020-01-01T10:00:00",
                "operation_risk": "low",
                "is_anomaly": False,
            },
            {
                "message_text": "two three four five six seven eight",
                "timestamp": "2020-01-01T10:01:00",
                "operation_risk": "low",
                "is_anomaly": False,
            },
            {
                "message_text": "export all records three four five",
                "timestamp": "2020-01-01T10:02:00",
                "operation_risk": "high",
                "is_anomaly": False,
            },
        ],
    )
    assert profile is not None
    assert profile.operational_profile.has_requested_sensitive_ops is True
