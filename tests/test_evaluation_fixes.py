"""Tests for evaluation.py fixes A (is_typical_active_time) and C (per-class metrics)."""

import sys
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
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


def _import_evaluation():
    if "evaluation" in sys.modules:
        del sys.modules["evaluation"]
    import evaluation  # noqa: F401
    return sys.modules["evaluation"]


# ---------- Fix A ----------

@patch("sentence_transformers.SentenceTransformer", _fake_sentence_transformer)
@patch("builtins.open", _open_dataset_stub)
def test_is_typical_active_time_uses_user_profile() -> None:
    """Fix A: night-shift worker's active hours come from profile, not hardcoded 9..21."""
    evaluation = _import_evaluation()

    # Mock a profile whose only consumed attribute is temporal_profile.most_active_hours_utc.
    night_shift_profile = SimpleNamespace(
        temporal_profile=SimpleNamespace(most_active_hours_utc=[0, 1, 2, 3, 4])
    )

    msg_hour_2 = {
        "message_text": "routine ping",
        "timestamp": "2024-01-01T02:00:00",
        "session_id": "s0",
    }
    cm2 = evaluation.message_to_current_message(msg_hour_2, None, user_profile=night_shift_profile)
    assert cm2.temporal_context.is_typical_active_time is True, (
        "hour 2 is in the learned [0,1,2,3,4] — must be typical for this user"
    )

    msg_hour_14 = {
        "message_text": "routine ping",
        "timestamp": "2024-01-01T14:00:00",
        "session_id": "s0",
    }
    cm14 = evaluation.message_to_current_message(msg_hour_14, None, user_profile=night_shift_profile)
    assert cm14.temporal_context.is_typical_active_time is False, (
        "hour 14 is NOT in learned [0,1,2,3,4] — must be atypical "
        "(opposite of the old 9<=h<=21 hardcode)"
    )


# ---------- Fix C ----------

@patch("sentence_transformers.SentenceTransformer", _fake_sentence_transformer)
@patch("builtins.open", _open_dataset_stub)
def test_compute_per_class_metrics_minimal_fixture() -> None:
    """Fix C: per-class F1 across two anomaly_type values + benign."""
    evaluation = _import_evaluation()

    # 4 messages covering two anomaly_type values plus benign.
    # Scheme:
    #   role_injection: 1 TP  (score 0.9, should_flag=True)  -> P=1, R=1, F1=1
    #   exfiltration:   1 FN  (score 0.4, should_flag=True)  -> P=0, R=0, F1=0
    #   benign:         1 TN  (score 0.2, should_flag=False) + 1 FP (score 0.8, should_flag=False)
    #                   -> TP=0, FP=1, FN=0, P=0, R=0, F1=0
    messages = [
        {"anomaly_type": "role_injection", "should_flag": True},
        {"anomaly_type": "exfiltration", "should_flag": True},
        {"anomaly_type": None, "should_flag": False},
        {"anomaly_type": "benign", "should_flag": False},
    ]
    results = [
        {"anomaly_score": 0.9},
        {"anomaly_score": 0.4},
        {"anomaly_score": 0.2},
        {"anomaly_score": 0.8},
    ]

    out = evaluation.compute_per_class_metrics(messages, results)

    assert "anomaly_type" in out and "attack_phase" in out
    at = out["anomaly_type"]

    assert at["role_injection"]["support"] == 1
    assert at["role_injection"]["precision"] == 1.0
    assert at["role_injection"]["recall"] == 1.0
    assert at["role_injection"]["f1"] == 1.0

    assert at["exfiltration"]["support"] == 1
    assert at["exfiltration"]["precision"] == 0.0
    assert at["exfiltration"]["recall"] == 0.0
    assert at["exfiltration"]["f1"] == 0.0

    # Both None and "benign" collapse to "benign".
    assert at["benign"]["support"] == 2
    # TP=0, FP=1, FN=0 -> precision=0, recall=0, f1=0 (zero-division guarded).
    assert at["benign"]["precision"] == 0.0
    assert at["benign"]["recall"] == 0.0
    assert at["benign"]["f1"] == 0.0

    # attack_phase absent from all messages -> everything collapses to "benign".
    ap = out["attack_phase"]
    assert set(ap.keys()) == {"benign"}
    assert ap["benign"]["support"] == 4
