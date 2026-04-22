"""Unit tests for operation risk classifier (P3.1)."""

import pytest

from behaviorguard.utils.operation_risk_classifier import classify_operation_risk


def test_delete_all_accounts_paraphrases():
    """Paraphrases of 'delete all accounts' must all be classified as critical."""
    paraphrases = [
        "delete all accounts",
        "remove every account",
        "wipe all user accounts",
        "erase all accounts",
        "terminate all user accounts",
        "I want to delete all my accounts",
        "please remove every account I have",
    ]
    for text in paraphrases:
        assert classify_operation_risk(text) == "critical", f"Failed for: {text!r}"


def test_disable_2fa_paraphrases():
    """Paraphrases of 'disable 2fa' must be classified as critical."""
    paraphrases = [
        "disable 2fa",
        "revoke my two-step authentication",
        "turn off two factor authentication",
        "deactivate 2fa",
        "remove 2fa",
        "bypass two-factor auth",
    ]
    for text in paraphrases:
        assert classify_operation_risk(text) == "critical", f"Failed for: {text!r}"


def test_reset_password_paraphrases():
    """Reset password requests should be critical."""
    paraphrases = [
        "reset my password",
        "change all passwords",
        "revoke credentials",
    ]
    for text in paraphrases:
        assert classify_operation_risk(text) == "critical", f"Failed for: {text!r}"


def test_normal_messages_low_risk():
    """Normal conversational messages should be low risk."""
    normal = [
        "How do I implement a BST in Python?",
        "What's the weather like today?",
        "Tell me about your favorite movie.",
        "I like to go to the park.",
    ]
    for text in normal:
        assert classify_operation_risk(text) == "low", f"Failed for: {text!r}"


def test_empty_message():
    """Empty message should be low risk."""
    assert classify_operation_risk("") == "low"
    assert classify_operation_risk("   ") == "low"
