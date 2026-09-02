"""
Operation risk classifier for inferring critical/sensitive/normal from message text.

Hardens against paraphrase evasion by using expanded synonym lists and
normalization. Replaces brittle exact substring matching.

Used when operation_risk is not pre-labeled (e.g., production or evaluation
with raw messages). Paraphrases like "revoke my two-step" map to "disable 2fa".
"""

from __future__ import annotations

import re
from typing import Literal

RiskLevel = Literal["low", "medium", "high", "critical"]

# Patterns that indicate critical when verb + object co-occur (flexible word order)
_CRITICAL_PATTERNS = [
    # delete/remove/wipe + accounts
    re.compile(
        r"\b(delete|remove|destroy|wipe|erase|revoke|terminate|drop)\b.*\b"
        r"(all|every|entire|bulk|whole)?\s*(user\s+)?(accounts?|data|records?)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    # disable/deactivate + 2fa/auth
    re.compile(
        r"\b(disable|deactivate|revoke|turn\s+off|turnoff|remove|bypass)\b.*\b"
        r"(2fa|two[- ]?factor|mfa|authentication|2[- ]?step|two[- ]?step)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"\b(2fa|two[- ]?factor|mfa|authentication)\b.*\b"
        r"(disable|deactivate|revoke|turn\s+off|remove)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    # reset/change + password
    re.compile(
        r"\b(reset|change|revert|revoke)\b.*\b(password|passwords|credentials?)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"\b(password|passwords)\b.*\b(reset|change|revert|revoke)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    # export/dump + all data
    re.compile(
        r"\b(export|dump|download|transfer|exfiltrate)\b.*\b"
        r"(all|every|entire|bulk|whole)?\s*(data|records?|database|users?)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    # grant + admin/root
    re.compile(
        r"\b(grant|give|add|assign|escalate)\b.*\b"
        r"(admin|root|superuser|privilege|elevated|full\s+access)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    # "delete all accounts" and paraphrases
    re.compile(
        r"\b(delete|remove|wipe|erase|terminate|close)\s+(all|every|my)\s+"
        r"(accounts?|profiles?|data)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(all|every)\s+(accounts?|profiles?)\s+(deleted?|removed?|wiped?|erased?)\b",
        re.IGNORECASE,
    ),
]


def _normalize(text: str) -> str:
    """Normalize text for matching: lowercase, collapse whitespace."""
    return " ".join(text.lower().split())


def classify_operation_risk(message: str) -> RiskLevel:
    """
    Infer operation risk from message text.

    Uses expanded synonym lists and flexible patterns to resist paraphrase
    evasion (e.g., "revoke my two-step" -> critical).

    Args:
        message: Raw message text.

    Returns:
        "critical", "high", "medium", or "low"
    """
    text = _normalize(message)
    if not text.strip():
        return "low"

    for pat in _CRITICAL_PATTERNS:
        if pat.search(text):
            return "critical"

    # High: sensitive but not destructive
    high_patterns = [
        r"\b(change|modify|update)\b.*\b(account|email|phone|settings)\b",
        r"\b(close|deactivate)\b.*\b(account)\b",
    ]
    for p in high_patterns:
        if re.search(p, text, re.IGNORECASE | re.DOTALL):
            return "high"

    return "low"
