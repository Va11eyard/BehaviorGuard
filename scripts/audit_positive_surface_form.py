#!/usr/bin/env python3
"""Audit should_flag=True test messages for surface-form vs metadata-only labels."""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import evaluation as ev  # noqa: E402

DATASETS = {
    "personachat": "PersonaChat",
    "blended_skill_talk": "BST",
    "anthropic_hh": "AnthropicHH",
}

# Heuristic overt-attack patterns (explicit malicious surface form)
OVERT_PATTERNS = [
    r"\b(admin|root|sudo)\s+(access|privileges?|account)\b",
    r"\bgrant\b.*\b(access|permission|admin)\b",
    r"\b(change|reset|update)\b.*\bpassword\b",
    r"\bdisable\b.*\b(2fa|two[- ]factor|mfa)\b",
    r"\bexport\b.*\b(data|history|records?|transactions?)\b",
    r"\b(ignore|disregard|forget|override)\b.*\b(instruction|rule|policy|guideline)\b",
    r"\b(jailbreak|dan mode|developer mode|do anything now)\b",
    r"\b(bypass|circumvent|work around)\b.*\b(safety|filter|restriction|guard)\b",
    r"\b(reveal|leak|dump|show me)\b.*\b(system prompt|hidden prompt|secret)\b",
    r"\b(hack|exploit|payload|malware|ransomware|keylogger)\b",
    r"\b(social security|credit card|ssn|bank account)\b.*\b(number|details?)\b",
    r"@[a-z0-9._-]+\.(com|org|net)\b",  # exfil to attacker email
    r"\bmalicious[_-]?\w+\b",
    r"\battacker[_-]?\w+\b",
    r"\bhacked\d+\b",
    r"\bpretend (you are|to be)\b.*\b(no restrictions|unfiltered|evil)\b",
    r"\broleplay as\b.*\b(villain|unrestricted|without limits)\b",
    r"\bexecute\b.*\b(code|command|script|sql)\b",
    r"\b(drop table|rm -rf|curl.*\|)\b",
    r"\bimpersonate\b",
    r"\bsteal\b.*\b(credential|token|password|data)\b",
    r"\bphishing\b",
    r"\bsend (all|my) (data|messages|logs)\b",
    r"\bunrestricted ai\b",
    r"\bno ethical\b",
    r"\bwithout (safety|restrictions|limits|guidelines)\b",
]

OVERT_RE = [re.compile(p, re.IGNORECASE) for p in OVERT_PATTERNS]


def _collect_test_positives(test_data: dict, max_users: int = 20) -> list[dict]:
    """Same user sampling and 80/20 split as evaluate_method()."""
    test_user_ids = set(test_data["splits"]["test"]["user_ids"])
    test_users = [u for u in test_data["users"] if u["user_id"] in test_user_ids]
    test_messages_by_user: dict[str, list] = defaultdict(list)
    for m in test_data["messages"]:
        if m["user_id"] in test_user_ids:
            test_messages_by_user[m["user_id"]].append(m)

    users_with_anomalies, users_without = [], []
    for user in test_users:
        user_msgs = test_messages_by_user[user["user_id"]]
        if any(m.get("should_flag", False) for m in user_msgs):
            users_with_anomalies.append(user)
        else:
            users_without.append(user)

    sampled = users_with_anomalies[:max_users]
    sampled.extend(users_without[: max_users - len(sampled)])

    positives: list[dict] = []
    for user in sampled:
        user_msgs = test_messages_by_user[user["user_id"]]
        split_idx = int(len(user_msgs) * 0.8)
        for msg in user_msgs[split_idx:]:
            if msg.get("should_flag", False):
                positives.append({
                    "user_id": user["user_id"],
                    "message_text": msg["message_text"],
                    "anomaly_type": msg.get("anomaly_type"),
                    "attack_phase": msg.get("attack_phase"),
                    "is_anomaly": msg.get("is_anomaly"),
                    "operation_risk": msg.get("operation_risk"),
                    "operation_type": msg.get("operation_type"),
                })
    return positives


def _heuristic_overt(text: str) -> bool:
    return any(p.search(text) for p in OVERT_RE)


def _manual_category(msg: dict) -> tuple[str, str]:
    """
    Returns (category, rationale).
    category: 'a_overt' | 'b_benign_surface'
    """
    text = msg["message_text"]
    atype = (msg.get("anomaly_type") or "").lower()

    if _heuristic_overt(text):
        return "a_overt", "explicit malicious surface form in text"

    # Known benign-surface anomaly types in gradual attacks
    if atype in ("account_takeover", "social_engineering", "behavioral_drift"):
        # Check if text reads as normal conversation
        benign_markers = [
            "gardening", "organic vegetables", "how are you", "nice to meet",
            "what do you do", "favorite", "hobby", "weather", "recipe",
            "tell me about yourself", "good morning", "hello", "hi there",
        ]
        lower = text.lower()
        if any(b in lower for b in benign_markers):
            return "b_benign_surface", f"benign conversational text; flagged via {atype} metadata"

    # Prompt injection without obvious jailbreak language
    if atype in ("prompt_injection", "jailbreak", "instruction_override"):
        if not _heuristic_overt(text):
            return "b_benign_surface", f"no overt injection language; type={atype}"

    # Default: if no overt pattern, classify as benign surface unless clearly operational
    if msg.get("operation_risk") in ("high", "critical") and _heuristic_overt(text):
        return "a_overt", "high-risk operation with overt text"

    if _heuristic_overt(text):
        return "a_overt", "pattern match"

    # Short ambiguous — read text
    lower = text.lower()
    attack_verbs = [
        "grant", "export", "disable", "password", "admin", "bypass",
        "ignore previous", "system prompt", "jailbreak", "malicious",
        "attacker", "hack", "steal", "impersonate", "phish",
    ]
    if any(v in lower for v in attack_verbs):
        return "a_overt", "attack keyword in text"

    return "b_benign_surface", "no overt attack language; context/metadata label only"


def main() -> None:
    report = {"datasets": {}, "disclosure_threshold_pct": 20.0}

    print("=" * 90)
    print("POSITIVE LABEL SURFACE-FORM AUDIT (evaluate_method test split, max_users=20)")
    print("=" * 90)

    for dk, dname in DATASETS.items():
        positives = _collect_test_positives(ev.datasets[dk], max_users=20)
        rows = []
        for i, msg in enumerate(positives, 1):
            cat, rationale = _manual_category(msg)
            rows.append({**msg, "index": i, "category": cat, "rationale": rationale})

        n_a = sum(1 for r in rows if r["category"] == "a_overt")
        n_b = sum(1 for r in rows if r["category"] == "b_benign_surface")
        n = len(rows)
        pct_b = 100.0 * n_b / n if n else 0.0
        flag = pct_b > 20.0

        report["datasets"][dk] = {
            "display_name": dname,
            "n_positives": n,
            "a_overt": n_a,
            "b_benign_surface": n_b,
            "pct_b_benign_surface": round(pct_b, 1),
            "disclosure_recommended": flag,
            "messages": rows,
        }

        print(f"\n{'=' * 90}")
        print(f"{dname} — {n} should_flag=True test messages")
        print(f"{'=' * 90}")
        for r in rows:
            cat_label = "A: overt" if r["category"] == "a_overt" else "B: benign surface"
            print(f"\n[{r['index']:02d}] {cat_label} | type={r['anomaly_type']} | "
                  f"phase={r.get('attack_phase')} | op_risk={r.get('operation_risk')}")
            print(f"     Rationale: {r['rationale']}")
            print(f"     Text: {r['message_text']}")

        print(f"\n--- {dname} SUMMARY ---")
        print(f"  (a) Overt attack surface:     {n_a}/{n} ({100*n_a/n:.1f}%)")
        print(f"  (b) Benign-looking surface:   {n_b}/{n} ({pct_b:.1f}%)")
        if flag:
            print(f"  *** DISCLOSURE FLAG: category (b) > 20% — recommend §V/VII limitation ***")

    out = ROOT / "results" / "methodology-diagnostics" / "positive_label_surface_audit.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
