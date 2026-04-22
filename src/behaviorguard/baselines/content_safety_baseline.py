"""
Content-only safety classifier baseline.

Replicates the behaviour of a Llama-Guard-style safety classifier that
evaluates each message *in isolation*, without any user-history context.
This is Baseline 4 from Section 5.2 of the BehaviorGuard paper.

The implementation is intentionally history-free: it scores a single
message against a fixed taxonomy of hazard categories and returns an
anomaly score, but it deliberately ignores whether the message is
anomalous *for this particular user*.  That blindspot is precisely what
BehaviorGuard addresses.

Hazard taxonomy (mirrors Llama Guard's 10 categories):
    S1  - Violent crimes
    S2  - Non-violent crimes
    S3  - Sex-related crimes
    S4  - Child sexual exploitation
    S5  - Defamation
    S6  - Specialised advice (financial, medical, legal)
    S7  - Privacy violations
    S8  - Intellectual property
    S9  - Indiscriminate weapons / CBRN
    S10 - Hate speech
    S11 - Suicide / self-harm
    S12 - Sexual content
    S13 - Elections / disinformation
    S14 - Code / cyberattack facilitation  ← added for this domain
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple


# ─── Hazard taxonomy ──────────────────────────────────────────────────────────

_HAZARD_PATTERNS: List[Tuple[str, float, List[str]]] = [
    # (category_id, base_weight, keyword_patterns)
    (
        "S1_violent_crimes",
        0.90,
        [
            r"\b(kill|murder|assassinate|shoot|stab|bomb|terrorist)\b",
            r"\b(mass.?shooting|genocide|homicide)\b",
        ],
    ),
    (
        "S2_nonviolent_crimes",
        0.70,
        [
            r"\b(fraud|embezzl|launder|counterfeit|forgery|bribe)\b",
            r"\b(theft|robbery|burglary|shoplifting)\b",
            r"\b(money.?laundering|wire.?fraud)\b",
        ],
    ),
    (
        "S6_specialised_advice",
        0.40,
        [
            r"\b(diagnos|prescri|medical.?advice|treat.?disease)\b",
            r"\b(legal.?advice|sue|lawsuit|attorney)\b",
            r"\b(invest.?advice|buy.?stock|financial.?advice)\b",
        ],
    ),
    (
        "S7_privacy_violations",
        0.65,
        [
            r"\b(doxx|personal.?data|private.?address|ssn|social.?security)\b",
            r"\b(deanonymize|de-anonymize|track.?location)\b",
        ],
    ),
    (
        "S9_cbrn",
        0.95,
        [
            r"\b(chemical.?weapon|bioweapon|nerve.?agent|sarin|vx.?gas)\b",
            r"\b(synthesize.*(poison|toxin|explosive))\b",
            r"\b(radiolog|dirty.?bomb|nuclear.?device)\b",
        ],
    ),
    (
        "S10_hate_speech",
        0.75,
        [
            r"\b(racial.?slur|ethnic.?cleansing|white.?supremac)\b",
            r"\b(hate.?group|neo.?nazi|kkk)\b",
        ],
    ),
    (
        "S11_self_harm",
        0.80,
        [
            r"\b(suicide|self.?harm|cut.?myself|overdose.?on)\b",
            r"\b(how.?to.?die|method.?to.?kill.?myself)\b",
        ],
    ),
    (
        "S14_cyberattack",
        0.85,
        [
            r"\b(exploit|payload|backdoor|rootkit|keylogger)\b",
            r"\b(sql.?injection|xss|cross.?site|buffer.?overflow)\b",
            r"\b(penetrat.?test|pentest|red.?team)\b",
            r"\b(malware|ransomware|trojan|worm|botnet)\b",
            r"\b(credential.?harvest|privilege.?escalat|lateral.?movement)\b",
            r"\b(exfiltrat|data.?breach|c2.?server|command.?and.?control)\b",
        ],
    ),
    (
        "S2_account_abuse",
        0.75,
        [
            r"\b(delete.?all.?account|wipe.?account|bulk.?delete)\b",
            r"\b(export.?all.?data|dump.?database|exfiltrate)\b",
            r"\b(disable.?2fa|bypass.?authentication|reset.?password)\b",
            r"\b(grant.?admin|escalate.?privilege|add.?admin)\b",
        ],
    ),
    (
        "S2_social_engineering",
        0.70,
        [
            r"\b(phish|spear.?phish|pretexting|vishing|smishing)\b",
            r"\b(impersonat|pretend.?to.?be|pose.?as)\b",
            r"\b(urgently|immediately|asap).{0,30}(account|password|transfer|access)\b",
        ],
    ),
]

# High-risk operation keywords that amplify scores
_HIGH_RISK_OPS = [
    r"\b(delete|remove|destroy|wipe|erase).{0,20}(all|bulk|every|entire)\b",
    r"\b(export|download|transfer).{0,20}(all|bulk|every|database|records)\b",
    r"\b(reset|change|disable).{0,20}(password|2fa|auth|access|credential)\b",
    r"\b(grant|give|add).{0,20}(admin|root|superuser|permission|access)\b",
]


class ContentSafetyBaseline:
    """
    Content-only safety classifier that mirrors a Llama-Guard-style detector.

    Key characteristics:
    - Evaluates messages in isolation (no user history, no behavioral context)
    - Uses a fixed hazard taxonomy with regex pattern matching
    - Returns a score that reflects content-level risk only
    - Does NOT detect whether a message is anomalous for *this* user

    This is intentionally blind to behavioral context, making it the baseline
    that BehaviorGuard is designed to outperform on account-takeover and
    social-engineering scenarios.
    """

    def __init__(self, amplify_high_risk_ops: bool = True):
        """
        Args:
            amplify_high_risk_ops: If True, messages requesting destructive
                operations receive an additional score boost even if they
                don't match a specific hazard category.
        """
        self.amplify_high_risk_ops = amplify_high_risk_ops
        # Pre-compile all patterns for efficiency
        self._compiled: List[Tuple[str, float, List[re.Pattern[str]]]] = [
            (
                cat_id,
                weight,
                [re.compile(p, re.IGNORECASE | re.DOTALL) for p in patterns],
            )
            for cat_id, weight, patterns in _HAZARD_PATTERNS
        ]
        self._high_risk_ops_compiled = [
            re.compile(p, re.IGNORECASE | re.DOTALL) for p in _HIGH_RISK_OPS
        ]

    def detect(self, message: str) -> Dict:
        """
        Classify a single message for content-level safety.

        Args:
            message: The raw message text to evaluate.

        Returns:
            Dict with:
                - anomaly_score: float in [0.0, 1.0]
                - is_anomaly: bool (score > 0.50)
                - triggered_categories: list of matched hazard category IDs
                - category_scores: per-category match details
                - note: reminder that this baseline ignores user context
        """
        triggered: List[str] = []
        category_scores: Dict[str, float] = {}
        combined_score = 0.0

        for cat_id, weight, patterns in self._compiled:
            matches = sum(1 for p in patterns if p.search(message))
            if matches:
                # Diminishing returns for multiple matches in same category
                cat_score = weight * min(1.0, 0.5 + 0.25 * matches)
                category_scores[cat_id] = round(cat_score, 3)
                triggered.append(cat_id)
                combined_score = max(combined_score, cat_score)

        # Additive boost from secondary category hits
        if len(triggered) > 1:
            combined_score = min(1.0, combined_score + 0.10 * (len(triggered) - 1))

        # Amplify for high-risk destructive operations
        if self.amplify_high_risk_ops:
            op_matches = sum(
                1 for p in self._high_risk_ops_compiled if p.search(message)
            )
            if op_matches:
                combined_score = min(1.0, combined_score + 0.15 * op_matches)
                if "S2_account_abuse" not in triggered:
                    triggered.append("high_risk_operation")
                    category_scores["high_risk_operation"] = round(0.15 * op_matches, 3)

        final_score = min(1.0, combined_score)

        return {
            "anomaly_score": round(final_score, 3),
            "is_anomaly": final_score > 0.50,
            "triggered_categories": triggered,
            "category_scores": category_scores,
            "num_categories_triggered": len(triggered),
            "note": (
                "Content-only classifier — ignores user behavioral context. "
                "Cannot detect account takeover or social-engineering that "
                "uses individually benign-looking requests."
            ),
        }

    def batch_detect(self, messages: List[str]) -> List[Dict]:
        """
        Classify a batch of messages.

        Args:
            messages: List of message texts.

        Returns:
            List of result dicts, one per message.
        """
        return [self.detect(msg) for msg in messages]
