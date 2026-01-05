"""Red flag detection for BehaviorGuard evaluations."""

from typing import List

from behaviorguard.models import ComponentScores, CurrentMessage, UserProfile


class RedFlagDetector:
    """Identifies specific suspicious indicators in messages."""

    def detect(
        self,
        current_message: CurrentMessage,
        user_profile: UserProfile,
        component_scores: ComponentScores,
    ) -> List[str]:
        """
        Identify specific suspicious indicators.

        Args:
            current_message: Current message being evaluated
            user_profile: User's behavioral profile
            component_scores: Component anomaly scores

        Returns:
            List of red flag descriptions
        """
        red_flags = []

        # 1. Extreme semantic deviation
        if component_scores.semantic > 0.85:
            red_flags.append("Extreme semantic deviation from typical topics")

        # 2. Critical operation without precedent
        if (
            current_message.requested_operation.risk_classification == "critical"
            and not user_profile.operational_profile.has_requested_sensitive_ops
        ):
            red_flags.append("Critical operation requested without historical precedent")

        # 3. Suspicious external destinations
        if self._has_suspicious_destinations(current_message):
            red_flags.append("Request involves suspicious external destinations")

        # 4. Meta-instructions (prompt injection)
        if self._contains_prompt_injection(current_message.text):
            red_flags.append("Message contains prompt injection attempt")

        # 5. Social engineering escalation patterns
        if self._shows_social_engineering(current_message, user_profile):
            red_flags.append("Social engineering escalation pattern detected")

        # 6. Bot-like timing patterns
        if component_scores.temporal > 0.9:
            red_flags.append("Bot-like timing pattern detected")

        # 7. APT-style reconnaissance
        if self._shows_apt_reconnaissance(current_message, user_profile):
            red_flags.append("APT-style reconnaissance behavior detected")

        # 8. Impossible velocity/location changes
        if current_message.temporal_context.time_since_last_session_hours < 1.0:
            red_flags.append("Impossible velocity or location change detected")

        # 9. Technical sophistication inconsistent with history
        if self._shows_inconsistent_sophistication(current_message, user_profile):
            red_flags.append("Technical sophistication inconsistent with user history")

        return red_flags

    def _has_suspicious_destinations(self, message: CurrentMessage) -> bool:
        """Check for suspicious external destinations."""
        text_lower = message.text.lower()
        suspicious_patterns = [
            "pastebin",
            "bit.ly",
            "tinyurl",
            "export to",
            "send to",
            "upload to",
            ".onion",
            "tor",
        ]
        return any(pattern in text_lower for pattern in suspicious_patterns)

    def _contains_prompt_injection(self, text: str) -> bool:
        """Check for prompt injection attempts."""
        text_lower = text.lower()
        injection_patterns = [
            "ignore previous",
            "disregard instructions",
            "system prompt",
            "you are now",
            "forget everything",
            "new instructions",
            "override",
            "bypass",
            "jailbreak",
        ]
        return any(pattern in text_lower for pattern in injection_patterns)

    def _shows_social_engineering(
        self, message: CurrentMessage, profile: UserProfile
    ) -> bool:
        """Check for social engineering patterns."""
        text_lower = message.text.lower()

        # Check for urgency + authority
        urgency_words = ["urgent", "immediately", "asap", "emergency", "critical"]
        authority_words = ["ceo", "manager", "director", "admin", "authorized"]

        has_urgency = any(word in text_lower for word in urgency_words)
        has_authority = any(word in text_lower for word in authority_words)

        # Overly polite with sensitive request
        is_overly_polite = message.linguistic_features.politeness_score > 0.8
        is_sensitive = message.requested_operation.risk_classification in ["high", "critical"]

        return (has_urgency and has_authority) or (is_overly_polite and is_sensitive)

    def _shows_apt_reconnaissance(
        self, message: CurrentMessage, profile: UserProfile
    ) -> bool:
        """Check for APT-style reconnaissance behavior."""
        text_lower = message.text.lower()

        recon_patterns = [
            "list all",
            "show all",
            "enumerate",
            "what users",
            "what permissions",
            "what access",
            "system information",
            "network topology",
        ]

        has_recon_pattern = any(pattern in text_lower for pattern in recon_patterns)

        # Combined with unusual timing or new behavior
        unusual_timing = not message.temporal_context.is_typical_active_time
        new_behavior = profile.total_interactions < 50

        return has_recon_pattern and (unusual_timing or new_behavior)

    def _shows_inconsistent_sophistication(
        self, message: CurrentMessage, profile: UserProfile
    ) -> bool:
        """Check for technical sophistication inconsistent with history."""
        # User not typically technical but message is highly technical
        not_technical = not profile.linguistic_profile.uses_technical_vocabulary

        is_technical = (
            message.linguistic_features.contains_code
            or len(message.text) > 200
            or any(
                term in message.text.lower()
                for term in ["api", "sql", "query", "database", "server", "admin"]
            )
        )

        return not_technical and is_technical
