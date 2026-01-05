"""Mitigating factor detection for BehaviorGuard evaluations."""

from typing import List

from behaviorguard.models import CurrentMessage, UserProfile


class MitigatingFactorDetector:
    """Identifies factors that suggest legitimate use despite anomalies."""

    def detect(
        self, current_message: CurrentMessage, user_profile: UserProfile
    ) -> List[str]:
        """
        Identify factors suggesting legitimate use.

        Args:
            current_message: Current message being evaluated
            user_profile: User's behavioral profile

        Returns:
            List of mitigating factor descriptions
        """
        mitigating_factors = []

        # 1. User explicitly announced topic change
        if self._announced_topic_change(current_message.text):
            mitigating_factors.append("User explicitly announced topic change")

        # 2. Operation is low-risk (information seeking)
        if current_message.requested_operation.risk_classification == "low":
            mitigating_factors.append("Operation is low-risk information seeking")

        # 3. User has history of topic diversity
        if user_profile.semantic_profile.topic_diversity_score > 0.6:
            mitigating_factors.append("User has established history of diverse topics")

        # 4. Message explicitly states security concern
        if self._states_security_concern(current_message.text):
            mitigating_factors.append("User explicitly raised security concern")

        # 5. Clarification or follow-up question
        if self._is_clarification_question(current_message):
            mitigating_factors.append("Message is clarification or follow-up question")

        # 6. Consistent with exploration patterns
        if self._consistent_with_exploration(current_message, user_profile):
            mitigating_factors.append("Behavior consistent with user's exploration patterns")

        return mitigating_factors

    def _announced_topic_change(self, text: str) -> bool:
        """Check if user explicitly announced topic change."""
        text_lower = text.lower()
        announcement_phrases = [
            "changing topic",
            "different subject",
            "new topic",
            "switch gears",
            "moving on to",
            "let's talk about",
            "i want to ask about",
            "switching to",
            "now about",
        ]
        return any(phrase in text_lower for phrase in announcement_phrases)

    def _states_security_concern(self, text: str) -> bool:
        """Check if message explicitly states security concern."""
        text_lower = text.lower()
        security_phrases = [
            "security",
            "suspicious",
            "unusual",
            "concerned about",
            "worried about",
            "is this safe",
            "is this normal",
        ]
        return any(phrase in text_lower for phrase in security_phrases)

    def _is_clarification_question(self, message: CurrentMessage) -> bool:
        """Check if message is a clarification question."""
        # Short message with question mark
        is_short = message.linguistic_features.message_length_tokens < 30
        has_question = "?" in message.text

        # Common clarification patterns
        text_lower = message.text.lower()
        clarification_patterns = [
            "what do you mean",
            "can you explain",
            "could you clarify",
            "i don't understand",
            "what is",
            "how does",
            "why",
        ]
        has_clarification = any(pattern in text_lower for pattern in clarification_patterns)

        return (is_short and has_question) or has_clarification

    def _consistent_with_exploration(
        self, message: CurrentMessage, profile: UserProfile
    ) -> bool:
        """Check if behavior is consistent with user's exploration patterns."""
        # User has diverse topic history
        has_diverse_history = profile.semantic_profile.topic_diversity_score > 0.5

        # Message is exploratory (questions, learning)
        is_exploratory = (
            "?" in message.text
            or any(
                word in message.text.lower()
                for word in ["how", "what", "why", "learn", "understand", "explain"]
            )
        )

        # Not requesting sensitive operations
        is_safe_operation = message.requested_operation.risk_classification in ["low", "medium"]

        return has_diverse_history and is_exploratory and is_safe_operation
