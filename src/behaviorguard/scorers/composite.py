"""Composite scorer for combining component scores with weights and overrides."""

from typing import List

from behaviorguard.models import (
    ComponentScores,
    CompositeScore,
    CurrentMessage,
    SystemConfig,
    UserProfile,
)


class CompositeScorer:
    """Combines component scores using configured weights and applies override conditions."""

    # Weight configurations by sensitivity level
    WEIGHTS = {
        "low": {"semantic": 0.5, "linguistic": 0.3, "temporal": 0.2},
        "medium": {"semantic": 0.4, "linguistic": 0.35, "temporal": 0.25},
        "high": {"semantic": 0.4, "linguistic": 0.3, "temporal": 0.3},
        "maximum": {"semantic": 0.35, "linguistic": 0.35, "temporal": 0.3},
    }

    def compute_score(
        self,
        component_scores: ComponentScores,
        system_config: SystemConfig,
        current_message: CurrentMessage,
        user_profile: UserProfile,
    ) -> CompositeScore:
        """
        Combine component scores with weights and apply override conditions.

        Args:
            component_scores: Semantic, linguistic, and temporal scores
            system_config: System configuration with sensitivity level
            current_message: Current message being evaluated
            user_profile: User's behavioral profile

        Returns:
            CompositeScore with anomaly score and applied overrides
        """
        applied_overrides = []

        # Check for instant HIGH_RISK overrides first
        high_risk_override = self._check_high_risk_overrides(
            component_scores, current_message, user_profile
        )
        if high_risk_override:
            applied_overrides.append(high_risk_override)
            return CompositeScore(anomaly_score=1.0, applied_overrides=applied_overrides)

        # Check for instant NORMAL overrides
        normal_override = self._check_normal_overrides(
            component_scores, current_message
        )
        if normal_override:
            applied_overrides.append(normal_override)
            # Calculate base score but cap it low
            base_score = self._calculate_weighted_score(
                component_scores, system_config.sensitivity_level
            )
            return CompositeScore(
                anomaly_score=min(0.15, base_score), applied_overrides=applied_overrides
            )

        # Calculate weighted composite score
        anomaly_score = self._calculate_weighted_score(
            component_scores, system_config.sensitivity_level
        )

        # Ensure score is bounded
        anomaly_score = max(0.0, min(1.0, anomaly_score))

        return CompositeScore(
            anomaly_score=anomaly_score, applied_overrides=applied_overrides
        )

    def _calculate_weighted_score(
        self, component_scores: ComponentScores, sensitivity_level: str
    ) -> float:
        """Calculate weighted combination of component scores."""
        weights = self.WEIGHTS[sensitivity_level]

        score = (
            weights["semantic"] * component_scores.semantic
            + weights["linguistic"] * component_scores.linguistic
            + weights["temporal"] * component_scores.temporal
        )

        return score

    def _check_high_risk_overrides(
        self,
        component_scores: ComponentScores,
        current_message: CurrentMessage,
        user_profile: UserProfile,
    ) -> str:
        """
        Check for conditions that trigger instant HIGH_RISK override.

        Returns:
            Override reason string if triggered, empty string otherwise
        """
        # Override 1: Semantic score >0.85 AND operation is critical
        if (
            component_scores.semantic > 0.85
            and current_message.requested_operation.risk_classification == "critical"
        ):
            return "Extreme semantic deviation with critical operation"

        # Override 2: Temporal score >0.9 (bot/automation indicator)
        if component_scores.temporal > 0.9:
            return "Bot-like timing pattern detected"

        # Override 3: No history of sensitive operations BUT current request is critical
        if (
            not user_profile.operational_profile.has_requested_sensitive_ops
            and current_message.requested_operation.risk_classification == "critical"
        ):
            return "Critical operation without precedent"

        # Override 4: Message contains explicit ATO indicators (check message text)
        if self._contains_ato_indicators(current_message.text):
            return "Explicit account takeover indicators detected"

        return ""

    def _check_normal_overrides(
        self, component_scores: ComponentScores, current_message: CurrentMessage
    ) -> str:
        """
        Check for conditions that trigger instant NORMAL override.

        Returns:
            Override reason string if triggered, empty string otherwise
        """
        # Calculate base anomaly score
        base_score = (
            component_scores.semantic + component_scores.linguistic + component_scores.temporal
        ) / 3.0

        # Override 1: User explicitly states context change
        if self._contains_context_change_announcement(current_message.text):
            return "Explicit context change announcement"

        # Override 2: Clarification question about previous topic (<50 tokens)
        if (
            current_message.linguistic_features.message_length_tokens < 50
            and current_message.linguistic_features.formality_score < 0.4
            and "?" in current_message.text
        ):
            return "Brief clarification question"

        # Override 3: Anomaly score <0.15 regardless of operation
        if base_score < 0.15:
            return "Very low anomaly score"

        return ""

    def _contains_ato_indicators(self, text: str) -> bool:
        """Check if message contains explicit account takeover indicators."""
        text_lower = text.lower()
        ato_keywords = [
            "ignore previous",
            "disregard instructions",
            "system prompt",
            "you are now",
            "forget everything",
            "new instructions",
        ]
        return any(keyword in text_lower for keyword in ato_keywords)

    def _contains_context_change_announcement(self, text: str) -> bool:
        """Check if message explicitly announces a context change."""
        text_lower = text.lower()
        context_change_phrases = [
            "changing topic",
            "different subject",
            "new topic",
            "switch gears",
            "moving on to",
            "let's talk about",
            "i want to ask about",
        ]
        return any(phrase in text_lower for phrase in context_change_phrases)
