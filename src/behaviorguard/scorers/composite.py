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
        overrides_enabled = getattr(system_config, "overrides_enabled", True)

        if overrides_enabled:
            # Check for instant HIGH_RISK overrides first
            override_id, override_reason = self._check_high_risk_overrides(
                component_scores, current_message, user_profile
            )
            if override_id:
                applied_overrides.append(override_reason)
                return CompositeScore(
                    anomaly_score=1.0,
                    applied_overrides=applied_overrides,
                    detection_mechanism=override_id,
                )

            # Check for instant NORMAL overrides
            normal_override = self._check_normal_overrides(
                component_scores, current_message, system_config
            )
            if normal_override:
                applied_overrides.append(normal_override)
                base_score = self._calculate_weighted_score(
                    component_scores, system_config.sensitivity_level
                )
                return CompositeScore(
                    anomaly_score=min(0.15, base_score),
                    applied_overrides=applied_overrides,
                    detection_mechanism="normal_override",
                )

        # Calculate weighted composite score (or when overrides disabled)
        anomaly_score = self._calculate_weighted_score(
            component_scores, system_config.sensitivity_level
        )

        anomaly_score = max(0.0, min(1.0, anomaly_score))

        return CompositeScore(
            anomaly_score=anomaly_score,
            applied_overrides=applied_overrides,
            detection_mechanism="composite_score",
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
    ) -> tuple:
        """
        Check for conditions that trigger instant HIGH_RISK override.

        Returns:
            (override_id, reason_str) if triggered, ("", "") otherwise.
            override_id: "override_1".."override_4" for attribution.
        """
        # Override 1: Semantic score >0.85 AND operation is critical
        if (
            component_scores.semantic > 0.85
            and current_message.requested_operation.risk_classification == "critical"
        ):
            return ("override_1", "Extreme semantic deviation with critical operation")

        # Override 2: Temporal score >0.9 (bot/automation indicator)
        if component_scores.temporal > 0.9:
            return ("override_2", "Bot-like timing pattern detected")

        # Override 3: No history of sensitive operations BUT current request is critical
        if (
            not user_profile.operational_profile.has_requested_sensitive_ops
            and current_message.requested_operation.risk_classification == "critical"
        ):
            return ("override_3", "Critical operation without precedent")

        # Override 4: Message contains explicit ATO indicators (check message text)
        if self._contains_ato_indicators(current_message.text):
            return ("override_4", "Explicit account takeover indicators detected")

        return ("", "")

    def _check_normal_overrides(
        self,
        component_scores: ComponentScores,
        current_message: CurrentMessage,
        system_config: SystemConfig,
    ) -> str:
        """
        Check for conditions that trigger instant NORMAL override.

        Returns:
            Override reason string if triggered, empty string otherwise
        """
        # Average over enabled components only. Including a zeroed (disabled)
        # component would artificially suppress the mean and fire the <0.15
        # override on genuine anomalies during component ablations.
        enabled_scores = [
            s
            for s, enabled in (
                (component_scores.semantic, system_config.enable_semantic_scoring),
                (component_scores.linguistic, system_config.enable_linguistic_scoring),
                (component_scores.temporal, system_config.enable_temporal_scoring),
            )
            if enabled
        ]
        base_score = sum(enabled_scores) / len(enabled_scores) if enabled_scores else 1.0

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
