"""Confidence assessment for BehaviorGuard evaluations."""

from behaviorguard.models import (
    ConfidenceAssessment,
    ConfidenceFactors,
    ConfidenceLevel,
    UserProfile,
)


class ConfidenceAssessor:
    """Assesses confidence level based on user history and adjusts thresholds."""

    def assess(self, user_profile: UserProfile) -> ConfidenceAssessment:
        """
        Assess confidence level based on user history.

        Args:
            user_profile: User's behavioral profile

        Returns:
            ConfidenceAssessment with level and contributing factors
        """
        interaction_count = user_profile.total_interactions

        # Determine confidence factors
        sufficient_history = interaction_count >= 20
        clear_patterns = self._has_clear_patterns(user_profile)
        high_signal_quality = self._has_high_signal_quality(user_profile)

        # Determine confidence level
        if interaction_count > 100 and clear_patterns and high_signal_quality:
            level = ConfidenceLevel.HIGH
        elif 20 <= interaction_count <= 100:
            level = ConfidenceLevel.MEDIUM
        else:
            level = ConfidenceLevel.LOW

        return ConfidenceAssessment(
            level=level,
            factors=ConfidenceFactors(
                sufficient_history=sufficient_history,
                clear_patterns=clear_patterns,
                high_signal_quality=high_signal_quality,
            ),
        )

    def adjust_thresholds(
        self, base_normal_threshold: float, base_suspicious_threshold: float, confidence: ConfidenceLevel
    ) -> tuple[float, float]:
        """
        Adjust thresholds based on confidence level.

        Args:
            base_normal_threshold: Base threshold for NORMAL classification (typically 0.25)
            base_suspicious_threshold: Base threshold for SUSPICIOUS classification (typically 0.60)
            confidence: Confidence level

        Returns:
            Tuple of (adjusted_normal_threshold, adjusted_suspicious_threshold)
        """
        if confidence == ConfidenceLevel.LOW:
            # Less aggressive - increase thresholds by 0.1
            return (base_normal_threshold + 0.1, base_suspicious_threshold + 0.1)
        elif confidence == ConfidenceLevel.HIGH:
            # More sensitive - decrease thresholds by 0.05
            return (base_normal_threshold - 0.05, base_suspicious_threshold - 0.05)
        else:
            # Medium - use standard thresholds
            return (base_normal_threshold, base_suspicious_threshold)

    def _has_clear_patterns(self, user_profile: UserProfile) -> bool:
        """Check if user has clear behavioral patterns."""
        # Check for consistent topics
        has_consistent_topics = len(user_profile.semantic_profile.typical_topics) >= 3

        # Check for stable linguistic patterns (low std deviation)
        has_stable_linguistic = (
            user_profile.linguistic_profile.lexical_diversity_std < 0.2
            and user_profile.linguistic_profile.formality_score_std < 0.2
        )

        # Check for regular temporal patterns
        has_regular_temporal = len(user_profile.temporal_profile.most_active_hours_utc) >= 3

        return has_consistent_topics and has_stable_linguistic and has_regular_temporal

    def _has_high_signal_quality(self, user_profile: UserProfile) -> bool:
        """Check if user profile has high signal quality."""
        # Check for diverse but consistent behavior
        has_topic_diversity = 0.3 <= user_profile.semantic_profile.topic_diversity_score <= 0.8

        # Check for reasonable message lengths
        has_reasonable_lengths = (
            10 <= user_profile.linguistic_profile.avg_message_length_tokens <= 500
        )

        # Check for regular activity
        has_regular_activity = user_profile.temporal_profile.typical_session_frequency_per_week >= 1

        return has_topic_diversity and has_reasonable_lengths and has_regular_activity
