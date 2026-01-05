"""Cold start handling for BehaviorGuard evaluations."""

from behaviorguard.models import ConfidenceLevel, UserProfile


class ColdStartHandler:
    """Handles cold start scenarios for new users with insufficient history."""

    def is_cold_start(self, user_profile: UserProfile) -> bool:
        """
        Check if user is in cold start scenario.

        Args:
            user_profile: User's behavioral profile

        Returns:
            True if user has insufficient history (<20 interactions)
        """
        return user_profile.total_interactions < 20

    def get_cold_start_thresholds(self) -> tuple[float, float]:
        """
        Get adjusted thresholds for cold start scenarios.

        Returns:
            Tuple of (normal_threshold, suspicious_threshold)
            Thresholds are increased by 0.1 to be less aggressive
        """
        # Base thresholds: 0.25, 0.60
        # Cold start: 0.35, 0.70
        return (0.35, 0.70)

    def get_cold_start_confidence(self) -> ConfidenceLevel:
        """
        Get confidence level for cold start scenarios.

        Returns:
            ConfidenceLevel.LOW for cold start users
        """
        return ConfidenceLevel.LOW

    def should_flag_only_extreme(self, anomaly_score: float) -> bool:
        """
        Check if score should be flagged in cold start scenario.

        In cold start, only flag violations >0.85 (extreme anomalies).

        Args:
            anomaly_score: Composite anomaly score

        Returns:
            True if score should be flagged
        """
        return anomaly_score > 0.85

    def get_cold_start_note(self) -> str:
        """
        Get note to include in output for cold start scenarios.

        Returns:
            Note explaining insufficient history
        """
        return "Insufficient user history (<20 interactions). Using generic heuristics with conservative thresholds."
