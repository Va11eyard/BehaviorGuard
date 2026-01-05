"""Monitoring recommendation generation for BehaviorGuard evaluations."""

from typing import List

from behaviorguard.models import MonitoringRecommendations, RiskLevel


class MonitoringRecommendationGenerator:
    """Generates monitoring recommendations for suspicious activity."""

    def generate(
        self, risk_level: RiskLevel, anomaly_score: float, red_flags: List[str]
    ) -> MonitoringRecommendations:
        """
        Generate monitoring recommendations.

        Args:
            risk_level: Classified risk level
            anomaly_score: Composite anomaly score
            red_flags: List of detected red flags

        Returns:
            MonitoringRecommendations with escalation conditions and watch patterns
        """
        escalate_if = []
        watch_for = []
        auto_clear_after = "manual"

        if risk_level == RiskLevel.HIGH_RISK:
            # High risk - manual review required
            escalate_if = [
                "Any additional suspicious activity",
                "Attempt to access sensitive resources",
                "Pattern repetition within 24 hours",
            ]
            watch_for = [
                "Similar anomaly patterns",
                "Escalation of privileges",
                "Data exfiltration attempts",
                "Coordination with other accounts",
            ]
            auto_clear_after = "manual"

        elif risk_level == RiskLevel.SUSPICIOUS:
            # Suspicious - monitor for pattern continuation
            escalate_if = [
                f"Next message anomaly score > {min(0.4, anomaly_score + 0.1):.2f}",
                "Critical operation attempted",
                "Multiple suspicious messages within 1 hour",
            ]
            watch_for = [
                "Topic deviation continuation",
                "Linguistic pattern changes",
                "Unusual timing patterns",
                "Sensitive operation requests",
            ]

            # Auto-clear based on score
            if anomaly_score < 0.35:
                auto_clear_after = "24 hours"
            else:
                auto_clear_after = "48 hours"

        else:
            # Normal - minimal monitoring
            escalate_if = ["Anomaly score exceeds 0.4", "Critical operation requested"]
            watch_for = ["Behavioral pattern changes"]
            auto_clear_after = "24 hours"

        # Add red flag specific monitoring
        if red_flags:
            for flag in red_flags[:2]:  # Top 2 red flags
                if "prompt injection" in flag.lower():
                    watch_for.append("Additional injection attempts")
                elif "bot" in flag.lower():
                    watch_for.append("Automated behavior patterns")
                elif "social engineering" in flag.lower():
                    watch_for.append("Escalation tactics")

        return MonitoringRecommendations(
            escalate_if=escalate_if, watch_for=watch_for, auto_clear_after=auto_clear_after
        )
