"""Risk classification for BehaviorGuard evaluations."""

from behaviorguard.models import RequestedOperation, RiskLevel


class RiskClassifier:
    """Classifies anomaly scores into risk levels with operation-based escalation."""

    def classify(
        self,
        anomaly_score: float,
        operation: RequestedOperation,
        normal_threshold: float = 0.25,
        suspicious_threshold: float = 0.60,
    ) -> RiskLevel:
        """
        Classify anomaly score into risk levels.

        Args:
            anomaly_score: Composite anomaly score (0.0-1.0)
            operation: Requested operation details
            normal_threshold: Threshold for NORMAL classification
            suspicious_threshold: Threshold for SUSPICIOUS classification

        Returns:
            RiskLevel classification
        """
        # Base classification
        if anomaly_score < normal_threshold:
            base_risk = RiskLevel.NORMAL
        elif anomaly_score < suspicious_threshold:
            base_risk = RiskLevel.SUSPICIOUS
        else:
            base_risk = RiskLevel.HIGH_RISK

        # Apply operation-based escalation
        return self._apply_escalation(base_risk, anomaly_score, operation)

    def _apply_escalation(
        self, base_risk: RiskLevel, anomaly_score: float, operation: RequestedOperation
    ) -> RiskLevel:
        """Apply operation-based risk escalation rules."""
        risk_class = operation.risk_classification

        # Rule 1: Critical operation + SUSPICIOUS → HIGH_RISK
        if risk_class == "critical" and base_risk == RiskLevel.SUSPICIOUS:
            return RiskLevel.HIGH_RISK

        # Rule 2: Critical operation + NORMAL (score >0.15) → SUSPICIOUS
        if risk_class == "critical" and base_risk == RiskLevel.NORMAL and anomaly_score > 0.15:
            return RiskLevel.SUSPICIOUS

        # Rule 3: High operation + SUSPICIOUS (score >0.45) → HIGH_RISK
        if risk_class == "high" and base_risk == RiskLevel.SUSPICIOUS and anomaly_score > 0.45:
            return RiskLevel.HIGH_RISK

        return base_risk
