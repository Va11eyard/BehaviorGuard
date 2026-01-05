"""Policy decision engine for BehaviorGuard evaluations."""

from behaviorguard.models import PolicyAction, RequestedOperation, RiskLevel


class PolicyDecisionEngine:
    """Determines recommended security action based on risk and context."""

    def determine_action(
        self,
        risk_level: RiskLevel,
        anomaly_score: float,
        operation: RequestedOperation,
        deployment_context: str,
    ) -> PolicyAction:
        """
        Determine recommended security action.

        Args:
            risk_level: Classified risk level
            anomaly_score: Composite anomaly score
            operation: Requested operation details
            deployment_context: Deployment context (consumer, enterprise, etc.)

        Returns:
            PolicyAction recommendation
        """
        # ESCALATE_TO_HUMAN: High-risk in sensitive contexts
        if self._should_escalate_to_human(
            risk_level, anomaly_score, operation, deployment_context
        ):
            return PolicyAction.ESCALATE_TO_HUMAN

        # BLOCK_AND_VERIFY_OOB: High-risk operations
        if risk_level == RiskLevel.HIGH_RISK and anomaly_score > 0.6:
            return PolicyAction.BLOCK_AND_VERIFY_OOB

        # ALLOW_WITH_CAUTION: Suspicious behavior
        if risk_level == RiskLevel.SUSPICIOUS and 0.25 <= anomaly_score <= 0.45:
            return PolicyAction.ALLOW_WITH_CAUTION

        # ALLOW_NORMAL: Normal behavior
        if risk_level == RiskLevel.NORMAL and anomaly_score < 0.2:
            return PolicyAction.ALLOW_NORMAL

        # Default to ALLOW_WITH_CAUTION for edge cases
        if risk_level == RiskLevel.SUSPICIOUS:
            return PolicyAction.ALLOW_WITH_CAUTION

        # Default to BLOCK for HIGH_RISK
        if risk_level == RiskLevel.HIGH_RISK:
            return PolicyAction.BLOCK_AND_VERIFY_OOB

        return PolicyAction.ALLOW_NORMAL

    def _should_escalate_to_human(
        self,
        risk_level: RiskLevel,
        anomaly_score: float,
        operation: RequestedOperation,
        deployment_context: str,
    ) -> bool:
        """Check if situation requires human escalation."""
        # Only escalate in sensitive deployment contexts
        sensitive_contexts = ["enterprise", "financial", "healthcare", "government"]
        if deployment_context not in sensitive_contexts:
            return False

        # Must be HIGH_RISK with high score
        if risk_level != RiskLevel.HIGH_RISK or anomaly_score <= 0.75:
            return False

        # Check for sensitive operations
        sensitive_operations = ["admin", "permission_change", "auth_change", "export", "financial"]
        if operation.type in sensitive_operations:
            return True

        # Check for bulk operations or high-value targets
        if operation.targets and len(operation.targets) > 1000:
            return True

        return False
