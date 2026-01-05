"""Output formatting for BehaviorGuard evaluations."""

import json
import uuid
from datetime import datetime

from behaviorguard.models import (
    ComponentScores,
    ConfidenceLevel,
    EvaluationResult,
    MonitoringRecommendations,
    PolicyAction,
    Rationale,
    RiskLevel,
)


class OutputFormatter:
    """Formats comprehensive evaluation results as structured JSON."""

    def format(
        self,
        user_id: str,
        anomaly_score: float,
        component_scores: ComponentScores,
        risk_level: RiskLevel,
        recommended_action: PolicyAction,
        confidence: ConfidenceLevel,
        rationale: Rationale,
        red_flags: list[str],
        mitigating_factors: list[str],
        monitoring_recommendations: MonitoringRecommendations,
        metadata: dict = None,
    ) -> EvaluationResult:
        """
        Format evaluation results.

        Args:
            user_id: User identifier
            anomaly_score: Composite anomaly score
            component_scores: Component scores
            risk_level: Risk classification
            recommended_action: Policy action recommendation
            confidence: Confidence level
            rationale: Detailed rationale
            red_flags: List of red flags
            mitigating_factors: List of mitigating factors
            monitoring_recommendations: Monitoring recommendations
            metadata: Additional metadata

        Returns:
            EvaluationResult with all formatted data
        """
        # Generate unique evaluation ID
        evaluation_id = str(uuid.uuid4())

        # Get current timestamp
        timestamp = datetime.utcnow().isoformat() + "Z"

        # Round scores to 3 decimal places
        anomaly_score = round(anomaly_score, 3)
        component_scores = ComponentScores(
            semantic=round(component_scores.semantic, 3),
            linguistic=round(component_scores.linguistic, 3),
            temporal=round(component_scores.temporal, 3),
        )

        # Prepare metadata
        if metadata is None:
            metadata = {}
        metadata["evaluation_id"] = evaluation_id

        return EvaluationResult(
            analysis_version="1.0",
            timestamp=timestamp,
            user_id=user_id,
            anomaly_score=anomaly_score,
            component_scores=component_scores,
            risk_level=risk_level,
            recommended_action=recommended_action,
            confidence=confidence,
            rationale=rationale,
            red_flags=red_flags,
            mitigating_factors=mitigating_factors,
            monitoring_recommendations=monitoring_recommendations,
            metadata=metadata,
        )

    def to_json(self, evaluation_result: EvaluationResult) -> str:
        """
        Convert evaluation result to JSON string.

        Args:
            evaluation_result: Evaluation result to format

        Returns:
            JSON string representation
        """
        return evaluation_result.model_dump_json(indent=2)
