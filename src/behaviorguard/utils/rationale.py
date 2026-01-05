"""Rationale generation for BehaviorGuard evaluations."""

from typing import Dict, List

from behaviorguard.models import ComponentScores, Rationale, RequestedOperation


class RationaleGenerator:
    """Generates comprehensive rationale for evaluation decisions."""

    def generate(
        self,
        component_scores: ComponentScores,
        component_reasonings: Dict[str, str],
        operation: RequestedOperation,
        red_flags: List[str],
        mitigating_factors: List[str],
    ) -> Rationale:
        """
        Generate comprehensive rationale.

        Args:
            component_scores: Component anomaly scores
            component_reasonings: Reasoning from each component
            operation: Requested operation details
            red_flags: List of red flags detected
            mitigating_factors: List of mitigating factors

        Returns:
            Rationale with detailed reasoning
        """
        # Select 1-3 primary factors
        primary_factors = self._select_primary_factors(
            component_scores, red_flags, mitigating_factors
        )

        # Generate component reasoning (1-2 sentences each)
        semantic_reasoning = self._format_reasoning(
            component_reasonings.get("semantic", "No semantic anomalies detected."), 2
        )
        linguistic_reasoning = self._format_reasoning(
            component_reasonings.get("linguistic", "Linguistic patterns are normal."), 2
        )
        temporal_reasoning = self._format_reasoning(
            component_reasonings.get("temporal", "Temporal patterns are normal."), 2
        )

        # Generate operation risk assessment (1 sentence)
        operation_risk_assessment = self._assess_operation_risk(operation)

        # Generate overall summary (2-3 sentences)
        overall_summary = self._generate_summary(
            component_scores, operation, red_flags, mitigating_factors
        )

        return Rationale(
            primary_factors=primary_factors,
            semantic_reasoning=semantic_reasoning,
            linguistic_reasoning=linguistic_reasoning,
            temporal_reasoning=temporal_reasoning,
            operation_risk_assessment=operation_risk_assessment,
            overall_summary=overall_summary,
        )

    def _select_primary_factors(
        self,
        component_scores: ComponentScores,
        red_flags: List[str],
        mitigating_factors: List[str],
    ) -> List[str]:
        """Select 1-3 most important factors."""
        factors = []

        # Add highest scoring components
        scores = [
            ("Semantic anomaly", component_scores.semantic),
            ("Linguistic anomaly", component_scores.linguistic),
            ("Temporal anomaly", component_scores.temporal),
        ]
        scores.sort(key=lambda x: x[1], reverse=True)

        # Add top 2 components if score > 0.3
        for name, score in scores[:2]:
            if score > 0.3:
                factors.append(f"{name} (score: {score:.2f})")

        # Add most critical red flag if present
        if red_flags:
            factors.append(red_flags[0])

        # Limit to 3 factors
        return factors[:3]

    def _format_reasoning(self, reasoning: str, max_sentences: int) -> str:
        """Format reasoning to specified sentence count."""
        # Split into sentences
        sentences = [s.strip() for s in reasoning.split(".") if s.strip()]

        # Take first max_sentences
        formatted = ". ".join(sentences[:max_sentences])
        if formatted and not formatted.endswith("."):
            formatted += "."

        return formatted

    def _assess_operation_risk(self, operation: RequestedOperation) -> str:
        """Generate operation risk assessment (1 sentence)."""
        risk_class = operation.risk_classification
        op_type = operation.type

        if risk_class == "critical":
            return f"Operation is critical ({op_type}) and requires heightened scrutiny."
        elif risk_class == "high":
            return f"Operation is high-risk ({op_type}) and warrants careful monitoring."
        elif risk_class == "medium":
            return f"Operation is medium-risk ({op_type}) with standard security controls."
        else:
            return f"Operation is low-risk ({op_type}) with minimal security concerns."

    def _generate_summary(
        self,
        component_scores: ComponentScores,
        operation: RequestedOperation,
        red_flags: List[str],
        mitigating_factors: List[str],
    ) -> str:
        """Generate overall summary (2-3 sentences)."""
        # Calculate average score
        avg_score = (
            component_scores.semantic + component_scores.linguistic + component_scores.temporal
        ) / 3.0

        # First sentence: Overall assessment
        if avg_score > 0.7:
            summary = "Evaluation indicates high anomaly levels across multiple dimensions. "
        elif avg_score > 0.4:
            summary = "Evaluation shows moderate anomalies requiring attention. "
        else:
            summary = "Evaluation shows minimal anomalies with normal behavioral patterns. "

        # Second sentence: Red flags or mitigating factors
        if red_flags:
            summary += f"Detected {len(red_flags)} red flag(s) including suspicious indicators. "
        elif mitigating_factors:
            summary += f"Identified {len(mitigating_factors)} mitigating factor(s) suggesting legitimate use. "

        # Third sentence: Recommendation context
        if operation.risk_classification in ["critical", "high"]:
            summary += "Given the sensitive nature of the operation, enhanced security measures are recommended."
        else:
            summary += "Standard security protocols are sufficient for this operation."

        return summary
