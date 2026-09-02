"""Main evaluation orchestrator for TurnShift system."""

from datetime import datetime

from turnshift.analyzers.linguistic import LinguisticAnalyzer
from turnshift.analyzers.semantic import SemanticAnalyzer
from turnshift.analyzers.temporal import TemporalAnalyzer
from turnshift.detectors.mitigating_factors import MitigatingFactorDetector
from turnshift.detectors.red_flags import RedFlagDetector
from turnshift.models import (
    ComponentScores,
    EvaluationInput,
    EvaluationResult,
)
from turnshift.scorers.composite import CompositeScorer
from turnshift.utils.cold_start import ColdStartHandler
from turnshift.utils.confidence import ConfidenceAssessor
from turnshift.utils.monitoring import MonitoringRecommendationGenerator
from turnshift.utils.output_formatter import OutputFormatter
from turnshift.utils.policy_engine import PolicyDecisionEngine
from turnshift.utils.rationale import RationaleGenerator
from turnshift.utils.risk_classifier import RiskClassifier


class TurnShiftEvaluator:
    """Main orchestrator for behavioral anomaly evaluation."""

    def __init__(self):
        """Initialize all components."""
        # Analyzers
        self.semantic_analyzer = SemanticAnalyzer()
        self.linguistic_analyzer = LinguisticAnalyzer()
        self.temporal_analyzer = TemporalAnalyzer()

        # Scorers
        self.composite_scorer = CompositeScorer()

        # Classifiers and decision engines
        self.confidence_assessor = ConfidenceAssessor()
        self.risk_classifier = RiskClassifier()
        self.policy_engine = PolicyDecisionEngine()

        # Detectors
        self.red_flag_detector = RedFlagDetector()
        self.mitigating_factor_detector = MitigatingFactorDetector()

        # Generators
        self.rationale_generator = RationaleGenerator()
        self.monitoring_generator = MonitoringRecommendationGenerator()

        # Utilities
        self.cold_start_handler = ColdStartHandler()
        self.output_formatter = OutputFormatter()

    def evaluate(self, evaluation_input: EvaluationInput) -> EvaluationResult:
        """
        Perform complete behavioral anomaly evaluation.

        Args:
            evaluation_input: Complete input with user profile, message, and config

        Returns:
            EvaluationResult with comprehensive analysis
        """
        user_profile = evaluation_input.user_profile
        current_message = evaluation_input.current_message
        system_config = evaluation_input.system_config

        # Check for cold start scenario
        is_cold_start = self.cold_start_handler.is_cold_start(user_profile)
        metadata = {}

        if is_cold_start:
            metadata["cold_start"] = True
            metadata["note"] = self.cold_start_handler.get_cold_start_note()

        # Step 1: Component Analysis
        semantic_result = self.semantic_analyzer.analyze(
            current_message, user_profile.semantic_profile
        )
        linguistic_result = self.linguistic_analyzer.analyze(
            current_message, user_profile.linguistic_profile
        )
        temporal_result = self.temporal_analyzer.analyze(
            current_message, user_profile.temporal_profile
        )

        component_scores = ComponentScores(
            semantic=semantic_result.score if system_config.enable_semantic_scoring else 0.0,
            linguistic=linguistic_result.score if system_config.enable_linguistic_scoring else 0.0,
            temporal=temporal_result.score if system_config.enable_temporal_scoring else 0.0,
        )

        # Step 2: Composite Scoring
        composite_result = self.composite_scorer.compute_score(
            component_scores, system_config, current_message, user_profile
        )
        anomaly_score = composite_result.anomaly_score

        # Step 3: Confidence Assessment
        if is_cold_start:
            confidence_assessment = self.confidence_assessor.assess(user_profile)
            confidence_assessment.level = self.cold_start_handler.get_cold_start_confidence()
            normal_threshold, suspicious_threshold = (
                self.cold_start_handler.get_cold_start_thresholds()
            )
        else:
            confidence_assessment = self.confidence_assessor.assess(user_profile)
            normal_threshold, suspicious_threshold = (
                self.confidence_assessor.adjust_thresholds(0.25, 0.60, confidence_assessment.level)
            )

        # Step 4: Risk Classification
        risk_level = self.risk_classifier.classify(
            anomaly_score, current_message.requested_operation, normal_threshold, suspicious_threshold
        )

        # Step 5: Policy Decision
        recommended_action = self.policy_engine.determine_action(
            risk_level,
            anomaly_score,
            current_message.requested_operation,
            system_config.deployment_context,
        )

        # Step 6: Red Flag Detection
        red_flags = self.red_flag_detector.detect(current_message, user_profile, component_scores)

        # Step 7: Mitigating Factor Detection
        mitigating_factors = self.mitigating_factor_detector.detect(current_message, user_profile)

        # Step 8: Rationale Generation
        component_reasonings = {
            "semantic": semantic_result.reasoning,
            "linguistic": linguistic_result.reasoning,
            "temporal": temporal_result.reasoning,
        }
        rationale = self.rationale_generator.generate(
            component_scores,
            component_reasonings,
            current_message.requested_operation,
            red_flags,
            mitigating_factors,
        )

        # Step 9: Monitoring Recommendations
        monitoring_recommendations = self.monitoring_generator.generate(
            risk_level, anomaly_score, red_flags
        )

        # Step 10: Output Formatting
        evaluation_result = self.output_formatter.format(
            user_id=user_profile.user_id,
            anomaly_score=anomaly_score,
            component_scores=component_scores,
            risk_level=risk_level,
            recommended_action=recommended_action,
            confidence=confidence_assessment.level,
            rationale=rationale,
            red_flags=red_flags,
            mitigating_factors=mitigating_factors,
            monitoring_recommendations=monitoring_recommendations,
            metadata=metadata,
        )

        return evaluation_result
