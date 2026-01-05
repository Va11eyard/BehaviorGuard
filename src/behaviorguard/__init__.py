"""BehaviorGuard: AI Security Agent for Behavioral Anomaly Detection."""

from behaviorguard.evaluator import BehaviorGuardEvaluator

# Try to import ML evaluator (requires sentence-transformers)
try:
    from behaviorguard.evaluator_ml import BehaviorGuardEvaluatorML
    ML_AVAILABLE = True
except ImportError:
    BehaviorGuardEvaluatorML = None
    ML_AVAILABLE = False

from behaviorguard.models import (
    ComponentScores,
    ConfidenceLevel,
    CurrentMessage,
    EvaluationInput,
    EvaluationResult,
    PolicyAction,
    RiskLevel,
    SystemConfig,
    UserProfile,
)
from behaviorguard.validator import InputValidator

__version__ = "1.0.0"

__all__ = [
    "BehaviorGuardEvaluator",
    "BehaviorGuardEvaluatorML",
    "ML_AVAILABLE",
    "InputValidator",
    "EvaluationInput",
    "EvaluationResult",
    "UserProfile",
    "CurrentMessage",
    "SystemConfig",
    "ComponentScores",
    "RiskLevel",
    "PolicyAction",
    "ConfidenceLevel",
]
