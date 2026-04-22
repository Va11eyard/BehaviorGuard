"""BehaviorGuard: AI Security Agent for Behavioral Anomaly Detection."""

from behaviorguard.evaluator import BehaviorGuardEvaluator

# Try to import ML evaluator (requires sentence-transformers)
try:
    from behaviorguard.evaluator_ml import BehaviorGuardEvaluatorML
    ML_AVAILABLE = True
except ImportError:
    BehaviorGuardEvaluatorML = None  # type: ignore[assignment]
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
from behaviorguard.profile_manager import ProfileManager, MessageRecord
from behaviorguard.utils.profile_store import ProfileStore

__version__ = "1.0.0"

__all__ = [
    "BehaviorGuardEvaluator",
    "BehaviorGuardEvaluatorML",
    "ML_AVAILABLE",
    "InputValidator",
    "ProfileManager",
    "MessageRecord",
    "ProfileStore",
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
