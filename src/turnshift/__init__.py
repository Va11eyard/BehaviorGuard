"""TurnShift: AI Security Agent for Behavioral Anomaly Detection."""

from turnshift.evaluator import TurnShiftEvaluator

# Try to import ML evaluator (requires sentence-transformers)
try:
    from turnshift.evaluator_ml import TurnShiftEvaluatorML
    ML_AVAILABLE = True
except ImportError:
    TurnShiftEvaluatorML = None  # type: ignore[assignment]
    ML_AVAILABLE = False

from turnshift.models import (
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
from turnshift.validator import InputValidator
from turnshift.profile_manager import ProfileManager, MessageRecord
from turnshift.utils.profile_store import ProfileStore

__version__ = "1.0.0"

__all__ = [
    "TurnShiftEvaluator",
    "TurnShiftEvaluatorML",
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
