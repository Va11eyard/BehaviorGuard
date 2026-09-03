"""TurnShift: AI Security Agent for Behavioral Anomaly Detection."""

# The ML evaluator imports without the optional stack; instantiating it needs
# sentence-transformers (pip install "turnshift[ml]"). ML_AVAILABLE reports
# whether that stack is installed, resolved without importing torch.
from turnshift.analyzers.semantic_ml import TRANSFORMERS_AVAILABLE as ML_AVAILABLE
from turnshift.evaluator import TurnShiftEvaluator
from turnshift.evaluator_ml import TurnShiftEvaluatorML
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
from turnshift.profile_manager import MessageRecord, ProfileManager
from turnshift.utils.profile_store import ProfileStore
from turnshift.validator import InputValidator

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
