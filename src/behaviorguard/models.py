"""Pydantic models for BehaviorGuard system."""

from enum import Enum
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


# Enums
class RiskLevel(str, Enum):
    """Risk level classification."""

    NORMAL = "NORMAL"
    SUSPICIOUS = "SUSPICIOUS"
    HIGH_RISK = "HIGH_RISK"


class PolicyAction(str, Enum):
    """Recommended policy action."""

    ALLOW_NORMAL = "ALLOW_NORMAL"
    ALLOW_WITH_CAUTION = "ALLOW_WITH_CAUTION"
    BLOCK_AND_VERIFY_OOB = "BLOCK_AND_VERIFY_OOB"
    ESCALATE_TO_HUMAN = "ESCALATE_TO_HUMAN"


class ConfidenceLevel(str, Enum):
    """Confidence level assessment."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# User Profile Models
class SemanticProfile(BaseModel):
    """Semantic profile of user behavior."""

    typical_topics: List[str]
    primary_domains: List[str]
    topic_diversity_score: float
    embedding_centroid_summary: str
    embedding_centroid: Optional[List[float]] = None  # Pre-computed EMA centroid when available


class LinguisticProfile(BaseModel):
    """Linguistic profile of user behavior."""

    avg_message_length_tokens: float
    avg_message_length_chars: float
    lexical_diversity_mean: float
    lexical_diversity_std: float
    formality_score_mean: float
    formality_score_std: float
    politeness_score_mean: float
    politeness_score_std: float
    question_ratio_mean: float
    uses_technical_vocabulary: bool
    uses_code_blocks: bool
    primary_languages: List[str]
    typical_sentence_complexity: Literal["simple", "moderate", "complex"]


class TemporalProfile(BaseModel):
    """Temporal profile of user behavior."""

    typical_session_duration_minutes: float
    typical_inter_message_gap_seconds: float
    most_active_hours_utc: List[int]
    most_active_days_of_week: List[str]
    average_messages_per_session: float
    longest_session_duration_minutes: float
    typical_session_frequency_per_week: float
    last_activity_timestamp: str


class OperationalProfile(BaseModel):
    """Operational profile of user behavior."""

    common_intent_types: List[str]
    tools_used_historically: List[str]
    has_requested_sensitive_ops: bool
    typical_risk_level: Literal["low", "medium", "high"]


class UserProfile(BaseModel):
    """Complete user behavioral profile."""

    user_id: str
    account_age_days: int
    total_interactions: int
    semantic_profile: SemanticProfile
    linguistic_profile: LinguisticProfile
    temporal_profile: TemporalProfile
    operational_profile: OperationalProfile


# Current Message Models
class RequestedOperation(BaseModel):
    """Operation requested in current message."""

    type: Literal[
        "read",
        "write",
        "delete",
        "export",
        "auth_change",
        "permission_change",
        "financial",
        "admin",
        "none",
    ]
    risk_classification: Literal["low", "medium", "high", "critical"]
    targets: Optional[List[str]] = None
    requires_auth: bool


class LinguisticFeatures(BaseModel):
    """Linguistic features of current message."""

    message_length_tokens: int
    message_length_chars: int
    lexical_diversity: float
    formality_score: float
    politeness_score: float
    contains_code: bool
    contains_urls: bool
    language: str


class TemporalContext(BaseModel):
    """Temporal context of current message."""

    hour_of_day_utc: int
    day_of_week: str
    is_typical_active_time: bool
    time_since_last_session_hours: float


class CurrentMessage(BaseModel):
    """Current message being evaluated."""

    text: str
    timestamp: str
    session_id: str
    message_sequence_in_session: int
    time_since_last_message_seconds: float
    detected_intent: Optional[str] = None
    requested_operation: RequestedOperation
    linguistic_features: LinguisticFeatures
    temporal_context: TemporalContext


# System Configuration
class SystemConfig(BaseModel):
    """System configuration for evaluation."""

    sensitivity_level: Literal["low", "medium", "high", "maximum"]
    deployment_context: Literal["consumer", "enterprise", "financial", "healthcare", "government"]
    enable_temporal_scoring: bool = True
    enable_linguistic_scoring: bool = True
    enable_semantic_scoring: bool = True
    overrides_enabled: bool = True


# Component Results
class ComponentScores(BaseModel):
    """Component anomaly scores."""

    semantic: float = Field(ge=0.0, le=1.0)
    linguistic: float = Field(ge=0.0, le=1.0)
    temporal: float = Field(ge=0.0, le=1.0)


class SemanticAnalysisResult(BaseModel):
    """Result of semantic analysis."""

    score: float = Field(ge=0.0, le=1.0)
    reasoning: str
    contributing_factors: List[str]


class LinguisticAnalysisResult(BaseModel):
    """Result of linguistic analysis."""

    score: float = Field(ge=0.0, le=1.0)
    reasoning: str
    contributing_factors: List[str]


class TemporalAnalysisResult(BaseModel):
    """Result of temporal analysis."""

    score: float = Field(ge=0.0, le=1.0)
    reasoning: str
    contributing_factors: List[str]


class CompositeScore(BaseModel):
    """Composite anomaly score with overrides."""

    anomaly_score: float = Field(ge=0.0, le=1.0)
    applied_overrides: List[str]
    detection_mechanism: Literal[
        "override_1", "override_2", "override_3", "override_4",
        "normal_override", "composite_score"
    ] = "composite_score"


class ConfidenceFactors(BaseModel):
    """Factors contributing to confidence assessment."""

    sufficient_history: bool
    clear_patterns: bool
    high_signal_quality: bool


class ConfidenceAssessment(BaseModel):
    """Confidence assessment result."""

    level: ConfidenceLevel
    factors: ConfidenceFactors


class Rationale(BaseModel):
    """Detailed rationale for evaluation."""

    primary_factors: List[str] = Field(min_length=1, max_length=3)
    semantic_reasoning: str
    linguistic_reasoning: str
    temporal_reasoning: str
    operation_risk_assessment: str
    overall_summary: str = Field(max_length=500)


class MonitoringRecommendations(BaseModel):
    """Monitoring recommendations for suspicious activity."""

    escalate_if: List[str]
    watch_for: List[str]
    auto_clear_after: str


# Evaluation Input/Output
class EvaluationInput(BaseModel):
    """Complete input for evaluation."""

    user_profile: UserProfile
    current_message: CurrentMessage
    system_config: SystemConfig


class EvaluationResult(BaseModel):
    """Complete evaluation result."""

    analysis_version: str = "1.0"
    timestamp: str
    user_id: str
    anomaly_score: float = Field(ge=0.0, le=1.0)
    component_scores: ComponentScores
    risk_level: RiskLevel
    recommended_action: PolicyAction
    confidence: ConfidenceLevel
    rationale: Rationale
    red_flags: List[str]
    mitigating_factors: List[str]
    monitoring_recommendations: MonitoringRecommendations
    metadata: dict


# Error Models
class ErrorDetail(BaseModel):
    """Error detail information."""

    type: str
    message: str
    details: List[str]
    timestamp: str


class ErrorResponse(BaseModel):
    """Error response format."""

    error: ErrorDetail


# Validation Result
class ValidationResult(BaseModel):
    """Result of input validation."""

    is_valid: bool
    errors: List[str]
