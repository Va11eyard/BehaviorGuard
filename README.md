# BehaviorGuard Anomaly Scoring System

A **machine learning-driven** AI security agent for detecting behavioral anomalies in conversational AI interactions.

## Overview

BehaviorGuard protects users and systems from:
- Account takeover (ATO)
- Social engineering attacks
- Unauthorized access
- AI-assisted cyber operations
- Prompt injection attempts
- Bot/automated attacks

The system evaluates user messages against established behavioral baselines to detect anomalous activity across semantic, linguistic, and temporal dimensions.

## Features

### Machine Learning Components
- **Neural Embeddings**: Uses sentence transformers for semantic understanding (cosine distance in embedding space)
- **Statistical Learning**: Mahalanobis distance for linguistic drift detection
- **Time-Series Analysis**: Z-scores and logistic functions for temporal anomalies
- **Learned Profiles**: User-specific baselines from interaction history
- **Ensemble Methods**: Weighted combination with adaptive thresholds
- **No Hardcoded Rules**: Learns patterns from data, not keyword matching

### System Capabilities
- **Multi-dimensional Analysis**: Semantic, linguistic, and temporal anomaly detection
- **Configurable Sensitivity**: Four sensitivity levels (low, medium, high, maximum)
- **Deployment Context Aware**: Supports consumer, enterprise, financial, healthcare, and government contexts
- **Comprehensive Output**: Detailed rationale, red flags, mitigating factors, and monitoring recommendations
- **Privacy-Preserving**: Focuses on behavioral patterns, not demographic attributes
- **Cold Start Handling**: Graceful handling of new users with insufficient history
- **Dual Implementation**: Rule-based (deterministic) and ML-based (learned) versions

## Installation

```bash
# Basic installation (rule-based system)
pip install -e .

# ML-based system (includes sentence transformers)
pip install -e . sentence-transformers numpy scikit-learn scipy

# Using Poetry
poetry install
```

## Quick Start

### ML-Based System (Recommended)

```python
from behaviorguard import BehaviorGuardEvaluatorML, ML_AVAILABLE
from behaviorguard.models import EvaluationInput, UserProfile, CurrentMessage, SystemConfig

# Create ML-based evaluator (uses sentence embeddings)
evaluator = BehaviorGuardEvaluatorML(embedding_model="all-MiniLM-L6-v2")

# Prepare input
evaluation_input = EvaluationInput(
    user_profile=user_profile,
    current_message=current_message,
    system_config=system_config
)

# Evaluate using ML (cosine distance, Mahalanobis distance, z-scores)
result = evaluator.evaluate(evaluation_input)

# Access results
print(f"Anomaly Score: {result.anomaly_score:.3f}")
print(f"Semantic Score: {result.component_scores.semantic:.3f} (cosine distance)")
print(f"Linguistic Score: {result.component_scores.linguistic:.3f} (Mahalanobis)")
print(f"Temporal Score: {result.component_scores.temporal:.3f} (z-scores)")
```

### Rule-Based System (Deterministic)

```python
from behaviorguard import BehaviorGuardEvaluator

# Create rule-based evaluator (deterministic, no ML dependencies)
evaluator = BehaviorGuardEvaluator()

# Same API as ML version
result = evaluator.evaluate(evaluation_input)
```

## Examples

### ML-Based Example (Recommended)

```bash
python example_ml.py
```

This demonstrates:
- Sentence embedding-based semantic analysis
- Mahalanobis distance for linguistic drift
- Z-score based temporal anomaly detection
- Learned user profiles
- No hardcoded keyword patterns

### Rule-Based Example

```bash
python example.py
```

This demonstrates the deterministic rule-based system.

## Architecture

The system follows a pipeline architecture with five main stages:

1. **Input Validation & Parsing**: Validates JSON input with Pydantic
2. **Component Scoring**: Computes semantic, linguistic, and temporal scores (0.0-1.0)
3. **Composite Scoring & Classification**: Combines scores with sensitivity-based weights
4. **Policy Decision**: Determines recommended action (ALLOW_NORMAL, ALLOW_WITH_CAUTION, BLOCK_AND_VERIFY_OOB, ESCALATE_TO_HUMAN)
5. **Output Generation**: Formats comprehensive JSON output with rationale and monitoring recommendations

### Components

**Analyzers:**
- `SemanticAnalyzer`: Topic and domain anomaly detection
- `LinguisticAnalyzer`: Writing style anomaly detection
- `TemporalAnalyzer`: Timing pattern anomaly detection

**Scorers:**
- `CompositeScorer`: Weighted combination with override conditions

**Classifiers:**
- `RiskClassifier`: NORMAL, SUSPICIOUS, HIGH_RISK classification
- `ConfidenceAssessor`: Confidence level assessment

**Detectors:**
- `RedFlagDetector`: 9 categories of suspicious indicators
- `MitigatingFactorDetector`: Factors suggesting legitimate use

**Utilities:**
- `PolicyDecisionEngine`: Policy action recommendations
- `RationaleGenerator`: Comprehensive reasoning generation
- `MonitoringRecommendationGenerator`: Escalation conditions and watch patterns
- `ColdStartHandler`: New user handling
- `OutputFormatter`: JSON formatting with 3 decimal precision

## Testing

### Run All Tests

```bash
pytest tests/ -v
```

**Current Status:** 61 tests passing, 93% coverage

### Run Specific Test Suites

```bash
# Component analyzers
pytest tests/test_semantic_analyzer.py -v
pytest tests/test_linguistic_analyzer.py -v
pytest tests/test_temporal_analyzer.py -v

# Composite scorer
pytest tests/test_composite_scorer.py -v

# Integration tests
pytest tests/test_integration.py -v
```

### Test Coverage

```bash
pytest tests/ --cov=src/behaviorguard --cov-report=html
```

### Property-Based Testing

BehaviorGuard uses Hypothesis for property-based testing to verify correctness across many randomly generated inputs:

- Component scores bounded [0.0, 1.0]
- Weighted score combination respects sensitivity
- Risk classification thresholds
- Deterministic evaluation
- Override conditions
- And more...

## Configuration

### Sensitivity Levels

| Level | Semantic (α) | Linguistic (β) | Temporal (γ) |
|-------|--------------|----------------|--------------|
| low | 0.5 | 0.3 | 0.2 |
| medium | 0.4 | 0.35 | 0.25 |
| high | 0.4 | 0.3 | 0.3 |
| maximum | 0.35 | 0.35 | 0.3 |

### Deployment Contexts

- **consumer**: Standard consumer applications
- **enterprise**: Enterprise environments
- **financial**: Financial services (escalates HIGH_RISK to human)
- **healthcare**: Healthcare systems (escalates HIGH_RISK to human)
- **government**: Government systems (escalates HIGH_RISK to human)

## Risk Classification

### Thresholds
- **NORMAL**: score < 0.25
- **SUSPICIOUS**: 0.25 ≤ score < 0.60
- **HIGH_RISK**: score ≥ 0.60

### Policy Actions
- **ALLOW_NORMAL**: Proceed normally with standard logging
- **ALLOW_WITH_CAUTION**: Rate limiting, enhanced logging, 24-48h watchlist
- **BLOCK_AND_VERIFY_OOB**: Block operation, trigger out-of-band verification
- **ESCALATE_TO_HUMAN**: Block immediately, alert security team, freeze account

## Development

### Project Structure

```
src/behaviorguard/
├── analyzers/          # Component analyzers
├── scorers/            # Composite scoring
├── detectors/          # Red flags and mitigating factors
├── utils/              # Utilities
├── models.py           # Pydantic data models
├── validator.py        # Input validation
└── evaluator.py        # Main orchestrator

tests/                  # 61 tests, 93% coverage
```

### Type Checking

```bash
mypy src/behaviorguard --strict
```

### Code Formatting

```bash
black src/ tests/
isort src/ tests/
```

## Performance

- Target latency: <500ms per evaluation
- Stateless design enables horizontal scaling
- Deterministic for audit replay

## Security & Privacy

- Never logs raw message content in production
- Hashes sensitive identifiers
- Focuses on behavioral patterns, not demographic attributes
- No discrimination based on protected attributes
- Complies with privacy regulations

## License

MIT
