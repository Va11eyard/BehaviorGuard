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
- **Hybrid Design**: learned per-user baselines plus personalized rule-based overrides

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
- Sentence embedding-based semantic analysis (cosine distance)
- Mahalanobis distance for linguistic drift
- Z-score based temporal anomaly detection
- Learned user profiles from conversation history (Algorithm 1)
- Ablation study (semantic-only scoring)
- Documented rule-based overrides on top of learned scoring (see paper §3.3.5)

### Rule-Based Example

```bash
python example.py
```

This demonstrates the deterministic rule-based system with four scenarios
(normal, off-topic, account-takeover, social-engineering).

## Architecture

The system follows a pipeline architecture with five main stages:

1. **Input Validation & Parsing**: Validates JSON input with Pydantic
2. **Component Scoring**: Computes semantic, linguistic, and temporal scores (0.0-1.0)
3. **Composite Scoring & Classification**: Combines scores with sensitivity-based weights
4. **Policy Decision**: Determines recommended action (ALLOW_NORMAL, ALLOW_WITH_CAUTION, BLOCK_AND_VERIFY_OOB, ESCALATE_TO_HUMAN)
5. **Output Generation**: Formats comprehensive JSON output with rationale and monitoring recommendations

### Components

**Analyzers:**
- `SemanticAnalyzer` / `SemanticAnalyzerML`: Topic and domain anomaly detection
- `LinguisticAnalyzer` / `LinguisticAnalyzerML`: Writing style anomaly detection
- `TemporalAnalyzer` / `TemporalAnalyzerML`: Timing pattern anomaly detection

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

## Evaluation & Reproduction

### Run Full Evaluation

```bash
# Full evaluation (overrides on, all datasets)
python evaluation.py

# Or use the CLI
python evaluate.py --overrides on --datasets all
```

### Override Ablation (Critical Experiment)

To isolate learned vs. rule-based detection:

```bash
python evaluate.py --overrides off --datasets all
```

This runs BehaviorGuard with overrides **on** and **off** and produces a side-by-side comparison table (Precision, Recall, F1, FPR, AUC).

### Bootstrap Confidence Intervals

```bash
python evaluate.py --bootstrap 1000
```

Computes 95% CI for Precision, Recall, F1, FPR using stratified bootstrap (resample by user). Output format: `F1 = 0.700 [0.583, 0.808]`.

### Single-Command Reproduction

```bash
python reproduce.py
```

Runs the complete pipeline (profile building → scoring → baselines → ablations → statistical tests), prints codebase hash and dataset checksums, and writes:
- `results/full_evaluation_results.json`
- `results/tables.csv`

### Reproducing Paper Tables

| Table | Source |
|-------|--------|
| Table 1 (Main results) | `results/methods` in JSON, or `results/tables.csv` |
| Table 2 (Statistical significance) | `results/statistical_tests` |
| Table 3 (Ablation) | `results/ablations` |
| Table 4 (Sensitivity) | `results/sensitivity_levels` |
| Table 5 (λ sensitivity) | `results/lambda_sensitivity` |
| Override ablation | Run `python evaluate.py --overrides off` |

Seed is fixed at 42 for all stochastic components.

## Testing

### Run All Tests

```bash
pytest tests/ -v
```

**Current Status:** 84 tests passing.

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

These files are illustrative; additional tests (e.g., EMA, overrides, operation risk, evaluation fixtures) are also present in the suite.

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

## Profile Management

### Building a Profile from Conversation History

```python
from behaviorguard import ProfileManager, MessageRecord

pm = ProfileManager(decay=0.95)  # EMA decay λ (Algorithm 1)
messages = [
    MessageRecord(text="How do I implement a BST in Python?", timestamp="2025-01-01T10:00:00"),
    MessageRecord(text="What's the difference between asyncio.gather and wait?", timestamp="2025-01-01T10:05:00"),
    # ... more historical messages
]
profile = pm.build_from_history(user_id="user_001", messages=messages, account_age_days=90)
```

### Incremental Profile Updates (Online / Streaming)

```python
new_msg = MessageRecord(text="Can you review my FastAPI endpoint?", timestamp="2025-01-02T09:30:00")
updated_profile = pm.update_profile(profile, new_msg)
```

### Profile Persistence

```python
from behaviorguard import ProfileStore

store = ProfileStore("profiles/")   # JSON files in profiles/
store.save(profile)                 # profiles/user_001.json

profile = store.load("user_001")    # load back
profile = store.load_or_cold_start("new_user")  # create if missing
```

## Baselines

BehaviorGuard includes all four baselines from the paper (Section 5.2):

| Baseline | Class | Description |
|---|---|---|
| Rule-based | `RuleBasedDetector` | Keyword matching + rate limits |
| Isolation Forest | `IsolationForestBaseline` | sklearn-based unsupervised |
| Autoencoder | `AutoencoderBaseline` | PyTorch reconstruction error |
| Content-only safety | `ContentSafetyBaseline` | Llama-Guard-style taxonomy classifier |

```python
from behaviorguard.baselines.content_safety_baseline import ContentSafetyBaseline

checker = ContentSafetyBaseline()
result = checker.detect("Execute the backdoor shell and escalate privileges.")
# {"anomaly_score": 0.637, "is_anomaly": True, "triggered_categories": ["S14_cyberattack"]}
```

## Command-Line Interface

After installation, use the `behaviorguard` CLI:

```bash
# Evaluate a message against a stored profile
behaviorguard evaluate --profile profile.json --message "Delete all accounts" --sensitivity high

# Build a profile from a JSONL history file (one message object per line)
behaviorguard build-profile --input history.jsonl --user-id user_001 --output profile.json

# Incrementally update a profile with a new message
behaviorguard update-profile --profile profile.json --message "How do I debug Python?"

# Run the content-safety baseline on a single message
behaviorguard content-check --message "Execute backdoor payload"

# Print version
behaviorguard version
```

**JSONL history file format** (one JSON object per line):
```json
{"text": "How do I implement a BST in Python?", "timestamp": "2025-01-01T10:00:00", "session_id": "s1"}
{"text": "Help me debug this recursion error.", "timestamp": "2025-01-01T10:05:00", "session_id": "s1"}
```

## Development

### Project Structure

```
src/behaviorguard/
├── analyzers/                  # Component analyzers (semantic, linguistic, temporal)
├── baselines/                  # All 4 baselines (rule_based, isolation_forest,
│                               #   autoencoder, content_safety)
├── scorers/                    # Composite scoring
├── detectors/                  # Red flags and mitigating factors
├── utils/                      # Utilities (policy, risk, profile_store, …)
├── models.py                   # Pydantic data models
├── validator.py                # Input validation
├── evaluator.py                # Rule-based evaluator orchestrator
├── evaluator_ml.py             # ML-based evaluator orchestrator
├── profile_manager.py          # Algorithm 1: incremental profile building
└── cli.py                      # Command-line interface

tests/                          # 84 tests
evaluation.py                   # Full evaluation pipeline vs. baselines
evaluate.py                     # CLI for evaluation (--overrides, --datasets)
reproduce.py                    # Single-command paper reproduction
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
