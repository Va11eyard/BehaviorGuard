"""
Full evaluation pipeline with baselines, ablations, and statistical tests.

This script runs:
1. TurnShift (full system)
2. Baselines (Rule-based, Isolation Forest, Autoencoder)
3. Ablation studies (7 configurations)
4. Statistical significance tests
5. Sensitivity level analysis
"""

import csv
import importlib
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
from datetime import datetime
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
from scipy import stats

# Windows consoles default to a non-UTF-8 codepage (e.g. cp1251) which cannot
# encode characters like the Greek lambda used in progress output; force UTF-8.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

# Set seeds for reproducibility
SEED = 42
np.random.seed(SEED)

print("="*80)
print("TURNSHIFT FULL EVALUATION PIPELINE")
print("="*80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Import TurnShift components
from turnshift import TurnShiftEvaluatorML
from turnshift.analyzers.semantic_ml import SemanticAnalyzerML
from turnshift.analyzers.linguistic_ml import LinguisticAnalyzerML
from turnshift.analyzers.temporal_ml import TemporalAnalyzerML
from turnshift.baselines.rule_based import RuleBasedDetector
from turnshift.baselines.isolation_forest_baseline import IsolationForestBaseline
from turnshift.baselines.autoencoder_baseline import AutoencoderBaseline
from turnshift.baselines.content_safety_baseline import ContentSafetyBaseline
from turnshift.models import (
    UserProfile, SemanticProfile, LinguisticProfile, 
    TemporalProfile, OperationalProfile,
    EvaluationInput, CurrentMessage, SystemConfig,
    RequestedOperation, LinguisticFeatures, TemporalContext
)

# Load datasets
print("\n[1/7] Loading datasets...")
datasets = {}
# Default to the de-confounded re-injection (tools/rebuild_injected_datasets.py).
# The original *_processed.json has tail-clustered injections and metadata leakage
# and is kept only as the rebuild input / archived-exhibit source.
dataset_files = {}
dataset_variant = {}
for name in ("personachat", "blended_skill_talk", "anthropic_hh"):
    corrected = f"datasets/{name}_processed_corrected.json"
    if os.path.exists(corrected):
        dataset_files[name] = corrected
        dataset_variant[name] = "corrected"
    else:
        dataset_files[name] = f"datasets/{name}_processed.json"
        dataset_variant[name] = "UNCORRECTED"
        print(
            f"  WARNING: {corrected} not found; loading the UNCORRECTED {dataset_files[name]} "
            "(confounded injection, enriched tail prevalence). Results are NOT comparable to "
            "the paper's realistic-prevalence figures. Build it with "
            "tools/rebuild_injected_datasets.py.",
            file=sys.stderr,
            flush=True,
        )

for name, filepath in dataset_files.items():
    with open(filepath, 'r', encoding="utf-8") as f:
        datasets[name] = json.load(f)
    print(f"  [OK] Loaded {name} ({dataset_variant[name]}: {filepath}): "
          f"{len(datasets[name]['users'])} users, {len(datasets[name]['messages'])} messages")

# Initialize components
print("\n[2/7] Initializing detectors...")
evaluator = TurnShiftEvaluatorML()
semantic_analyzer = SemanticAnalyzerML()
linguistic_analyzer = LinguisticAnalyzerML()
temporal_analyzer = TemporalAnalyzerML()
rule_based = RuleBasedDetector()
content_safety = ContentSafetyBaseline()
print("  [OK] All detectors initialized")

# Helper functions
def build_user_profile(user_data: Dict, user_messages: List[Dict]) -> UserProfile:
    """Build user profile from training messages."""
    normal_msgs = [m for m in user_messages if not m.get("is_anomaly", False)]
    
    if len(normal_msgs) < 3:
        return None
    
    # Semantic profile
    texts = [m["message_text"] for m in normal_msgs[:50]]
    embeddings = semantic_analyzer.model.encode(texts, convert_to_numpy=True)
    
    # Linguistic profile — same proxies as ProfileManager.build_from_history so
    # profile statistics match the message features from message_to_current_message
    lengths = [len(m["message_text"].split()) for m in normal_msgs]
    formality_vals = [min(1.0, n / 50.0) for n in lengths]
    politeness_vals = [
        float(
            any(
                w in m["message_text"].lower()
                for w in ("please", "thank", "sorry", "could you", "would you")
            )
        )
        for m in normal_msgs
    ]
    
    # Temporal profile
    from datetime import datetime as dt
    hours = [dt.fromisoformat(m["timestamp"]).hour for m in normal_msgs]
    active_hours = list(set(hours))

    has_requested_sensitive_ops = any(
        m.get("operation_risk") in ("high", "critical")
        for m in normal_msgs
        if m.get("operation_risk") is not None
    )
    
    return UserProfile(
        user_id=user_data["user_id"],
        account_age_days=user_data.get("account_age_days", 100),
        total_interactions=len(normal_msgs),
        semantic_profile=SemanticProfile(
            typical_topics=user_data.get("typical_topics", ["general"]),
            primary_domains=["conversation"],
            topic_diversity_score=0.5,
            embedding_centroid_summary="User profile"
        ),
        linguistic_profile=LinguisticProfile(
            avg_message_length_tokens=float(np.mean(lengths)),
            avg_message_length_chars=float(np.mean([len(m["message_text"]) for m in normal_msgs])),
            avg_message_length_tokens_std=max(float(np.std(lengths)), 1.0),
            lexical_diversity_mean=0.7,
            lexical_diversity_std=0.1,
            formality_score_mean=float(np.mean(formality_vals)),
            formality_score_std=max(float(np.std(formality_vals)), 0.01),
            politeness_score_mean=float(np.mean(politeness_vals)),
            politeness_score_std=max(float(np.std(politeness_vals)), 0.01),
            question_ratio_mean=0.3,
            uses_technical_vocabulary=True,
            uses_code_blocks=False,
            primary_languages=["en"],
            typical_sentence_complexity="moderate"
        ),
        temporal_profile=TemporalProfile(
            typical_session_duration_minutes=40.0,
            typical_inter_message_gap_seconds=30.0,
            most_active_hours_utc=active_hours,
            most_active_days_of_week=["Monday", "Tuesday", "Wednesday"],
            average_messages_per_session=10.0,
            longest_session_duration_minutes=80.0,
            typical_session_frequency_per_week=5.0,
            last_activity_timestamp=normal_msgs[-1]["timestamp"]
        ),
        operational_profile=OperationalProfile(
            common_intent_types=["information_seeking"],
            tools_used_historically=["search"],
            has_requested_sensitive_ops=has_requested_sensitive_ops,
            typical_risk_level="low"
        )
    )

_VALID_OPERATION_TYPES = frozenset(
    (
        "read",
        "write",
        "delete",
        "export",
        "auth_change",
        "permission_change",
        "financial",
        "admin",
        "none",
    )
)


def message_to_current_message(
    msg: dict,
    prev_msg: dict | None = None,
    user_profile: UserProfile | None = None,
) -> CurrentMessage:
    """Convert dataset message to CurrentMessage.

    is_typical_active_time is derived from user_profile.temporal_profile.most_active_hours_utc
    when available. Falls back to range(9, 22) only for cold-start users whose profile has
    no learned active hours (ProfileManager.build_from_history and build_user_profile
    populate most_active_hours_utc for any user with >=3 normal messages).
    """
    from datetime import datetime as dt
    from turnshift.utils.operation_risk_classifier import classify_operation_risk

    text = msg["message_text"]
    timestamp = dt.fromisoformat(msg["timestamp"])
    session_id = msg.get("session_id", "session_0")

    if (
        "time_since_last_message_seconds" in msg
        and msg["time_since_last_message_seconds"] is not None
    ):
        time_since_last_message_seconds = float(msg["time_since_last_message_seconds"])
    elif prev_msg is not None and prev_msg.get("session_id", "session_0") == session_id:
        prev_ts = dt.fromisoformat(prev_msg["timestamp"])
        time_since_last_message_seconds = (timestamp - prev_ts).total_seconds()
    else:
        time_since_last_message_seconds = 30.0

    raw_type = msg.get("operation_type", "read")
    op_type = raw_type if raw_type in _VALID_OPERATION_TYPES else "read"

    op_risk = msg.get("operation_risk")
    if op_risk is not None and isinstance(op_risk, str):
        s = op_risk.strip().lower()
        if s in ("low", "medium", "high", "critical"):
            risk_classification = s
        else:
            risk_classification = classify_operation_risk(text)
    else:
        risk_classification = classify_operation_risk(text)

    n_tokens = len(text.split())
    n_chars = len(text)
    message_length_tokens = int(msg["message_length_tokens"]) if "message_length_tokens" in msg else n_tokens
    message_length_chars = int(msg["message_length_chars"]) if "message_length_chars" in msg else n_chars
    if "lexical_diversity" in msg:
        lexical_diversity = float(msg["lexical_diversity"])
    else:
        lexical_diversity = len(set(text.split())) / max(len(text.split()), 1)
    lower = text.lower()
    if "formality_score" in msg:
        formality_score = float(msg["formality_score"])
    else:
        formality_score = min(1.0, n_tokens / 50.0)
    if "politeness_score" in msg:
        politeness_score = float(msg["politeness_score"])
    else:
        politeness_score = float(
            any(w in lower for w in ("please", "thank", "sorry", "could you", "would you"))
        )

    return CurrentMessage(
        text=text,
        timestamp=msg["timestamp"],
        session_id=session_id,
        message_sequence_in_session=msg.get("sequence_in_session", 1),
        time_since_last_message_seconds=time_since_last_message_seconds,
        requested_operation=RequestedOperation(
            type=op_type,
            risk_classification=risk_classification,
            targets=msg.get("targets"),
            requires_auth=bool(msg.get("requires_auth", False)),
        ),
        linguistic_features=LinguisticFeatures(
            message_length_tokens=message_length_tokens,
            message_length_chars=message_length_chars,
            lexical_diversity=lexical_diversity,
            formality_score=formality_score,
            politeness_score=politeness_score,
            contains_code=bool(msg.get("contains_code", False)),
            contains_urls=bool(msg.get("contains_urls", False)),
            language=str(msg.get("language", "en")),
        ),
        temporal_context=TemporalContext(
            hour_of_day_utc=timestamp.hour,
            day_of_week=timestamp.strftime("%A"),
            is_typical_active_time=_is_typical_active_time(timestamp.hour, user_profile),
            time_since_last_session_hours=float(msg.get("time_since_last_session_hours", 24.0)),
        ),
    )


def _is_typical_active_time(hour: int, user_profile: UserProfile | None) -> bool:
    """Return True iff hour is in the user's learned active hours.

    Falls back to range(9, 22) only when the profile is missing or its
    most_active_hours_utc is empty (cold-start safety net).
    """
    active_hours = (
        user_profile.temporal_profile.most_active_hours_utc
        if user_profile is not None and user_profile.temporal_profile.most_active_hours_utc
        else list(range(9, 22))
    )
    return hour in active_hours

def extract_features_for_baselines(msg: Dict, profile: UserProfile) -> np.ndarray:
    """Extract feature vector for baseline methods."""
    # Semantic features (embedding)
    embedding = semantic_analyzer.model.encode(msg["message_text"], convert_to_numpy=True)
    
    # Linguistic features
    text = msg["message_text"]
    ling_features = [
        len(text.split()),  # word count
        len(text),  # char count
        len(set(text.split())) / max(len(text.split()), 1),  # lexical diversity
        text.count('?') / max(len(text.split('.')), 1),  # question ratio
        float('code' in text.lower() or 'def ' in text or 'function' in text),  # code presence
    ]
    
    # Temporal features
    from datetime import datetime as dt
    timestamp = dt.fromisoformat(msg["timestamp"])
    temp_features = [
        timestamp.hour / 24.0,  # normalized hour
        timestamp.weekday() / 7.0,  # normalized day
    ]
    
    # Concatenate all features
    feature_vector = np.concatenate([embedding, ling_features, temp_features])
    return feature_vector

def compute_bootstrap_ci(
    predictions: List[Dict],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Dict[str, Tuple[float, float]]:
    """
    Compute 95% bootstrap CI for metrics using stratified resampling by user.

    Resamples users (not individual messages) to avoid leakage. Returns
    (lower, upper) for each of precision, recall, f1, fpr.
    """
    rng = np.random.default_rng(seed)
    # Group predictions by user_id
    by_user: Dict[str, List[Dict]] = defaultdict(list)
    for p in predictions:
        by_user[p["user_id"]].append(p)
    user_ids = list(by_user.keys())
    n_users = len(user_ids)

    boot_precision, boot_recall, boot_f1, boot_fpr = [], [], [], []
    for _ in range(n_bootstrap):
        # Resample users with replacement
        sampled_ids = rng.choice(user_ids, size=n_users, replace=True)
        flat = []
        for uid in sampled_ids:
            flat.extend(by_user[uid])
        y_true = [x["true_label"] for x in flat]
        y_pred = [x["predicted_label"] for x in flat]
        y_scores = [x["predicted_score"] for x in flat]
        m = compute_metrics(y_true, y_pred, y_scores)
        boot_precision.append(m["precision"])
        boot_recall.append(m["recall"])
        boot_f1.append(m["f1"])
        boot_fpr.append(m["fpr"])

    def ci(arr: List[float]) -> Tuple[float, float]:
        return (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))

    return {
        "precision": ci(boot_precision),
        "recall": ci(boot_recall),
        "f1": ci(boot_f1),
        "fpr": ci(boot_fpr),
    }


def get_bootstrap_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float = 0.60,
    n: int = 1000,
    seed: int = 42,
) -> dict:
    """Message-level bootstrap CIs for ROC-AUC, F1, precision, and recall."""
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    aucs, f1s, precs, recs = [], [], [], []
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    for _ in range(n):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        yt, ys = y_true[idx], y_scores[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, ys))
        yp = (ys >= threshold).astype(int)
        tp = ((yp == 1) & (yt == 1)).sum()
        fp = ((yp == 1) & (yt == 0)).sum()
        fn = ((yp == 0) & (yt == 1)).sum()
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s.append(2 * p * r / (p + r) if (p + r) > 0 else 0.0)
        precs.append(p)
        recs.append(r)

    def ci(arr):
        a = np.array(arr)
        return float(np.mean(a)), float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))

    return {"auc": ci(aucs), "f1": ci(f1s), "prec": ci(precs), "rec": ci(recs)}


def compute_metrics(y_true: List[bool], y_pred: List[bool], y_scores: List[float]) -> Dict:
    """Compute comprehensive metrics."""
    from sklearn.metrics import (
        precision_score, recall_score, f1_score, accuracy_score,
        roc_auc_score, average_precision_score, matthews_corrcoef,
        confusion_matrix
    )
    
    # Handle edge cases
    if len(set(y_true)) < 2:
        auc_roc = 0.5
        auc_pr = 0.5
        tn, fp, fn, tp = 0, 0, 0, 0
    else:
        auc_roc = roc_auc_score(y_true, y_scores)
        auc_pr = average_precision_score(y_true, y_scores)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "fpr": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        "fnr": float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0,
        "tpr": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        "tnr": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "roc_auc": float(auc_roc),
        "pr_auc": float(auc_pr),
    }

def compute_per_class_metrics(messages: List[Dict], results: List) -> Dict:
    """
    Compute precision, recall, F1, support per anomaly_type and per attack_phase.

    messages: raw test-message dicts (may contain 'anomaly_type', 'attack_phase', 'should_flag')
    results: list aligned by index with 'messages'. Each item is either a
             TurnShiftResult (has .anomaly_score) or a dict with
             'anomaly_score' or 'predicted_score'.

    Returns:
        {
          "anomaly_type": {cls: {"precision": f, "recall": f, "f1": f, "support": n}, ...},
          "attack_phase": {cls: {"precision": f, "recall": f, "f1": f, "support": n}, ...},
        }
        Missing / None / "" class labels are reported under the key "benign".
    """
    def _score(r) -> float:
        if hasattr(r, "anomaly_score"):
            return float(r.anomaly_score)
        if isinstance(r, dict):
            if "anomaly_score" in r:
                return float(r["anomaly_score"])
            if "predicted_score" in r:
                return float(r["predicted_score"])
        return 0.0

    out: Dict[str, Dict[str, Dict]] = {}
    for key in ("anomaly_type", "attack_phase"):
        by_class: Dict[str, List[Tuple[bool, bool]]] = defaultdict(list)
        for msg, res in zip(messages, results):
            cls = msg.get(key)
            if cls is None or cls == "" or cls == "benign":
                cls = "benign"
            y_true = bool(msg.get("should_flag", False))
            y_pred = _score(res) > 0.60
            by_class[cls].append((y_true, y_pred))

        group: Dict[str, Dict] = {}
        for cls, items in by_class.items():
            tp = sum(1 for y, p in items if y and p)
            fp = sum(1 for y, p in items if not y and p)
            fn = sum(1 for y, p in items if y and not p)
            support = len(items)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            group[cls] = {
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "support": int(support),
            }
        out[key] = group
    return out


def _build_profile_with_pm(decay: float):
    """Return a profile builder that uses ProfileManager with given decay."""
    from turnshift import ProfileManager, MessageRecord

    pm = ProfileManager(decay=decay)

    def builder(user_data: Dict, user_messages: List[Dict]) -> Optional[UserProfile]:
        normal_msgs = [m for m in user_messages if not m.get("is_anomaly", False)]
        if len(normal_msgs) < 3:
            return None
        records = [
            MessageRecord(
                text=m["message_text"],
                timestamp=m["timestamp"],
                session_id=m.get("session_id", "s0"),
                is_anomaly=m.get("is_anomaly", False),
                operation_risk=m.get("operation_risk"),
            )
            for m in normal_msgs
        ]
        return pm.build_from_history(
            user_data["user_id"],
            records,
            account_age_days=user_data.get("account_age_days", 100),
        )

    builder.profile_manager = pm  # exposed so callers can verify the decay actually in use
    return builder


def evaluate_method(
    method_name: str,
    dataset_name: str,
    test_data: Dict,
    max_users: int | None = None,
    config: SystemConfig = None,
    enable_semantic: bool = True,
    enable_linguistic: bool = True,
    enable_temporal: bool = True,
    overrides_enabled: bool = True,
    profile_builder=None,
    canonical_metrics: bool = False,
    contamination: float | str = 0.1,
) -> Dict:
    """Evaluate a single method on a dataset.

    Args:
        max_users: Cap on sampled test users. ``None`` (default) evaluates the
            full test split; otherwise a seeded uniform random subsample of that
            size is used, preserving the split's natural anomaly prevalence.
            Historical paper runs used ``max_users=20`` with anomaly-first
            filling (positive-enriched, ~48% prevalence); that sampling was
            removed as a diagnosed base-rate artifact and is retained only as
            the archived exhibit in results/archived-per-message-study/.
    """
    print(f"\n  Evaluating {method_name} on {dataset_name}...")
    
    # Prepare test data
    test_user_ids = set(test_data["splits"]["test"]["user_ids"])
    test_users = [u for u in test_data["users"] if u["user_id"] in test_user_ids]
    test_messages_by_user = defaultdict(list)
    for m in test_data["messages"]:
        if m["user_id"] in test_user_ids:
            test_messages_by_user[m["user_id"]].append(m)
    
    # Sample users with anomalies
    users_with_anomalies = []
    users_without_anomalies = []
    
    for user in test_users:
        user_msgs = test_messages_by_user[user["user_id"]]
        has_anomaly = any(m.get("should_flag", False) for m in user_msgs)
        if has_anomaly:
            users_with_anomalies.append(user)
        else:
            users_without_anomalies.append(user)

    if max_users is None or max_users >= len(test_users):
        sampled_test_users = users_with_anomalies + users_without_anomalies
    else:
        # Uniform random subsample at the split's natural prevalence. max_users is a
        # compute cap only. Filling it with anomaly-containing users first (the
        # historical Table III protocol) is the positive-enrichment sampling the
        # paper diagnoses as a base-rate artifact; it must not be reintroduced here.
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(test_users), size=max_users, replace=False)
        sampled_test_users = [test_users[i] for i in sorted(idx)]
    
    # Build profiles and track cold-start coverage
    test_user_profiles = {}
    cold_start_count = 0
    for user in sampled_test_users:
        user_msgs = test_messages_by_user[user["user_id"]]
        split_idx = int(len(user_msgs) * 0.8)
        train_msgs = user_msgs[:split_idx]
        n_train = len(train_msgs)
        maturity_bin = "cold_start" if n_train < 10 else "stable"

        if n_train < 10:
            cold_start_count += 1
        builder = profile_builder if profile_builder else build_user_profile
        profile = builder(user, train_msgs)
        if profile:
            test_user_profiles[user["user_id"]] = {
                "profile": profile,
                "test_messages": user_msgs[split_idx:],
                "maturity_bin": maturity_bin,
            }
    
    # Collect predictions
    predictions = []
    aligned_test_msgs: List[Dict] = []
    latencies = []
    n_train_features: Optional[int] = None
    
    # For baseline methods, collect training features (content_safety needs no training)
    if method_name in ["isolation_forest", "autoencoder"]:
        train_features = []
        for user_id, user_data in test_user_profiles.items():
            profile = user_data["profile"]
            # Use first 80% of messages for training baseline
            train_user_ids = set(test_data["splits"]["train"]["user_ids"])
            train_users = [u for u in test_data["users"] if u["user_id"] in train_user_ids][:20]
            
            for train_user in train_users:
                train_user_msgs = [m for m in test_data["messages"] 
                                  if m["user_id"] == train_user["user_id"] and not m.get("is_anomaly", False)]
                for msg in train_user_msgs[:10]:
                    feat = extract_features_for_baselines(msg, profile)
                    train_features.append(feat)
        
        train_features = np.array(train_features)
        
        if method_name == "isolation_forest":
            n_train_features = len(train_features)
            iso_forest = IsolationForestBaseline(
                contamination=contamination, random_state=SEED
            )
            iso_forest.fit(train_features)
        elif method_name == "autoencoder":
            autoencoder = AutoencoderBaseline(
                input_dim=train_features.shape[1],
                random_seed=SEED,
            )
            autoencoder.fit(train_features, verbose=False)
    
    # Evaluate on test messages
    for user_id, user_data in test_user_profiles.items():
        profile = user_data["profile"]
        test_msgs = user_data["test_messages"]
        maturity_bin = user_data.get("maturity_bin", "stable")

        for i, msg in enumerate(test_msgs):
            start = time.perf_counter()
            detection_mechanism = "n/a"
            cur_session = msg.get("session_id", "session_0")
            if i > 0:
                p = test_msgs[i - 1]
                prev_msg = p if p.get("session_id", "session_0") == cur_session else None
            else:
                prev_msg = None

            if method_name == "behaviorguard":
                current_msg = message_to_current_message(msg, prev_msg, user_profile=profile)
                base_config = SystemConfig(
                    sensitivity_level="medium",
                    deployment_context="enterprise",
                    enable_temporal_scoring=enable_temporal,
                    enable_linguistic_scoring=enable_linguistic,
                    enable_semantic_scoring=enable_semantic,
                    overrides_enabled=overrides_enabled,
                )
                result = evaluator.evaluate(EvaluationInput(
                    user_profile=profile,
                    current_message=current_msg,
                    system_config=config or base_config,
                ))
                score = result.anomaly_score
                detection_mechanism = result.metadata.get("detection_mechanism", "composite_score")
                
            elif method_name == "rule_based":
                from datetime import datetime as dt
                timestamp = dt.fromisoformat(msg["timestamp"])
                result = rule_based.detect(user_id, msg["message_text"], timestamp)
                score = result["anomaly_score"]
                
            elif method_name == "isolation_forest":
                feat = extract_features_for_baselines(msg, profile)
                result = iso_forest.detect_single(feat)
                score = result["anomaly_score"]
                
            elif method_name == "autoencoder":
                feat = extract_features_for_baselines(msg, profile)
                result = autoencoder.detect_single(feat)
                score = result["anomaly_score"]

            elif method_name == "content_safety":
                result = content_safety.detect(msg["message_text"])
                score = result["anomaly_score"]

            else:
                score = 0.0

            latency = (time.perf_counter() - start) * 1000
            
            predictions.append({
                "true_label": msg.get("should_flag", False),
                "predicted_score": score,
                "predicted_label": score > 0.60,
                "anomaly_type": msg.get("anomaly_type"),
                "detection_mechanism": detection_mechanism if method_name == "behaviorguard" else "n/a",
                "user_id": user_id,
                "maturity_bin": maturity_bin,
            })
            aligned_test_msgs.append(msg)
            latencies.append(latency)
    
    # Compute metrics
    y_true = [p["true_label"] for p in predictions]
    y_pred = [p["predicted_label"] for p in predictions]
    y_scores = [p["predicted_score"] for p in predictions]
    
    metrics = compute_metrics(y_true, y_pred, y_scores)
    metrics["per_class_metrics"] = compute_per_class_metrics(aligned_test_msgs, predictions)
    metrics["latency_mean_ms"] = float(np.mean(latencies))
    metrics["latency_median_ms"] = float(np.median(latencies))
    metrics["latency_p95_ms"] = float(np.percentile(latencies, 95))
    metrics["num_predictions"] = len(predictions)
    n_test_users = len(sampled_test_users)
    metrics["cold_start_pct"] = (
        100.0 * cold_start_count / n_test_users if n_test_users > 0 else 0.0
    )
    metrics["cold_start_count"] = cold_start_count
    metrics["n_test_users"] = n_test_users
    if n_train_features is not None:
        metrics["n_train_features"] = n_train_features

    if canonical_metrics:
        boot = get_bootstrap_metrics(np.asarray(y_true), np.asarray(y_scores))
        metrics["bootstrap"] = boot
        metrics["auc_mean"] = boot["auc"][0]
        metrics["auc_ci_low"] = boot["auc"][1]
        metrics["auc_ci_high"] = boot["auc"][2]
        metrics["f1_ci_low"] = boot["f1"][1]
        metrics["f1_ci_high"] = boot["f1"][2]
        metrics["prec_ci_low"] = boot["prec"][1]
        metrics["prec_ci_high"] = boot["prec"][2]
        metrics["rec_ci_low"] = boot["rec"][1]
        metrics["rec_ci_high"] = boot["rec"][2]
        metrics["maturity_breakdown"] = {
            "cold_start": _aggregate_maturity_predictions(predictions, "cold_start"),
            "stable": _aggregate_maturity_predictions(predictions, "stable"),
        }

    # FPR verification: FPR = FP / (FP + TN), per-dataset breakdown
    fp, tn = metrics["false_positives"], metrics["true_negatives"]
    n_negatives = fp + tn
    metrics["fpr_denominator"] = n_negatives
    if n_negatives > 0:
        metrics["fpr_verified"] = fp / n_negatives
    print(f"    [FPR] FP={fp}, TN={tn}, FPR={metrics['fpr']:.4f} (FP/(FP+TN)={fp}/{n_negatives})")

    # Attribution breakdown: for TurnShift true-positive detections only
    if method_name == "behaviorguard":
        tp_mechanisms = [
            p["detection_mechanism"]
            for p in predictions
            if p["true_label"] and p["predicted_label"]
        ]
        if tp_mechanisms:
            mechanism_counts = {}
            for m in tp_mechanisms:
                mechanism_counts[m] = mechanism_counts.get(m, 0) + 1
            total_tp = len(tp_mechanisms)
            metrics["tp_attribution"] = {
                k: {"count": v, "pct": round(100.0 * v / total_tp, 1)}
                for k, v in mechanism_counts.items()
            }
            # Paper-reproducible attribution breakdown
            override_labels = {
                "override_1": "Override 1 (semantic>0.85 + critical op)",
                "override_2": "Override 2 (temporal>0.9, bot-like)",
                "override_3": "Override 3 (critical op, no sensitive history)",
                "override_4": "Override 4 (ATO keywords)",
                "composite_score": "Composite score only",
            }
            print(f"    [Attribution] Detection mechanism breakdown ({total_tp} TPs):")
            printed = set()
            for k in ["override_1", "override_2", "override_3", "override_4", "composite_score"]:
                v = mechanism_counts.get(k, 0)
                if v > 0:
                    label = override_labels.get(k, k)
                    print(f"      {label}: {v} ({100.0*v/total_tp:.1f}%)")
                    printed.add(k)
            for k, v in sorted(mechanism_counts.items()):
                if k in printed:
                    continue
                print(f"      {k}: {v} ({100.0*v/total_tp:.1f}%)")
        else:
            metrics["tp_attribution"] = {}

    print(f"    [OK] F1: {metrics['f1']:.3f}, Precision: {metrics['precision']:.3f}, "
          f"Recall: {metrics['recall']:.3f}")
    if n_test_users > 0:
        print(f"    [Cold-start] {cold_start_count}/{n_test_users} test users "
              f"({metrics['cold_start_pct']:.1f}%) had fewer than 10 prior messages")

    return metrics, predictions

# Results container (populated by run_evaluation)
results = {
    "metadata": {
        "evaluation_timestamp": datetime.now().isoformat(),
        "seed": SEED,
        "ml_based": True,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2@1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
    },
    "methods": {},
    "ablations": {},
    "override_ablations": {},
    "sensitivity_levels": {},
    "statistical_tests": {},
}

CSV_RESULT_COLUMNS = [
    "dataset",
    "experiment_type",
    "precision",
    "recall",
    "f1",
    "detection_mechanism_breakdown",
    "auc_mean",
    "auc_ci_low",
    "auc_ci_high",
    "f1_ci_low",
    "f1_ci_high",
    "prec_ci_low",
    "prec_ci_high",
    "rec_ci_low",
    "rec_ci_high",
]

MATURITY_DISPLAY_NAMES = {
    "personachat": "PersonaChat",
    "blended_skill_talk": "BST",
    "anthropic_hh": "HH",
}


def _empty_maturity_bin() -> Dict[str, float]:
    return {
        "n_users": 0,
        "n_messages": 0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "fpr": 0.0,
    }


def _aggregate_maturity_predictions(predictions: List[Dict], bin_name: str) -> Dict:
    bin_preds = [p for p in predictions if p.get("maturity_bin") == bin_name]
    if not bin_preds:
        return _empty_maturity_bin()
    y_true = [p["true_label"] for p in bin_preds]
    y_pred = [p["predicted_label"] for p in bin_preds]
    y_scores = [p["predicted_score"] for p in bin_preds]
    m = compute_metrics(y_true, y_pred, y_scores)
    return {
        "n_users": len({p["user_id"] for p in bin_preds}),
        "n_messages": len(bin_preds),
        "precision": m["precision"],
        "recall": m["recall"],
        "f1": m["f1"],
        "fpr": m["fpr"],
    }


def _rewrite_csv_if_needed(csv_path: str) -> None:
    """Upgrade an existing CSV header when bootstrap columns are added."""
    if not os.path.isfile(csv_path):
        return
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows or rows[0] == CSV_RESULT_COLUMNS:
        return
    upgraded = [CSV_RESULT_COLUMNS]
    for row in rows[1:]:
        upgraded.append(row + [""] * (len(CSV_RESULT_COLUMNS) - len(row)))
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(upgraded)

OVERRIDE_ABLATION_CONFIGS = {
    "overrides_off_full": (True, True, True),
    "overrides_off_no_semantic": (False, True, True),
    "overrides_off_no_linguistic": (True, False, True),
    "overrides_off_no_temporal": (True, True, False),
    "overrides_off_semantic_only": (True, False, False),
    "overrides_off_linguistic_only": (False, True, False),
    "overrides_off_temporal_only": (False, False, True),
}

# Canonical EMA decay used for the override-ablation table so its overrides_off_full
# row lands exactly on the F1-vs-λ surface (plateau optimum for BST/HH, near-optimum
# for PersonaChat). Keeps the ablation table and the λ sweep on one profile builder.
CANONICAL_LAMBDA = 0.50


def _format_tp_attribution(metrics: Dict) -> str:
    tp_attr = metrics.get("tp_attribution") or {}
    if not tp_attr:
        return "no_tp"
    return "|".join(
        f"{k}:{v['count']}({v['pct']:.1f}%)"
        for k, v in sorted(tp_attr.items())
    )


def _row_ci_completeness(row: List) -> int:
    """Count populated bootstrap / AUC columns (indices 6 onward)."""
    padded = row + [""] * max(0, len(CSV_RESULT_COLUMNS) - len(row))
    return sum(1 for value in padded[6 : len(CSV_RESULT_COLUMNS)] if str(value).strip() != "")


def _metrics_to_csv_row(dataset: str, experiment_type: str, metrics: Dict) -> List:
    return [
        dataset,
        experiment_type,
        metrics.get("precision", 0.0),
        metrics.get("recall", 0.0),
        metrics.get("f1", 0.0),
        _format_tp_attribution(metrics),
        metrics.get("auc_mean", ""),
        metrics.get("auc_ci_low", ""),
        metrics.get("auc_ci_high", ""),
        metrics.get("f1_ci_low", ""),
        metrics.get("f1_ci_high", ""),
        metrics.get("prec_ci_low", ""),
        metrics.get("prec_ci_high", ""),
        metrics.get("rec_ci_low", ""),
        metrics.get("rec_ci_high", ""),
    ]


def _load_results_csv_rows(csv_path: str) -> Dict[Tuple[str, str], List]:
    """Read evaluation_results.csv into a dict keyed by (dataset, experiment_type)."""
    _rewrite_csv_if_needed(csv_path)
    if not os.path.isfile(csv_path):
        return {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        raw_rows = list(csv.reader(f))
    if len(raw_rows) < 2:
        return {}
    keyed: Dict[Tuple[str, str], List] = {}
    for row in raw_rows[1:]:
        if len(row) < 2:
            continue
        key = (row[0], row[1])
        padded = (row + [""] * (len(CSV_RESULT_COLUMNS) - len(row)))[: len(CSV_RESULT_COLUMNS)]
        existing = keyed.get(key)
        if existing is None or _row_ci_completeness(padded) >= _row_ci_completeness(existing):
            keyed[key] = padded
    return keyed


def _write_results_csv(csv_path: str, keyed_rows: Dict[Tuple[str, str], List]) -> None:
    ordered = sorted(keyed_rows.values(), key=lambda r: (r[0], r[1]))
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_RESULT_COLUMNS)
        writer.writerows(ordered)


def append_results_csv(
    dataset: str,
    experiment_type: str,
    metrics: Dict,
    csv_path: str = "results/evaluation_results.csv",
) -> None:
    """Upsert one experiment row in results/evaluation_results.csv keyed on (dataset, experiment_type).

    Live output is separate from the frozen archived copy under
    results/archived-per-message-study/evaluation_results.csv.

    Re-running the same experiment replaces the prior row instead of appending a duplicate.
    """
    keyed = _load_results_csv_rows(csv_path)
    keyed[(dataset, experiment_type)] = _metrics_to_csv_row(dataset, experiment_type, metrics)
    _write_results_csv(csv_path, keyed)


def _print_override_ablation_summary(run_datasets: Dict) -> None:
    """Print learned-composite-only metrics (overrides_off_full) per dataset."""
    print("\n" + "=" * 80)
    print("OVERRIDE ABLATION SUMMARY (overrides_off_full — learned composite only)")
    print("=" * 80)
    off_full = results.get("override_ablations", {}).get("overrides_off_full", {})
    on_full = results.get("override_ablations", {}).get("overrides_on_full", {})
    for ds_name in run_datasets:
        m_off = off_full.get(ds_name, {}).get("metrics", {})
        m_on = on_full.get(ds_name, {}).get("metrics", {})
        print(f"\n  {ds_name}:")
        print(
            f"    overrides ON:  P={m_on.get('precision', 0):.3f} "
            f"R={m_on.get('recall', 0):.3f} F1={m_on.get('f1', 0):.3f} "
            f"TP attr: {_format_tp_attribution(m_on)}"
        )
        print(
            f"    overrides OFF: P={m_off.get('precision', 0):.3f} "
            f"R={m_off.get('recall', 0):.3f} F1={m_off.get('f1', 0):.3f} "
            f"TP attr: {_format_tp_attribution(m_off)}"
        )


def run_override_ablation_experiment(run_datasets: Dict) -> None:
    """PRIORITY 1: overrides disabled × component ablation matrix + CSV logging.

    Uses the EMA ProfileManager builder at CANONICAL_LAMBDA so this table shares one
    profile builder with the λ sensitivity sweep; overrides_off_full therefore lands
    on the F1-vs-λ surface at λ=CANONICAL_LAMBDA.
    """
    print(f"\n[5b/7] Override ablation (overrides disabled, component matrix, "
          f"EMA λ={CANONICAL_LAMBDA})...")
    results["override_ablations"] = {}
    pm_builder = _build_profile_with_pm(CANONICAL_LAMBDA)

    for dataset_name in run_datasets:
        metrics, preds = evaluate_method(
            "behaviorguard", dataset_name, datasets[dataset_name],
            overrides_enabled=True,
            profile_builder=pm_builder,
        )
        results["override_ablations"].setdefault("overrides_on_full", {})[dataset_name] = {
            "metrics": metrics,
            "predictions": preds,
        }
        append_results_csv(dataset_name, "overrides_on_full", metrics)

    for ablation_name, (sem, ling, temp) in OVERRIDE_ABLATION_CONFIGS.items():
        print(f"\n  Override ablation: {ablation_name}")
        for dataset_name in run_datasets:
            metrics, preds = evaluate_method(
                "behaviorguard",
                dataset_name,
                datasets[dataset_name],
                enable_semantic=sem,
                enable_linguistic=ling,
                enable_temporal=temp,
                overrides_enabled=False,
                profile_builder=pm_builder,
                canonical_metrics=(ablation_name == "overrides_off_full"),
            )
            results["override_ablations"].setdefault(ablation_name, {})[dataset_name] = {
                "metrics": metrics,
                "config": {"semantic": sem, "linguistic": ling, "temporal": temp},
                "predictions": preds,
            }
            append_results_csv(dataset_name, ablation_name, metrics)

    maturity_output: Dict[str, Dict] = {}
    for dataset_name in run_datasets:
        display_name = MATURITY_DISPLAY_NAMES.get(dataset_name, dataset_name)
        off_metrics = (
            results["override_ablations"]
            .get("overrides_off_full", {})
            .get(dataset_name, {})
            .get("metrics", {})
        )
        maturity_output[display_name] = off_metrics.get(
            "maturity_breakdown",
            {"cold_start": _empty_maturity_bin(), "stable": _empty_maturity_bin()},
        )
    with open("maturity_analysis.json", "w", encoding="utf-8") as f:
        json.dump(maturity_output, f, indent=2)
    print("    [OK] maturity_analysis.json written")

    _print_override_ablation_summary(run_datasets)


def _plot_lambda_sensitivity(sweep: Dict, run_datasets: Dict, out_path: str) -> Optional[str]:
    """Plot F1 vs λ per dataset. Returns saved path or None if matplotlib missing."""
    try:
        matplotlib = importlib.import_module("matplotlib")
        matplotlib.use("Agg")
        plt = importlib.import_module("matplotlib.pyplot")
    except ImportError:
        print("    [plot] matplotlib not installed; skipping figure (CSV/JSON still written).")
        return None

    decay_keys = sorted(sweep.keys(), key=float)
    xs = [float(k) for k in decay_keys]
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for ds in run_datasets:
        ys = [sweep[k][ds]["metrics"]["f1"] for k in decay_keys]
        ax.plot(xs, ys, marker="o", label=ds)
    ax.set_xlabel("EMA decay λ")
    ax.set_ylabel("F1 (overrides disabled)")
    ax.set_title("Learned-composite F1 vs EMA decay λ")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def run_lambda_sensitivity_sweep(
    run_datasets: Dict,
    lambda_decays: Optional[Tuple[float, ...]] = None,
    overrides_enabled: bool = False,
) -> None:
    """PRIORITY 4: sweep EMA decay λ from 0.0 to 1.0 (step 0.1) per dataset.

    Overrides are disabled by default so that λ's effect on the learned EMA
    semantic centroid is observable; with overrides enabled, override_3
    (λ-independent) dominates TP attribution and masks the λ response.
    """
    if lambda_decays is None:
        lambda_decays = tuple(round(v, 2) for v in np.arange(0.0, 1.0001, 0.1))
    mode = "overrides OFF" if not overrides_enabled else "overrides ON"
    print(f"\n[6b/7] Running λ sensitivity sweep ({mode}, λ=0.0..1.0 step 0.1)...")
    results["lambda_sensitivity"] = {}

    for decay_val in lambda_decays:
        decay_key = f"{decay_val:.2f}"
        builder = _build_profile_with_pm(decay_val)
        print(f"\n  λ = {decay_key}")
        for dataset_name in run_datasets:
            metrics, _ = evaluate_method(
                "behaviorguard",
                dataset_name,
                datasets[dataset_name],
                profile_builder=builder,
                overrides_enabled=overrides_enabled,
            )
            results["lambda_sensitivity"].setdefault(decay_key, {})[dataset_name] = {
                "metrics": metrics
            }
            append_results_csv(dataset_name, f"lambda_{decay_key}_{'off' if not overrides_enabled else 'on'}", metrics)

    print("\n  λ sensitivity table (F1):")
    decay_keys = sorted(results["lambda_sensitivity"].keys(), key=float)
    for decay_key in decay_keys:
        cells = " | ".join(
            f"{ds}={results['lambda_sensitivity'][decay_key][ds]['metrics']['f1']:.3f}"
            for ds in run_datasets
        )
        print(f"    λ={decay_key}: {cells}")

    # Report per-dataset argmax λ to show optimum is interior / stable
    print("\n  Optimal λ per dataset (max F1):")
    for ds in run_datasets:
        best_key = max(decay_keys, key=lambda k: results["lambda_sensitivity"][k][ds]["metrics"]["f1"])
        best_f1 = results["lambda_sensitivity"][best_key][ds]["metrics"]["f1"]
        print(f"    {ds}: λ*={best_key} (F1={best_f1:.3f})")

    fig_path = _plot_lambda_sensitivity(
        results["lambda_sensitivity"], run_datasets, os.path.join("paper", "figures", "lambda_sensitivity.png")
    )
    if fig_path:
        print(f"    [plot] F1-vs-λ figure saved to {fig_path}")


def convert_to_json_serializable(obj):
    """Recursively convert numpy types to Python types."""
    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    else:
        return obj


def run_evaluation(dataset_filter=None):
    """Run full evaluation pipeline. dataset_filter: set of names or None for all."""
    global results
    results["metadata"]["evaluation_timestamp"] = datetime.now().isoformat()

    run_datasets = (
        {k: v for k, v in datasets.items() if k in dataset_filter}
        if dataset_filter
        else datasets
    )

    # [3/7] Evaluate TurnShift (full system)
    print("\n[3/7] Evaluating TurnShift (full system)...")
    for dataset_name in run_datasets:
        metrics, preds = evaluate_method("behaviorguard", dataset_name, datasets[dataset_name])
        if "behaviorguard" not in results["methods"]:
            results["methods"]["behaviorguard"] = {}
        results["methods"]["behaviorguard"][dataset_name] = {
            "metrics": metrics,
            "predictions": preds
        }

    # [4/7] Evaluate baselines
    print("\n[4/7] Evaluating baselines...")
    for method in ["rule_based", "isolation_forest", "autoencoder", "content_safety"]:
        for dataset_name in run_datasets:
            metrics, preds = evaluate_method(method, dataset_name, datasets[dataset_name])
            if method not in results["methods"]:
                results["methods"][method] = {}
            results["methods"][method][dataset_name] = {
                "metrics": metrics,
                "predictions": preds
            }

    # [5/7] Ablation studies
    print("\n[5/7] Running ablation studies...")
    ablation_configs = {
        "no_semantic": (False, True, True),
        "no_linguistic": (True, False, True),
        "no_temporal": (True, True, False),
        "semantic_only": (True, False, False),
        "linguistic_only": (False, True, False),
        "temporal_only": (False, False, True),
    }

    for ablation_name, (sem, ling, temp) in ablation_configs.items():
        print(f"\n  Ablation: {ablation_name}")
        for dataset_name in run_datasets:
            metrics, preds = evaluate_method(
                "behaviorguard", dataset_name, datasets[dataset_name],
                enable_semantic=sem, enable_linguistic=ling, enable_temporal=temp
            )
            if ablation_name not in results["ablations"]:
                results["ablations"][ablation_name] = {}
            results["ablations"][ablation_name][dataset_name] = {
                "metrics": metrics,
                "config": {"semantic": sem, "linguistic": ling, "temporal": temp}
            }

    run_override_ablation_experiment(run_datasets)

    # [6/7] Sensitivity level analysis
    print("\n[6/7] Running sensitivity level analysis...")
    sensitivity_levels = ["low", "medium", "high", "maximum"]

    for sensitivity in sensitivity_levels:
        print(f"\n  Sensitivity: {sensitivity}")
        config = SystemConfig(
            sensitivity_level=sensitivity,
            deployment_context="enterprise",
            enable_temporal_scoring=True,
            enable_linguistic_scoring=True,
            enable_semantic_scoring=True,
            overrides_enabled=True,
        )

        for dataset_name in run_datasets:
            metrics, preds = evaluate_method(
                "behaviorguard", dataset_name, datasets[dataset_name], config=config
            )
            if sensitivity not in results["sensitivity_levels"]:
                results["sensitivity_levels"][sensitivity] = {}
            results["sensitivity_levels"][sensitivity][dataset_name] = {
                "metrics": metrics
            }

    # [6b/7] Lambda sensitivity (EMA decay) across all datasets
    run_lambda_sensitivity_sweep(run_datasets)

    # [7/7] Statistical significance tests
    print("\n[7/7] Computing statistical significance tests...")

    def compute_statistical_tests(method1_preds: List[Dict], method2_preds: List[Dict]) -> Dict:
        """Compute statistical tests between two methods."""
        method1_correct = [
            float(p["true_label"] == p["predicted_label"]) for p in method1_preds
        ]
        method2_correct = [
            float(p["true_label"] == p["predicted_label"]) for p in method2_preds
        ]
        t_stat, p_value = stats.ttest_rel(method1_correct, method2_correct)
        diff = np.array(method1_correct) - np.array(method2_correct)
        cohens_d = np.mean(diff) / (np.std(diff) + 1e-8)
        ci = stats.t.interval(0.95, len(diff)-1, loc=np.mean(diff), scale=stats.sem(diff))
        return {
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "cohens_d": float(cohens_d),
            "confidence_interval_95": [float(ci[0]), float(ci[1])],
            "significant": p_value < 0.05,
            "practical_significance": abs(cohens_d) > 0.2
        }

    # Compare TurnShift vs baselines
    for baseline in ["rule_based", "isolation_forest", "autoencoder", "content_safety"]:
        comparison_name = f"behaviorguard_vs_{baseline}"
        results["statistical_tests"][comparison_name] = {}
        for dataset_name in run_datasets:
            bg_preds = results["methods"]["behaviorguard"][dataset_name]["predictions"]
            baseline_preds = results["methods"][baseline][dataset_name]["predictions"]
            min_len = min(len(bg_preds), len(baseline_preds))
            bg_preds = bg_preds[:min_len]
            baseline_preds = baseline_preds[:min_len]
            test_results = compute_statistical_tests(bg_preds, baseline_preds)
            results["statistical_tests"][comparison_name][dataset_name] = test_results
            print(f"  {comparison_name} on {dataset_name}: "
                  f"p={test_results['p_value']:.4f}, d={test_results['cohens_d']:.3f}")

    return results


def run_evaluation_override_ablations_only(dataset_filter=None) -> Dict:
    """Run only the PRIORITY 1 override ablation matrix (EMA builder, faster path).

    overrides_on_full is computed inside the ablation with the canonical EMA builder,
    so the redundant non-EMA [3/7] TurnShift run is intentionally skipped here.
    """
    global results
    results["metadata"]["evaluation_timestamp"] = datetime.now().isoformat()
    run_datasets = (
        {k: v for k, v in datasets.items() if k in dataset_filter}
        if dataset_filter
        else datasets
    )
    run_override_ablation_experiment(run_datasets)
    return results


def run_evaluation_diagnostic_harness_only(dataset_filter=None) -> Dict:
    """Run s_ling saturation audit + λ-sweep via real analyzers (80/20 split)."""
    from scripts.diagnostic_harness import run_full_diagnostic

    path_map = {
        "personachat": ROOT / "datasets/personachat_processed_corrected.json",
        "blended_skill_talk": ROOT / "datasets/blended_skill_talk_processed_corrected.json",
        "anthropic_hh": ROOT / "datasets/anthropic_hh_processed_corrected.json",
    }
    if dataset_filter:
        keys = [dataset_filter] if isinstance(dataset_filter, str) else list(dataset_filter)
    else:
        keys = ["personachat"]

    harness_results: Dict[str, Any] = {}
    for key in keys:
        ds_path = path_map.get(key)
        if ds_path is None or not ds_path.exists():
            print(f"  [SKIP] no corrected dataset for {key}")
            continue
        print(f"\n[diagnostic harness] {key} ...")
        harness_results[key] = run_full_diagnostic(dataset_path=ds_path)

    out_path = ROOT / "results" / "diagnostic_harness_output.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(convert_to_json_serializable(harness_results), indent=2),
        encoding="utf-8",
    )
    print(f"\n[OK] Diagnostic harness results saved to {out_path}")
    return {"diagnostic_harness": harness_results}


ROOT = Path(__file__).resolve().parent


def run_evaluation_lambda_sweep_only(dataset_filter=None) -> Dict:
    """Run only the PRIORITY 4 λ sensitivity sweep (faster path)."""
    global results
    results["metadata"]["evaluation_timestamp"] = datetime.now().isoformat()
    run_datasets = (
        {k: v for k, v in datasets.items() if k in dataset_filter}
        if dataset_filter
        else datasets
    )
    run_lambda_sensitivity_sweep(run_datasets)
    return results


# Run when executed directly
if __name__ == "__main__":
    if os.environ.get("BG_DIAGNOSTIC_HARNESS"):
        run_evaluation_diagnostic_harness_only(
            dataset_filter=os.environ.get("BG_DIAGNOSTIC_DATASET")
        )
    elif os.environ.get("BG_OVERRIDE_ABLATION_ONLY"):
        run_evaluation_override_ablations_only()
    elif os.environ.get("BG_LAMBDA_SWEEP_ONLY"):
        run_evaluation_lambda_sweep_only()
    else:
        run_evaluation()
    output_file = "full_evaluation_results.json"
    results_serializable = convert_to_json_serializable(results)
    with open(output_file, "w") as f:
        json.dump(results_serializable, f, indent=2)
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nMethods evaluated: {len(results['methods'])}")
    print(f"Ablation studies: {len(results['ablations'])}")
    print(f"Override ablations: {len(results.get('override_ablations', {}))}")
    print(f"Sensitivity levels: {len(results['sensitivity_levels'])}")
    print(f"Statistical tests: {len(results['statistical_tests'])}")
    print(f"\n[OK] Full results saved to {output_file}")
