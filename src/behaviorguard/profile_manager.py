"""
BehaviorGuard Profile Manager
==============================
Implements Algorithm 1 from the paper: incremental user profile updates
using exponential moving averages (EMA) and decay-weighted aggregation.

The profile manager is the component that bridges raw conversation history
and the UserProfile consumed by the evaluators.  It provides:

  - ProfileManager.update_profile()    — update an existing profile with a
                                         new message (online / streaming use)
  - ProfileManager.build_from_history() — build a fresh profile from a list
                                          of historical messages (batch use)
  - ProfileManager.cold_start_profile() — minimal profile for new users
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import List, Literal, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator

from behaviorguard.models import (
    LinguisticProfile,
    OperationalProfile,
    SemanticProfile,
    TemporalProfile,
    UserProfile,
)


# ─────────────────────────────────────────────────────────────────────────────
# Internal data structures
# ─────────────────────────────────────────────────────────────────────────────

class _RunningStats:
    """
    Online Welford algorithm for mean and variance.

    Allows incrementally updating statistics without storing all samples.
    Reference: Welford, B.P. (1962). Technometrics, 4(3), 419-420.
    """

    def __init__(self, n: int = 0, mean: float = 0.0, m2: float = 0.0):
        self.n = n
        self.mean = mean
        self.m2 = m2  # sum of squared deviations from mean

    def update(self, value: float) -> None:
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        delta2 = value - self.mean
        self.m2 += delta * delta2

    @property
    def variance(self) -> float:
        return self.m2 / max(self.n - 1, 1)

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)

    def to_dict(self) -> dict:
        return {"n": self.n, "mean": self.mean, "m2": self.m2}

    @classmethod
    def from_dict(cls, d: dict) -> "_RunningStats":
        return cls(n=d["n"], mean=d["mean"], m2=d["m2"])


class _EmbeddingAccumulator:
    """
    Maintains an EMA-weighted embedding centroid using standard exponential
    moving average: µ_new = λ * µ_old + (1 - λ) * e_new.

    decay (λ) controls how much history is retained. Lower λ weights recent
    messages more heavily (more responsive); higher λ weights history more
    (more stable). λ=0.95 gives ~20-message effective window.
    """

    def __init__(self, decay: float = 0.95):
        self.decay = decay
        self._centroid: Optional[np.ndarray] = None
        self._count: int = 0

    def update(self, embedding: np.ndarray) -> None:
        e = embedding.astype(np.float64)
        if self._centroid is None:
            self._centroid = e.copy()
        else:
            # Standard EMA: µ_new = λ * µ_old + (1-λ) * e_new
            self._centroid = self.decay * self._centroid + (1.0 - self.decay) * e
        self._count += 1

    @property
    def centroid(self) -> Optional[np.ndarray]:
        if self._centroid is None:
            return None
        norm = np.linalg.norm(self._centroid)
        return self._centroid / norm if norm > 0 else self._centroid

    @property
    def count(self) -> int:
        return self._count


# ─────────────────────────────────────────────────────────────────────────────
# Public message DTO
# ─────────────────────────────────────────────────────────────────────────────

OperationRisk = Literal["low", "medium", "high", "critical"]


class MessageRecord(BaseModel):
    """
    A single observed user message passed to the profile manager.

    JSON / model_validate round-trip supported (see operation_risk).
    """

    model_config = ConfigDict(extra="forbid", frozen=False)

    text: str
    timestamp: str
    session_id: str = "default"
    is_anomaly: bool = False
    operation_risk: Optional[OperationRisk] = None

    @field_validator("operation_risk", mode="before")
    @classmethod
    def _normalize_risk(cls, v: object) -> str | None:
        if v is None or v == "":
            return None
        if isinstance(v, str):
            s = v.strip().lower()
            if s in ("low", "medium", "high", "critical"):
                return s
        return None

    def parsed_timestamp(self) -> datetime:
        ts = datetime.fromisoformat(self.timestamp)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts


# ─────────────────────────────────────────────────────────────────────────────
# Profile Manager
# ─────────────────────────────────────────────────────────────────────────────

class ProfileManager:
    """
    Manages incremental and batch creation of UserProfile objects.

    This is the implementation of Algorithm 1 from the BehaviorGuard paper:

        Input:  m_new (new message), t (timestamp), P_u (current profile)
        Output: P_u' (updated profile)

        1. e ← encode(m_new)
        2. ling ← extract_linguistic_features(m_new)
        3. w ← decay^(interaction_count)
        4. μ_u ← (μ_u · w_sum + e · w) / (w_sum + w)   # EMA centroid
        5. update_second_moment(Σ_u, e, w)
        6. update_stats(P_u.ling_stats)                  # Welford online stats
        7. temporal ← update_temporal(P_u, t)
        8. return P_u'

    The manager deliberately avoids storing raw message text; it only retains
    aggregate statistics (embedding centroid, running stats, counters).

    Args:
        decay:           EMA decay factor λ ∈ (0, 1].  Default 0.95 gives
                         approximately 20-message effective window.
        embedding_model: Sentence-transformer model name.  If None, profile
                         learning from text is disabled (you must supply
                         pre-computed embeddings via update_with_embedding).
    """

    def __init__(
        self,
        decay: float = 0.95,
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.decay = decay
        self._model = None
        self._embedding_model_name = embedding_model

    # ── lazy model loading ────────────────────────────────────────────────────

    @property
    def model(self):
        """Lazily load sentence transformer model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore[import]
                self._model = SentenceTransformer(self._embedding_model_name)
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is required for ProfileManager text encoding.\n"
                    "Install with: pip install sentence-transformers"
                ) from exc
        return self._model

    # ── public API ────────────────────────────────────────────────────────────

    def build_from_history(
        self,
        user_id: str,
        messages: List[MessageRecord],
        account_age_days: int = 0,
    ) -> UserProfile:
        """
        Build a complete UserProfile from a list of historical messages.

        Only messages with is_anomaly=False are used to construct the normal
        behavioral baseline.  Anomalous messages are excluded (they would
        corrupt the centroid and statistics).

        Args:
            user_id:          Unique user identifier.
            messages:         Ordered list of MessageRecord objects.
            account_age_days: How old the account is (metadata only).

        Returns:
            A UserProfile ready to be used by BehaviorGuardEvaluatorML.
        """
        normal = [m for m in messages if not m.is_anomaly]
        if not normal:
            return self.cold_start_profile(user_id, account_age_days)

        embedder = _EmbeddingAccumulator(decay=self.decay)
        len_tokens = _RunningStats()
        len_chars = _RunningStats()
        lex_div = _RunningStats()
        formality = _RunningStats()
        politeness = _RunningStats()
        question_ratio = _RunningStats()

        hours: List[int] = []
        days: List[str] = []
        gaps: List[float] = []
        sessions: dict[str, list[datetime]] = {}

        prev_ts: Optional[datetime] = None

        texts = [m.text for m in normal]
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

        for i, msg in enumerate(normal):
            ts = msg.parsed_timestamp()

            # Embedding update
            embedder.update(embeddings[i])

            # Linguistic stats (lightweight proxies — no external NLP required)
            words = msg.text.split()
            n_words = len(words)
            n_chars = len(msg.text)
            uniq_ratio = len(set(w.lower() for w in words)) / max(n_words, 1)
            q_ratio = msg.text.count("?") / max(msg.text.count(".") + 1, 1)

            len_tokens.update(n_words)
            len_chars.update(n_chars)
            lex_div.update(uniq_ratio)
            # Simple heuristics for formality (sentence length proxy) and
            # politeness (presence of polite markers)
            formality_proxy = min(1.0, n_words / 50.0)
            politeness_proxy = float(
                any(w in msg.text.lower() for w in ("please", "thank", "sorry", "could you", "would you"))
            )
            formality.update(formality_proxy)
            politeness.update(politeness_proxy)
            question_ratio.update(q_ratio)

            # Temporal stats
            hours.append(ts.hour)
            days.append(ts.strftime("%A"))
            if prev_ts is not None:
                gap = (ts - prev_ts).total_seconds()
                if 0 < gap < 3600:  # cap gaps at 1 hour (session boundary)
                    gaps.append(gap)
            prev_ts = ts

            # Session tracking
            sessions.setdefault(msg.session_id, []).append(ts)

        # Compute session statistics
        session_lengths = []
        for sess_timestamps in sessions.values():
            if len(sess_timestamps) >= 2:
                sess_ts = sorted(sess_timestamps)
                duration_min = (sess_ts[-1] - sess_ts[0]).total_seconds() / 60.0
                session_lengths.append(duration_min)

        avg_session_min = float(np.mean(session_lengths)) if session_lengths else 30.0
        longest_session_min = float(max(session_lengths)) if session_lengths else 60.0
        avg_msgs_per_session = len(normal) / max(len(sessions), 1)
        typical_gap_s = float(np.mean(gaps)) if gaps else 30.0

        # Active hours / days (most frequent)
        hour_counts: dict[int, int] = {}
        for h in hours:
            hour_counts[h] = hour_counts.get(h, 0) + 1
        active_hours = sorted(hour_counts, key=lambda h: -hour_counts[h])[:6]

        day_counts: dict[str, int] = {}
        for d in days:
            day_counts[d] = day_counts.get(d, 0) + 1
        active_days = sorted(day_counts, key=lambda d: -day_counts[d])[:5]

        # Top topic words (frequency-ranked)
        all_words = [
            w.lower()
            for m in normal
            for w in m.text.split()
            if len(w) > 4 and w.isalpha()
        ]
        word_freq: dict[str, int] = {}
        for w in all_words:
            word_freq[w] = word_freq.get(w, 0) + 1
        top_words = sorted(word_freq, key=lambda w: -word_freq[w])[:10]

        uses_code = any("def " in m.text or "```" in m.text for m in normal)
        uses_tech = any(
            any(kw in m.text.lower() for kw in ("api", "code", "function", "class", "import"))
            for m in normal
        )

        last_ts = max(m.parsed_timestamp() for m in normal).isoformat()

        has_requested_sensitive_ops = any(
            m.operation_risk in ("high", "critical")
            for m in normal
            if m.operation_risk is not None
        )

        centroid = embedder.centroid
        return UserProfile(
            user_id=user_id,
            account_age_days=account_age_days,
            total_interactions=len(normal),
            semantic_profile=SemanticProfile(
                typical_topics=top_words or ["general"],
                primary_domains=["learned_from_history"],
                topic_diversity_score=min(1.0, len(set(top_words)) / 10.0),
                embedding_centroid_summary=(
                    f"EMA centroid (decay={self.decay}) over {embedder.count} messages"
                ),
                embedding_centroid=centroid.tolist() if centroid is not None else None,
            ),
            linguistic_profile=LinguisticProfile(
                avg_message_length_tokens=len_tokens.mean,
                avg_message_length_chars=len_chars.mean,
                lexical_diversity_mean=lex_div.mean,
                lexical_diversity_std=max(lex_div.std, 0.01),
                formality_score_mean=formality.mean,
                formality_score_std=max(formality.std, 0.01),
                politeness_score_mean=politeness.mean,
                politeness_score_std=max(politeness.std, 0.01),
                question_ratio_mean=question_ratio.mean,
                uses_technical_vocabulary=uses_tech,
                uses_code_blocks=uses_code,
                primary_languages=["en"],
                typical_sentence_complexity=(
                    "complex" if len_tokens.mean > 40
                    else "moderate" if len_tokens.mean > 15
                    else "simple"
                ),
            ),
            temporal_profile=TemporalProfile(
                typical_session_duration_minutes=avg_session_min,
                typical_inter_message_gap_seconds=typical_gap_s,
                most_active_hours_utc=active_hours or list(range(9, 18)),
                most_active_days_of_week=active_days or ["Monday", "Tuesday", "Wednesday"],
                average_messages_per_session=avg_msgs_per_session,
                longest_session_duration_minutes=longest_session_min,
                typical_session_frequency_per_week=min(len(sessions) / max(account_age_days / 7.0, 1), 14.0),
                last_activity_timestamp=last_ts,
            ),
            operational_profile=OperationalProfile(
                common_intent_types=["information_seeking"],
                tools_used_historically=["search"],
                has_requested_sensitive_ops=has_requested_sensitive_ops,
                typical_risk_level="low",
            ),
        )

    def update_profile(
        self,
        profile: UserProfile,
        message: MessageRecord,
    ) -> UserProfile:
        """
        Incrementally update an existing profile with a single new message.

        This is the online variant of Algorithm 1.  It returns a *new*
        UserProfile object (profiles are immutable Pydantic models).

        The update uses the EMA decay so that:
            μ_u ← (μ_u · n + e_new · 1) / (n + 1)
        with the decay factor applied to weight recent messages more.

        Args:
            profile: The current UserProfile.
            message: The new (non-anomalous) message to incorporate.

        Returns:
            Updated UserProfile with statistics reflecting the new message.
        """
        if message.is_anomaly:
            # Never update the normal baseline with anomalous messages
            return profile

        text = message.text
        words = text.split()
        n_words = len(words)
        n_chars = len(text)
        n = profile.total_interactions

        # Update linguistic running stats (online Welford)
        lp = profile.linguistic_profile

        def _welford_update(mean: float, m2_proxy: float, n: int, new_val: float):
            """Returns (new_mean, new_std) using simplified online update."""
            new_n = n + 1
            delta = new_val - mean
            new_mean = mean + delta / new_n
            # Approximate std update (proper Welford needs m2 stored)
            delta2 = new_val - new_mean
            new_m2_proxy = m2_proxy + abs(delta * delta2)
            new_std = math.sqrt(new_m2_proxy / max(new_n - 1, 1))
            return new_mean, max(new_std, 0.001)

        uniq_ratio = len(set(w.lower() for w in words)) / max(n_words, 1)
        q_ratio = text.count("?") / max(text.count(".") + 1, 1)
        formality_proxy = min(1.0, n_words / 50.0)
        politeness_proxy = float(
            any(w in text.lower() for w in ("please", "thank", "sorry", "could you"))
        )

        new_avg_tokens, _ = _welford_update(lp.avg_message_length_tokens, 0.0, n, float(n_words))
        new_avg_chars, _ = _welford_update(lp.avg_message_length_chars, 0.0, n, float(n_chars))
        new_lex_mean, new_lex_std = _welford_update(
            lp.lexical_diversity_mean, lp.lexical_diversity_std ** 2 * max(n - 1, 1), n, uniq_ratio
        )
        new_form_mean, new_form_std = _welford_update(
            lp.formality_score_mean, lp.formality_score_std ** 2 * max(n - 1, 1), n, formality_proxy
        )
        new_pol_mean, new_pol_std = _welford_update(
            lp.politeness_score_mean, lp.politeness_score_std ** 2 * max(n - 1, 1), n, politeness_proxy
        )
        new_qr_mean, _ = _welford_update(lp.question_ratio_mean, 0.0, n, q_ratio)

        # Update semantic topics (add new top words if novel)
        all_topic_words = list(profile.semantic_profile.typical_topics)
        new_words = [w.lower() for w in words if len(w) > 4 and w.isalpha()]
        for w in new_words[:3]:
            if w not in all_topic_words:
                all_topic_words.append(w)
        all_topic_words = all_topic_words[:15]  # cap at 15

        # Update EMA centroid
        e_new = self.model.encode(text, convert_to_numpy=True, show_progress_bar=False)
        old_centroid = profile.semantic_profile.embedding_centroid
        if old_centroid is not None:
            mu_old = np.array(old_centroid, dtype=np.float64)
            mu_new = self.decay * mu_old + (1.0 - self.decay) * e_new
        else:
            mu_new = e_new.astype(np.float64)
        norm = np.linalg.norm(mu_new)
        new_centroid = (mu_new / norm) if norm > 0 else mu_new

        # Update temporal profile
        ts = message.parsed_timestamp()
        tp = profile.temporal_profile
        active_hours = list(tp.most_active_hours_utc)
        if ts.hour not in active_hours and len(active_hours) < 12:
            active_hours.append(ts.hour)
        active_days = list(tp.most_active_days_of_week)
        day_name = ts.strftime("%A")
        if day_name not in active_days and len(active_days) < 7:
            active_days.append(day_name)

        return UserProfile(
            user_id=profile.user_id,
            account_age_days=profile.account_age_days,
            total_interactions=n + 1,
            semantic_profile=SemanticProfile(
                typical_topics=all_topic_words,
                primary_domains=profile.semantic_profile.primary_domains,
                topic_diversity_score=min(1.0, len(set(all_topic_words)) / 10.0),
                embedding_centroid_summary=(
                    f"EMA centroid over {n + 1} messages "
                    f"(decay={self.decay})"
                ),
                embedding_centroid=new_centroid.tolist(),
            ),
            linguistic_profile=LinguisticProfile(
                avg_message_length_tokens=new_avg_tokens,
                avg_message_length_chars=new_avg_chars,
                lexical_diversity_mean=new_lex_mean,
                lexical_diversity_std=new_lex_std,
                formality_score_mean=new_form_mean,
                formality_score_std=new_form_std,
                politeness_score_mean=new_pol_mean,
                politeness_score_std=new_pol_std,
                question_ratio_mean=new_qr_mean,
                uses_technical_vocabulary=lp.uses_technical_vocabulary or (
                    any(kw in text.lower() for kw in ("api", "code", "function", "class", "import"))
                ),
                uses_code_blocks=lp.uses_code_blocks or ("def " in text or "```" in text),
                primary_languages=lp.primary_languages,
                typical_sentence_complexity=lp.typical_sentence_complexity,
            ),
            temporal_profile=TemporalProfile(
                typical_session_duration_minutes=tp.typical_session_duration_minutes,
                typical_inter_message_gap_seconds=tp.typical_inter_message_gap_seconds,
                most_active_hours_utc=active_hours,
                most_active_days_of_week=active_days,
                average_messages_per_session=tp.average_messages_per_session,
                longest_session_duration_minutes=tp.longest_session_duration_minutes,
                typical_session_frequency_per_week=tp.typical_session_frequency_per_week,
                last_activity_timestamp=ts.isoformat(),
            ),
            operational_profile=profile.operational_profile,
        )

    def cold_start_profile(
        self,
        user_id: str,
        account_age_days: int = 0,
    ) -> UserProfile:
        """
        Create a minimal profile for a brand-new user with no history.

        The evaluator's ColdStartHandler will detect insufficient history
        (total_interactions < 20) and apply conservative thresholds.

        Args:
            user_id:          Unique user identifier.
            account_age_days: Account age in days (0 for brand-new).

        Returns:
            Minimal UserProfile with population-level priors.
        """
        return UserProfile(
            user_id=user_id,
            account_age_days=account_age_days,
            total_interactions=0,
            semantic_profile=SemanticProfile(
                typical_topics=["general"],
                primary_domains=["unknown"],
                topic_diversity_score=0.5,
                embedding_centroid_summary="No history — cold start",
            ),
            linguistic_profile=LinguisticProfile(
                avg_message_length_tokens=20.0,
                avg_message_length_chars=100.0,
                lexical_diversity_mean=0.70,
                lexical_diversity_std=0.10,
                formality_score_mean=0.50,
                formality_score_std=0.15,
                politeness_score_mean=0.60,
                politeness_score_std=0.15,
                question_ratio_mean=0.30,
                uses_technical_vocabulary=False,
                uses_code_blocks=False,
                primary_languages=["en"],
                typical_sentence_complexity="moderate",
            ),
            temporal_profile=TemporalProfile(
                typical_session_duration_minutes=30.0,
                typical_inter_message_gap_seconds=30.0,
                most_active_hours_utc=list(range(9, 18)),
                most_active_days_of_week=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
                average_messages_per_session=8.0,
                longest_session_duration_minutes=60.0,
                typical_session_frequency_per_week=3.0,
                last_activity_timestamp=datetime.now(timezone.utc).isoformat(),
            ),
            operational_profile=OperationalProfile(
                common_intent_types=["information_seeking"],
                tools_used_historically=[],
                has_requested_sensitive_ops=False,
                typical_risk_level="low",
            ),
        )
