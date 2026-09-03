"""ML-based semantic analyzer using sentence embeddings."""

from __future__ import annotations

import importlib.util

import numpy as np

from turnshift.models import (
    CurrentMessage,
    SemanticAnalysisResult,
    SemanticProfile,
    SystemConfig,
)
from turnshift.utils.torch_device import embedding_device

# Resolved without importing: sentence-transformers drags in torch and transformers,
# and `import turnshift` must not pay for that. The real import happens in __init__.
TRANSFORMERS_AVAILABLE = importlib.util.find_spec("sentence_transformers") is not None


class SemanticAnalyzerML:
    """
    ML-based semantic analyzer using neural embeddings.

    Supports cosine distance (default) or Mahalanobis distance when the profile
    carries a tracked covariance matrix from ProfileManager.
    """

    def __init__(self, model_name: str | None = None):
        from turnshift.embedding_config import (
            EMBEDDING_MODEL_HF_ID,
            EMBEDDING_MODEL_NAME,
            EMBEDDING_MODEL_REVISION,
            load_sentence_transformer,
        )

        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers not installed. "
                'Install with: pip install "turnshift[ml]"'
            )

        self.model_name = model_name or EMBEDDING_MODEL_NAME
        if model_name is None or model_name in (EMBEDDING_MODEL_NAME, EMBEDDING_MODEL_HF_ID):
            self.model = load_sentence_transformer()
        else:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(model_name, device=embedding_device())
        self._embedding_revision = EMBEDDING_MODEL_REVISION
        self._embedding_cache: dict[str, np.ndarray] = {}
        self._sigma_inv_cache: dict[tuple, np.ndarray] = {}

    def analyze(
        self,
        current_message: CurrentMessage,
        semantic_profile: SemanticProfile,
        system_config: SystemConfig | None = None,
        total_interactions: int = 0,
    ) -> SemanticAnalysisResult:
        config = system_config or SystemConfig(
            sensitivity_level="medium",
            deployment_context="enterprise",
            overrides_enabled=True,
        )
        current_embedding = self._get_embedding(current_message.text)

        if config.semantic_scoring_mode == "mahalanobis":
            score, detail = self._mahalanobis_score(
                current_embedding,
                semantic_profile,
                total_interactions,
                config.mahalanobis_shrinkage,
            )
            reasoning = (
                f"Mahalanobis semantic deviation (d²={detail:.3f}, n={semantic_profile.embedding_sample_count})."
            )
            factors = [f"Mahalanobis d²={detail:.3f}"]
        else:
            profile_embedding = self._compute_profile_centroid(semantic_profile)
            cosine_sim = self._cosine_similarity(current_embedding, profile_embedding)
            cosine_distance = 1.0 - cosine_sim
            score = min(1.0, cosine_distance / 2.0)
            reasoning = "Message embedding compared to user centroid via cosine distance."
            factors = [f"Cosine distance: {cosine_distance:.3f}"]

        if current_message.requested_operation.risk_classification == "critical":
            score = min(1.0, score * 1.2)

        if current_message.message_sequence_in_session > 1 and score > 0.5:
            score = min(1.0, score + 0.1)
            factors.append("Mid-session semantic shift detected")

        return SemanticAnalysisResult(
            score=score,
            reasoning=reasoning,
            contributing_factors=factors,
        )

    def _mahalanobis_score(
        self,
        embedding: np.ndarray,
        profile: SemanticProfile,
        total_interactions: int,
        shrinkage: float,
    ) -> tuple[float, float]:
        e = embedding.astype(np.float64)
        d = e.shape[0]
        n = profile.embedding_sample_count or total_interactions

        if profile.embedding_mean is not None:
            mu = np.array(profile.embedding_mean, dtype=np.float64)
        elif profile.embedding_centroid is not None:
            mu = np.array(profile.embedding_centroid, dtype=np.float64)
        else:
            return 0.0, 0.0

        if profile.embedding_covariance is not None and len(profile.embedding_covariance) == d * d:
            sigma = np.array(profile.embedding_covariance, dtype=np.float64).reshape(d, d)
        else:
            sigma = np.eye(d) * 1e-4

        reg = shrinkage if n < 10 else shrinkage * 0.1
        sigma_reg = sigma + reg * np.eye(d)
        # Cache inverse per profile object (stable within a user's test window).
        cache_key = (id(profile), reg, d)
        sigma_inv = self._sigma_inv_cache.get(cache_key)
        if sigma_inv is None:
            try:
                sigma_inv = np.linalg.inv(sigma_reg)
            except np.linalg.LinAlgError:
                sigma_inv = np.linalg.pinv(sigma_reg)
            self._sigma_inv_cache[cache_key] = sigma_inv
        diff = e - mu
        maha_sq = float(diff @ sigma_inv @ diff)
        score = 1.0 - float(np.exp(-0.5 * maha_sq / max(d, 1)))
        return min(1.0, max(0.0, score)), maha_sq

    def encode_message(self, text: str) -> np.ndarray:
        embedding = self._get_embedding(text)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            return embedding / norm
        return embedding

    def _get_embedding(self, text: str) -> np.ndarray:
        if text in self._embedding_cache:
            return self._embedding_cache[text]
        embedding = self.model.encode(text, convert_to_numpy=True)
        self._embedding_cache[text] = embedding
        return embedding

    def _compute_profile_centroid(self, semantic_profile: SemanticProfile) -> np.ndarray:
        dim = self.model.get_sentence_embedding_dimension()
        if semantic_profile.embedding_centroid is not None:
            arr = np.array(semantic_profile.embedding_centroid, dtype=np.float64)
            norm = np.linalg.norm(arr)
            return arr / norm if norm > 0 else np.zeros(dim)
        if not semantic_profile.typical_topics:
            return np.zeros(dim)
        topic_embeddings = [
            self._get_embedding(topic) for topic in semantic_profile.typical_topics
        ]
        centroid = np.mean(topic_embeddings, axis=0)
        norm = np.linalg.norm(centroid)
        return centroid / norm if norm > 0 else centroid

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        similarity = np.dot(vec1 / norm1, vec2 / norm2)
        return float(np.clip(similarity, -1.0, 1.0))

    def learn_profile_from_history(self, message_history: list[str]) -> np.ndarray:
        if not message_history:
            return np.zeros(self.model.get_sentence_embedding_dimension())
        embeddings = self.model.encode(message_history, convert_to_numpy=True)
        weights = np.exp(np.linspace(-1, 0, len(embeddings)))
        weights = weights / weights.sum()
        centroid = np.average(embeddings, axis=0, weights=weights)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        return centroid
