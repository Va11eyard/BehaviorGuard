"""ML-based semantic analyzer using sentence embeddings and cosine distance."""

import numpy as np
from typing import List, Optional

try:
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from behaviorguard.models import (
    CurrentMessage,
    SemanticAnalysisResult,
    SemanticProfile,
)


class SemanticAnalyzerML:
    """
    ML-based semantic analyzer using neural embeddings.
    
    Uses sentence transformers to compute embeddings and measures
    cosine distance from user's historical semantic centroid.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize semantic analyzer with sentence transformer model.
        
        Args:
            model_name: Name of sentence transformer model to use
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
        
        self.model = SentenceTransformer(model_name)
        self._embedding_cache = {}

    def analyze(
        self, current_message: CurrentMessage, semantic_profile: SemanticProfile
    ) -> SemanticAnalysisResult:
        """
        Detect semantic anomalies using cosine distance in embedding space.

        Args:
            current_message: Current message being evaluated
            semantic_profile: User's semantic behavioral profile

        Returns:
            SemanticAnalysisResult with score, reasoning, and contributing factors
        """
        # Compute embedding for current message
        current_embedding = self._get_embedding(current_message.text)
        
        # Get user's semantic centroid from profile
        # In production, this would be pre-computed from user's message history
        profile_embedding = self._compute_profile_centroid(semantic_profile)
        
        # Compute cosine distance (1 - cosine similarity)
        cosine_sim = self._cosine_similarity(current_embedding, profile_embedding)
        cosine_distance = 1.0 - cosine_sim
        
        # Map distance to anomaly score [0, 1]
        # Distance of 0 (identical) -> score 0
        # Distance of 1 (orthogonal) -> score 0.5
        # Distance of 2 (opposite) -> score 1.0
        score = min(1.0, cosine_distance / 2.0)
        
        # Apply operation-based scaling
        if current_message.requested_operation.risk_classification == "critical":
            score = min(1.0, score * 1.2)  # Amplify for critical operations
        
        # Generate reasoning and contributing factors
        contributing_factors = []
        
        if score > 0.8:
            reasoning = "Message embedding shows extreme deviation from user's semantic centroid in embedding space."
            contributing_factors.append(f"Cosine distance: {cosine_distance:.3f}")
            contributing_factors.append("Semantic representation highly dissimilar to user profile")
        elif score > 0.6:
            reasoning = "Message embedding shows significant deviation from typical semantic patterns."
            contributing_factors.append(f"Cosine distance: {cosine_distance:.3f}")
            contributing_factors.append("Moderate semantic drift detected")
        elif score > 0.3:
            reasoning = "Message shows mild semantic deviation from user's typical topics."
            contributing_factors.append(f"Cosine distance: {cosine_distance:.3f}")
        else:
            reasoning = "Message embedding is consistent with user's semantic profile."
        
        # Check for mid-session context switch (additional penalty)
        if current_message.message_sequence_in_session > 1 and score > 0.5:
            score = min(1.0, score + 0.1)
            contributing_factors.append("Mid-session semantic shift detected")
        
        return SemanticAnalysisResult(
            score=score,
            reasoning=reasoning,
            contributing_factors=contributing_factors,
        )

    def _get_embedding(self, text: str) -> np.ndarray:
        """
        Get sentence embedding for text.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        # Cache embeddings for efficiency
        if text in self._embedding_cache:
            return self._embedding_cache[text]
        
        embedding = self.model.encode(text, convert_to_numpy=True)
        self._embedding_cache[text] = embedding
        return embedding

    def _compute_profile_centroid(self, semantic_profile: SemanticProfile) -> np.ndarray:
        """
        Compute semantic centroid from user's profile.

        Uses pre-computed embedding_centroid when available (from ProfileManager EMA).
        Otherwise falls back to encoding typical_topics.
        
        Args:
            semantic_profile: User's semantic profile
            
        Returns:
            Centroid embedding vector
        """
        dim = self.model.get_sentence_embedding_dimension()
        if semantic_profile.embedding_centroid is not None:
            arr = np.array(semantic_profile.embedding_centroid, dtype=np.float64)
            norm = np.linalg.norm(arr)
            return arr / norm if norm > 0 else np.zeros(dim)
        if not semantic_profile.typical_topics:
            return np.zeros(dim)
        # Fallback: compute from typical topics
        topic_embeddings = [
            self._get_embedding(topic) for topic in semantic_profile.typical_topics
        ]
        centroid = np.mean(topic_embeddings, axis=0)
        norm = np.linalg.norm(centroid)
        return centroid / norm if norm > 0 else centroid

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity in [-1, 1]
        """
        # Normalize vectors
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        vec1_normalized = vec1 / norm1
        vec2_normalized = vec2 / norm2
        
        # Compute dot product
        similarity = np.dot(vec1_normalized, vec2_normalized)
        
        # Clip to [-1, 1] to handle numerical errors
        return float(np.clip(similarity, -1.0, 1.0))

    def learn_profile_from_history(self, message_history: List[str]) -> np.ndarray:
        """
        Learn semantic profile centroid from user's message history.
        
        This would be called during profile building/updating.
        
        Args:
            message_history: List of user's historical messages
            
        Returns:
            Learned centroid embedding
        """
        if not message_history:
            return np.zeros(self.model.get_sentence_embedding_dimension())
        
        # Compute embeddings for all messages
        embeddings = self.model.encode(message_history, convert_to_numpy=True)
        
        # Compute centroid with exponential moving average for recent messages
        # Weight recent messages more heavily
        weights = np.exp(np.linspace(-1, 0, len(embeddings)))
        weights = weights / weights.sum()
        
        centroid = np.average(embeddings, axis=0, weights=weights)
        
        # Normalize
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        
        return centroid
