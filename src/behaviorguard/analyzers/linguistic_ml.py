"""ML-based linguistic analyzer using Mahalanobis distance and statistical modeling."""

import numpy as np
from typing import List, Tuple

from behaviorguard.models import (
    CurrentMessage,
    LinguisticAnalysisResult,
    LinguisticProfile,
)


class LinguisticAnalyzerML:
    """
    ML-based linguistic analyzer using Gaussian distribution modeling.
    
    Uses Mahalanobis-like distance to detect changes in writing style,
    formality, and complexity based on learned statistical profiles.
    """

    def analyze(
        self, current_message: CurrentMessage, linguistic_profile: LinguisticProfile
    ) -> LinguisticAnalysisResult:
        """
        Detect linguistic anomalies using Mahalanobis distance.

        Args:
            current_message: Current message being evaluated
            linguistic_profile: User's linguistic behavioral profile

        Returns:
            LinguisticAnalysisResult with score, reasoning, and contributing factors
        """
        features = current_message.linguistic_features
        
        # Extract feature vector from current message
        current_vector = self._extract_feature_vector(features)
        
        # Extract mean vector and covariance from profile
        mean_vector, std_vector = self._extract_profile_statistics(linguistic_profile)
        
        # Compute Mahalanobis-like distance
        # Using diagonal covariance (independent features) for simplicity
        mahal_distance = self._mahalanobis_distance(
            current_vector, mean_vector, std_vector
        )
        
        # Map distance to anomaly score [0, 1]
        # Use logistic function to map distance to probability
        # Distance of 0 -> score ~0
        # Distance of 3 (3 std devs) -> score ~0.5
        # Distance of 6+ -> score ~1.0
        score = self._distance_to_score(mahal_distance)
        
        # Generate reasoning and contributing factors
        contributing_factors = []
        reasoning_parts = []
        
        # Analyze individual feature deviations
        feature_deviations = self._compute_feature_deviations(
            current_vector, mean_vector, std_vector
        )
        
        # Identify significant deviations (>2 std devs)
        significant_features = [
            (name, dev) for name, dev in feature_deviations.items() if abs(dev) > 2.0
        ]
        
        if significant_features:
            for name, dev in significant_features[:3]:  # Top 3
                contributing_factors.append(
                    f"{name}: {abs(dev):.2f} standard deviations from mean"
                )
        
        # Generate reasoning based on score
        if score > 0.7:
            reasoning_parts.append(
                f"Linguistic features show extreme deviation (Mahalanobis distance: {mahal_distance:.2f})"
            )
        elif score > 0.4:
            reasoning_parts.append(
                f"Moderate linguistic drift detected (Mahalanobis distance: {mahal_distance:.2f})"
            )
        else:
            reasoning_parts.append("Linguistic patterns are consistent with user profile")
        
        # Check for language switching
        if features.language not in linguistic_profile.primary_languages:
            score = min(1.0, score + 0.3)
            contributing_factors.append(
                f"Language switch detected: {features.language} not in primary languages"
            )
            reasoning_parts.append("Unexpected language detected")
        
        reasoning = ". ".join(reasoning_parts) + "."
        
        return LinguisticAnalysisResult(
            score=score,
            reasoning=reasoning,
            contributing_factors=contributing_factors,
        )

    def _extract_feature_vector(self, features) -> np.ndarray:
        """
        Extract feature vector from linguistic features.
        
        Args:
            features: LinguisticFeatures object
            
        Returns:
            Feature vector [length_tokens, lexical_diversity, formality, politeness]
        """
        return np.array([
            float(features.message_length_tokens),
            features.lexical_diversity,
            features.formality_score,
            features.politeness_score,
        ])

    def _extract_profile_statistics(
        self, profile: LinguisticProfile
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract mean and std vectors from linguistic profile.
        
        Args:
            profile: LinguisticProfile object
            
        Returns:
            Tuple of (mean_vector, std_vector)
        """
        mean_vector = np.array([
            profile.avg_message_length_tokens,
            profile.lexical_diversity_mean,
            profile.formality_score_mean,
            profile.politeness_score_mean,
        ])
        
        std_vector = np.array([
            max(profile.lexical_diversity_std * profile.avg_message_length_tokens, 1.0),
            max(profile.lexical_diversity_std, 0.01),
            max(profile.formality_score_std, 0.01),
            max(profile.politeness_score_std, 0.01),
        ])
        
        return mean_vector, std_vector

    def _mahalanobis_distance(
        self, x: np.ndarray, mean: np.ndarray, std: np.ndarray
    ) -> float:
        """
        Compute Mahalanobis-like distance with diagonal covariance.
        
        Args:
            x: Current feature vector
            mean: Mean feature vector from profile
            std: Standard deviation vector from profile
            
        Returns:
            Mahalanobis distance
        """
        # Compute standardized differences
        diff = x - mean
        standardized_diff = diff / std
        
        # Compute Euclidean distance in standardized space
        # This is equivalent to Mahalanobis with diagonal covariance
        distance = np.sqrt(np.sum(standardized_diff ** 2))
        
        return float(distance)

    def _distance_to_score(self, distance: float) -> float:
        """
        Map Mahalanobis distance to anomaly score using logistic function.
        
        Args:
            distance: Mahalanobis distance
            
        Returns:
            Anomaly score in [0, 1]
        """
        # Logistic function: 1 / (1 + exp(-k * (d - d0)))
        # k controls steepness, d0 is inflection point
        k = 0.5  # Steepness parameter
        d0 = 3.0  # Inflection at 3 std devs
        
        score = 1.0 / (1.0 + np.exp(-k * (distance - d0)))
        return float(score)

    def _compute_feature_deviations(
        self, current: np.ndarray, mean: np.ndarray, std: np.ndarray
    ) -> dict:
        """
        Compute standardized deviations for each feature.
        
        Args:
            current: Current feature vector
            mean: Mean feature vector
            std: Standard deviation vector
            
        Returns:
            Dictionary mapping feature names to z-scores
        """
        feature_names = [
            "message_length",
            "lexical_diversity",
            "formality",
            "politeness",
        ]
        
        z_scores = (current - mean) / std
        
        return {name: float(z) for name, z in zip(feature_names, z_scores)}

    def learn_profile_from_history(
        self, message_features: List[dict]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Learn linguistic profile statistics from message history.
        
        This would be called during profile building/updating.
        
        Args:
            message_features: List of feature dictionaries from historical messages
            
        Returns:
            Tuple of (mean_vector, std_vector)
        """
        if not message_features:
            # Return default statistics
            return np.array([50.0, 0.7, 0.5, 0.6]), np.array([10.0, 0.1, 0.1, 0.1])
        
        # Extract feature vectors
        vectors = []
        for features in message_features:
            vector = np.array([
                float(features.get("length_tokens", 50)),
                features.get("lexical_diversity", 0.7),
                features.get("formality", 0.5),
                features.get("politeness", 0.6),
            ])
            vectors.append(vector)
        
        vectors = np.array(vectors)
        
        # Compute statistics with exponential moving average for recent messages
        weights = np.exp(np.linspace(-1, 0, len(vectors)))
        weights = weights / weights.sum()
        
        # Weighted mean
        mean_vector = np.average(vectors, axis=0, weights=weights)
        
        # Weighted standard deviation
        variance = np.average((vectors - mean_vector) ** 2, axis=0, weights=weights)
        std_vector = np.sqrt(variance)
        
        # Ensure minimum std to avoid division by zero
        std_vector = np.maximum(std_vector, 0.01)
        
        return mean_vector, std_vector
