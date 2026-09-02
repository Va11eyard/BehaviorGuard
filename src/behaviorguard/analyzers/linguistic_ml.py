"""ML-based linguistic analyzer using Mahalanobis distance and stylometric features."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from behaviorguard.models import (
    CurrentMessage,
    LinguisticAnalysisResult,
    LinguisticProfile,
)

STOPWORDS = frozenset(
    "a an the i me my we you your he she it they this that is are was were be been "
    "do does did have has had will would can could to of in on at for with and or "
    "but not no so if then than as from by about just really very".split()
)
FIRST_PERSON = frozenset(("i", "me", "my", "mine", "im", "i'm", "we", "our"))
POLITE = ("please", "thank", "sorry", "could you", "would you")

STYLO_FEATURE_NAMES = [
    "n_tokens",
    "mean_word_len",
    "type_token_ratio",
    "punct_rate",
    "question_rate",
    "exclaim",
    "caps_rate",
    "stopword_rate",
    "first_person_rate",
    "politeness",
]


def stylo_features(text: str) -> np.ndarray:
    """10 content-light stylometric features (same as sequential ATO study)."""
    words = text.split()
    n = max(len(words), 1)
    lower_words = [w.lower().strip(".,!?;:'\"") for w in words]
    chars = max(len(text), 1)
    letters = [c for c in text if c.isalpha()]
    n_letters = max(len(letters), 1)
    return np.array(
        [
            float(len(words)),
            float(np.mean([len(w) for w in words])) if words else 0.0,
            len(set(lower_words)) / n,
            sum(1 for c in text if c in ".,!?;:'\"-") / chars,
            text.count("?") / n,
            float("!" in text),
            sum(1 for c in letters if c.isupper()) / n_letters,
            sum(1 for w in lower_words if w in STOPWORDS) / n,
            sum(1 for w in lower_words if w in FIRST_PERSON) / n,
            float(any(p in text.lower() for p in POLITE)),
        ],
        dtype=np.float64,
    )


class LinguisticAnalyzerML:
    """
    ML-based linguistic analyzer using Gaussian distribution modeling.

    Default feature set is the 10-dimensional stylometric vector derived from
    raw message text (replacing the saturating 4-feature length/TTR/formality/
    politeness set). Profile means for overlapping dimensions are taken from
    LinguisticProfile; remaining dimensions use population priors with wide
    std floors so cold profiles do not saturate.
    """

    def __init__(self, use_stylometric: bool = True):
        self.use_stylometric = use_stylometric

    def analyze(
        self, current_message: CurrentMessage, linguistic_profile: LinguisticProfile
    ) -> LinguisticAnalysisResult:
        features = current_message.linguistic_features

        if self.use_stylometric:
            current_vector = stylo_features(current_message.text)
            mean_vector, std_vector = self._stylometric_profile_statistics(
                linguistic_profile
            )
            feature_names = STYLO_FEATURE_NAMES
        else:
            current_vector = self._extract_feature_vector_legacy(features)
            mean_vector, std_vector = self._extract_profile_statistics_legacy(
                linguistic_profile
            )
            feature_names = [
                "message_length",
                "lexical_diversity",
                "formality",
                "politeness",
            ]

        mahal_distance = self._mahalanobis_distance(
            current_vector, mean_vector, std_vector
        )
        score = self._distance_to_score(mahal_distance)

        contributing_factors = []
        reasoning_parts = []

        feature_deviations = self._compute_feature_deviations(
            current_vector, mean_vector, std_vector, feature_names
        )
        significant_features = [
            (name, dev) for name, dev in feature_deviations.items() if abs(dev) > 2.0
        ]
        if significant_features:
            for name, dev in significant_features[:3]:
                contributing_factors.append(
                    f"{name}: {abs(dev):.2f} standard deviations from mean"
                )

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

    def _extract_feature_vector_legacy(self, features) -> np.ndarray:
        return np.array(
            [
                float(features.message_length_tokens),
                features.lexical_diversity,
                features.formality_score,
                features.politeness_score,
            ]
        )

    def _extract_profile_statistics_legacy(
        self, profile: LinguisticProfile
    ) -> Tuple[np.ndarray, np.ndarray]:
        mean_vector = np.array(
            [
                profile.avg_message_length_tokens,
                profile.lexical_diversity_mean,
                profile.formality_score_mean,
                profile.politeness_score_mean,
            ]
        )
        std_vector = np.array(
            [
                max(profile.avg_message_length_tokens_std, 1.0),
                max(profile.lexical_diversity_std, 0.05),
                max(profile.formality_score_std, 0.05),
                max(profile.politeness_score_std, 0.05),
            ]
        )
        return mean_vector, std_vector

    def _stylometric_profile_statistics(
        self, profile: LinguisticProfile
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Map LinguisticProfile fields onto the 10-d stylo mean/std, with priors."""
        # Population priors (wide) for dimensions not stored in LinguisticProfile
        mean = np.array(
            [
                float(profile.avg_message_length_tokens),
                4.5,  # mean word length
                float(profile.lexical_diversity_mean),
                0.05,  # punct rate
                float(profile.question_ratio_mean),
                0.1,  # exclaim
                0.05,  # caps
                0.4,  # stopword
                0.1,  # first person
                float(profile.politeness_score_mean),
            ],
            dtype=np.float64,
        )
        std = np.array(
            [
                max(profile.avg_message_length_tokens_std, 1.0),
                1.0,
                max(profile.lexical_diversity_std, 0.05),
                0.05,
                0.15,
                0.2,
                0.05,
                0.15,
                0.1,
                max(profile.politeness_score_std, 0.05),
            ],
            dtype=np.float64,
        )
        return mean, std

    def _mahalanobis_distance(
        self, x: np.ndarray, mean: np.ndarray, std: np.ndarray
    ) -> float:
        diff = x - mean
        standardized_diff = diff / std
        distance = np.sqrt(np.sum(standardized_diff**2))
        return float(distance)

    def _distance_to_score(self, distance: float) -> float:
        k = 0.8
        d0 = 2.5
        score = 1.0 / (1.0 + np.exp(-k * (distance - d0)))
        return float(score)

    def _compute_feature_deviations(
        self,
        current: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> dict:
        names = feature_names or STYLO_FEATURE_NAMES
        z_scores = (current - mean) / std
        return {name: float(z) for name, z in zip(names, z_scores)}

    def learn_profile_from_history(
        self, message_features: List[dict]
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not message_features:
            return (
                np.array([50.0, 4.5, 0.7, 0.05, 0.3, 0.1, 0.05, 0.4, 0.1, 0.6]),
                np.array([10.0, 1.0, 0.1, 0.05, 0.15, 0.2, 0.05, 0.15, 0.1, 0.1]),
            )

        vectors = []
        for features in message_features:
            if "text" in features:
                vectors.append(stylo_features(features["text"]))
            else:
                vectors.append(
                    np.array(
                        [
                            float(features.get("length_tokens", 50)),
                            4.5,
                            features.get("lexical_diversity", 0.7),
                            0.05,
                            0.3,
                            0.1,
                            0.05,
                            0.4,
                            0.1,
                            features.get("politeness", 0.6),
                        ]
                    )
                )
        vectors = np.array(vectors)
        weights = np.exp(np.linspace(-1, 0, len(vectors)))
        weights = weights / weights.sum()
        mean_vector = np.average(vectors, axis=0, weights=weights)
        variance = np.average((vectors - mean_vector) ** 2, axis=0, weights=weights)
        std_vector = np.maximum(np.sqrt(variance), 0.01)
        return mean_vector, std_vector
