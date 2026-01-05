"""Linguistic analyzer for detecting writing style anomalies."""

from typing import List

from behaviorguard.models import (
    CurrentMessage,
    LinguisticAnalysisResult,
    LinguisticProfile,
)


class LinguisticAnalyzer:
    """Analyzes linguistic anomalies in user messages."""

    def analyze(
        self, current_message: CurrentMessage, linguistic_profile: LinguisticProfile
    ) -> LinguisticAnalysisResult:
        """
        Detect linguistic anomalies by analyzing writing style patterns.

        Args:
            current_message: Current message being evaluated
            linguistic_profile: User's linguistic behavioral profile

        Returns:
            LinguisticAnalysisResult with score, reasoning, and contributing factors
        """
        score = 0.0
        contributing_factors = []
        reasoning_parts = []

        features = current_message.linguistic_features

        # 1. Message length deviation (>3 std devs → +0.15)
        length_penalty = self._check_message_length_deviation(
            features.message_length_tokens,
            linguistic_profile.avg_message_length_tokens,
            linguistic_profile.lexical_diversity_std,
        )
        if length_penalty > 0:
            score += length_penalty
            contributing_factors.append(
                f"Message length deviates significantly from typical ({length_penalty:.2f})"
            )
            reasoning_parts.append("Message length is unusually different from user's typical pattern")

        # 2. Lexical diversity shift (+0.1 to +0.3)
        diversity_penalty = self._check_lexical_diversity_shift(
            features.lexical_diversity,
            linguistic_profile.lexical_diversity_mean,
            linguistic_profile.lexical_diversity_std,
        )
        if diversity_penalty > 0:
            score += diversity_penalty
            contributing_factors.append(
                f"Lexical diversity shift detected ({diversity_penalty:.2f})"
            )
            reasoning_parts.append("Vocabulary usage pattern differs from baseline")

        # 3. Formality mismatch (+0.1 to +0.2)
        formality_penalty = self._check_formality_mismatch(
            features.formality_score,
            linguistic_profile.formality_score_mean,
            linguistic_profile.formality_score_std,
        )
        if formality_penalty > 0:
            score += formality_penalty
            contributing_factors.append(
                f"Formality level mismatch ({formality_penalty:.2f})"
            )
            reasoning_parts.append("Writing formality differs from typical style")

        # 4. Politeness inversion (+0.1 to +0.15)
        politeness_penalty = self._check_politeness_inversion(
            features.politeness_score,
            linguistic_profile.politeness_score_mean,
        )
        if politeness_penalty > 0:
            score += politeness_penalty
            contributing_factors.append(
                f"Politeness pattern change detected ({politeness_penalty:.2f})"
            )
            reasoning_parts.append("Politeness level shows unexpected shift")

        # 5. Language switching (+0.2 to +0.4)
        language_penalty = self._check_language_switching(
            features.language, linguistic_profile.primary_languages
        )
        if language_penalty > 0:
            score += language_penalty
            contributing_factors.append(
                f"Language switching detected ({language_penalty:.2f})"
            )
            reasoning_parts.append("Message uses different language than typical")

        # 6. Technical vocabulary mismatch (+0.2)
        tech_penalty = self._check_technical_vocabulary_mismatch(
            features, linguistic_profile
        )
        if tech_penalty > 0:
            score += tech_penalty
            contributing_factors.append(
                f"Technical vocabulary inconsistency ({tech_penalty:.2f})"
            )
            reasoning_parts.append("Technical language usage inconsistent with history")

        # Ensure score is bounded
        score = max(0.0, min(1.0, score))

        # Generate reasoning
        if reasoning_parts:
            reasoning = ". ".join(reasoning_parts) + "."
        else:
            reasoning = "Linguistic patterns are consistent with user's typical writing style."

        return LinguisticAnalysisResult(
            score=score, reasoning=reasoning, contributing_factors=contributing_factors
        )

    def _check_message_length_deviation(
        self, current_length: int, mean_length: float, std_dev: float
    ) -> float:
        """Check if message length deviates >3 standard deviations."""
        if std_dev == 0:
            # No variation in history - any deviation is suspicious
            if abs(current_length - mean_length) > mean_length * 0.5:
                return 0.15
            return 0.0

        deviation = abs(current_length - mean_length) / std_dev

        if deviation >= 3.0:
            return 0.15
        return 0.0

    def _check_lexical_diversity_shift(
        self, current_diversity: float, mean_diversity: float, std_dev: float
    ) -> float:
        """Check for dramatic vocabulary change."""
        if std_dev == 0:
            std_dev = 0.1  # Default for zero variance

        deviation = abs(current_diversity - mean_diversity) / std_dev

        if deviation > 3.0:
            return 0.3
        elif deviation > 2.0:
            return 0.2
        elif deviation > 1.5:
            return 0.1
        return 0.0

    def _check_formality_mismatch(
        self, current_formality: float, mean_formality: float, std_dev: float
    ) -> float:
        """Check for formality level mismatch (casual ↔ formal shift)."""
        if std_dev == 0:
            std_dev = 0.1

        deviation = abs(current_formality - mean_formality)

        # Large shift in formality
        if deviation > 0.4:
            return 0.2
        elif deviation > 0.3:
            return 0.15
        elif deviation > 0.2:
            return 0.1
        return 0.0

    def _check_politeness_inversion(
        self, current_politeness: float, mean_politeness: float
    ) -> float:
        """Check for politeness inversion (polite → rude or direct → overly polite)."""
        change = current_politeness - mean_politeness

        # Polite → demanding/rude (decrease)
        if change < -0.3:
            return 0.1

        # Direct → overly polite (increase) - social engineering indicator
        if change > 0.3:
            return 0.15

        return 0.0

    def _check_language_switching(
        self, current_language: str, primary_languages: List[str]
    ) -> float:
        """Check for language switching without history."""
        if current_language not in primary_languages:
            # Different language without history
            if len(primary_languages) == 1:
                # User only uses one language historically
                return 0.4
            else:
                # User uses multiple languages but this is new
                return 0.2
        return 0.0

    def _check_technical_vocabulary_mismatch(
        self, features: "LinguisticFeatures", profile: LinguisticProfile
    ) -> float:
        """Check for inconsistent technical term usage."""
        current_has_technical = features.contains_code or features.contains_urls
        historical_technical = profile.uses_technical_vocabulary or profile.uses_code_blocks

        # Mismatch: technical content but user not typically technical
        if current_has_technical and not historical_technical:
            return 0.2

        # Mismatch: no technical content but user typically technical
        # (less suspicious, could be casual conversation)
        if not current_has_technical and historical_technical:
            return 0.05

        return 0.0
