"""Semantic analyzer for detecting topic and domain anomalies."""

from typing import List, Set

from behaviorguard.models import (
    CurrentMessage,
    SemanticAnalysisResult,
    SemanticProfile,
)


class SemanticAnalyzer:
    """Analyzes semantic anomalies in user messages."""

    def analyze(
        self, current_message: CurrentMessage, semantic_profile: SemanticProfile
    ) -> SemanticAnalysisResult:
        """
        Detect semantic anomalies by comparing message topics against user profile.

        Args:
            current_message: Current message being evaluated
            semantic_profile: User's semantic behavioral profile

        Returns:
            SemanticAnalysisResult with score, reasoning, and contributing factors
        """
        score = 0.0
        contributing_factors = []

        # Extract topics from current message (simplified - in production use NLP)
        current_topics = self._extract_topics(current_message.text)
        typical_topics = set(semantic_profile.typical_topics)
        primary_domains = set(semantic_profile.primary_domains)

        # Calculate topic overlap
        topic_overlap = self._calculate_topic_overlap(current_topics, typical_topics)

        # Score based on deviation levels
        if topic_overlap >= 0.8:
            # 0.0-0.2: Fully consistent with typical topics
            score = 0.0 + (1.0 - topic_overlap) * 0.2
            reasoning = "Message topics are fully consistent with user's typical conversation themes."
        elif topic_overlap >= 0.5:
            # 0.3-0.5: Mild deviation - related but new sub-topic
            score = 0.3 + (0.8 - topic_overlap) * 0.67
            contributing_factors.append("New sub-topic within familiar domains")
            reasoning = "Message explores a related but new sub-topic within familiar domains."
        elif topic_overlap >= 0.2:
            # 0.6-0.8: Significant deviation - completely different domain
            score = 0.6 + (0.5 - topic_overlap) * 0.67
            contributing_factors.append("Completely different domain from typical topics")
            reasoning = "Message discusses topics in a completely different domain from user's typical interests."
        else:
            # 0.9-1.0: Extreme deviation - radically inconsistent
            score = 0.9 + (0.2 - topic_overlap) * 0.5
            contributing_factors.append("Extreme topic deviation from all historical patterns")
            reasoning = "Message shows extreme deviation from all historical conversation patterns."

        # Check for domain expertise mismatch
        if self._detect_domain_expertise_mismatch(current_message, semantic_profile):
            score = min(1.0, score + 0.15)
            contributing_factors.append("Technical expertise inconsistent with history")

        # Check for mid-session context switch (more suspicious)
        if current_message.message_sequence_in_session > 1 and topic_overlap < 0.3:
            score = min(1.0, score + 0.1)
            contributing_factors.append("Abrupt mid-session topic change")

        # Critical operation with extreme deviation
        if (
            score >= 0.85
            and current_message.requested_operation.risk_classification == "critical"
        ):
            score = min(1.0, score + 0.05)
            contributing_factors.append("Critical operation with extreme semantic deviation")

        # Ensure score is bounded
        score = max(0.0, min(1.0, score))

        return SemanticAnalysisResult(
            score=score, reasoning=reasoning, contributing_factors=contributing_factors
        )

    def _extract_topics(self, text: str) -> Set[str]:
        """
        Extract topics from message text.

        In production, this would use NLP/embeddings. For now, use simple keyword extraction.
        """
        # Simplified topic extraction - in production use proper NLP
        # Remove punctuation and convert to lowercase
        import re
        text_clean = re.sub(r'[^\w\s]', ' ', text.lower())
        keywords = text_clean.split()
        
        # Filter common words (simplified stopword removal)
        stopwords = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of",
            "with", "by", "from", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could", "should",
            "may", "might", "can", "this", "that", "these", "those", "i", "you", "he",
            "she", "it", "we", "they", "what", "which", "who", "when", "where", "why", "how",
            "my", "your", "their", "our", "his", "her", "its", "all", "some", "any", "each",
            "every", "both", "few", "more", "most", "other", "such", "no", "nor", "not",
            "only", "own", "same", "so", "than", "too", "very", "just", "use", "using"
        }
        return {word for word in keywords if word not in stopwords and len(word) > 2}

    def _calculate_topic_overlap(
        self, current_topics: Set[str], typical_topics: Set[str]
    ) -> float:
        """
        Calculate overlap between current and typical topics.

        Returns:
            Overlap ratio between 0.0 and 1.0
        """
        if not typical_topics:
            return 0.5  # Neutral score for no history

        if not current_topics:
            return 0.0

        intersection = len(current_topics & typical_topics)
        
        # Use Jaccard similarity (intersection / union)
        union = len(current_topics | typical_topics)
        if union == 0:
            return 0.0

        jaccard = intersection / union
        
        # Also consider what percentage of typical topics are present
        typical_coverage = intersection / len(typical_topics) if typical_topics else 0.0
        
        # Weight both metrics
        return (jaccard * 0.6) + (typical_coverage * 0.4)

    def _detect_domain_expertise_mismatch(
        self, current_message: CurrentMessage, semantic_profile: SemanticProfile
    ) -> bool:
        """
        Detect if message shows technical expertise inconsistent with user history.

        Returns:
            True if mismatch detected
        """
        # Check for technical vocabulary in message
        has_technical_content = (
            current_message.linguistic_features.contains_code
            or len(current_message.text) > 200
        )

        # Check if user typically uses technical vocabulary
        typical_technical = (
            "technology" in semantic_profile.primary_domains
            or "programming" in semantic_profile.primary_domains
            or "engineering" in semantic_profile.primary_domains
        )

        # Mismatch if technical content but user not typically technical
        return has_technical_content and not typical_technical
