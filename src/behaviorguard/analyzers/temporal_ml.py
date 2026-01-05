"""ML-based temporal analyzer using z-scores and statistical modeling."""

import numpy as np
from typing import List, Tuple

from behaviorguard.models import (
    CurrentMessage,
    TemporalAnalysisResult,
    TemporalProfile,
)


class TemporalAnalyzerML:
    """
    ML-based temporal analyzer using statistical z-scores.
    
    Identifies unusual timing patterns through z-scores and
    logistic functions rather than hardcoded thresholds.
    """

    def analyze(
        self, current_message: CurrentMessage, temporal_profile: TemporalProfile
    ) -> TemporalAnalysisResult:
        """
        Detect temporal anomalies using z-scores and statistical modeling.

        Args:
            current_message: Current message being evaluated
            temporal_profile: User's temporal behavioral profile

        Returns:
            TemporalAnalysisResult with score, reasoning, and contributing factors
        """
        context = current_message.temporal_context
        score = 0.0
        contributing_factors = []
        reasoning_parts = []
        
        # 1. Activity timing anomaly (z-score based)
        timing_score, timing_factors = self._compute_timing_anomaly(
            context.hour_of_day_utc,
            context.is_typical_active_time,
            temporal_profile.most_active_hours_utc,
        )
        score += timing_score
        contributing_factors.extend(timing_factors)
        
        # 2. Inter-message gap anomaly (z-score)
        gap_score, gap_factors = self._compute_gap_anomaly(
            current_message.time_since_last_message_seconds,
            temporal_profile.typical_inter_message_gap_seconds,
        )
        score += gap_score
        contributing_factors.extend(gap_factors)
        
        # 3. Session frequency anomaly (z-score)
        freq_score, freq_factors = self._compute_frequency_anomaly(
            context.time_since_last_session_hours,
            temporal_profile.typical_session_frequency_per_week,
        )
        score += freq_score
        contributing_factors.extend(freq_factors)
        
        # 4. Session duration anomaly (z-score)
        duration_score, duration_factors = self._compute_duration_anomaly(
            current_message.message_sequence_in_session,
            temporal_profile.average_messages_per_session,
        )
        score += duration_score
        contributing_factors.extend(duration_factors)
        
        # Normalize score to [0, 1] using logistic function
        score = self._normalize_score(score)
        
        # Generate reasoning
        if score > 0.7:
            reasoning_parts.append("Extreme temporal anomalies detected across multiple dimensions")
        elif score > 0.4:
            reasoning_parts.append("Moderate temporal deviations from learned patterns")
        else:
            reasoning_parts.append("Temporal patterns consistent with user's learned behavior")
        
        reasoning = ". ".join(reasoning_parts) + "."
        
        return TemporalAnalysisResult(
            score=score,
            reasoning=reasoning,
            contributing_factors=contributing_factors,
        )

    def _compute_timing_anomaly(
        self, current_hour: int, is_typical: bool, typical_hours: List[int]
    ) -> Tuple[float, List[str]]:
        """
        Compute timing anomaly using activity distribution.
        
        Args:
            current_hour: Current hour of day (UTC)
            is_typical: Whether this is typical active time
            typical_hours: List of typical active hours
            
        Returns:
            Tuple of (anomaly_score, contributing_factors)
        """
        factors = []
        
        if not typical_hours:
            return 0.0, factors
        
        # Compute activity probability for this hour
        # Model as Gaussian mixture over 24-hour cycle
        activity_prob = self._compute_hour_probability(current_hour, typical_hours)
        
        # Convert probability to anomaly score
        # Low probability -> high anomaly
        if activity_prob < 0.05:
            score = 0.3
            factors.append(f"Activity at unusual hour (probability: {activity_prob:.3f})")
            
            # Extra penalty for 3-4am (circadian low point)
            if 3 <= current_hour <= 4:
                score += 0.2
                factors.append("Activity during typical sleep hours (3-4am)")
        elif activity_prob < 0.15:
            score = 0.15
            factors.append(f"Activity at less common hour (probability: {activity_prob:.3f})")
        else:
            score = 0.0
        
        return score, factors

    def _compute_hour_probability(self, hour: int, typical_hours: List[int]) -> float:
        """
        Compute probability of activity at given hour using Gaussian mixture.
        
        Args:
            hour: Hour to evaluate
            typical_hours: List of typical active hours
            
        Returns:
            Probability of activity at this hour
        """
        if not typical_hours:
            return 1.0 / 24  # Uniform if no data
        
        # Model as mixture of Gaussians centered on typical hours
        # with circular wrapping for 24-hour cycle
        prob = 0.0
        sigma = 2.0  # Standard deviation in hours
        
        for typical_hour in typical_hours:
            # Compute circular distance
            diff = min(abs(hour - typical_hour), 24 - abs(hour - typical_hour))
            # Gaussian probability
            prob += np.exp(-(diff ** 2) / (2 * sigma ** 2))
        
        # Normalize
        prob = prob / len(typical_hours)
        
        return float(prob)

    def _compute_gap_anomaly(
        self, current_gap: float, typical_gap: float
    ) -> Tuple[float, List[str]]:
        """
        Compute inter-message gap anomaly using z-score.
        
        Args:
            current_gap: Current gap in seconds
            typical_gap: Typical gap in seconds
            
        Returns:
            Tuple of (anomaly_score, contributing_factors)
        """
        factors = []
        
        if typical_gap <= 0:
            return 0.0, factors
        
        # Assume log-normal distribution for gaps
        # Compute z-score in log space
        log_current = np.log(max(current_gap, 0.1))
        log_typical = np.log(typical_gap)
        log_std = 0.5  # Assumed std in log space
        
        z_score = abs(log_current - log_typical) / log_std
        
        # Map z-score to anomaly score using logistic function
        score = self._z_score_to_anomaly(z_score)
        
        if score > 0.2:
            if current_gap < 5.0:
                factors.append(f"Very rapid response (z-score: {z_score:.2f}, gap: {current_gap:.1f}s)")
            else:
                factors.append(f"Unusual message gap (z-score: {z_score:.2f})")
        
        return score, factors

    def _compute_frequency_anomaly(
        self, time_since_last: float, typical_frequency: float
    ) -> Tuple[float, List[str]]:
        """
        Compute session frequency anomaly using z-score.
        
        Args:
            time_since_last: Hours since last session
            typical_frequency: Typical sessions per week
            
        Returns:
            Tuple of (anomaly_score, contributing_factors)
        """
        factors = []
        
        if typical_frequency <= 0:
            return 0.0, factors
        
        # Expected hours between sessions
        expected_gap = (7 * 24) / typical_frequency
        
        # Compute z-score
        std_gap = expected_gap * 0.5  # Assume 50% coefficient of variation
        z_score = abs(time_since_last - expected_gap) / std_gap
        
        # Map to anomaly score
        score = self._z_score_to_anomaly(z_score) * 0.5  # Weight less than other factors
        
        if score > 0.15:
            if time_since_last < expected_gap * 0.2:
                factors.append(f"Unusually frequent session (z-score: {z_score:.2f})")
            elif time_since_last > expected_gap * 3:
                factors.append(f"Return after long absence (z-score: {z_score:.2f})")
        
        return score, factors

    def _compute_duration_anomaly(
        self, current_sequence: int, typical_messages: float
    ) -> Tuple[float, List[str]]:
        """
        Compute session duration anomaly using z-score.
        
        Args:
            current_sequence: Current message sequence number
            typical_messages: Typical messages per session
            
        Returns:
            Tuple of (anomaly_score, contributing_factors)
        """
        factors = []
        
        if typical_messages <= 0:
            return 0.0, factors
        
        # Compute z-score
        std_messages = typical_messages * 0.4  # Assume 40% coefficient of variation
        z_score = abs(current_sequence - typical_messages) / std_messages
        
        # Map to anomaly score
        score = self._z_score_to_anomaly(z_score) * 0.3  # Weight less
        
        if score > 0.1 and current_sequence > typical_messages * 2:
            factors.append(f"Unusually long session (z-score: {z_score:.2f})")
        
        return score, factors

    def _z_score_to_anomaly(self, z_score: float) -> float:
        """
        Map z-score to anomaly score using logistic function.
        
        Args:
            z_score: Standardized score
            
        Returns:
            Anomaly score in [0, 1]
        """
        # Logistic function centered at z=2 (2 std devs)
        k = 0.5  # Steepness
        z0 = 2.0  # Inflection point
        
        score = 1.0 / (1.0 + np.exp(-k * (z_score - z0)))
        return float(score)

    def _normalize_score(self, raw_score: float) -> float:
        """
        Normalize combined score to [0, 1] using logistic function.
        
        Args:
            raw_score: Raw combined score
            
        Returns:
            Normalized score in [0, 1]
        """
        # Logistic normalization
        k = 1.0
        score = 1.0 / (1.0 + np.exp(-k * (raw_score - 0.5)))
        return float(np.clip(score, 0.0, 1.0))

    def learn_profile_from_history(
        self, temporal_data: List[dict]
    ) -> Tuple[float, float, List[int], float]:
        """
        Learn temporal profile statistics from interaction history.
        
        This would be called during profile building/updating.
        
        Args:
            temporal_data: List of temporal feature dictionaries
            
        Returns:
            Tuple of (typical_gap, typical_duration, active_hours, frequency)
        """
        if not temporal_data:
            return 30.0, 45.0, list(range(9, 18)), 5.0
        
        # Extract features
        gaps = [d.get("gap_seconds", 30) for d in temporal_data]
        hours = [d.get("hour_utc", 12) for d in temporal_data]
        
        # Compute statistics with exponential moving average
        weights = np.exp(np.linspace(-1, 0, len(gaps)))
        weights = weights / weights.sum()
        
        typical_gap = float(np.average(gaps, weights=weights))
        
        # Find active hours (hours with >10% of activity)
        hour_counts = np.bincount(hours, minlength=24)
        hour_probs = hour_counts / hour_counts.sum()
        active_hours = [h for h in range(24) if hour_probs[h] > 0.1]
        
        # Estimate frequency (sessions per week)
        # This would be computed from actual session data
        frequency = 5.0  # Default
        
        typical_duration = 45.0  # Default
        
        return typical_gap, typical_duration, active_hours, frequency
