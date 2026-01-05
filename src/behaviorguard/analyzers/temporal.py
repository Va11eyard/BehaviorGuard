"""Temporal analyzer for detecting timing and activity pattern anomalies."""

from typing import List

from behaviorguard.models import (
    CurrentMessage,
    TemporalAnalysisResult,
    TemporalProfile,
)


class TemporalAnalyzer:
    """Analyzes temporal anomalies in user activity patterns."""

    def analyze(
        self, current_message: CurrentMessage, temporal_profile: TemporalProfile
    ) -> TemporalAnalysisResult:
        """
        Detect temporal anomalies by analyzing timing patterns.

        Args:
            current_message: Current message being evaluated
            temporal_profile: User's temporal behavioral profile

        Returns:
            TemporalAnalysisResult with score, reasoning, and contributing factors
        """
        score = 0.0
        contributing_factors = []
        reasoning_parts = []

        context = current_message.temporal_context

        # 1. Activity at unusual hours
        unusual_hours_penalty = self._check_unusual_hours(
            context.hour_of_day_utc,
            temporal_profile.most_active_hours_utc,
            context.is_typical_active_time,
        )
        if unusual_hours_penalty > 0:
            score += unusual_hours_penalty
            contributing_factors.append(
                f"Activity during unusual hours ({unusual_hours_penalty:.2f})"
            )
            reasoning_parts.append("Message sent during atypical hours for this user")

        # 2. Impossible velocity (location/timezone changes)
        velocity_penalty = self._check_impossible_velocity(
            context.time_since_last_session_hours
        )
        if velocity_penalty > 0:
            score += velocity_penalty
            contributing_factors.append(
                f"Impossible velocity detected ({velocity_penalty:.2f})"
            )
            reasoning_parts.append("Session timing suggests impossible location change")

        # 3. Session duration anomaly
        duration_penalty = self._check_session_duration_anomaly(
            current_message.message_sequence_in_session,
            temporal_profile.average_messages_per_session,
            temporal_profile.longest_session_duration_minutes,
        )
        if duration_penalty > 0:
            score += duration_penalty
            contributing_factors.append(
                f"Session duration anomaly ({duration_penalty:.2f})"
            )
            reasoning_parts.append("Session duration exceeds typical patterns")

        # 4. Bot-like timing patterns
        bot_timing_penalty = self._check_bot_like_timing(
            current_message.time_since_last_message_seconds,
            temporal_profile.typical_inter_message_gap_seconds,
        )
        if bot_timing_penalty > 0:
            score += bot_timing_penalty
            contributing_factors.append(
                f"Bot-like timing pattern detected ({bot_timing_penalty:.2f})"
            )
            reasoning_parts.append("Inter-message timing suggests automated behavior")

        # 5. Session frequency anomaly
        frequency_penalty = self._check_session_frequency_anomaly(
            context.time_since_last_session_hours,
            temporal_profile.typical_session_frequency_per_week,
        )
        if frequency_penalty > 0:
            score += frequency_penalty
            contributing_factors.append(
                f"Session frequency anomaly ({frequency_penalty:.2f})"
            )
            reasoning_parts.append("Session frequency deviates from typical pattern")

        # Ensure score is bounded
        score = max(0.0, min(1.0, score))

        # Generate reasoning
        if reasoning_parts:
            reasoning = ". ".join(reasoning_parts) + "."
        else:
            reasoning = "Temporal patterns are consistent with user's typical activity."

        return TemporalAnalysisResult(
            score=score, reasoning=reasoning, contributing_factors=contributing_factors
        )

    def _check_unusual_hours(
        self, current_hour: int, typical_hours: List[int], is_typical: bool
    ) -> float:
        """Check for activity during unusual hours."""
        # If it's marked as typical time, no penalty
        if is_typical:
            return 0.0

        # Calculate activity percentage in this hour
        if not typical_hours:
            # No history - neutral
            return 0.0

        # Check if current hour is in typical hours
        if current_hour not in typical_hours:
            # Activity during hours with <5% historical activity
            activity_percentage = len(typical_hours) / 24.0

            # Check if it's 3-4am (extra suspicious)
            if 3 <= current_hour <= 4:
                return 0.3
            
            # Other unusual hours
            if activity_percentage < 0.05:
                # Very rare activity time
                return 0.2
            else:
                # Somewhat unusual but not extremely rare
                return 0.1

        return 0.0

    def _check_impossible_velocity(self, time_since_last_session_hours: float) -> float:
        """Check for impossible velocity (different timezone/location within 2 hours)."""
        # In production, this would check actual location/timezone changes
        # For now, check for very rapid session changes that might indicate
        # different locations

        if time_since_last_session_hours < 2.0:
            # Rapid session change - could indicate location change
            # In production, would check actual location data
            # For now, apply moderate penalty for very rapid changes
            if time_since_last_session_hours < 0.5:
                return 0.6  # <30 min between sessions from different locations
            elif time_since_last_session_hours < 1.0:
                return 0.5
            else:
                return 0.4

        return 0.0

    def _check_session_duration_anomaly(
        self,
        current_sequence: int,
        avg_messages_per_session: float,
        longest_session_minutes: float,
    ) -> float:
        """Check for session duration anomalies."""
        # Check if current session is unusually long
        if current_sequence > avg_messages_per_session * 2:
            # Session is >2x typical maximum
            if current_sequence > avg_messages_per_session * 3:
                return 0.3  # Extremely long session
            return 0.1

        # Sustained high-frequency activity
        if current_sequence > 50:  # Very long session
            return 0.3

        return 0.0

    def _check_bot_like_timing(
        self, current_gap: float, typical_gap: float
    ) -> float:
        """Check for bot-like timing patterns (<5s gaps consistently)."""
        # Bot-like: consistently <5 seconds between messages
        if current_gap < 5.0:
            # Very fast response - could be bot/script
            if current_gap < 2.0:
                return 0.3  # Extremely fast
            return 0.25

        # Also check for unusually consistent timing
        if typical_gap > 0 and abs(current_gap - typical_gap) < 1.0:
            # Timing is suspiciously consistent (within 1 second of typical)
            # Could indicate automation
            if current_gap < 10.0:
                return 0.1

        return 0.0

    def _check_session_frequency_anomaly(
        self, time_since_last_session_hours: float, typical_frequency_per_week: float
    ) -> float:
        """Check for session frequency anomalies."""
        # Convert typical frequency to hours between sessions
        if typical_frequency_per_week == 0:
            return 0.0

        typical_hours_between = (7 * 24) / typical_frequency_per_week

        # Check for unusually frequent sessions
        if time_since_last_session_hours < typical_hours_between * 0.1:
            # 10x more frequent than typical
            return 0.3

        # Check for return after long absence (6+ months = 4320 hours)
        if time_since_last_session_hours >= 4320.0:
            # Inactive 6+ months, suddenly returns
            return 0.2

        return 0.0
