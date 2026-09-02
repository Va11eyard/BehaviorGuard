"""Rule-based baseline detector using keyword matching and heuristics."""

import re
from typing import Dict, List, Set
from datetime import datetime
from collections import defaultdict


class RuleBasedDetector:
    """
    Simple rule-based anomaly detector using keywords and patterns.
    
    This baseline represents traditional security approaches that rely on
    static rules rather than learned behavioral patterns.
    """
    
    def __init__(self):
        # High-risk keywords
        self.high_risk_keywords = {
            'delete', 'remove', 'reset', 'password', 'account',
            'export', 'transfer', 'admin', 'sudo', 'root',
            'credential', 'token', 'api_key', 'secret', 'key',
            'hack', 'exploit', 'breach', 'penetrate', 'inject',
            'backdoor', 'shell', 'payload', 'vulnerability',
            'bypass', 'disable', '2fa', 'verification', 'grant',
            'access', 'permission', 'privilege', 'escalate',
            'funds', 'urgent', 'immediately', 'asap', 'critical'
        }
        
        # Suspicious patterns (regex)
        self.suspicious_patterns = [
            r'(?i)\b(hack|exploit|penetrate|breach)\b',
            r'(?i)\b(inject|payload|shell|backdoor)\b',
            r'SELECT.*FROM.*WHERE',  # SQL injection
            r'<script.*?>.*?</script>',  # XSS
            r'(?i)\b(password|passwd|pwd)\s*[:=]',  # Password exposure
            r'(?i)api[_-]?key\s*[:=]',  # API key exposure
        ]
        
        # Rate limiting tracking
        self.user_message_times: Dict[str, List[datetime]] = defaultdict(list)
        self.rate_limit_window = 60  # seconds
        self.rate_limit_threshold = 10  # messages per window
        
        # Message length anomaly
        self.typical_length_min = 10
        self.typical_length_max = 500
    
    def detect(
        self,
        user_id: str,
        message: str,
        timestamp: datetime = None
    ) -> Dict:
        """
        Apply rule-based detection.
        
        Args:
            user_id: User identifier
            message: Message text
            timestamp: Message timestamp (optional)
        
        Returns:
            Dict with:
                - anomaly_score: float in [0, 1]
                - detected_rules: list of triggered rules
                - is_anomaly: bool (score > 0.5)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        detected_rules = []
        score = 0.0
        
        # 1. Keyword matching
        message_lower = message.lower()
        keyword_matches = sum(
            1 for keyword in self.high_risk_keywords
            if keyword in message_lower
        )
        if keyword_matches > 0:
            detected_rules.append(f"high_risk_keywords: {keyword_matches}")
            score += min(keyword_matches * 0.15, 0.4)
        
        # 2. Pattern matching
        for pattern in self.suspicious_patterns:
            if re.search(pattern, message):
                detected_rules.append(f"suspicious_pattern: {pattern[:30]}")
                score += 0.25
        
        # 3. Rate limiting
        self.user_message_times[user_id].append(timestamp)
        
        # Remove old timestamps
        cutoff = timestamp.timestamp() - self.rate_limit_window
        self.user_message_times[user_id] = [
            t for t in self.user_message_times[user_id]
            if t.timestamp() >= cutoff
        ]
        
        message_count = len(self.user_message_times[user_id])
        if message_count > self.rate_limit_threshold:
            detected_rules.append(f"rate_limit_exceeded: {message_count} msg/min")
            score += 0.3
        
        # 4. Message length anomaly
        msg_length = len(message)
        if msg_length < self.typical_length_min:
            detected_rules.append(f"message_too_short: {msg_length} chars")
            score += 0.1
        elif msg_length > self.typical_length_max:
            detected_rules.append(f"message_too_long: {msg_length} chars")
            score += 0.15
        
        # 5. All caps (shouting)
        if len(message) > 20 and message.isupper():
            detected_rules.append("all_caps_message")
            score += 0.1
        
        # 6. Excessive punctuation
        punct_count = sum(1 for c in message if c in '!?.')
        if punct_count > 10:
            detected_rules.append(f"excessive_punctuation: {punct_count}")
            score += 0.1
        
        # Cap score at 1.0
        final_score = min(score, 1.0)
        
        return {
            'anomaly_score': final_score,
            'detected_rules': detected_rules,
            'is_anomaly': final_score > 0.5,
            'component_scores': {
                'semantic': 0.0,  # Not applicable
                'linguistic': final_score,  # All rules are linguistic
                'temporal': 0.0,  # Not applicable
                'overall': final_score
            }
        }
    
    def reset_user_history(self, user_id: str = None):
        """Reset rate limiting history for a user or all users."""
        if user_id:
            self.user_message_times[user_id] = []
        else:
            self.user_message_times.clear()
