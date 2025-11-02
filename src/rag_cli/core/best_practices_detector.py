"""Best practices detection for queries.

This module provides specialized detection for queries about best practices,
recommendations, and anti-patterns.
"""

import re
from typing import List, Optional
from dataclasses import dataclass
from enum import Enum

from rag_cli_plugin.services.logger import get_logger

logger = get_logger(__name__)


class BestPracticeType(Enum):
    """Types of best practice queries."""
    PRESCRIPTIVE = "prescriptive"  # "how should I...", "what's the right way"
    EVALUATIVE = "evaluative"  # "is it good to...", "should I avoid"
    COMPARATIVE = "comparative"  # "X vs Y best practices", "which is better"
    ANTI_PATTERN = "anti_pattern"  # what NOT to do
    GENERAL = "general"  # general best practices


@dataclass
class BestPracticeDetection:
    """Result of best practices detection."""
    is_best_practice_query: bool
    practice_type: Optional[BestPracticeType]
    confidence: float
    keywords: List[str]
    requires_authoritative_source: bool  # Should prioritize official docs


class BestPracticesDetector:
    """Detector for best practices queries."""

    # Prescriptive patterns (asking for recommended approach)
    PRESCRIPTIVE_PATTERNS = [
        r'\bhow\s+should\s+i\b',
        r'\bwhat\'?s\s+the\s+(right|correct|proper|best)\s+way\b',
        r'\brecommended\s+(way|approach|method|practice)\b',
        r'\bidiomatic\s+way\b',
        r'\bproper\s+way\b',
        r'\bstandard\s+(way|approach|method)\b',
        r'\bconvention\s+for\b',
    ]

    # Evaluative patterns (asking for evaluation)
    EVALUATIVE_PATTERNS = [
        r'\bis\s+it\s+(good|bad|safe|dangerous|okay)\s+to\b',
        r'\bshould\s+i\s+(use|avoid|consider)\b',
        r'\bcan\s+i\s+safely\b',
        r'\bis\s+this\s+(recommended|advised|acceptable)\b',
        r'\bwould\s+it\s+be\s+(better|worse)\b',
        r'\bis\s+there\s+a\s+better\b',
    ]

    # Comparative patterns
    COMPARATIVE_PATTERNS = [
        r'\b(which|what)\s+is\s+(better|safer|faster|more\s+efficient)\b',
        r'\bvs\.?\b.*\bbest\s+practice',
        r'\bcompare\b.*\b(approach|method|practice)',
        r'\bwhen\s+to\s+use\s+.*\s+(vs|versus|over)\b',
        r'\b(x|y)\s+or\s+(y|x)\s+for\b',
    ]

    # Anti-pattern indicators
    ANTI_PATTERN_PATTERNS = [
        r'\banti[- ]pattern\b',
        r'\bavoid\b',
        r'\bdon\'t\b.*\bdo\b',
        r'\bshould\s+not\b',
        r'\bshouldn\'t\b',
        r'\bbad\s+practice\b',
        r'\bpitfall\b',
        r'\bcommon\s+mistake\b',
        r'\bwhat\s+not\s+to\b',
    ]

    # General best practice keywords
    BEST_PRACTICE_KEYWORDS = [
        'best practice', 'recommended', 'should', 'idiomatic', 'convention',
        'standard', 'guideline', 'pattern', 'approach', 'methodology'
    ]

    # Context indicators (when combined with technical terms, indicate best practice query)
    CONTEXT_INDICATORS = [
        'production', 'scale', 'performance', 'security', 'maintainable',
        'robust', 'efficient', 'clean', 'professional', 'industry'
    ]

    def __init__(self, confidence_threshold: float = 0.5):
        """Initialize best practices detector.

        Args:
            confidence_threshold: Minimum confidence to classify as best practice query
        """
        self.confidence_threshold = confidence_threshold

    def detect(self, query: str) -> BestPracticeDetection:
        """Detect if query is about best practices.

        Args:
            query: User query

        Returns:
            BestPracticeDetection result
        """
        query_lower = query.lower()

        # Check each pattern type
        prescriptive_score = self._check_patterns(query_lower, self.PRESCRIPTIVE_PATTERNS)
        evaluative_score = self._check_patterns(query_lower, self.EVALUATIVE_PATTERNS)
        comparative_score = self._check_patterns(query_lower, self.COMPARATIVE_PATTERNS)
        anti_pattern_score = self._check_patterns(query_lower, self.ANTI_PATTERN_PATTERNS)

        # Check keywords
        keyword_matches = [kw for kw in self.BEST_PRACTICE_KEYWORDS if kw in query_lower]
        keyword_score = min(len(keyword_matches) * 0.2, 0.6)

        # Check context indicators
        context_matches = [ci for ci in self.CONTEXT_INDICATORS if ci in query_lower]
        context_boost = min(len(context_matches) * 0.1, 0.3)

        # Determine type and confidence
        scores = {
            BestPracticeType.PRESCRIPTIVE: prescriptive_score,
            BestPracticeType.EVALUATIVE: evaluative_score,
            BestPracticeType.COMPARATIVE: comparative_score,
            BestPracticeType.ANTI_PATTERN: anti_pattern_score,
        }

        # Get primary type
        max_type = max(scores.items(), key=lambda x: x[1])
        practice_type = max_type[0] if max_type[1] > 0 else BestPracticeType.GENERAL

        # Calculate overall confidence
        pattern_confidence = max(scores.values())
        total_confidence = min(pattern_confidence + keyword_score + context_boost, 1.0)

        # Determine if this is a best practice query
        is_best_practice = total_confidence >= self.confidence_threshold

        # Check if requires authoritative source
        # Prescriptive and anti-pattern queries should use official docs
        requires_authoritative = practice_type in [
            BestPracticeType.PRESCRIPTIVE,
            BestPracticeType.ANTI_PATTERN
        ]

        logger.debug(
            "Best practice detection",
            is_best_practice=is_best_practice,
            type=practice_type.value if is_best_practice else None,
            confidence=total_confidence
        )

        return BestPracticeDetection(
            is_best_practice_query=is_best_practice,
            practice_type=practice_type if is_best_practice else None,
            confidence=total_confidence,
            keywords=keyword_matches + context_matches,
            requires_authoritative_source=requires_authoritative
        )

    def _check_patterns(self, query: str, patterns: List[str]) -> float:
        """Check query against patterns and return score.

        Args:
            query: Normalized query
            patterns: List of regex patterns

        Returns:
            Score between 0 and 1
        """
        matches = 0
        for pattern in patterns:
            if re.search(pattern, query, re.IGNORECASE):
                matches += 1

        if matches == 0:
            return 0.0

        # Each match adds to score, capped at 1.0
        return min(matches * 0.4, 1.0)


def get_best_practices_detector(confidence_threshold: float = 0.5) -> BestPracticesDetector:
    """Get or create best practices detector instance.

    Args:
        confidence_threshold: Minimum confidence threshold

    Returns:
        BestPracticesDetector instance
    """
    return BestPracticesDetector(confidence_threshold=confidence_threshold)
