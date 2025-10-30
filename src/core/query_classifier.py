"""Unified query classification system for RAG-CLI.

This module provides intent detection, confidence scoring, and entity extraction
for user queries to enable intelligent routing and retrieval.
"""

import re
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum

from src.monitoring.logger import get_logger

logger = get_logger(__name__)


class QueryIntent(Enum):
    """Query intent categories."""
    CODE_EXPLANATION = "code_explanation"
    TROUBLESHOOTING = "troubleshooting"
    HOW_TO = "how_to"
    BEST_PRACTICES = "best_practices"
    COMPARISON = "comparison"
    TECHNICAL_DOCS = "technical_docs"
    CONCEPTUAL = "conceptual"
    GENERAL_QA = "general_qa"


class TechnicalDepth(Enum):
    """Technical depth levels."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


@dataclass
class TechnicalEntity:
    """Represents a detected technical entity."""
    type: str  # 'language', 'framework', 'library', 'tool', 'version'
    name: str
    version: Optional[str] = None
    confidence: float = 1.0


@dataclass
class QueryClassification:
    """Result of query classification."""
    primary_intent: QueryIntent
    all_intents: Dict[QueryIntent, float] = field(default_factory=dict)  # intent -> confidence
    technical_depth: TechnicalDepth = TechnicalDepth.INTERMEDIATE
    entities: List[TechnicalEntity] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    is_technical: bool = True
    confidence: float = 0.0  # Overall classification confidence


class QueryClassifier:
    """Classifies user queries for intelligent RAG routing."""

    # Intent detection patterns
    INTENT_PATTERNS = {
        QueryIntent.CODE_EXPLANATION: {
            'patterns': [
                r'\bexplain\s+(this\s+)?code\b',
                r'\bwhat\s+does\s+this\s+code\b',
                r'\bhow\s+does\s+(this\s+|the\s+)?\w+\s+work',
                r'\bexplain\s+how\s+(this\s+|the\s+)?\w+\s+works?\b',
                r'\bwalk\s+through\b',
                r'\bbreak\s+down\b',
                r'\bunderstand\s+code\b',
                r'\bwhat\s+is\s+(a\s+|an\s+|the\s+)?\w+\s+doing\b',
            ],
            'keywords': ['explain', 'code', 'function', 'class', 'method', 'works', 'does', 'how does', 'explain how'],
            'weight': 1.1
        },
        QueryIntent.TROUBLESHOOTING: {
            'patterns': [
                r'\berror\b',
                r'\bexception\b',
                r'\bfailed\b',
                r'\bnot\s+working\b',
                r'\bissue\b',
                r'\bbug\b',
                r'\bproblem\b',
                r'\bfix\b',
                r'\bdebug\b',
                r'\bcrash\b',
                r'\bi\'?m\s+getting\s+(a|an)\s+\w+error\b',
                r'\bgetting\s+(a|an)\s+\w+error\b',
                r'\b(type|syntax|runtime|value|attribute|key|index)error\b',
                r'\bthrows?\s+(an?\s+)?error\b',
                r'\bfailing\s+with\b',
            ],
            'keywords': ['error', 'exception', 'failed', 'not working', 'issue', 'bug', 'fix', 'debug', 'getting', 'throws', 'TypeError'],
            'weight': 1.3  # Higher weight for error queries
        },
        QueryIntent.HOW_TO: {
            'patterns': [
                r'\bhow\s+to\b',
                r'\bhow\s+do\s+i\b',
                r'\bhow\s+can\s+i\b',
                r'\bsteps\s+to\b',
                r'\bguide\s+to\b',
                r'\btutorial\b',
                r'\bwalk\s*through\b',
            ],
            'keywords': ['how to', 'how do', 'steps', 'guide', 'tutorial', 'create', 'build', 'implement'],
            'weight': 1.0
        },
        QueryIntent.BEST_PRACTICES: {
            'patterns': [
                r'\bbest\s+practice',
                r'\bwhat\s+(are|is)\s+(the\s+)?best\s+practice',
                r'\brecommended\s+(way|approach|method|practices?)\b',
                r'\bshould\s+i\b',
                r'\bis\s+it\s+good\b',
                r'\bidiomatic\b',
                r'\bconvention\b',
                r'\bstandard\b',
                r'\banti[- ]pattern\b',
                r'\bavoid\b',
                r'\bdon\'t\b',
                r'\bwhat\'s\s+the\s+(right|correct|proper|better)\b',
                r'\bhow\s+should\s+i\b',
                r'\bgood\s+practice',
            ],
            'keywords': ['best practice', 'recommended', 'should', 'idiomatic', 'convention', 'standard', 'avoid', 'what are', 'good practice'],
            'weight': 1.2
        },
        QueryIntent.COMPARISON: {
            'patterns': [
                r'\bvs\.?\b',
                r'\bversus\b',
                r'\bcompare\b',
                r'\bdifference\s+between\b',
                r'\bwhich\s+is\s+better\b',
                r'\bwhen\s+to\s+use\b',
                r'\bor\b.*\bor\b',  # "X or Y or Z"
            ],
            'keywords': ['vs', 'versus', 'compare', 'difference', 'better', 'which'],
            'weight': 1.0
        },
        QueryIntent.TECHNICAL_DOCS: {
            'patterns': [
                r'\bapi\b',
                r'\bparameter\b',
                r'\bargument\b',
                r'\bconfiguration\b',
                r'\bsyntax\b',
                r'\bsignature\b',
                r'\breference\b',
                r'\bdocumentation\b',
            ],
            'keywords': ['api', 'parameter', 'argument', 'config', 'syntax', 'reference', 'docs'],
            'weight': 1.0
        },
        QueryIntent.CONCEPTUAL: {
            'patterns': [
                r'\bwhat\s+is\b',
                r'\bwhat\s+are\b',
                r'\bwhy\s+(does|is|do)\b',
                r'\bconcept\b',
                r'\barchitecture\b',
                r'\bdesign\s+pattern\b',
                r'\bexplain\s+(the\s+)?concept\b',
            ],
            'keywords': ['what is', 'what are', 'why', 'concept', 'architecture', 'design', 'theory'],
            'weight': 1.0
        }
    }

    # Technical entity patterns
    ENTITY_PATTERNS = {
        'language': {
            'python': r'\b(python|py)\b',
            'javascript': r'\b(javascript|js|node\.?js)\b',
            'typescript': r'\b(typescript|ts)\b',
            'rust': r'\brust\b',
            'go': r'\b(go|golang)\b',
            'java': r'\bjava\b',
            'cpp': r'\b(c\+\+|cpp)\b',
            'c': r'\b\bc\b',
            'ruby': r'\bruby\b',
            'php': r'\bphp\b',
        },
        'framework': {
            'django': r'\bdjango\b',
            'flask': r'\bflask\b',
            'fastapi': r'\bfastapi\b',
            'react': r'\breact\b',
            'vue': r'\b(vue|vuejs)\b',
            'angular': r'\bangular\b',
            'express': r'\bexpress\b',
            'nextjs': r'\b(next\.?js|nextjs)\b',
            'langchain': r'\blangchain\b',
        },
        'library': {
            'numpy': r'\bnumpy\b',
            'pandas': r'\bpandas\b',
            'pytorch': r'\b(pytorch|torch)\b',
            'tensorflow': r'\btensorflow\b',
            'anthropic': r'\b(anthropic|claude)\b',
            'faiss': r'\bfaiss\b',
        }
    }

    # Version pattern
    VERSION_PATTERN = r'\b(v?\d+\.\d+(?:\.\d+)?(?:\.\d+)?)\b'

    # Technical depth indicators
    BEGINNER_INDICATORS = [
        'beginner', 'basic', 'simple', 'intro', 'introduction', 'getting started',
        'first time', 'new to', 'start', 'learn'
    ]

    ADVANCED_INDICATORS = [
        'advanced', 'optimize', 'performance', 'internals', 'deep dive',
        'architecture', 'implement', 'custom', 'extend', 'low-level'
    ]

    # Non-technical patterns (to filter out)
    NON_TECHNICAL_PATTERNS = [
        r'\bhello\b', r'\bhi\b', r'\bthank', r'\bthanks\b',
        r'\bplease\b', r'\bsorry\b', r'\bcan\s+you\b',
        r'\bwhat\'s\s+up\b', r'\bhow\s+are\s+you\b',
    ]

    def __init__(self, confidence_threshold: float = 0.3):
        """Initialize query classifier.

        Args:
            confidence_threshold: Minimum confidence for intent detection
        """
        self.confidence_threshold = confidence_threshold

    def classify(self, query: str) -> QueryClassification:
        """Classify a user query.

        Args:
            query: User query string

        Returns:
            QueryClassification with intents, entities, and metadata
        """
        query_lower = query.lower()

        # Check if technical query
        is_technical = self._is_technical_query(query_lower)

        # Detect intents with confidence scores
        intent_scores = self._detect_intents(query_lower)

        # Get primary intent
        primary_intent = max(intent_scores.items(), key=lambda x: x[1])[0] if intent_scores else QueryIntent.GENERAL_QA

        # Extract technical entities
        entities = self._extract_entities(query_lower)

        # Determine technical depth
        technical_depth = self._determine_depth(query_lower)

        # Extract keywords
        keywords = self._extract_keywords(query_lower)

        # Calculate overall confidence
        confidence = intent_scores.get(primary_intent, 0.0) if intent_scores else 0.5

        classification = QueryClassification(
            primary_intent=primary_intent,
            all_intents=intent_scores,
            technical_depth=technical_depth,
            entities=entities,
            keywords=keywords,
            is_technical=is_technical,
            confidence=confidence
        )

        logger.debug(
            f"Classified query",
            intent=primary_intent.value,
            confidence=confidence,
            entities=len(entities),
            is_technical=is_technical
        )

        return classification

    def _is_technical_query(self, query: str) -> bool:
        """Determine if query is technical.

        Args:
            query: Normalized query string

        Returns:
            True if technical query
        """
        # Check for non-technical patterns
        for pattern in self.NON_TECHNICAL_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                return False

        # Check for technical entities
        for entity_type, patterns in self.ENTITY_PATTERNS.items():
            for name, pattern in patterns.items():
                if re.search(pattern, query, re.IGNORECASE):
                    return True

        # Check for code-related keywords
        code_keywords = ['function', 'class', 'method', 'variable', 'import', 'module', 'package']
        if any(keyword in query for keyword in code_keywords):
            return True

        return True  # Default to technical

    def _detect_intents(self, query: str) -> Dict[QueryIntent, float]:
        """Detect all applicable intents with confidence scores.

        Args:
            query: Normalized query string

        Returns:
            Dictionary of intent -> confidence score
        """
        intent_scores = {}

        for intent, config in self.INTENT_PATTERNS.items():
            score = 0.0
            weight = config['weight']

            # Check regex patterns
            pattern_matches = 0
            for pattern in config['patterns']:
                if re.search(pattern, query, re.IGNORECASE):
                    pattern_matches += 1

            if pattern_matches > 0:
                # Each pattern match increases confidence
                score += min(pattern_matches * 0.3, 0.9)

            # Check keywords
            keyword_matches = sum(1 for kw in config['keywords'] if kw in query)
            if keyword_matches > 0:
                score += min(keyword_matches * 0.1, 0.4)

            # Apply weight and normalize
            if score > 0:
                score = min(score * weight, 1.0)
                if score >= self.confidence_threshold:
                    intent_scores[intent] = score

        # Default to GENERAL_QA if no strong intent detected
        if not intent_scores:
            intent_scores[QueryIntent.GENERAL_QA] = 0.5

        return intent_scores

    def _extract_entities(self, query: str) -> List[TechnicalEntity]:
        """Extract technical entities from query.

        Args:
            query: Normalized query string

        Returns:
            List of detected technical entities
        """
        entities = []

        for entity_type, patterns in self.ENTITY_PATTERNS.items():
            for name, pattern in patterns.items():
                if re.search(pattern, query, re.IGNORECASE):
                    # Check for version
                    version = None
                    version_match = re.search(f"{name}\\s*{self.VERSION_PATTERN}", query, re.IGNORECASE)
                    if version_match:
                        version = version_match.group(1)

                    entities.append(TechnicalEntity(
                        type=entity_type,
                        name=name.title(),
                        version=version,
                        confidence=0.9
                    ))

        return entities

    def _determine_depth(self, query: str) -> TechnicalDepth:
        """Determine technical depth of query.

        Args:
            query: Normalized query string

        Returns:
            Technical depth level
        """
        # Check for explicit indicators
        if any(indicator in query for indicator in self.BEGINNER_INDICATORS):
            return TechnicalDepth.BEGINNER

        if any(indicator in query for indicator in self.ADVANCED_INDICATORS):
            return TechnicalDepth.ADVANCED

        # Default to intermediate
        return TechnicalDepth.INTERMEDIATE

    def _extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from query.

        Args:
            query: Normalized query string

        Returns:
            List of keywords
        """
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'is', 'are', 'was', 'were'}

        # Tokenize and filter
        words = re.findall(r'\b\w+\b', query)
        keywords = [word for word in words if word not in stop_words and len(word) > 2]

        return keywords[:10]  # Return top 10


def get_query_classifier(confidence_threshold: float = 0.3) -> QueryClassifier:
    """Get or create query classifier instance.

    Args:
        confidence_threshold: Minimum confidence for intent detection

    Returns:
        QueryClassifier instance
    """
    return QueryClassifier(confidence_threshold=confidence_threshold)
