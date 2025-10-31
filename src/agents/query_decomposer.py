"""Query Decomposer Agent for RAG-CLI.

Breaks complex multi-part queries into atomic sub-queries for parallel execution.

DECOMPOSITION STRATEGIES:
1. Pattern-based: Uses regex to detect conjunctions (AND, OR, +)
2. Punctuation-based: Splits on question marks, semicolons
3. List-based: Detects numbered/bulleted lists
4. MAF-assisted: Uses MAF Architect for complex planning (optional)

EXAMPLES:
Input: "How to implement FastAPI with async DB and CORS?"
Output: ["How to implement FastAPI", "FastAPI async database", "FastAPI CORS"]

Input: "What are: 1) best RAG practices 2) chunking strategies 3) embedding models"
Output: ["best RAG practices", "chunking strategies", "embedding models"]
"""

import re
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from monitoring.logger import get_logger
from integrations.maf_connector import get_maf_connector

logger = get_logger(__name__)


class DecompositionStrategy(Enum):
    """Query decomposition strategies."""
    PATTERN_BASED = "pattern"
    PUNCTUATION_BASED = "punctuation"
    LIST_BASED = "list"
    MAF_ASSISTED = "maf_assisted"
    NO_DECOMPOSITION = "none"


@dataclass
class SubQuery:
    """Represents a decomposed sub-query."""
    text: str
    index: int
    original_context: str
    dependencies: List[int]  # Indices of sub-queries this depends on
    priority: int  # Execution priority (lower = higher priority)
    metadata: Dict[str, Any]


@dataclass
class DecompositionResult:
    """Result of query decomposition."""
    original_query: str
    sub_queries: List[SubQuery]
    strategy_used: DecompositionStrategy
    confidence: float
    is_complex: bool
    metadata: Dict[str, Any]


class QueryDecomposer:
    """Decomposes complex queries into atomic sub-queries."""

    def __init__(self):
        """Initialize query decomposer."""
        self.maf_connector = get_maf_connector()
        self.use_maf = self.maf_connector.is_available()

        # Complexity thresholds
        self.min_query_length_for_complex = 50  # Chars
        self.min_sub_queries = 2
        self.max_sub_queries = 5  # Limit to avoid over-decomposition

        # Patterns for detection
        self.conjunction_patterns = [
            r'\b(and|or|plus|\+|also|additionally|furthermore)\b',
            r'\b(as well as|along with|together with)\b',
            r'\b(including|such as|like)\b'
        ]

        self.question_patterns = [
            r'\?',  # Question marks
            r'\b(how|what|why|when|where|who|which)\b'
        ]

        self.list_patterns = [
            r'^\s*\d+[\.)]\s+',  # Numbered lists: 1) or 1.
            r'^\s*[a-z][\.)]\s+',  # Lettered lists: a) or a.
            r'^\s*[-**]\s+',  # Bulleted lists
            r'\b(first|second|third|fourth|fifth)\b',  # Ordinal words
            r'\b(firstly|secondly|thirdly)\b'
        ]

        logger.info(
            "Query decomposer initialized",
            maf_available=self.use_maf,
            max_sub_queries=self.max_sub_queries
        )

    async def decompose(
        self,
        query: str,
        use_maf: Optional[bool] = None
    ) -> DecompositionResult:
        """Decompose query into sub-queries if complex.

        Args:
            query: Input query to decompose
            use_maf: Whether to use MAF Architect (defaults to availability)

        Returns:
            DecompositionResult with sub-queries or original query
        """
        logger.info("Analyzing query for decomposition", length=len(query))

        # Step 1: Check if query is complex
        complexity_score, indicators = self._analyze_complexity(query)
        is_complex = complexity_score >= 0.6

        logger.debug(
            "Complexity analysis",
            score=complexity_score,
            is_complex=is_complex,
            indicators=indicators
        )

        if not is_complex:
            # Simple query, no decomposition needed
            return DecompositionResult(
                original_query=query,
                sub_queries=[SubQuery(
                    text=query,
                    index=0,
                    original_context=query,
                    dependencies=[],
                    priority=0,
                    metadata={}
                )],
                strategy_used=DecompositionStrategy.NO_DECOMPOSITION,
                confidence=1.0,
                is_complex=False,
                metadata={'complexity_score': complexity_score}
            )

        # Step 2: Try pattern-based decomposition first
        pattern_result = self._decompose_by_patterns(query)

        if pattern_result and len(pattern_result) >= self.min_sub_queries:
            logger.info(
                "Pattern-based decomposition successful",
                sub_queries=len(pattern_result)
            )
            return self._create_result(
                query,
                pattern_result,
                DecompositionStrategy.PATTERN_BASED,
                0.8
            )

        # Step 3: Try MAF-assisted decomposition if available
        if (use_maf is None and self.use_maf) or use_maf:
            maf_result = await self._decompose_with_maf(query)
            if maf_result and len(maf_result) >= self.min_sub_queries:
                logger.info(
                    "MAF-assisted decomposition successful",
                    sub_queries=len(maf_result)
                )
                return self._create_result(
                    query,
                    maf_result,
                    DecompositionStrategy.MAF_ASSISTED,
                    0.9
                )

        # Step 4: Fallback to simple splitting
        simple_result = self._simple_split(query)
        if simple_result and len(simple_result) >= self.min_sub_queries:
            logger.info(
                "Simple split decomposition",
                sub_queries=len(simple_result)
            )
            return self._create_result(
                query,
                simple_result,
                DecompositionStrategy.PUNCTUATION_BASED,
                0.6
            )

        # No decomposition possible or beneficial
        logger.debug("No decomposition possible, treating as single query")
        return DecompositionResult(
            original_query=query,
            sub_queries=[SubQuery(
                text=query,
                index=0,
                original_context=query,
                dependencies=[],
                priority=0,
                metadata={}
            )],
            strategy_used=DecompositionStrategy.NO_DECOMPOSITION,
            confidence=1.0,
            is_complex=False,
            metadata={'complexity_score': complexity_score}
        )

    def _analyze_complexity(self, query: str) -> Tuple[float, List[str]]:
        """Analyze query complexity.

        Args:
            query: Input query

        Returns:
            Tuple of (complexity_score, indicators)
        """
        indicators = []
        score = 0.0

        # Length indicator
        if len(query) >= self.min_query_length_for_complex:
            score += 0.2
            indicators.append("long_query")

        # Multiple questions
        question_count = len(re.findall(r'\?', query))
        if question_count > 1:
            score += 0.3
            indicators.append(f"multiple_questions({question_count})")

        # Conjunctions
        conjunction_count = 0
        for pattern in self.conjunction_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            conjunction_count += len(matches)

        if conjunction_count >= 2:
            score += 0.3
            indicators.append(f"conjunctions({conjunction_count})")

        # List patterns
        list_count = 0
        for pattern in self.list_patterns:
            if re.search(pattern, query, re.IGNORECASE | re.MULTILINE):
                list_count += 1

        if list_count >= 1:
            score += 0.4
            indicators.append(f"list_structure({list_count})")

        # Semicolons or line breaks (multiple statements)
        statement_separators = query.count(';') + query.count('\n')
        if statement_separators >= 1:
            score += 0.2
            indicators.append(f"statements({statement_separators + 1})")

        # Cap score at 1.0
        score = min(score, 1.0)

        return score, indicators

    def _decompose_by_patterns(self, query: str) -> Optional[List[str]]:
        """Decompose query using pattern matching.

        Args:
            query: Input query

        Returns:
            List of sub-queries or None
        """
        sub_queries = []

        # Strategy 1: Split by numbered/bulleted lists
        list_pattern = r'(?:^|\n)\s*(?:\d+[\.)]|[a-z][\.)]|[-**])\s+(.+?)(?=(?:\n\s*(?:\d+[\.)]|[a-z][\.)]|[-**]))|$)'
        list_matches = re.findall(list_pattern, query, re.MULTILINE)

        if len(list_matches) >= 2:
            sub_queries = [match.strip() for match in list_matches if match.strip()]
            logger.debug(f"List-based decomposition: {len(sub_queries)} items")
            return sub_queries[:self.max_sub_queries]

        # Strategy 2: Split by multiple questions
        if query.count('?') > 1:
            parts = query.split('?')
            sub_queries = [
                part.strip() + '?' for part in parts[:-1] if part.strip()
            ]
            # Add last part if it's not empty
            if parts[-1].strip():
                sub_queries.append(parts[-1].strip())

            if len(sub_queries) >= 2:
                logger.debug(f"Question-based decomposition: {len(sub_queries)} questions")
                return sub_queries[:self.max_sub_queries]

        # Strategy 3: Split by strong conjunctions with contextual keywords
        conjunction_split_pattern = r'\b(?:and|or|plus|\+)\s+(?:how|what|explain|describe|implement)'
        if re.search(conjunction_split_pattern, query, re.IGNORECASE):
            # Try to split intelligently
            parts = re.split(r'\b(and|or|plus|\+)\b', query, flags=re.IGNORECASE)

            # Reconstruct meaningful sub-queries
            current = []
            for i, part in enumerate(parts):
                if part.lower() in ['and', 'or', 'plus', '+']:
                    if current:
                        sub_queries.append(' '.join(current).strip())
                        current = []
                else:
                    current.append(part.strip())

            if current:
                sub_queries.append(' '.join(current).strip())

            # Filter out too short fragments
            sub_queries = [sq for sq in sub_queries if len(sq) > 10]

            if len(sub_queries) >= 2:
                logger.debug(f"Conjunction-based decomposition: {len(sub_queries)} parts")
                return sub_queries[:self.max_sub_queries]

        return None

    async def _decompose_with_maf(self, query: str) -> Optional[List[str]]:
        """Use MAF Architect to decompose query.

        Args:
            query: Input query

        Returns:
            List of sub-queries or None
        """
        if not self.use_maf:
            return None

        try:
            logger.debug("Attempting MAF-assisted decomposition")

            # Execute MAF Architect
            result = await self.maf_connector.execute_architect(
                query=query,
                complexity='complex'
            )

            if not result or result.status != 'completed':
                logger.debug("MAF decomposition failed or incomplete")
                return None

            # Parse MAF result for sub-queries
            # Expected format: MAF should return structured decomposition
            # For now, use simple parsing
            content = result.content

            # Look for numbered list in MAF output
            sub_queries = []
            lines = content.split('\n')

            for line in lines:
                # Match patterns like "1. ", "- ", etc.
                match = re.match(r'^\s*(?:\d+[\.)]|[-**])\s+(.+)$', line)
                if match:
                    sub_query = match.group(1).strip()
                    if len(sub_query) > 10:  # Filter short fragments
                        sub_queries.append(sub_query)

            if len(sub_queries) >= 2:
                logger.info(f"MAF provided {len(sub_queries)} sub-queries")
                return sub_queries[:self.max_sub_queries]

            return None

        except Exception as e:
            logger.error(f"MAF-assisted decomposition failed: {e}")
            return None

    def _simple_split(self, query: str) -> Optional[List[str]]:
        """Simple fallback splitting by punctuation.

        Args:
            query: Input query

        Returns:
            List of sub-queries or None
        """
        # Split by semicolons, line breaks, or commas in long queries
        separators = [';', '\n']

        for separator in separators:
            if separator in query:
                parts = query.split(separator)
                sub_queries = [part.strip() for part in parts if len(part.strip()) > 15]

                if len(sub_queries) >= 2:
                    logger.debug(f"Simple split by '{separator}': {len(sub_queries)} parts")
                    return sub_queries[:self.max_sub_queries]

        return None

    def _create_result(
        self,
        original_query: str,
        sub_query_texts: List[str],
        strategy: DecompositionStrategy,
        confidence: float
    ) -> DecompositionResult:
        """Create DecompositionResult from sub-query texts.

        Args:
            original_query: Original query
            sub_query_texts: List of sub-query text strings
            strategy: Strategy used
            confidence: Decomposition confidence

        Returns:
            DecompositionResult
        """
        sub_queries = []

        for i, text in enumerate(sub_query_texts):
            # Clean up text
            text = text.strip()

            # Ensure complete question format
            if not any(text.endswith(p) for p in ['.', '?', '!']):
                # Check if it's asking a question
                if any(text.lower().startswith(q) for q in ['how', 'what', 'why', 'when', 'where', 'who', 'which']):
                    text += '?'
                else:
                    text += '.'

            sub_query = SubQuery(
                text=text,
                index=i,
                original_context=original_query,
                dependencies=[],  # Future: analyze dependencies
                priority=i,  # Sequential by default
                metadata={
                    'source_strategy': strategy.value,
                    'length': len(text)
                }
            )
            sub_queries.append(sub_query)

        return DecompositionResult(
            original_query=original_query,
            sub_queries=sub_queries,
            strategy_used=strategy,
            confidence=confidence,
            is_complex=True,
            metadata={
                'num_sub_queries': len(sub_queries),
                'avg_length': sum(len(sq.text) for sq in sub_queries) / len(sub_queries)
            }
        )


# Singleton instance
_decomposer: Optional[QueryDecomposer] = None


def get_query_decomposer() -> QueryDecomposer:
    """Get or create the global query decomposer instance.

    Returns:
        Query decomposer instance
    """
    global _decomposer

    if _decomposer is None:
        _decomposer = QueryDecomposer()

    return _decomposer


async def test_decomposer():
    """Test query decomposer functionality."""
    print("Testing Query Decomposer...")
    print("=" * 70)

    decomposer = get_query_decomposer()

    # Test queries
    test_cases = [
        (
            "How to implement FastAPI with async database connections and handle CORS?",
            "Complex query with conjunctions"
        ),
        (
            "What are: 1) best RAG practices 2) chunking strategies 3) embedding models",
            "List-based query"
        ),
        (
            "How does vector search work? What about keyword search? How do you combine them?",
            "Multiple questions"
        ),
        (
            "Explain RAG systems",
            "Simple query (should not decompose)"
        ),
        (
            "Implement authentication; Add database migrations; Set up CORS",
            "Semicolon-separated tasks"
        )
    ]

    for query, description in test_cases:
        print(f"\nQuery: {query}")
        print(f"Type: {description}")
        print("-" * 70)

        result = await decomposer.decompose(query)

        print(f"Complex: {result.is_complex}")
        print(f"Strategy: {result.strategy_used.value}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Sub-queries: {len(result.sub_queries)}")

        if result.is_complex:
            for sq in result.sub_queries:
                print(f"  [{sq.index}] {sq.text}")

    print("\n" + "=" * 70)
    print("Query decomposer test complete!")


if __name__ == "__main__":
    asyncio.run(test_decomposer())
