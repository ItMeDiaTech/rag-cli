"""Unit tests for query classifier.

Tests intent detection, entity extraction, confidence scoring, and edge cases.
"""

import pytest
from core.query_classifier import (
    QueryClassifier,
    QueryIntent,
    TechnicalDepth,
    TechnicalEntity,
    QueryClassification,
    get_query_classifier
)


class TestQueryClassifier:
    """Test suite for QueryClassifier."""

    @pytest.fixture
    def classifier(self):
        """Create classifier instance for tests."""
        return QueryClassifier(confidence_threshold=0.3)

    def test_code_explanation_intent(self, classifier):
        """Test detection of code explanation queries."""
        queries = [
            "Explain this code",
            "What does this function do?",
            "How does this class work?",
            "Break down this method for me"
        ]

        for query in queries:
            result = classifier.classify(query)
            assert result.primary_intent == QueryIntent.CODE_EXPLANATION
            assert result.confidence > 0.3

    def test_troubleshooting_intent(self, classifier):
        """Test detection of troubleshooting queries."""
        queries = [
            "I'm getting an error",
            "Exception thrown when running",
            "Bug in my code",
            "Not working as expected",
            "Failed to execute"
        ]

        for query in queries:
            result = classifier.classify(query)
            assert result.primary_intent == QueryIntent.TROUBLESHOOTING
            assert result.confidence > 0.3

    def test_how_to_intent(self, classifier):
        """Test detection of how-to queries."""
        queries = [
            "How to create a Flask application",
            "How do I setup Django?",
            "Steps to configure FastAPI",
            "Guide to building React app"
        ]

        for query in queries:
            result = classifier.classify(query)
            assert result.primary_intent == QueryIntent.HOW_TO
            assert result.confidence > 0.3

    def test_best_practices_intent(self, classifier):
        """Test detection of best practices queries."""
        queries = [
            "What are best practices for Python error handling?",
            "Recommended way to structure Flask project",
            "Should I use async in FastAPI?",
            "What's the idiomatic approach?",
            "Is it good to use global variables?"
        ]

        for query in queries:
            result = classifier.classify(query)
            assert result.primary_intent == QueryIntent.BEST_PRACTICES
            assert result.confidence > 0.3

    def test_comparison_intent(self, classifier):
        """Test detection of comparison queries."""
        queries = [
            "Flask vs Django",
            "Compare React and Vue",
            "Difference between REST and GraphQL",
            "Which is better: PostgreSQL or MySQL?"
        ]

        for query in queries:
            result = classifier.classify(query)
            assert result.primary_intent == QueryIntent.COMPARISON
            assert result.confidence > 0.3

    def test_technical_docs_intent(self, classifier):
        """Test detection of technical documentation queries."""
        queries = [
            "FastAPI configuration options",
            "Django settings reference",
            "Python API documentation",
            "What are the parameters for requests.get?"
        ]

        for query in queries:
            result = classifier.classify(query)
            assert result.primary_intent == QueryIntent.TECHNICAL_DOCS
            assert result.confidence > 0.3

    def test_conceptual_intent(self, classifier):
        """Test detection of conceptual queries."""
        queries = [
            "What is REST architecture?",
            "What are microservices?",
            "Why use async programming?",
            "Explain the concept of middleware"
        ]

        for query in queries:
            result = classifier.classify(query)
            assert result.primary_intent == QueryIntent.CONCEPTUAL
            assert result.confidence > 0.3

    def test_entity_extraction_languages(self, classifier):
        """Test extraction of programming language entities."""
        query = "How to write Python code for JavaScript interop?"
        result = classifier.classify(query)

        entity_names = [e.name.lower() for e in result.entities]
        assert 'python' in entity_names
        assert 'javascript' in entity_names

    def test_entity_extraction_frameworks(self, classifier):
        """Test extraction of framework entities."""
        query = "Build a Django app with React frontend"
        result = classifier.classify(query)

        entity_names = [e.name.lower() for e in result.entities]
        assert 'django' in entity_names
        assert 'react' in entity_names

    def test_technical_depth_beginner(self, classifier):
        """Test detection of beginner-level queries."""
        queries = [
            "I'm new to Python, how do I start?",
            "Basic introduction to Flask",
            "Getting started with React",
            "Simple tutorial for beginners"
        ]

        for query in queries:
            result = classifier.classify(query)
            assert result.technical_depth == TechnicalDepth.BEGINNER

    def test_technical_depth_advanced(self, classifier):
        """Test detection of advanced-level queries."""
        queries = [
            "How to optimize performance in production?",
            "Advanced architectural patterns",
            "Deep dive into internals",
            "Custom implementation of middleware"
        ]

        for query in queries:
            result = classifier.classify(query)
            assert result.technical_depth == TechnicalDepth.ADVANCED

    def test_technical_depth_intermediate(self, classifier):
        """Test default intermediate depth."""
        query = "How to create a REST API with FastAPI?"
        result = classifier.classify(query)

        assert result.technical_depth == TechnicalDepth.INTERMEDIATE

    def test_non_technical_query_detection(self, classifier):
        """Test filtering of non-technical queries."""
        queries = [
            "Hello, how are you?",
            "Thank you for your help",
            "What's up?",
            "Sorry about that"
        ]

        for query in queries:
            result = classifier.classify(query)
            assert result.is_technical == False

    def test_technical_query_detection(self, classifier):
        """Test detection of technical queries."""
        queries = [
            "How to use Python's asyncio?",
            "FastAPI route configuration",
            "Django ORM queries",
            "React component lifecycle"
        ]

        for query in queries:
            result = classifier.classify(query)
            assert result.is_technical == True

    def test_confidence_scoring(self, classifier):
        """Test confidence scores are reasonable."""
        # Strong intent query
        strong_query = "What are the best practices for error handling in Python?"
        strong_result = classifier.classify(strong_query)

        # Weak intent query
        weak_query = "Tell me something"
        weak_result = classifier.classify(weak_query)

        assert strong_result.confidence > weak_result.confidence
        assert 0.0 <= strong_result.confidence <= 1.0
        assert 0.0 <= weak_result.confidence <= 1.0

    def test_multi_intent_detection(self, classifier):
        """Test detection of multiple intents in single query."""
        query = "How to fix error in FastAPI and what are best practices?"
        result = classifier.classify(query)

        # Should detect multiple intents
        assert len(result.all_intents) > 1
        assert QueryIntent.TROUBLESHOOTING in result.all_intents
        assert QueryIntent.BEST_PRACTICES in result.all_intents

    def test_keyword_extraction(self, classifier):
        """Test extraction of important keywords."""
        query = "How to optimize database queries in Django?"
        result = classifier.classify(query)

        assert len(result.keywords) > 0
        assert 'optimize' in result.keywords or 'database' in result.keywords or 'django' in result.keywords

    def test_confidence_threshold(self):
        """Test custom confidence threshold."""
        high_threshold_classifier = QueryClassifier(confidence_threshold=0.7)

        weak_query = "maybe something about code"
        result = high_threshold_classifier.classify(weak_query)

        # With high threshold, weak queries should have fewer detected intents
        assert len(result.all_intents) <= 1

    def test_empty_query(self, classifier):
        """Test handling of empty query."""
        result = classifier.classify("")

        assert result.primary_intent == QueryIntent.GENERAL_QA
        assert result.confidence >= 0.0

    def test_very_short_query(self, classifier):
        """Test handling of very short queries."""
        result = classifier.classify("Python?")

        assert result.is_technical == True
        assert len(result.entities) > 0

    def test_very_long_query(self, classifier):
        """Test handling of very long queries."""
        long_query = "How do I " + "implement " * 50 + "a feature in Flask?"
        result = classifier.classify(long_query)

        assert result.primary_intent in [QueryIntent.HOW_TO, QueryIntent.GENERAL_QA]
        assert result.is_technical == True

    def test_query_with_code(self, classifier):
        """Test handling of queries with code snippets."""
        query = "Explain this code: def foo(): return 42"
        result = classifier.classify(query)

        assert result.primary_intent == QueryIntent.CODE_EXPLANATION
        assert result.is_technical == True

    def test_query_with_numbers(self, classifier):
        """Test handling of queries with version numbers."""
        query = "How to use Python 3.11 features?"
        result = classifier.classify(query)

        assert result.is_technical == True
        python_entities = [e for e in result.entities if e.name.lower() == 'python']
        assert len(python_entities) > 0

    def test_get_query_classifier_function(self):
        """Test the get_query_classifier factory function."""
        classifier = get_query_classifier(confidence_threshold=0.5)

        assert isinstance(classifier, QueryClassifier)
        assert classifier.confidence_threshold == 0.5

    def test_entity_confidence(self, classifier):
        """Test entity confidence scores."""
        query = "How to use FastAPI with Python?"
        result = classifier.classify(query)

        for entity in result.entities:
            assert 0.0 <= entity.confidence <= 1.0

    def test_case_insensitivity(self, classifier):
        """Test that classification is case-insensitive."""
        query_lower = "how to use python?"
        query_upper = "HOW TO USE PYTHON?"
        query_mixed = "How To Use Python?"

        result_lower = classifier.classify(query_lower)
        result_upper = classifier.classify(query_upper)
        result_mixed = classifier.classify(query_mixed)

        assert result_lower.primary_intent == result_upper.primary_intent == result_mixed.primary_intent

    def test_special_characters(self, classifier):
        """Test handling of special characters."""
        query = "How to use React's useState() hook?"
        result = classifier.classify(query)

        assert result.primary_intent == QueryIntent.HOW_TO
        assert result.is_technical == True

    def test_multiple_languages_in_query(self, classifier):
        """Test detection of multiple programming languages."""
        query = "How to call Python from JavaScript and integrate with Go?"
        result = classifier.classify(query)

        entity_names = [e.name.lower() for e in result.entities]
        assert 'python' in entity_names
        assert 'javascript' in entity_names
        assert 'go' in entity_names


@pytest.mark.parametrize("query,expected_intent", [
    ("How to fix ImportError?", QueryIntent.TROUBLESHOOTING),
    ("Explain async/await", QueryIntent.CODE_EXPLANATION),
    ("Python vs JavaScript", QueryIntent.COMPARISON),
    ("FastAPI best practices", QueryIntent.BEST_PRACTICES),
    ("What is Docker?", QueryIntent.CONCEPTUAL),
    ("How to setup Django?", QueryIntent.HOW_TO),
    ("API reference for requests library", QueryIntent.TECHNICAL_DOCS),
])
def test_intent_detection_parameterized(query, expected_intent):
    """Parameterized test for various query intents."""
    classifier = QueryClassifier()
    result = classifier.classify(query)

    assert result.primary_intent == expected_intent


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
