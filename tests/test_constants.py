"""Tests for constants module."""

import pytest
from rag_cli.core import constants


class TestCacheConstants:
    """Tests for cache-related constants."""

    def test_tcp_check_cache_seconds(self):
        """Test TCP cache timeout constant."""
        assert constants.TCP_CHECK_CACHE_SECONDS == 30
        assert isinstance(constants.TCP_CHECK_CACHE_SECONDS, int)

    def test_response_cache_max_size(self):
        """Test response cache size constant."""
        assert constants.RESPONSE_CACHE_MAX_SIZE == 100
        assert isinstance(constants.RESPONSE_CACHE_MAX_SIZE, int)

    def test_embedding_cache_size(self):
        """Test embedding cache size constant."""
        assert constants.EMBEDDING_CACHE_SIZE == 1000
        assert isinstance(constants.EMBEDDING_CACHE_SIZE, int)


class TestTokenConstants:
    """Tests for token estimation constants."""

    def test_chars_per_token(self):
        """Test characters per token constant."""
        assert constants.CHARS_PER_TOKEN == 4
        assert isinstance(constants.CHARS_PER_TOKEN, int)

    def test_token_estimation_ratio(self):
        """Test token estimation ratio constant."""
        assert constants.TOKEN_ESTIMATION_RATIO == 0.25
        assert isinstance(constants.TOKEN_ESTIMATION_RATIO, float)

    def test_token_estimation_consistency(self):
        """Test token estimation constants are consistent."""
        # CHARS_PER_TOKEN * TOKEN_ESTIMATION_RATIO should equal 1
        assert constants.CHARS_PER_TOKEN * constants.TOKEN_ESTIMATION_RATIO == 1.0


class TestSearchConstants:
    """Tests for search-related constants."""

    def test_default_top_k(self):
        """Test default top_k constant."""
        assert constants.DEFAULT_TOP_K == 5
        assert isinstance(constants.DEFAULT_TOP_K, int)
        assert constants.DEFAULT_TOP_K > 0

    def test_max_top_k(self):
        """Test max top_k constant."""
        assert constants.MAX_TOP_K == 100
        assert isinstance(constants.MAX_TOP_K, int)
        assert constants.MAX_TOP_K >= constants.DEFAULT_TOP_K

    def test_max_query_length(self):
        """Test max query length constant."""
        assert constants.MAX_QUERY_LENGTH == 10000
        assert isinstance(constants.MAX_QUERY_LENGTH, int)


class TestRetrievalWeights:
    """Tests for retrieval weight constants."""

    def test_default_vector_weight(self):
        """Test default vector weight constant."""
        assert constants.DEFAULT_VECTOR_WEIGHT == 0.7
        assert isinstance(constants.DEFAULT_VECTOR_WEIGHT, float)
        assert 0 <= constants.DEFAULT_VECTOR_WEIGHT <= 1

    def test_default_keyword_weight(self):
        """Test default keyword weight constant."""
        assert constants.DEFAULT_KEYWORD_WEIGHT == 0.3
        assert isinstance(constants.DEFAULT_KEYWORD_WEIGHT, float)
        assert 0 <= constants.DEFAULT_KEYWORD_WEIGHT <= 1

    def test_weights_sum_to_one(self):
        """Test that vector and keyword weights sum to 1.0."""
        total = constants.DEFAULT_VECTOR_WEIGHT + constants.DEFAULT_KEYWORD_WEIGHT
        assert abs(total - 1.0) < 1e-10  # Use epsilon for float comparison


class TestFileProcessingConstants:
    """Tests for file processing constants."""

    def test_chunk_size_tokens(self):
        """Test chunk size constant."""
        assert constants.CHUNK_SIZE_TOKENS == 500
        assert isinstance(constants.CHUNK_SIZE_TOKENS, int)

    def test_chunk_overlap_tokens(self):
        """Test chunk overlap constant."""
        assert constants.CHUNK_OVERLAP_TOKENS == 100
        assert isinstance(constants.CHUNK_OVERLAP_TOKENS, int)
        assert constants.CHUNK_OVERLAP_TOKENS < constants.CHUNK_SIZE_TOKENS

    def test_max_file_size_mb(self):
        """Test max file size constant."""
        assert constants.MAX_FILE_SIZE_MB == 10
        assert isinstance(constants.MAX_FILE_SIZE_MB, int)


class TestVectorStoreThresholds:
    """Tests for vector store threshold constants."""

    def test_hnsw_threshold_vectors(self):
        """Test HNSW threshold constant."""
        assert constants.HNSW_THRESHOLD_VECTORS == 2000
        assert isinstance(constants.HNSW_THRESHOLD_VECTORS, int)

    def test_ivf_threshold_vectors(self):
        """Test IVF threshold constant."""
        assert constants.IVF_THRESHOLD_VECTORS == 1_000_000
        assert isinstance(constants.IVF_THRESHOLD_VECTORS, int)
        assert constants.IVF_THRESHOLD_VECTORS > constants.HNSW_THRESHOLD_VECTORS


class TestPerformanceTuning:
    """Tests for performance tuning constants."""

    def test_default_batch_size(self):
        """Test default batch size constant."""
        assert constants.DEFAULT_BATCH_SIZE == 32
        assert isinstance(constants.DEFAULT_BATCH_SIZE, int)
        assert constants.DEFAULT_BATCH_SIZE > 0

    def test_max_workers(self):
        """Test max workers constant."""
        assert constants.MAX_WORKERS == 4
        assert isinstance(constants.MAX_WORKERS, int)
        assert constants.MAX_WORKERS > 0


class TestMonitoringLimits:
    """Tests for monitoring limit constants."""

    def test_max_event_history(self):
        """Test max event history constant."""
        assert constants.MAX_EVENT_HISTORY == 100
        assert isinstance(constants.MAX_EVENT_HISTORY, int)

    def test_metrics_history_size(self):
        """Test metrics history size constant."""
        assert constants.METRICS_HISTORY_SIZE == 1000
        assert isinstance(constants.METRICS_HISTORY_SIZE, int)


class TestAPILimits:
    """Tests for API limit constants."""

    def test_tavily_free_tier_limit(self):
        """Test Tavily free tier limit constant."""
        assert constants.TAVILY_FREE_TIER_LIMIT == 1000
        assert isinstance(constants.TAVILY_FREE_TIER_LIMIT, int)

    def test_claude_rate_limit_requests(self):
        """Test Claude rate limit constant."""
        assert constants.CLAUDE_RATE_LIMIT_REQUESTS == 100
        assert isinstance(constants.CLAUDE_RATE_LIMIT_REQUESTS, int)


class TestTimeouts:
    """Tests for timeout constants."""

    def test_default_http_timeout(self):
        """Test default HTTP timeout constant."""
        assert constants.DEFAULT_HTTP_TIMEOUT == 30
        assert isinstance(constants.DEFAULT_HTTP_TIMEOUT, int)

    def test_embedding_timeout(self):
        """Test embedding timeout constant."""
        assert constants.EMBEDDING_TIMEOUT == 60
        assert isinstance(constants.EMBEDDING_TIMEOUT, int)

    def test_search_timeout(self):
        """Test search timeout constant."""
        assert constants.SEARCH_TIMEOUT == 10
        assert isinstance(constants.SEARCH_TIMEOUT, int)

    def test_cache_stale_threshold(self):
        """Test cache stale threshold constant."""
        assert constants.CACHE_STALE_THRESHOLD_SECONDS == 300
        assert isinstance(constants.CACHE_STALE_THRESHOLD_SECONDS, int)


class TestConstantTypes:
    """Tests for ensuring constants have correct types."""

    def test_all_int_constants(self):
        """Test all integer constants are actually integers."""
        int_constants = [
            'TCP_CHECK_CACHE_SECONDS',
            'RESPONSE_CACHE_MAX_SIZE',
            'EMBEDDING_CACHE_SIZE',
            'CACHE_STALE_THRESHOLD_SECONDS',
            'CHARS_PER_TOKEN',
            'DEFAULT_TOP_K',
            'MAX_TOP_K',
            'MAX_QUERY_LENGTH',
            'CHUNK_SIZE_TOKENS',
            'CHUNK_OVERLAP_TOKENS',
            'MAX_FILE_SIZE_MB',
            'HNSW_THRESHOLD_VECTORS',
            'IVF_THRESHOLD_VECTORS',
            'DEFAULT_BATCH_SIZE',
            'MAX_WORKERS',
            'MAX_EVENT_HISTORY',
            'METRICS_HISTORY_SIZE',
            'TAVILY_FREE_TIER_LIMIT',
            'CLAUDE_RATE_LIMIT_REQUESTS',
            'DEFAULT_HTTP_TIMEOUT',
            'EMBEDDING_TIMEOUT',
            'SEARCH_TIMEOUT',
        ]

        for const_name in int_constants:
            value = getattr(constants, const_name)
            assert isinstance(value, int), f"{const_name} should be int, got {type(value)}"

    def test_all_float_constants(self):
        """Test all float constants are actually floats."""
        float_constants = [
            'TOKEN_ESTIMATION_RATIO',
            'DEFAULT_VECTOR_WEIGHT',
            'DEFAULT_KEYWORD_WEIGHT',
        ]

        for const_name in float_constants:
            value = getattr(constants, const_name)
            assert isinstance(value, float), f"{const_name} should be float, got {type(value)}"


class TestConstantRanges:
    """Tests for validating constant value ranges."""

    def test_positive_constants(self):
        """Test constants that should be positive."""
        positive_constants = [
            'TCP_CHECK_CACHE_SECONDS',
            'RESPONSE_CACHE_MAX_SIZE',
            'EMBEDDING_CACHE_SIZE',
            'CACHE_STALE_THRESHOLD_SECONDS',
            'CHARS_PER_TOKEN',
            'DEFAULT_TOP_K',
            'MAX_TOP_K',
            'MAX_QUERY_LENGTH',
            'CHUNK_SIZE_TOKENS',
            'CHUNK_OVERLAP_TOKENS',
            'MAX_FILE_SIZE_MB',
            'HNSW_THRESHOLD_VECTORS',
            'IVF_THRESHOLD_VECTORS',
            'DEFAULT_BATCH_SIZE',
            'MAX_WORKERS',
            'MAX_EVENT_HISTORY',
            'METRICS_HISTORY_SIZE',
            'TAVILY_FREE_TIER_LIMIT',
            'CLAUDE_RATE_LIMIT_REQUESTS',
            'DEFAULT_HTTP_TIMEOUT',
            'EMBEDDING_TIMEOUT',
            'SEARCH_TIMEOUT',
        ]

        for const_name in positive_constants:
            value = getattr(constants, const_name)
            assert value > 0, f"{const_name} should be positive, got {value}"

    def test_probability_constants(self):
        """Test constants that represent probabilities (0-1 range)."""
        probability_constants = [
            'TOKEN_ESTIMATION_RATIO',
            'DEFAULT_VECTOR_WEIGHT',
            'DEFAULT_KEYWORD_WEIGHT',
        ]

        for const_name in probability_constants:
            value = getattr(constants, const_name)
            assert 0 <= value <= 1, f"{const_name} should be in [0, 1], got {value}"


class TestConstantUsageExamples:
    """Tests demonstrating how constants should be used."""

    def test_token_estimation_usage(self):
        """Example: Using token estimation constants."""
        text = "This is a test string with some words"
        char_count = len(text)

        # Estimate tokens
        estimated_tokens = char_count * constants.TOKEN_ESTIMATION_RATIO

        assert estimated_tokens == char_count / constants.CHARS_PER_TOKEN

    def test_chunk_size_usage(self):
        """Example: Using chunk size constants."""
        # Simulate document chunking
        total_tokens = 1500
        chunks_needed = (total_tokens // (constants.CHUNK_SIZE_TOKENS - constants.CHUNK_OVERLAP_TOKENS)) + 1

        assert chunks_needed > 0

    def test_vector_store_threshold_usage(self):
        """Example: Using vector store thresholds."""
        def get_index_type(vector_count: int) -> str:
            """Determine index type based on vector count."""
            if vector_count < constants.HNSW_THRESHOLD_VECTORS:
                return "flat"
            elif vector_count < constants.IVF_THRESHOLD_VECTORS:
                return "hnsw"
            else:
                return "ivf"

        assert get_index_type(1000) == "flat"
        assert get_index_type(5000) == "hnsw"
        assert get_index_type(2_000_000) == "ivf"
