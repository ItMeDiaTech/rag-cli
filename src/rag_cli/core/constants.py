"""Global constants for RAG-CLI.

This module contains all magic numbers and configuration constants used throughout
the RAG-CLI codebase, making them easier to tune and maintain.
"""

# Cache configuration
TCP_CHECK_CACHE_SECONDS = 30
"""Time to cache TCP server availability check (seconds)"""

RESPONSE_CACHE_MAX_SIZE = 100
"""Maximum number of responses to cache"""

CACHE_STALE_THRESHOLD_SECONDS = 300
"""Time before cache is considered stale (seconds)"""

EMBEDDING_CACHE_SIZE = 1000
"""Maximum number of embeddings to cache"""

# Token estimation
CHARS_PER_TOKEN = 4
"""Approximate characters per token for quick estimation"""

TOKEN_ESTIMATION_RATIO = 0.25
"""Token/character ratio (1 token ≈ 4 chars)"""

# Search configuration
DEFAULT_TOP_K = 5
"""Default number of search results to return"""

MAX_TOP_K = 100
"""Maximum number of search results allowed"""

MAX_QUERY_LENGTH = 10000
"""Maximum query length in characters"""

# Retrieval weights
DEFAULT_VECTOR_WEIGHT = 0.7
"""Default weight for vector search in hybrid retrieval"""

DEFAULT_KEYWORD_WEIGHT = 0.3
"""Default weight for keyword search in hybrid retrieval"""

# File processing
MAX_FILE_SIZE_MB = 10
"""Maximum file size to process (megabytes)"""

CHUNK_SIZE_TOKENS = 500
"""Target chunk size in tokens"""

CHUNK_OVERLAP_TOKENS = 100
"""Overlap between chunks in tokens"""

# Vector store
HNSW_THRESHOLD_VECTORS = 2000
"""Minimum vectors to use HNSW index instead of flat"""

IVF_THRESHOLD_VECTORS = 1000000
"""Minimum vectors to use IVF index instead of HNSW"""

# Performance
DEFAULT_BATCH_SIZE = 32
"""Default batch size for embedding generation"""

MAX_WORKERS = 4
"""Default maximum worker threads/processes"""

# Monitoring
MAX_EVENT_HISTORY = 100
"""Maximum events to keep in history"""

METRICS_HISTORY_SIZE = 1000
"""Maximum metrics data points to keep"""

# API limits
TAVILY_FREE_TIER_LIMIT = 1000
"""Tavily API free tier monthly limit"""

CLAUDE_RATE_LIMIT_REQUESTS = 100
"""Claude API rate limit (requests per minute)"""

# Timeouts
DEFAULT_HTTP_TIMEOUT = 30
"""Default HTTP request timeout (seconds)"""

EMBEDDING_TIMEOUT = 60
"""Timeout for embedding generation (seconds)"""

SEARCH_TIMEOUT = 10
"""Timeout for vector search (seconds)"""

# Similarity thresholds
SIMILARITY_THRESHOLD = 0.85
"""Default similarity threshold for deduplication and matching"""

# Cache TTL
RESPONSE_CACHE_TTL_SECONDS = 300
"""Response cache TTL (seconds) - 5 minutes"""

# Backoff configuration
MAX_BACKOFF_SECONDS = 240
"""Maximum backoff time for exponential backoff (seconds) - 4 minutes"""
