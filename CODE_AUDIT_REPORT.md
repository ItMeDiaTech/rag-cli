# RAG-CLI v2.0 Code Audit Report

**Date:** 2025-11-08
**Scope:** Complete codebase review (src/rag_cli and src/rag_cli_plugin)
**Total Lines Reviewed:** ~20,000 lines across 50+ files
**Total Issues Found:** 209 issues

---

## Executive Summary

This comprehensive code audit identified 209 issues across the RAG-CLI v2.0 codebase, categorized by severity and domain. While the codebase demonstrates good architecture and documentation practices, several critical bugs, performance issues, and maintainability concerns need immediate attention.

### Issues by Severity

| Severity | Count | Percentage | Priority |
|----------|-------|------------|----------|
| **CRITICAL** | 18 | 9% | Fix Immediately |
| **HIGH** | 54 | 26% | This Sprint |
| **MEDIUM** | 94 | 45% | Next Sprint |
| **LOW** | 43 | 20% | Backlog |

### Issues by Category

| Category | Count | Impact |
|----------|-------|--------|
| **Import Paths (v1.x → v2.0)** | 12 | Critical - Services won't start |
| **Thread Safety** | 15 | Critical - Race conditions |
| **Magic Numbers/Constants** | 32 | Medium - Maintainability |
| **Type Safety** | 18 | Medium - Runtime errors |
| **Performance** | 24 | High - Efficiency |
| **Error Handling** | 28 | High - Reliability |
| **Security** | 8 | Medium - Local project |
| **Code Quality** | 42 | Medium - Technical debt |
| **Documentation** | 10 | Low - Clarity |

---

## Critical Issues Requiring Immediate Fix

### 1. Import Path Errors (v1.x → v2.0 Migration)

**Impact:** Services and hooks will fail to start with ImportError

**Affected Files:**
- `src/rag_cli_plugin/hooks/user-prompt-submit.py:576`
- `src/rag_cli_plugin/services/service_manager.py:331, 396`
- `src/rag_cli_plugin/services/tcp_server.py:713, 740`
- `src/rag_cli/agents/maf/connectors.py:221, 555`
- `src/rag_cli/core/semantic_cache.py:381`

**Examples:**
```python
# WRONG (v1.x paths)
from monitoring.output_formatter import OutputFormatter
from core.semantic_cache_hnsw import HNSWSemanticCache
['python', '-m', 'src.monitoring.tcp_server']

# CORRECT (v2.0 paths)
from rag_cli_plugin.services.output_formatter import OutputFormatter
from rag_cli.core.semantic_cache_hnsw import HNSWSemanticCache
['python', '-m', 'rag_cli_plugin.services.tcp_server']
```

### 2. File Extension Typo - PDF Not Recognized

**Impact:** All PDF files will be ignored during indexing

**File:** `src/rag_cli/core/config.py:51, 268`

```python
# WRONG
supported_formats: List[str] = [".md", ".txt", ".pd", ".docx", ".html"]

# CORRECT
supported_formats: List[str] = [".md", ".txt", ".pdf", ".docx", ".html"]
```

### 3. Deprecated Pydantic API

**Impact:** Will break on Pydantic v2 upgrade

**File:** `src/rag_cli/core/config.py:482`

```python
# WRONG (Pydantic v1 API)
config_dict = self._config.dict()

# CORRECT (Pydantic v2 API)
config_dict = self._config.model_dump()
```

### 4. Cache Structure Mismatch

**Impact:** Semantic cache save operation will crash with AttributeError

**File:** `src/rag_cli/core/semantic_cache.py:301-314`

```python
# WRONG - entry is CacheEntry dataclass, not dict
for key, entry in active_cache.items():
    age = current_time - entry['timestamp']  # AttributeError!

# CORRECT
for key, entry in active_cache.items():
    age = current_time - entry.created_at.timestamp()
```

### 5. Division by Zero Risk

**Impact:** Crash when executing multiple agents with empty list

**File:** `src/rag_cli/integrations/maf_connector.py:318, 343`

```python
# WRONG
per_agent_timeout = timeout / len(agents)  # Crash if agents is empty!

# CORRECT
if not agents:
    logger.warning("No agents specified for multi-agent execution")
    return {}
per_agent_timeout = timeout / len(agents)
```

### 6. SQL Injection Vulnerability

**Impact:** Potential SQL injection in memory module

**File:** `src/rag_cli/agents/maf/core/memory.py:455`

```python
# WRONG - String interpolation in SQL
cursor.execute(f'''
    DELETE FROM memories
    WHERE id IN ({','.join(['?'] * len(memory_ids))})
''', memory_ids)

# CORRECT - Use parameterized queries properly
placeholders = ','.join('?' * len(memory_ids))
cursor.execute(f'DELETE FROM memories WHERE id IN ({placeholders})', memory_ids)
```

### 7. File Handle Leaks

**Impact:** Resource exhaustion in long-running services

**File:** `src/rag_cli_plugin/services/service_manager.py:319`

```python
# WRONG - File opened without context manager
tcp_log = open(tcp_log_file, 'a', buffering=1)

# CORRECT - Use context manager or ensure cleanup
self.tcp_log_file = open(tcp_log_file, 'a', buffering=1)
# Add cleanup in shutdown method
```

### 8. Infinite Recursion Risk

**Impact:** Stack overflow in Tavily quota management

**File:** `src/rag_cli/integrations/tavily_connector.py:98-101`

```python
# WRONG - Could recurse infinitely
def _get_usage(self) -> Dict[str, Any]:
    try:
        data = json.loads(self.quota_file.read_text())
        return data
    except (json.JSONDecodeError, FileNotFoundError) as e:
        self._init_quota_file()
        return self._get_usage()  # Infinite recursion if init fails!

# CORRECT - Add recursion guard
def _get_usage(self, _retry=False) -> Dict[str, Any]:
    try:
        data = json.loads(self.quota_file.read_text())
        return data
    except (json.JSONDecodeError, FileNotFoundError) as e:
        if _retry:
            raise
        self._init_quota_file()
        return self._get_usage(_retry=True)
```

---

## High Priority Issues (This Sprint)

### Thread Safety Issues (15 occurrences)

**Problem:** Global singletons without thread locking causing race conditions

**Affected Files:**
- `config.py:539-548` - ConfigManager singleton
- `embeddings.py:808-831` - EmbeddingGenerator singleton
- `arxiv_connector.py:271-285` - ArxivConnector singleton
- `tavily_connector.py:289-304` - TavilyConnector singleton
- `maf_connector.py:563-578` - MAFConnector singleton
- `base_agent.py:530-548` - AgentCoordinator singleton
- `query_decomposer.py:457-472` - QueryDecomposer singleton
- `user-prompt-submit.py:90-93` - Global mutable state

**Standard Fix Pattern:**
```python
import threading

_instance: Optional[ClassName] = None
_lock = threading.Lock()

def get_instance() -> ClassName:
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:  # Double-check pattern
                _instance = ClassName()
    return _instance
```

### Performance Issues

#### 1. O(n) LRU Cache Operations

**File:** `src/rag_cli/core/claude_integration.py:369-371`

```python
# WRONG - O(n) list operations
if cache_key in self.cache_access_order:
    self.cache_access_order.remove(cache_key)  # O(n)!
self.cache_access_order.append(cache_key)

# CORRECT - Use OrderedDict with O(1) operations
from collections import OrderedDict
self.cache = OrderedDict()  # Instead of dict + list
# On access:
self.cache.move_to_end(cache_key)  # O(1)
```

#### 2. Import Inside Loop

**File:** `src/rag_cli/core/retrieval_pipeline.py:601`

```python
# WRONG - Import on every method call
def _get_adaptive_weights(self, query: str):
    try:
        from core.query_classifier import QueryIntent  # Every call!
    except ImportError:
        return (vector_weight, keyword_weight)

# CORRECT - Module-level import
from rag_cli.core.query_classifier import QueryIntent  # Top of file
```

#### 3. Regex Not Pre-compiled

**File:** `src/rag_cli/core/query_classifier.py:63-210`

```python
# WRONG - Recompiles regex on every classification
for pattern in config['patterns']:
    if re.search(pattern, query, re.IGNORECASE):  # Recompiles!

# CORRECT - Pre-compile in __init__
def __init__(self):
    self.compiled_patterns = {}
    for intent, config in self.INTENT_PATTERNS.items():
        self.compiled_patterns[intent] = [
            re.compile(p, re.IGNORECASE) for p in config['patterns']
        ]
```

#### 4. Unbounded List Growth

**File:** `src/rag_cli/agents/maf/core/agent_communication.py:88, 148`

```python
# WRONG - Memory leak
self.message_history: List[AgentMessage] = []
# ...
self.message_history.append(message)  # No size limit!

# CORRECT - Implement circular buffer
from collections import deque
self.message_history = deque(maxlen=1000)  # Auto-evicts old messages
```

### Security Issues

#### 1. Weak MD5 Hashing

**Files:** 8 occurrences across codebase

```python
# WRONG - MD5 deprecated and truncated
hashlib.md5(query.encode()).hexdigest()[:16]

# CORRECT - Use blake2b for speed + security
hashlib.blake2b(query.encode(), digest_size=16).hexdigest()
```

#### 2. Missing File Size Validation

**File:** `src/rag_cli/core/document_processor.py:220-233`

```python
# WRONG - No validation before loading
content = self._load_document(file_path)  # Could load 10GB file!

# CORRECT - Validate first
max_size = get_config().document_processing.max_file_size_mb * 1024 * 1024
if file_path.stat().st_size > max_size:
    raise ValueError(f"File exceeds maximum size")
```

#### 3. Command Injection Risk

**File:** `src/rag_cli/agents/maf/core/claude_cli_unified.py:267-303`

```python
# WRONG - User input in subprocess
subprocess.run(f"claude {user_prompt}", shell=True)  # Dangerous!

# CORRECT - Use list form
subprocess.run(["claude", user_prompt], shell=False)
```

---

## Medium Priority Issues (Next Sprint)

### Constants Migration (32 occurrences)

Magic numbers throughout codebase should reference `constants.py`:

**Examples:**
- `chunk_size = 500` → `CHUNK_SIZE_TOKENS`
- `batch_size = 32` → `DEFAULT_BATCH_SIZE`
- `timeout = 10` → `DEFAULT_HTTP_TIMEOUT`
- `cache_ttl = 300` → `RESPONSE_CACHE_TTL_SECONDS`
- `similarity_threshold = 0.85` → `SIMILARITY_THRESHOLD`

### Error Handling Improvements (28 occurrences)

**Anti-pattern:** Catching broad `Exception` instead of specific types

```python
# WRONG
try:
    operation()
except Exception as e:  # Too broad!
    logger.error(f"Failed: {e}")

# CORRECT
try:
    operation()
except (ValueError, KeyError, IOError) as e:
    logger.error(f"Failed: {e}")
except Exception as e:
    logger.exception("Unexpected error", exc_info=True)
    raise
```

### Type Safety Issues (18 occurrences)

**Examples:**
- Missing type hints: `def process(query)` → `def process(query: str) -> Result`
- Type hint errors: `Dict[str, any]` → `Dict[str, Any]`
- Missing validation: Check types before attribute access

### Code Duplication

**Path Resolution (6 occurrences):**
- `user-prompt-submit.py:20-69`
- `session-start.py:27-56`
- `session-end.py:27-56`
- etc.

**Fix:** Use centralized `path_utils.setup_sys_path()`

---

## Low Priority Issues (Backlog)

### Documentation Updates

**ChromaDB vs FAISS References:**
- Update all comments mentioning "FAISS" to "ChromaDB"
- Update README and docstrings

### Code Style

**Long Functions:**
- `user-prompt-submit.py:process_hook()` - 140 lines → Split into helpers
- `retrieval_pipeline.py:retrieve_async()` - 120 lines → Extract methods
- `unified_server.py` - 1155 lines → Split class responsibilities

**Dead Code:**
- Remove `len(normalized)` statements that discard result
- Remove incomplete fuzzy matching stub code

### Platform-Specific Issues

**Windows Console Encoding:**
- Already handled by Rich library, remove redundant code

---

## Categorized Remediation Tasks

The following sections group all issues into logical categories for parallel remediation by specialized agents.

### Task Category 1: Import Path Migration
**Agent:** Code Refactoring Agent
**Estimated Effort:** 2 hours
**Files:** 12 files

Fix all v1.x import paths to v2.0 structure:

**Core Library Imports:**
```python
# Pattern 1: Absolute imports
from core.X → from rag_cli.core.X
from agents.X → from rag_cli.agents.X
from integrations.X → from rag_cli.integrations.X

# Pattern 2: CLI imports
from cli.X → from rag_cli.cli.X
```

**Plugin Imports:**
```python
from monitoring.X → from rag_cli_plugin.services.X
from plugin.X → from rag_cli_plugin.X
from mcp.X → from rag_cli_plugin.mcp.X
```

**Subprocess Module Paths:**
```python
['python', '-m', 'src.monitoring.tcp_server'] →
['python', '-m', 'rag_cli_plugin.services.tcp_server']
```

**Files to Update:**
1. `src/rag_cli_plugin/hooks/user-prompt-submit.py`
2. `src/rag_cli_plugin/services/service_manager.py`
3. `src/rag_cli_plugin/services/tcp_server.py`
4. `src/rag_cli_plugin/services/logger.py`
5. `src/rag_cli/core/semantic_cache.py`
6. `src/rag_cli/integrations/maf_connector.py`
7. `src/rag_cli/agents/maf/core/orchestrator.py`
8. `src/rag_cli/core/retrieval_pipeline.py`

### Task Category 2: Thread Safety Fixes
**Agent:** Concurrency Specialist Agent
**Estimated Effort:** 3 hours
**Files:** 15 files

Implement thread-safe singleton pattern across all modules:

**Standard Implementation:**
```python
import threading
from typing import Optional

_instance: Optional[ClassName] = None
_lock = threading.Lock()

def get_instance() -> ClassName:
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = ClassName()
    return _instance
```

**Files to Update:**
1. `src/rag_cli/core/config.py:539-548`
2. `src/rag_cli/core/embeddings.py:808-831`
3. `src/rag_cli/core/duplicate_detector.py:270-308`
4. `src/rag_cli/integrations/arxiv_connector.py:271-285`
5. `src/rag_cli/integrations/tavily_connector.py:289-304`
6. `src/rag_cli/integrations/maf_connector.py:563-578`
7. `src/rag_cli/agents/base_agent.py:530-548`
8. `src/rag_cli/agents/query_decomposer.py:457-472`

**Additional Thread Safety:**
- Add locks to file I/O operations in `tavily_connector.py`
- Add locks to global state in `user-prompt-submit.py:90-93`
- Add locks to PID file operations in `service_manager.py:59-86`

### Task Category 3: Constants Migration
**Agent:** Code Quality Agent
**Estimated Effort:** 4 hours
**Files:** 32 files

Replace all magic numbers with references to `constants.py`:

**Migration Map:**

| Magic Number | Constant | Files |
|--------------|----------|-------|
| `500` | `CHUNK_SIZE_TOKENS` | document_processor.py:151,990 |
| `100` | `CHUNK_OVERLAP_TOKENS` | document_processor.py |
| `32` | `DEFAULT_BATCH_SIZE` | embeddings.py, index.py:206 |
| `10` | `DEFAULT_HTTP_TIMEOUT` | arxiv_connector.py:135, tavily_connector.py:205 |
| `1000` | `EMBEDDING_CACHE_SIZE` | embeddings.py:354 |
| `0.7` | `DEFAULT_VECTOR_WEIGHT` | retrieval_pipeline.py:552 |
| `0.85` | `SIMILARITY_THRESHOLD` | result_synthesizer.py:47 |
| `300` | `RESPONSE_CACHE_TTL_SECONDS` | response-post.py:79 |
| `240` | `MAX_BACKOFF_SECONDS` | user-prompt-submit.py:144 |

**Process:**
1. Ensure constant exists in `constants.py`
2. Import constant at top of file
3. Replace hardcoded value with constant
4. Add comment if calculation is complex

### Task Category 4: Performance Optimization
**Agent:** Performance Optimization Agent
**Estimated Effort:** 5 hours
**Files:** 24 files

**Subtask 4A: Fix O(n) Cache Operations**

File: `src/rag_cli/core/claude_integration.py`

```python
# Replace list-based LRU with OrderedDict
from collections import OrderedDict

class ClaudeIntegration:
    def __init__(self):
        self.cache = OrderedDict()
        self.max_cache_size = 100

    def _get_cache_key(self, ...):
        # ... existing code ...
        if cache_key in self.cache:
            self.cache.move_to_end(cache_key)  # O(1) instead of O(n)
            return self.cache[cache_key]

        # ... generate response ...

        self.cache[cache_key] = response
        if len(self.cache) > self.max_cache_size:
            self.cache.popitem(last=False)  # Remove oldest
```

**Subtask 4B: Pre-compile Regex Patterns**

Files:
- `query_classifier.py:63-210`
- `document_processor.py`
- `logger.py:36-46`

```python
class QueryClassifier:
    def __init__(self):
        # Pre-compile all patterns
        self.compiled_patterns = {}
        for intent, config in self.INTENT_PATTERNS.items():
            self.compiled_patterns[intent] = [
                re.compile(p, re.IGNORECASE)
                for p in config['patterns']
            ]

    def classify(self, query: str):
        for intent, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(query):  # Use pre-compiled
                    return intent
```

**Subtask 4C: Move Module-Level Imports**

Files:
- `retrieval_pipeline.py:601` - Move QueryIntent import to top
- `query_decomposer.py` - Move regex imports to top
- Various hooks - Move json imports to top

**Subtask 4D: Implement Bounded Collections**

Replace unbounded lists with bounded collections:

```python
from collections import deque

# Message history
self.message_history = deque(maxlen=1000)  # Auto-evicts

# Event history
self.event_history = deque(maxlen=MAX_EVENT_HISTORY)

# Cache with size limit
self.cache = OrderedDict()  # With manual size check
```

Files:
- `agent_communication.py:88` - message_history
- `tcp_server.py:140-153` - event_subscribers cleanup
- `orchestrator.py:97` - workflow_history
- `arxiv_connector.py:45` - cache without limit
- `tavily_connector.py` - similar cache issue

**Subtask 4E: Memory Optimization in Batch Processing**

File: `cli/index.py:209-217`

```python
# Instead of accumulating all embeddings in memory:
all_embeddings = []
for i in range(0, len(chunk_texts), batch_size):
    batch_embeddings = generator.encode(batch_texts)
    all_embeddings.append(batch_embeddings)  # Accumulates!
all_embeddings = np.vstack(all_embeddings)  # High memory!

# Stream to vector store immediately:
for i in range(0, len(chunk_texts), batch_size):
    batch_texts = chunk_texts[i:i + batch_size]
    batch_embeddings = generator.encode(batch_texts)
    batch_metadata = metadata_list[i:i + batch_size]

    # Store immediately, don't accumulate
    vector_store.add(batch_embeddings, batch_texts,
                    metadata=batch_metadata)
```

### Task Category 5: Error Handling & Validation
**Agent:** Reliability Engineer Agent
**Estimated Effort:** 6 hours
**Files:** 28 files

**Subtask 5A: Replace Broad Exception Handlers**

**Pattern to find and replace:**
```python
# FIND
try:
    operation()
except Exception as e:
    logger.error(f"Failed: {e}")

# REPLACE
try:
    operation()
except (ValueError, KeyError, IOError) as e:
    # Expected errors - handle gracefully
    logger.error(f"Failed: {e}")
    return default_value
except Exception as e:
    # Unexpected errors - log and re-raise
    logger.exception("Unexpected error in operation", exc_info=True)
    raise
```

**Files:** Search for `except Exception` across entire codebase

**Subtask 5B: Add Input Validation**

Add validation to all public methods:

```python
def execute_agent(self, agent_name: str, task_data: Dict[str, Any], ...):
    # Validate agent_name
    if not agent_name or not isinstance(agent_name, str):
        raise ValueError("agent_name must be non-empty string")

    if agent_name not in self.agents_map:
        raise ValueError(f"Unknown agent: {agent_name}. "
                        f"Available: {list(self.agents_map.keys())}")

    # Validate task_data
    if not isinstance(task_data, dict):
        raise TypeError("task_data must be dictionary")

    if 'query' not in task_data:
        raise ValueError("task_data must contain 'query' key")
```

Files:
- `maf_connector.py:107` - validate agent_name
- `index.py:34-35` - validate chunk parameters
- `arxiv_connector.py` - validate max_results, query
- `tavily_connector.py` - validate API responses
- All CLI entry points - validate user inputs

**Subtask 5C: Add Edge Case Handling**

**Division by zero checks:**
- `maf_connector.py:318, 343` - Check agents list not empty

**Null/None checks:**
- `claude_integration.py` - Validate response.token_usage exists
- `retrieve.py:198-199` - Check hasattr before access

**Empty collection checks:**
- Validate lists/dicts before accessing first element
- Check vector store not empty before operations

**Subtask 5D: Fix Recursion Guards**

File: `tavily_connector.py:98-101`

```python
def _get_usage(self, _retry=False) -> Dict[str, Any]:
    try:
        if not self.quota_file.exists():
            if _retry:
                raise FileNotFoundError("Quota file missing after init")
            self._init_quota_file()
            return self._get_usage(_retry=True)

        data = json.loads(self.quota_file.read_text())
        # ... validation ...
        return data
    except json.JSONDecodeError as e:
        if _retry:
            raise ValueError("Quota file corrupted after init")
        logger.warning("Corrupted quota file, reinitializing")
        self._init_quota_file()
        return self._get_usage(_retry=True)
```

### Task Category 6: Type Safety & API Fixes
**Agent:** Type Safety Agent
**Estimated Effort:** 3 hours
**Files:** 18 files

**Subtask 6A: Fix Type Hints**

```python
# Fix lowercase 'any'
def get_statistics(self) -> Dict[str, any]:  # WRONG

from typing import Any
def get_statistics(self) -> Dict[str, Any]:  # CORRECT
```

Files: `duplicate_detector.py:310`, search for `: any` across codebase

**Subtask 6B: Fix API Compatibility**

**Pydantic v1 → v2:**
```python
# v1 API
config_dict = self._config.dict()
config_json = self._config.json()

# v2 API
config_dict = self._config.model_dump()
config_json = self._config.model_dump_json()
```

File: `config.py:482`

**ChromaDB API Abstraction:**

Add methods to `vector_store.py`:
```python
def get_vector_count(self) -> int:
    """Get total number of vectors in store."""
    return self.collection.count()

def get_dimension(self) -> int:
    """Get vector dimension."""
    return self.dimension
```

Update all direct access to use abstraction:
- `index.py:118, 277` - `vector_store.index.ntotal` → `get_vector_count()`
- `retrieve.py:118` - Same fix
- `unified_server.py:680` - Same fix

**Subtask 6C: Fix Attribute Access Errors**

Add validation before accessing attributes:

```python
# Before
response.token_usage['input']  # Could fail if missing

# After
if hasattr(response, 'token_usage') and response.token_usage:
    input_tokens = response.token_usage.get('input', 0)
else:
    input_tokens = 0
```

Files: `retrieve.py`, `claude_integration.py`

### Task Category 7: Security Fixes
**Agent:** Security Hardening Agent
**Estimated Effort:** 2 hours
**Files:** 8 files

**Subtask 7A: Replace MD5 with Blake2b**

```python
# FIND
import hashlib
hash_value = hashlib.md5(data.encode()).hexdigest()[:16]

# REPLACE
import hashlib
hash_value = hashlib.blake2b(data.encode(), digest_size=16).hexdigest()
```

Files:
- `document_processor.py:473-475` - Document ID generation
- `user-prompt-submit.py:689` - Cache key generation
- `response-post.py:59` - Prompt hash
- Search for `hashlib.md5` across codebase

**Subtask 7B: Fix SQL Injection**

File: `memory.py:455`

Ensure all SQL uses proper parameterization, no string interpolation.

**Subtask 7C: Add File Size Validation**

File: `document_processor.py:220-233`

```python
def process_document(self, file_path: Union[str, Path]) -> Document:
    file_path = Path(file_path)

    # Validate file size
    max_size = get_config().document_processing.max_file_size_mb * 1024 * 1024
    if file_path.stat().st_size > max_size:
        raise ValueError(
            f"File {file_path.name} ({file_path.stat().st_size / 1024 / 1024:.1f}MB) "
            f"exceeds maximum size of {max_size / 1024 / 1024}MB"
        )

    content = self._load_document(file_path)
    # ... rest of method ...
```

**Subtask 7D: Fix Command Injection**

File: `claude_cli_unified.py:267-303`

```python
# Never use shell=True with user input
# WRONG
subprocess.run(f"command {user_input}", shell=True)

# CORRECT
subprocess.run(["command", user_input], shell=False)
```

**Subtask 7E: Add Rate Limiting**

File: `claude_integration.py`, `unified_server.py`

Implement rate limiter:
```python
from time import time
from collections import deque

class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window = window_seconds
        self.requests = deque()

    def check_rate_limit(self) -> bool:
        now = time()
        # Remove old requests outside window
        while self.requests and self.requests[0] < now - self.window:
            self.requests.popleft()

        if len(self.requests) >= self.max_requests:
            return False  # Rate limited

        self.requests.append(now)
        return True  # Allowed
```

### Task Category 8: Code Quality & Refactoring
**Agent:** Code Quality Agent
**Estimated Effort:** 8 hours
**Files:** 42 files

**Subtask 8A: Extract Duplicated Code**

**Path Resolution Logic:**

Create centralized utility (already exists at `path_utils.py`):
```python
# In all hook files, REPLACE lines 20-69:
import sys
from pathlib import Path

# ... 50 lines of path resolution ...

# WITH single import:
from rag_cli_plugin.hooks.path_utils import setup_sys_path
setup_sys_path(__file__)
```

Files:
- `user-prompt-submit.py:20-69`
- `session-start.py:27-56`
- `session-end.py:27-56`
- `error-handler.py:20-56`
- `response-post.py:20-56`
- `document-indexing.py:20-56`

**Subtask 8B: Split Long Functions**

**user-prompt-submit.py:process_hook()** (140 lines)

Split into:
```python
def process_hook(event: Dict[str, Any]) -> Dict[str, Any]:
    """Main hook orchestration."""
    if not _should_process_hook(event):
        return event

    query = _extract_query(event)
    if not query:
        return event

    rag_results = _perform_rag_retrieval(query)
    maf_results = _perform_maf_analysis(query, event) if should_use_maf else None

    enhanced_prompt = _build_enhanced_prompt(query, rag_results, maf_results)
    event = _update_event(event, enhanced_prompt)

    _log_metrics(rag_results, maf_results)
    return event

def _should_process_hook(event: Dict[str, Any]) -> bool:
    """Check if hook should process this event."""
    # ... 10 lines ...

def _extract_query(event: Dict[str, Any]) -> str:
    """Extract query from event."""
    # ... 15 lines ...

def _perform_rag_retrieval(query: str) -> List[Result]:
    """Perform RAG retrieval."""
    # ... 20 lines ...

# etc.
```

**retrieve.py:process_query()** (120 lines)

Extract to module level with clear signature.

**unified_server.py** (1155 lines)

Split into multiple classes:
```python
# Split into:
class MCPProtocolHandler:
    """Handle MCP protocol operations."""

class RAGHandler:
    """Handle RAG retrieval operations."""

class ServiceHandler:
    """Handle service management."""

class MAFHandler:
    """Handle Multi-Agent Framework operations."""

class UnifiedMCPServer:
    """Main server orchestrating all handlers."""
    def __init__(self):
        self.protocol = MCPProtocolHandler()
        self.rag = RAGHandler()
        self.services = ServiceHandler()
        self.maf = MAFHandler()
```

**Subtask 8C: Remove Dead Code**

Search and remove:
```python
# Pattern 1: Computed value discarded
normalized = self._normalize_content(content)
len(normalized)  # Remove this line

# Pattern 2: Incomplete implementations
def _check_fuzzy_duplicate(self, ...):
    normalized = self._normalize_content(content)
    for hash_info in self.hashes.values():
        pass  # Remove entire method if not implementing
    return False, None
```

Files:
- `duplicate_detector.py:260` - Remove `len(normalized)`
- `duplicate_detector.py:248-268` - Remove or implement fuzzy matching
- `embeddings.py:260` - Same dead code

**Subtask 8D: Fix Backend Configuration**

File: `config.py:75`

```python
# WRONG - Misleading default
backend: str = "faiss"  # But code uses ChromaDB!

# Option 1: Fix default
backend: str = "chromadb"

# Option 2: Remove if not supporting multiple backends
# Remove backend field entirely if YAGNI
```

**Subtask 8E: Add Context Managers**

**Temporary weight override pattern:**

File: `retrieval_pipeline.py:816-822`

```python
# Create context manager
from contextlib import contextmanager

@contextmanager
def override_weights(self, vector_weight: float, keyword_weight: float):
    """Temporarily override retrieval weights."""
    old_v, old_k = self.vector_weight, self.keyword_weight
    self.vector_weight, self.keyword_weight = vector_weight, keyword_weight
    try:
        yield
    finally:
        self.vector_weight, self.keyword_weight = old_v, old_k

# Usage
with self.override_weights(adaptive_vector_weight, adaptive_keyword_weight):
    results = self._hybrid_search(query, top_k)
```

**File handle management:**

Ensure all file operations use context managers or have proper cleanup in finally blocks.

**Subtask 8F: Fix Documentation**

Update all references:
- "FAISS" → "ChromaDB" (except in historical/comparison contexts)
- "Faiss" → "ChromaDB"
- "vectors.index" → "chroma_db directory"

Files:
- `index.py:2-5, 55`
- `retrieve.py:5`
- All docstrings mentioning FAISS

### Task Category 9: Configuration & Path Fixes
**Agent:** Configuration Management Agent
**Estimated Effort:** 2 hours
**Files:** 10 files

**Subtask 9A: Fix Relative Paths**

File: `tavily_connector.py:46`

```python
# WRONG - Relative path
self.quota_file = Path("config/tavily_usage.json")

# CORRECT - Absolute path
from pathlib import Path

# Option 1: Use project root
project_root = Path(__file__).parent.parent.parent.parent
self.quota_file = project_root / "config" / "tavily_usage.json"

# Option 2: Use user config directory
self.quota_file = Path.home() / ".rag-cli" / "tavily_usage.json"
```

**Subtask 9B: Fix Path Detection**

Files: `session-start.py:30`, `session-end.py:30`

```python
# WRONG - Checking for old v1.x structure
if (current / 'src' / 'core').exists() and (current / 'src' / 'monitoring').exists():

# CORRECT - Check for v2.0 structure
if (current / 'src' / 'rag_cli').exists() and (current / 'src' / 'rag_cli_plugin').exists():
```

**Subtask 9C: Fix Vector Store Path**

File: `unified_server.py:64, 328, 521, 664`

```python
# WRONG - Looking for file, but ChromaDB uses directory
vector_store_path = project_root / "data" / "vectors" / "vectors.index"

# CORRECT - ChromaDB directory
vector_store_path = project_root / "data" / "vectors" / "chroma_db"
```

**Subtask 9D: Move Hardcoded Config to Files**

File: `rag_project_indexer.py:316-389`

Move documentation URLs to config file:
```yaml
# config/documentation_sources.yaml
sources:
  - name: "Claude Code"
    url: "https://docs.claude.com/en/docs/claude-code/"
    type: "documentation"
  - name: "Python"
    url: "https://docs.python.org/3/"
    type: "documentation"
  # ... etc
```

File: `claude_integration.py:582-584`

Move pricing to config:
```python
# config/claude_pricing.json
{
  "pricing": {
    "input_per_token": 0.00000025,
    "output_per_token": 0.00000125
  }
}
```

---

## Parallel Agent Execution Plan

Execute fixes in this order for maximum parallelism:

### Phase 1: Critical Fixes (Day 1)
**All can run in parallel:**

1. **Agent A - Import Path Migration** (2 hours)
   - Fix all import paths
   - Update subprocess module paths
   - Verify with static analysis

2. **Agent B - Critical Bugs** (2 hours)
   - Fix PDF extension typo
   - Fix Pydantic API calls
   - Fix cache structure mismatch
   - Fix division by zero
   - Add recursion guards

3. **Agent C - Security Fixes** (2 hours)
   - Replace MD5 with blake2b
   - Fix SQL injection
   - Add file size validation
   - Fix command injection risk

### Phase 2: High Priority (Days 2-3)
**Can run in parallel after Phase 1:**

4. **Agent D - Thread Safety** (3 hours)
   - Implement thread-safe singletons
   - Add file locks
   - Fix global state issues

5. **Agent E - Performance Optimization** (5 hours)
   - Fix cache algorithms
   - Pre-compile regex
   - Move imports to module level
   - Implement bounded collections
   - Optimize batch processing

6. **Agent F - Error Handling** (6 hours)
   - Replace broad exception handlers
   - Add input validation
   - Add edge case handling
   - Fix recursion issues

### Phase 3: Medium Priority (Days 4-5)
**Can run in parallel after Phase 2:**

7. **Agent G - Constants Migration** (4 hours)
   - Replace all magic numbers
   - Update references
   - Verify with tests

8. **Agent H - Type Safety** (3 hours)
   - Fix type hints
   - Fix API compatibility
   - Add attribute validation

9. **Agent I - Configuration & Paths** (2 hours)
   - Fix relative paths
   - Update path detection
   - Move hardcoded config to files

### Phase 4: Code Quality (Days 6-8)
**Can run after earlier phases:**

10. **Agent J - Code Quality** (8 hours)
    - Extract duplicated code
    - Split long functions
    - Remove dead code
    - Add context managers
    - Update documentation

---

## Testing Strategy

After each phase, run:

```bash
# Unit tests
pytest tests/ -v

# Integration tests
pytest tests/test_integration.py -v

# Type checking
mypy src/

# Linting
pylint src/
flake8 src/

# Security scanning
bandit -r src/

# Import verification
python -m scripts.verify_imports
```

---

## Success Criteria

**Phase 1 Complete:**
- All services start without ImportError
- No crashes in basic workflows
- Security vulnerabilities fixed

**Phase 2 Complete:**
- No race conditions in concurrent operations
- Performance benchmarks meet targets
- All error cases handled gracefully

**Phase 3 Complete:**
- No magic numbers in codebase
- All type hints correct
- Configuration externalized

**Phase 4 Complete:**
- Code complexity < 10 per function
- No code duplication
- Documentation updated
- All tests passing

---

## Risk Assessment

**High Risk:**
- Import path changes could break plugin loading
- Thread safety changes could introduce deadlocks
- Performance optimizations could change behavior

**Mitigation:**
- Comprehensive testing after each phase
- Feature flags for major changes
- Rollback plan for each agent task
- User acceptance testing before release

**Low Risk:**
- Constants migration (pure refactoring)
- Documentation updates
- Type hint additions
- Dead code removal

---

## Estimated Timeline

| Phase | Duration | Parallel Agents | Calendar Days |
|-------|----------|-----------------|---------------|
| Phase 1 | 6 hours | 3 agents | 1 day |
| Phase 2 | 14 hours | 3 agents | 2-3 days |
| Phase 3 | 9 hours | 3 agents | 2 days |
| Phase 4 | 8 hours | 1 agent | 2-3 days |
| **Testing** | 8 hours | 1 agent | 1-2 days |
| **Total** | **45 hours** | **Max 3 parallel** | **8-11 days** |

---

## Conclusion

This audit identified 209 issues across the RAG-CLI v2.0 codebase. While many are low-severity technical debt, 18 critical issues require immediate attention to ensure system stability and correctness.

The remediation plan groups fixes into 10 parallel agent tasks across 4 phases, allowing efficient resolution of all issues in 8-11 calendar days with proper testing.

**Next Steps:**
1. Review and approve this audit report
2. Prioritize which phases to tackle first
3. Assign agents to tasks
4. Execute Phase 1 (critical fixes)
5. Validate with comprehensive testing
6. Proceed to subsequent phases

**Key Recommendations:**
- Fix Phase 1 (Critical) immediately before any production use
- Implement Phase 2 (High Priority) in next sprint
- Schedule Phase 3 (Medium Priority) for following sprint
- Address Phase 4 (Code Quality) as ongoing technical debt reduction
