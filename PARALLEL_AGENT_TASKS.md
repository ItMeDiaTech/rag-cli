# RAG-CLI Code Audit - Parallel Agent Tasks

This document provides a quick reference for executing the 10 parallel agent tasks identified in the comprehensive code audit.

---

## Quick Stats

- **Total Issues Found:** 209
- **Critical Issues:** 18 (fix immediately)
- **High Priority:** 54 (this sprint)
- **Medium Priority:** 94 (next sprint)
- **Low Priority:** 43 (backlog)

---

## Execution Phases

### Phase 1: Critical Fixes (Day 1) - 3 Agents in Parallel

#### Agent A: Import Path Migration
**Duration:** 2 hours | **Files:** 12 | **Severity:** CRITICAL

**Task:** Fix all v1.x import paths to v2.0 structure

**Pattern Replacements:**
```python
# Core library
from core.X → from rag_cli.core.X
from agents.X → from rag_cli.agents.X

# Plugin
from monitoring.X → from rag_cli_plugin.services.X
from plugin.X → from rag_cli_plugin.X

# Subprocess paths
['python', '-m', 'src.monitoring.tcp_server'] →
['python', '-m', 'rag_cli_plugin.services.tcp_server']
```

**Files:**
1. `src/rag_cli_plugin/hooks/user-prompt-submit.py:576`
2. `src/rag_cli_plugin/services/service_manager.py:331, 396`
3. `src/rag_cli_plugin/services/tcp_server.py:713, 740`
4. `src/rag_cli/core/semantic_cache.py:381`
5. `src/rag_cli/integrations/maf_connector.py:221, 555`
6. `src/rag_cli/agents/maf/core/orchestrator.py:629-630`
7. `src/rag_cli/core/retrieval_pipeline.py:601`

**Command to Launch Agent:**
```bash
# Agent A will fix all import paths
```

---

#### Agent B: Critical Bugs
**Duration:** 2 hours | **Files:** 8 | **Severity:** CRITICAL

**Task:** Fix critical bugs that cause crashes or data loss

**Issues to Fix:**
1. PDF extension typo: `config.py:51, 268` - `.pd` → `.pdf`
2. Pydantic API: `config.py:482` - `.dict()` → `.model_dump()`
3. Cache structure: `semantic_cache.py:301-314` - Fix AttributeError
4. Division by zero: `maf_connector.py:318, 343` - Check for empty list
5. Infinite recursion: `tavily_connector.py:98-101` - Add retry guard
6. File handle leaks: `service_manager.py:319` - Add cleanup
7. Relative path: `tavily_connector.py:46` - Make absolute
8. Wrong key reference: `maf_connector.py:591, 596` - Fix test function

**Validation:**
```bash
# After fixes, run:
pytest tests/ -v
python scripts/verify_installation.py
```

---

#### Agent C: Security Fixes
**Duration:** 2 hours | **Files:** 8 | **Severity:** HIGH

**Task:** Fix security vulnerabilities

**Issues to Fix:**
1. Replace MD5 with blake2b (8 files)
   - `document_processor.py:473-475`
   - `user-prompt-submit.py:689`
   - `response-post.py:59`
   - Search for `hashlib.md5`

2. Fix SQL injection: `memory.py:455`

3. Add file size validation: `document_processor.py:220-233`

4. Fix command injection: `claude_cli_unified.py:267-303`

5. Add rate limiting: `claude_integration.py`, `unified_server.py`

**Pattern:**
```python
# Replace
hashlib.md5(data.encode()).hexdigest()[:16]

# With
hashlib.blake2b(data.encode(), digest_size=16).hexdigest()
```

**Validation:**
```bash
bandit -r src/ -ll
```

---

### Phase 2: High Priority (Days 2-3) - 3 Agents in Parallel

#### Agent D: Thread Safety
**Duration:** 3 hours | **Files:** 15 | **Severity:** HIGH

**Task:** Implement thread-safe singleton pattern across all modules

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
9. `src/rag_cli_plugin/hooks/user-prompt-submit.py:90-93` (global state)
10. `src/rag_cli_plugin/services/service_manager.py:59-86` (PID files)

**Additional:**
- Add file locking to `tavily_connector.py` quota writes
- Fix asyncio/threading mixing in `base_agent.py`

---

#### Agent E: Performance Optimization
**Duration:** 5 hours | **Files:** 24 | **Severity:** HIGH

**Task:** Fix performance bottlenecks and inefficiencies

**Subtasks:**

**E1. Fix O(n) Cache Operations** (1 hour)
- File: `claude_integration.py:369-371`
- Replace list-based LRU with OrderedDict

**E2. Pre-compile Regex Patterns** (1 hour)
- Files: `query_classifier.py:63-210`, `logger.py:36-46`
- Compile all patterns in `__init__`

**E3. Move Module-Level Imports** (0.5 hours)
- `retrieval_pipeline.py:601` - Move QueryIntent import to top
- Search for imports inside functions/methods

**E4. Implement Bounded Collections** (1.5 hours)
- Replace unbounded lists with `deque(maxlen=N)`
- Files: `agent_communication.py`, `tcp_server.py`, `orchestrator.py`

**E5. Optimize Batch Processing** (1 hour)
- File: `cli/index.py:209-217`
- Stream to vector store instead of accumulating in memory

**Pattern:**
```python
# Replace
from collections import OrderedDict
self.cache = OrderedDict()
self.cache.move_to_end(key)  # O(1) instead of list.remove() O(n)
```

---

#### Agent F: Error Handling & Validation
**Duration:** 6 hours | **Files:** 28 | **Severity:** HIGH

**Task:** Improve error handling and input validation

**Subtasks:**

**F1. Replace Broad Exception Handlers** (2 hours)
- Search for `except Exception` across codebase
- Replace with specific exception types

**Pattern:**
```python
# Replace
try:
    operation()
except Exception as e:
    logger.error(f"Failed: {e}")

# With
try:
    operation()
except (ValueError, KeyError, IOError) as e:
    logger.error(f"Failed: {e}")
    return default_value
except Exception as e:
    logger.exception("Unexpected error", exc_info=True)
    raise
```

**F2. Add Input Validation** (2 hours)
- Add validation to all public methods
- Files: `maf_connector.py`, `index.py`, `arxiv_connector.py`, `tavily_connector.py`

**F3. Add Edge Case Handling** (1 hour)
- Division by zero checks
- Null/None checks
- Empty collection checks

**F4. Fix Recursion Guards** (1 hour)
- Already covered in Agent B, but verify all recursive functions

---

### Phase 3: Medium Priority (Days 4-5) - 3 Agents in Parallel

#### Agent G: Constants Migration
**Duration:** 4 hours | **Files:** 32 | **Severity:** MEDIUM

**Task:** Replace all magic numbers with references to `constants.py`

**Migration Map:**

| Find | Replace With | Files |
|------|--------------|-------|
| `500` | `CHUNK_SIZE_TOKENS` | document_processor.py |
| `100` | `CHUNK_OVERLAP_TOKENS` | document_processor.py |
| `32` | `DEFAULT_BATCH_SIZE` | embeddings.py, index.py |
| `10` | `DEFAULT_HTTP_TIMEOUT` | connectors |
| `1000` | `EMBEDDING_CACHE_SIZE` | embeddings.py |
| `0.7` | `DEFAULT_VECTOR_WEIGHT` | retrieval_pipeline.py |
| `0.85` | `SIMILARITY_THRESHOLD` | result_synthesizer.py |
| `300` | `RESPONSE_CACHE_TTL_SECONDS` | response-post.py |
| `240` | `MAX_BACKOFF_SECONDS` | user-prompt-submit.py |

**Process:**
1. Ensure constant exists in `constants.py`
2. Import constant at top of file
3. Replace hardcoded value with constant
4. Run tests to verify

---

#### Agent H: Type Safety & API Fixes
**Duration:** 3 hours | **Files:** 18 | **Severity:** MEDIUM

**Task:** Fix type hints and API compatibility issues

**Subtasks:**

**H1. Fix Type Hints** (1 hour)
```python
# Find
def get_statistics(self) -> Dict[str, any]:  # WRONG

# Replace
from typing import Any
def get_statistics(self) -> Dict[str, Any]:  # CORRECT
```

**H2. Fix API Compatibility** (1 hour)
- Pydantic v1 → v2: Already in Agent B
- ChromaDB API Abstraction:
  ```python
  # Add to vector_store.py
  def get_vector_count(self) -> int:
      return self.collection.count()

  # Update all callers
  vector_store.index.ntotal → vector_store.get_vector_count()
  ```

**H3. Fix Attribute Access** (1 hour)
- Add hasattr() checks before accessing optional attributes
- Files: `retrieve.py`, `claude_integration.py`

---

#### Agent I: Configuration & Path Fixes
**Duration:** 2 hours | **Files:** 10 | **Severity:** MEDIUM

**Task:** Fix path issues and externalize configuration

**Subtasks:**

**I1. Fix Relative Paths** (0.5 hours)
```python
# tavily_connector.py:46
# Replace
self.quota_file = Path("config/tavily_usage.json")

# With
project_root = Path(__file__).parent.parent.parent.parent
self.quota_file = project_root / "config" / "tavily_usage.json"
```

**I2. Fix Path Detection** (0.5 hours)
- `session-start.py:30`, `session-end.py:30`
- Check for v2.0 structure instead of v1.x

**I3. Fix Vector Store Path** (0.5 hours)
- `unified_server.py:64, 328, 521, 664`
- Change from `vectors.index` file to `chroma_db` directory

**I4. Move Hardcoded Config to Files** (0.5 hours)
- Documentation URLs → YAML file
- API pricing → JSON file

---

### Phase 4: Code Quality (Days 6-8) - 1 Agent

#### Agent J: Code Quality & Refactoring
**Duration:** 8 hours | **Files:** 42 | **Severity:** LOW-MEDIUM

**Task:** Improve code quality and reduce technical debt

**Subtasks:**

**J1. Extract Duplicated Code** (2 hours)
- Path resolution logic in 6 hook files
- Replace with `from rag_cli_plugin.hooks.path_utils import setup_sys_path`

**J2. Split Long Functions** (3 hours)
- `user-prompt-submit.py:process_hook()` - 140 lines
- `retrieve.py:process_query()` - 120 lines
- `unified_server.py` - 1155 lines (split into multiple classes)

**J3. Remove Dead Code** (1 hour)
- `len(normalized)` statements that discard result
- Incomplete fuzzy matching implementation
- Unused variables

**J4. Add Context Managers** (1 hour)
- Temporary weight override pattern
- File handle management

**J5. Fix Documentation** (1 hour)
- Update "FAISS" → "ChromaDB" references
- Fix docstrings
- Update comments

---

## Execution Commands

### Phase 1: Launch All 3 Agents in Parallel

```bash
# In separate terminals or use tmux/screen

# Terminal 1 - Agent A
echo "Agent A: Import Path Migration"
# Agent A execution here

# Terminal 2 - Agent B
echo "Agent B: Critical Bugs"
# Agent B execution here

# Terminal 3 - Agent C
echo "Agent C: Security Fixes"
# Agent C execution here
```

### Testing After Each Phase

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
bandit -r src/ -ll

# Import verification
python -c "
import sys
sys.path.insert(0, 'src')
from rag_cli.core.config import get_config
from rag_cli_plugin.services.service_manager import ServiceManager
print('✓ All imports successful')
"
```

---

## Progress Tracking

### Phase 1 Checklist
- [ ] Agent A: Import Path Migration (2h)
- [ ] Agent B: Critical Bugs (2h)
- [ ] Agent C: Security Fixes (2h)
- [ ] Phase 1 Testing (1h)
- [ ] **Total: 7 hours → ~1 day**

### Phase 2 Checklist
- [ ] Agent D: Thread Safety (3h)
- [ ] Agent E: Performance Optimization (5h)
- [ ] Agent F: Error Handling (6h)
- [ ] Phase 2 Testing (2h)
- [ ] **Total: 16 hours → 2-3 days**

### Phase 3 Checklist
- [ ] Agent G: Constants Migration (4h)
- [ ] Agent H: Type Safety (3h)
- [ ] Agent I: Configuration & Paths (2h)
- [ ] Phase 3 Testing (1h)
- [ ] **Total: 10 hours → 2 days**

### Phase 4 Checklist
- [ ] Agent J: Code Quality (8h)
- [ ] Phase 4 Testing (1h)
- [ ] **Total: 9 hours → 2-3 days**

---

## Risk Mitigation

**Before Starting Each Agent:**
1. Create a git branch for the agent's work
2. Commit current state
3. Document expected changes

**After Each Agent Completes:**
1. Run full test suite
2. Manual smoke test of key features
3. Review changes with git diff
4. Merge to feature branch if tests pass

**Rollback Plan:**
```bash
# If agent introduces issues:
git checkout <branch-before-agent>
git branch -D <agent-branch>

# Start agent again with fixes
```

---

## Success Criteria

**Phase 1 Complete:**
- ✓ All services start without ImportError
- ✓ No crashes in basic workflows
- ✓ Security scan shows no critical issues
- ✓ All unit tests pass

**Phase 2 Complete:**
- ✓ Concurrent operations work correctly
- ✓ Performance benchmarks improved
- ✓ Error handling comprehensive
- ✓ Integration tests pass

**Phase 3 Complete:**
- ✓ No magic numbers in codebase
- ✓ All type hints correct (mypy passes)
- ✓ Configuration externalized

**Phase 4 Complete:**
- ✓ Code complexity < 10 per function
- ✓ No code duplication
- ✓ Documentation updated
- ✓ Code coverage > 70%

---

## Timeline Summary

| Phase | Duration | Parallel Agents | Calendar Days |
|-------|----------|-----------------|---------------|
| Phase 1 | 7 hours | 3 agents | 1 day |
| Phase 2 | 16 hours | 3 agents | 2-3 days |
| Phase 3 | 10 hours | 3 agents | 2 days |
| Phase 4 | 9 hours | 1 agent | 2-3 days |
| **Total** | **42 hours** | **Max 3 parallel** | **7-9 days** |

---

## Next Steps

1. **Review this task breakdown**
2. **Decide which phases to execute**
3. **Assign agents (can use Claude Code Task tool for each)**
4. **Execute Phase 1 first** (critical fixes)
5. **Validate with comprehensive testing**
6. **Proceed to subsequent phases**

**Recommended Approach:**
- Execute Phase 1 immediately (critical bugs)
- Schedule Phase 2 for next sprint
- Plan Phase 3 for following sprint
- Address Phase 4 as ongoing technical debt

---

## Agent Launch Commands

Use these commands to launch each agent using Claude Code's Task tool:

### Phase 1 Agents

**Agent A Launch:**
```
Task: Import Path Migration
Prompt: Fix all v1.x import paths to v2.0 structure in RAG-CLI.
Follow the patterns in PARALLEL_AGENT_TASKS.md under Agent A.
Update 12 files with correct import paths for rag_cli and rag_cli_plugin.
Test all imports after completion.
```

**Agent B Launch:**
```
Task: Critical Bug Fixes
Prompt: Fix 8 critical bugs in RAG-CLI as specified in Agent B section.
Fix PDF extension typo, Pydantic API, cache structure, division by zero,
infinite recursion, file handle leaks, relative paths, and wrong key references.
Run pytest after each fix.
```

**Agent C Launch:**
```
Task: Security Fixes
Prompt: Fix security vulnerabilities in RAG-CLI per Agent C specification.
Replace MD5 with blake2b in 8 locations, fix SQL injection, add file size
validation, fix command injection, and implement rate limiting.
Run bandit security scanner after completion.
```

---

**This document provides the complete task breakdown for parallel execution of all code audit fixes.**
