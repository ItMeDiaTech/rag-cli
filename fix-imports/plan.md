# Import Fixing Plan - RAG-CLI v2.0

**Session Started:** 2025-11-02
**Total Broken Imports:** 28+
**Files Affected:** 25
**Status:** In Progress

---

## Critical Architectural Issue

The core library (`rag_cli`) is importing from the plugin package (`rag_cli_plugin`), creating a hard dependency that violates the v2.0 architecture. The core should be platform-agnostic.

**Root Cause:** Logger and monitoring utilities are in `rag_cli_plugin.services` but needed by core.

**Solution Strategy:**
1. Extract logger to `rag_cli.utils.logger`
2. Extract monitoring utilities to `rag_cli.utils.monitoring`
3. Update all core imports to use `rag_cli.utils`
4. Keep `rag_cli_plugin.services` versions as wrappers for backward compatibility

---

## Import Fixes Needed

### Phase 1: Extract Shared Utilities (PRIORITY)

#### 1.1 Logger Extraction
- [ ] **NEW FILE:** `src/rag_cli/utils/logger.py`
  - Move from: `rag_cli_plugin.services.logger`
  - Contains: `get_logger()`, `get_metrics_logger()`, `log_api_call()`, `log_execution_time()`
  - Confidence: HIGH
  - Approach: Copy and adapt for platform-agnostic use

#### 1.2 Monitoring Utilities
- [ ] **NEW FILE:** `src/rag_cli/utils/metrics.py`
  - Move from: `rag_cli_plugin.services.tcp_server.metrics_collector`
  - Confidence: HIGH

- [ ] **NEW FILE:** `src/rag_cli/utils/latency_tracker.py`
  - Move from: `rag_cli_plugin.services.latency_tracker`
  - Contains: `get_latency_tracker()`, `time_operation()`
  - Confidence: HIGH

- [ ] **NEW FILE:** `src/rag_cli/utils/error_tracker.py`
  - Move from: `rag_cli_plugin.services.error_tracker`
  - Contains: `get_error_tracker()`
  - Confidence: HIGH

#### 1.3 Output Formatting
- [ ] **NEW FILE:** `src/rag_cli/utils/output_formatter.py`
  - Move from: `rag_cli_plugin.services.output_formatter`
  - Contains: `OutputFormatter` class
  - Confidence: MEDIUM (may need adaptation)

---

### Phase 2: Update Core Library Imports (23 imports in 15 files)

#### 2.1 Core Modules - Logger Only (12 files)

- [ ] **src/rag_cli/core/claude_code_adapter.py:13**
  - Current: `from rag_cli_plugin.services.logger import get_logger`
  - Fix: `from rag_cli.utils.logger import get_logger`
  - Status: PENDING

- [ ] **src/rag_cli/core/best_practices_detector.py:12**
  - Current: `from rag_cli_plugin.services.logger import get_logger`
  - Fix: `from rag_cli.utils.logger import get_logger`
  - Status: PENDING

- [ ] **src/rag_cli/core/hyde.py:15**
  - Current: `from rag_cli_plugin.services.logger import get_logger`
  - Fix: `from rag_cli.utils.logger import get_logger`
  - Status: PENDING

- [ ] **src/rag_cli/core/prompt_templates.py:11**
  - Current: `from rag_cli_plugin.services.logger import get_logger`
  - Fix: `from rag_cli.utils.logger import get_logger`
  - Status: PENDING

- [ ] **src/rag_cli/core/query_classifier.py:12**
  - Current: `from rag_cli_plugin.services.logger import get_logger`
  - Fix: `from rag_cli.utils.logger import get_logger`
  - Status: PENDING

- [ ] **src/rag_cli/core/query_enhancer.py:11**
  - Current: `from rag_cli_plugin.services.logger import get_logger`
  - Fix: `from rag_cli.utils.logger import get_logger`
  - Status: PENDING

- [ ] **src/rag_cli/core/semantic_cache.py:21**
  - Current: `from rag_cli_plugin.services.logger import get_logger`
  - Fix: `from rag_cli.utils.logger import get_logger`
  - Status: PENDING

- [ ] **src/rag_cli/core/semantic_cache_hnsw.py:19**
  - Current: `from rag_cli_plugin.services.logger import get_logger`
  - Fix: `from rag_cli.utils.logger import get_logger`
  - Status: PENDING

- [ ] **src/rag_cli/core/web_scraper.py:22**
  - Current: `from rag_cli_plugin.services.logger import get_logger`
  - Fix: `from rag_cli.utils.logger import get_logger`
  - Status: PENDING

- [ ] **src/rag_cli/core/claude_integration.py:27**
  - Current: `from rag_cli_plugin.services.logger import get_logger, get_metrics_logger, log_api_call`
  - Fix: `from rag_cli.utils.logger import get_logger, get_metrics_logger, log_api_call`
  - Status: PENDING

- [ ] **src/rag_cli/core/document_processor.py:36**
  - Current: `from rag_cli_plugin.services.logger import get_logger, get_metrics_logger, log_execution_time`
  - Fix: `from rag_cli.utils.logger import get_logger, get_metrics_logger, log_execution_time`
  - Status: PENDING

- [ ] **src/rag_cli/core/embeddings.py:23**
  - Current: `from rag_cli_plugin.services.logger import get_logger, get_metrics_logger, log_execution_time`
  - Fix: `from rag_cli.utils.logger import get_logger, get_metrics_logger, log_execution_time`
  - Status: PENDING

- [ ] **src/rag_cli/core/vector_store.py:33**
  - Current: `from rag_cli_plugin.services.logger import get_logger, get_metrics_logger, log_execution_time`
  - Fix: `from rag_cli.utils.logger import get_logger, get_metrics_logger, log_execution_time`
  - Status: PENDING

#### 2.2 Core Modules - Multiple Monitoring Imports (2 files)

- [ ] **src/rag_cli/core/retrieval_pipeline.py** (Lines 32-36)
  - Line 32: `from rag_cli_plugin.services.logger import get_logger, get_metrics_logger, log_execution_time`
    - Fix: `from rag_cli.utils.logger import get_logger, get_metrics_logger, log_execution_time`
  - Line 33: `from rag_cli_plugin.services.tcp_server import metrics_collector`
    - Fix: `from rag_cli.utils.metrics import metrics_collector`
  - Line 34: `from rag_cli_plugin.services.latency_tracker import get_latency_tracker, time_operation`
    - Fix: `from rag_cli.utils.latency_tracker import get_latency_tracker, time_operation`
  - Line 36: `from rag_cli_plugin.services.error_tracker import get_error_tracker`
    - Fix: `from rag_cli.utils.error_tracker import get_error_tracker`
  - Status: PENDING

- [ ] **src/rag_cli/core/agent_orchestrator.py** (Lines 37-38)
  - Line 37: `from rag_cli_plugin.services.logger import get_logger, get_metrics_logger`
    - Fix: `from rag_cli.utils.logger import get_logger, get_metrics_logger`
  - Line 38: `from rag_cli_plugin.services.output_formatter import OutputFormatter`
    - Fix: `from rag_cli.utils.output_formatter import OutputFormatter`
  - Status: PENDING

#### 2.3 Agents (3 files)

- [ ] **src/rag_cli/agents/base_agent.py:36**
  - Current: `from rag_cli_plugin.services.logger import get_logger`
  - Fix: `from rag_cli.utils.logger import get_logger`
  - Status: PENDING

- [ ] **src/rag_cli/agents/query_decomposer.py:25**
  - Current: `from rag_cli_plugin.services.logger import get_logger`
  - Fix: `from rag_cli.utils.logger import get_logger`
  - Status: PENDING

- [ ] **src/rag_cli/agents/result_synthesizer.py:26**
  - Current: `from rag_cli_plugin.services.logger import get_logger`
  - Fix: `from rag_cli.utils.logger import get_logger`
  - Status: PENDING

#### 2.4 CLI (2 files)

- [ ] **src/rag_cli/cli/index.py:23**
  - Current: `from rag_cli_plugin.services.logger import get_logger`
  - Fix: `from rag_cli.utils.logger import get_logger`
  - Status: PENDING

- [ ] **src/rag_cli/cli/retrieve.py:25**
  - Current: `from rag_cli_plugin.services.logger import get_logger`
  - Fix: `from rag_cli.utils.logger import get_logger`
  - Status: PENDING

#### 2.5 Integrations (3 files)

- [ ] **src/rag_cli/integrations/arxiv_connector.py:14**
  - Current: `from rag_cli_plugin.services.logger import get_logger`
  - Fix: `from rag_cli.utils.logger import get_logger`
  - Status: PENDING

- [ ] **src/rag_cli/integrations/maf_connector.py:23**
  - Current: `from rag_cli_plugin.services.logger import get_logger`
  - Fix: `from rag_cli.utils.logger import get_logger`
  - Status: PENDING

- [ ] **src/rag_cli/integrations/tavily_connector.py:17**
  - Current: `from rag_cli_plugin.services.logger import get_logger`
  - Fix: `from rag_cli.utils.logger import get_logger`
  - Status: PENDING

---

### Phase 3: Update Legacy Scripts (2 files, 12 imports)

- [ ] **scripts/utils/index.py** (Lines 21-26)
  - Line 21: `from core.config import load_config, get_config`
    - Fix: `from rag_cli.core.config import load_config, get_config`
  - Line 22: `from core.document_processor import get_document_processor`
    - Fix: `from rag_cli.core.document_processor import get_document_processor`
  - Line 23: `from core.embeddings import get_embedding_generator`
    - Fix: `from rag_cli.core.embeddings import get_embedding_generator`
  - Line 24: `from core.vector_store import get_vector_store`
    - Fix: `from rag_cli.core.vector_store import get_vector_store`
  - Line 25: `from core.retrieval_pipeline import get_retriever`
    - Fix: `from rag_cli.core.retrieval_pipeline import get_retriever`
  - Line 26: `from monitoring.logger import get_logger`
    - Fix: `from rag_cli.utils.logger import get_logger`
  - Status: PENDING

- [ ] **scripts/utils/retrieve.py** (Lines 23-28)
  - Line 23: `from core.config import load_config, get_config`
    - Fix: `from rag_cli.core.config import load_config, get_config`
  - Line 24: `from core.embeddings import get_embedding_generator`
    - Fix: `from rag_cli.core.embeddings import get_embedding_generator`
  - Line 25: `from core.vector_store import get_vector_store`
    - Fix: `from rag_cli.core.vector_store import get_vector_store`
  - Line 26: `from core.retrieval_pipeline import get_retriever`
    - Fix: `from rag_cli.core.retrieval_pipeline import get_retriever`
  - Line 27: `from core.claude_integration import get_claude_integration`
    - Fix: `from rag_cli.core.claude_integration import get_claude_integration`
  - Line 28: `from monitoring.logger import get_logger`
    - Fix: `from rag_cli.utils.logger import get_logger`
  - Status: PENDING

---

### Phase 4: Update Plugin Services (Backward Compatibility)

After extracting utilities, update `rag_cli_plugin.services` to import from `rag_cli.utils`:

- [ ] **src/rag_cli_plugin/services/logger.py**
  - Add: `from rag_cli.utils.logger import *` (re-export for compatibility)
  - Status: PENDING

- [ ] **src/rag_cli_plugin/services/latency_tracker.py**
  - Add: `from rag_cli.utils.latency_tracker import *`
  - Status: PENDING

- [ ] **src/rag_cli_plugin/services/error_tracker.py**
  - Add: `from rag_cli.utils.error_tracker import *`
  - Status: PENDING

---

### Phase 5: Verification

- [ ] Run import checker: `python -m py_compile <all modified files>`
- [ ] Run tests: `pytest tests/`
- [ ] Verify no circular imports: Custom script
- [ ] Check all imports resolve: `python -m rag_cli.core.config`
- [ ] Update CHANGELOG.md with import fixes

---

## Resolution Decisions Log

1. **Logger Location:** Moved to `rag_cli.utils.logger` (platform-agnostic)
2. **Monitoring Utilities:** Extracted to `rag_cli.utils.*` packages
3. **Backward Compatibility:** Plugin services re-export from utils
4. **Script Updates:** Changed to use proper v2.0 package paths

---

## Progress Tracking

- **Phase 1 (Extract):** 0/4 files created
- **Phase 2 (Core):** 0/23 imports fixed
- **Phase 3 (Scripts):** 0/12 imports fixed
- **Phase 4 (Plugin):** 0/3 files updated
- **Phase 5 (Verify):** 0/5 checks passed

**Overall:** 0/47 tasks completed (0%)
