# Hook Refactoring Plan

## Current State

**File:** `src/rag_cli_plugin/hooks/user-prompt-submit.py`
**Lines:** 876 (target: ~200-300)
**Complexity:** HIGH - violates Single Responsibility Principle

## Problem Analysis

The hook currently handles:
1. **Path Resolution** (lines 20-69): 50 lines of multi-strategy path finding
2. **TCP Server Communication** (lines 89-193): 100+ lines of health checking + event submission
3. **RAG Settings Management** (lines 195-237): Configuration loading/saving
4. **Query Classification** (lines 239-313): Determining if query should be enhanced
5. **Context Retrieval** (lines 315-413): Document retrieval and filtering
6. **Query Enhancement** (lines 415-430): Context formatting
7. **Helper Functions** (lines 433-724): Process orchestration
8. **Main Hook Logic** (lines 726-876): Event processing

## Completed Refactoring Steps

### ✅ Step 1: Path Resolution Utility (DONE)
**Module:** `src/rag_cli_plugin/hooks/path_utils.py`
**Status:** Already exists and is comprehensive

**Functions:**
- `get_rag_cli_root(hook_file)`: Multi-strategy path resolution
- `setup_sys_path(hook_file)`: Configure sys.path for imports

**Benefits:**
- Reusable across all hooks
- Eliminates 50 lines per hook
- Single source of truth for path logic

### ✅ Step 2: Event Submission Service (DONE)
**Module:** `src/rag_cli_plugin/services/event_submitter.py`
**Status:** Created

**Classes:**
- `EventSubmitter`: Handles TCP communication with exponential backoff
  - `is_server_available()`: Health check with caching
  - `submit_event()`: Send events to monitoring server
  - `reset_backoff()`: Manual backoff reset

**Functions:**
- `get_event_submitter()`: Singleton accessor

**Benefits:**
- Eliminates 100+ lines of TCP logic from hook
- Centralized backoff state management
- Testable in isolation
- Reusable across all hooks

## Remaining Refactoring Steps

### Step 3: RAG Settings Service
**Module:** `src/rag_cli_plugin/services/rag_settings.py` (to create)
**Estimated Effort:** 2-3 hours

**Scope:**
- Move settings load/save logic (lines 195-237)
- Add validation and defaults
- Implement settings caching
- Thread-safe access

**Functions to create:**
```python
class RAGSettingsManager:
    def __init__(self, settings_file: Path):
        ...

    def load_settings(self) -> Dict[str, Any]:
        """Load settings with caching and validation."""

    def save_settings(self, settings: Dict[str, Any]):
        """Save settings with validation."""

    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get individual setting value."""

    def update_setting(self, key: str, value: Any):
        """Update individual setting."""

    def reload(self):
        """Force reload from disk."""

def get_settings_manager() -> RAGSettingsManager:
    """Singleton accessor."""
```

### Step 4: Query Enhancement Service
**Module:** `src/rag_cli_plugin/services/query_enhancer.py` (to create)
**Estimated Effort:** 4-5 hours

**Scope:**
- Move query classification logic (lines 239-313)
- Move context retrieval (lines 315-413)
- Move query formatting (lines 415-430)
- Integrate event submission

**Classes to create:**
```python
class QueryEnhancer:
    def __init__(
        self,
        settings_manager: RAGSettingsManager,
        event_submitter: EventSubmitter,
        project_root: Path
    ):
        ...

    def should_enhance_query(self, query: str) -> Tuple[bool, Optional[QueryClassification]]:
        """Determine if query should be enhanced."""

    def retrieve_context(
        self,
        query: str,
        classification: Optional[QueryClassification] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant context documents."""

    def format_enhanced_query(self, query: str, documents: List[Dict[str, Any]]) -> str:
        """Format enhanced query with context."""

    def enhance_query(self, query: str) -> Tuple[str, Dict[str, Any]]:
        """Full enhancement pipeline. Returns (enhanced_query, metadata)."""

def get_query_enhancer() -> QueryEnhancer:
    """Singleton accessor."""
```

### Step 5: Refactor Hook to Coordination-Only
**File:** `src/rag_cli_plugin/hooks/user-prompt-submit.py` (to modify)
**Target Lines:** ~200-250
**Estimated Effort:** 2-3 hours

**New Structure:**
```python
#!/usr/bin/env python3
"""UserPromptSubmit hook - coordinates RAG enhancement."""

import sys
import os
from typing import Dict, Any

# Suppress console logging
os.environ['CLAUDE_HOOK_CONTEXT'] = '1'
os.environ['RAG_CLI_SUPPRESS_CONSOLE'] = '1'

# Setup imports using path_utils
from rag_cli_plugin.hooks.path_utils import setup_sys_path
project_root = setup_sys_path()

# Import services
from rag_cli_plugin.services.event_submitter import get_event_submitter
from rag_cli_plugin.services.rag_settings import get_settings_manager
from rag_cli_plugin.services.query_enhancer import get_query_enhancer
from rag_cli_plugin.services.service_manager import ensure_services_running
from rag_cli_plugin.services.logger import get_logger

logger = get_logger(__name__)

# Initialize services
event_submitter = get_event_submitter()
settings_manager = get_settings_manager(project_root / "config" / "rag_settings.json")
query_enhancer = get_query_enhancer()


def process_hook(event: Dict[str, Any]) -> Dict[str, Any]:
    """Process UserPromptSubmit event with RAG enhancement.

    This hook coordinates between services to enhance user queries with
    relevant context from the document knowledge base.

    Args:
        event: Hook event data with 'prompt' field

    Returns:
        Modified event with enhanced prompt (if applicable)
    """
    try:
        # Ensure monitoring services are running
        try:
            ensure_services_running()
        except Exception as e:
            logger.debug(f"Service startup check failed: {e}")

        # Extract query
        query = event.get("prompt", "")
        if not query:
            return event

        # Emit query received event
        event_submitter.submit_event("activity", {
            "activity": "query_received",
            "component": "user_prompt_hook",
            "metadata": {
                "query_length": len(query),
                "word_count": len(query.split())
            }
        })

        # Enhance query using service
        enhanced_query, enhancement_metadata = query_enhancer.enhance_query(query)

        # Update event if enhanced
        if enhanced_query != query:
            event["prompt"] = enhanced_query
            logger.info("Query enhanced with RAG context",
                       original_length=len(query),
                       enhanced_length=len(enhanced_query),
                       documents=enhancement_metadata.get("document_count", 0))

            # Emit enhancement event
            event_submitter.submit_event("query_enhancement", {
                "original_query": query[:100],
                "enhanced": True,
                "metadata": enhancement_metadata
            })
        else:
            # Emit skip event with reason
            event_submitter.submit_event("reasoning", {
                "reasoning": enhancement_metadata.get("skip_reason", "Query not enhanced"),
                "component": "user_prompt_hook"
            })

        return event

    except Exception as e:
        logger.error(f"Hook processing error: {e}", exc_info=True)
        return event  # Return original event on error


def main():
    """Main entry point for hook execution."""
    import json

    try:
        # Read event from stdin
        event = json.load(sys.stdin)

        # Process event
        result = process_hook(event)

        # Write result to stdout
        json.dump(result, sys.stdout)
        sys.stdout.flush()

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON input: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Hook execution failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
```

## Implementation Timeline

### Phase 1: Preparation (Day 1, 2-3 hours)
- [ ] Create backup of current hook
- [ ] Write unit tests for existing hook functionality
- [ ] Create integration test suite
- [ ] Document current behavior and edge cases

### Phase 2: Service Creation (Day 1-2, 6-8 hours)
- [ ] Create RAGSettingsManager service
- [ ] Create QueryEnhancer service
- [ ] Write unit tests for each service
- [ ] Test services in isolation

### Phase 3: Hook Refactoring (Day 2, 2-3 hours)
- [ ] Refactor hook to use new services
- [ ] Remove duplicated code
- [ ] Simplify error handling
- [ ] Update imports and dependencies

### Phase 4: Testing & Validation (Day 2-3, 3-4 hours)
- [ ] Run unit tests (should all pass)
- [ ] Run integration tests
- [ ] Manual testing of common scenarios
- [ ] Performance testing (ensure no regressions)
- [ ] Edge case testing

### Phase 5: Documentation & Cleanup (Day 3, 1-2 hours)
- [ ] Update hook documentation
- [ ] Document new service APIs
- [ ] Update CLAUDE.md with new structure
- [ ] Remove old commented code
- [ ] Code review and final cleanup

**Total Estimated Time:** 14-20 hours over 2-3 days

## Success Criteria

1. **Line Count:** Hook reduced from 876 to 200-300 lines
2. **Maintainability:** Each module has single, clear responsibility
3. **Testability:** Services can be tested in isolation
4. **Performance:** No degradation in hook execution time
5. **Functionality:** All existing features work identically
6. **Code Coverage:** Services have >80% test coverage

## Testing Strategy

### Unit Tests
```python
# test_event_submitter.py
def test_exponential_backoff():
    submitter = EventSubmitter()
    # Simulate failures
    for i in range(5):
        submitter.is_server_available()
        # Verify backoff increases: 30s, 60s, 120s, 240s, 240s

# test_rag_settings.py
def test_load_settings_with_defaults():
    manager = RAGSettingsManager("/nonexistent/path.json")
    settings = manager.load_settings()
    assert settings["enabled"] == False

# test_query_enhancer.py
def test_should_enhance_technical_query():
    enhancer = QueryEnhancer(...)
    should_enhance, classification = enhancer.should_enhance_query(
        "How do I configure ChromaDB persistence?"
    )
    assert should_enhance == True
```

### Integration Tests
```python
# test_hook_integration.py
def test_full_enhancement_pipeline():
    event = {"prompt": "Explain vector store operations"}
    result = process_hook(event)
    assert len(result["prompt"]) > len(event["prompt"])
    assert "Context:" in result["prompt"]

def test_hook_with_services_unavailable():
    # Stop TCP server
    event = {"prompt": "Test query"}
    result = process_hook(event)
    # Should still work, just without events
    assert result is not None
```

## Risk Mitigation

### Risk 1: Breaking Changes
**Mitigation:**
- Comprehensive test suite before refactoring
- Feature flag to toggle between old/new implementation
- Parallel deployment period

### Risk 2: Performance Degradation
**Mitigation:**
- Benchmark before/after
- Profile service initialization overhead
- Optimize singleton patterns
- Cache service instances

### Risk 3: Hidden Dependencies
**Mitigation:**
- Thorough code review of all imports
- Test in isolated environment
- Check for global state dependencies
- Document all service interfaces

## Benefits

### Immediate Benefits
1. **Reduced Complexity:** Single file goes from 876 to ~250 lines
2. **Better Testing:** Services testable in isolation
3. **Code Reuse:** Services usable by other hooks
4. **Easier Debugging:** Clear separation of concerns

### Long-term Benefits
1. **Maintainability:** New developers can understand system faster
2. **Extensibility:** Easy to add new enhancement strategies
3. **Performance:** Services can be optimized independently
4. **Reliability:** Better error handling and recovery

## Next Steps

1. **Review this plan** with team/stakeholders
2. **Schedule refactoring sprint** (2-3 days focused work)
3. **Assign ownership** for each service module
4. **Create tracking issues** for each phase
5. **Set up monitoring** for hook performance metrics

## References

- Original Issue: Predictive Code Analysis identified 855-line hook as critical issue
- Services Created:
  - `src/rag_cli_plugin/hooks/path_utils.py` (existing)
  - `src/rag_cli_plugin/services/event_submitter.py` (new)
- Related Documentation:
  - `CLAUDE.md`: Project structure
  - `SECURITY.md`: Security best practices
