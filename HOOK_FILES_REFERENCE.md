# Hook Files Reference - RAG-CLI v1.2.0

This document explains all hook files in the `src/plugin/hooks/` directory and their registration status.

---

## Active Hooks (Registered in hooks.json)

These hooks are active and execute as part of the plugin lifecycle.

### 1. slash-command-blocker.py
**Purpose**: Prevents Claude from providing commentary on slash command execution
**Trigger**: UserPromptSubmit (priority: 150)
**Status**: [OK] Registered and active
**Behavior**: Detects slash commands and blocks Claude's normal response processing

---

### 2. user-prompt-submit.py
**Purpose**: Main orchestration hook - routes queries to RAG and/or MAF agents
**Trigger**: UserPromptSubmit (priority: 100)
**Status**: [OK] Registered and active
**Behavior**:
- Classifies query intent
- Routes to RAG, MAF, or both (parallel execution)
- Manages multi-agent orchestration
- Handles fallback to RAG-only mode

---

### 3. response-post.py
**Purpose**: Injects citations into Claude's response
**Trigger**: ResponsePost (priority: 80)
**Status**: [OK] Registered and active
**Behavior**:
- Adds source citations from RAG retrieval
- Formats citations based on configuration
- Manages citation style (inline, footnotes, collapsible)

---

### 4. plugin-state-change.py
**Purpose**: Persists plugin settings and state
**Trigger**: PluginStateChange (priority: 60)
**Status**: [OK] Registered and active
**Behavior**:
- Saves configuration changes to disk
- Manages plugin enable/disable state
- Updates hook settings

---

### 5. document-indexing.py
**Purpose**: Auto-indexes documents when files are created or modified
**Trigger**: FileCreated, FileModified (priority: 50)
**Status**: [OK] Registered and active
**Behavior**:
- Watches project files for changes
- Automatically indexes new/modified documents
- Updates FAISS vector store
- Maintains knowledge base freshness

---

## Inactive Hooks (Not Registered)

These hooks are implemented but not currently registered in `hooks.json`. They exist for future use or as examples.

### 1. error-handler.py
**Purpose**: Graceful error handling with inline warnings
**Status**: [ERROR] Not registered (optional feature)
**Implementation**: Complete
**When to use**:
- If you want custom error message formatting
- If you prefer warnings instead of failures
- For testing error recovery flows

**To activate**: Add to `hooks.json` under an appropriate trigger (e.g., `ErrorOccurred`)

```json
{
  "ErrorOccurred": [
    {
      "hooks": [
        {
          "type": "command",
          "command": "python",
          "args": ["${CLAUDE_PLUGIN_ROOT}/src/plugin/hooks/error-handler.py"]
        }
      ]
    }
  ]
}
```

---

### 2. update-rag-hook.py
**Purpose**: Synchronizes plugin files and configuration
**Status**: [ERROR] Not registered (functionality moved to `/update-rag` command)
**Implementation**: Complete
**Notes**:
- Functionality is now available via `/update-rag` slash command
- Hook version is kept for reference but not needed
- Command version provides better user control

**To activate**: Add to `hooks.json` if you want automatic sync on plugin state changes

```json
{
  "PluginStateChange": [
    {
      "hooks": [
        {
          "type": "command",
          "command": "python",
          "args": ["${CLAUDE_PLUGIN_ROOT}/src/plugin/hooks/update-rag-hook.py"]
        }
      ]
    }
  ]
}
```

**Note**: Both versions can coexist - the command version gives users explicit control while the hook version could provide automatic sync.

---

## Hook Lifecycle

```
User Input
    v
[1] UserPromptSubmit hooks (priority order)
     slash-command-blocker.py (150) - Block commentary
     user-prompt-submit.py (100)    - Route query
    v
Query Processing (RAG/MAF)
    v
Claude Response
    v
[2] ResponsePost hook
     response-post.py (80) - Add citations
    v
[3] PluginStateChange hook (on settings change)
     plugin-state-change.py (60) - Save state
    v
[4] File monitoring hooks
     FileCreated -> document-indexing.py (50)
     FileModified -> document-indexing.py (50)
```

---

## Hook Configuration

### Priority System
Hooks execute in priority order (highest first):
- **150**: slash-command-blocker (must run first)
- **100**: user-prompt-submit (main orchestration)
- **80**: response-post (citation injection)
- **70**: error-handler (if activated)
- **60**: plugin-state-change (save state)
- **50**: document-indexing (auto-index files)

### Hook Files Requirements

All hook files must have:
1. **Shebang**: `#!/usr/bin/env python3` (first line)
2. **Executable permission**: `chmod +x hook-file.py`
3. **Metadata docstring**: Priority, enabled status, triggers
4. **Main execution logic**: Process input and manage output
5. **Environment setup**: Suppress console output in hooks

Example structure:
```python
#!/usr/bin/env python3
"""HookName hook for RAG-CLI.

Description of what this hook does.

Metadata:
  priority: 70
  enabled: true
  triggers: ["trigger_type"]
"""

import sys
import os

# Suppress console logging
os.environ['CLAUDE_HOOK_CONTEXT'] = '1'

# ... implementation ...

if __name__ == "__main__":
    # Process input
    # Perform actions
    # Return output
```

---

## Adding New Hooks

To create a new hook:

1. **Create hook file**: `src/plugin/hooks/hook-name.py`
2. **Add shebang and metadata**
3. **Implement logic**
4. **Make executable**: `chmod +x src/plugin/hooks/hook-name.py`
5. **Register in hooks.json** (if not optional)
6. **Test thoroughly**

Example new hook:

```python
#!/usr/bin/env python3
"""MyHook hook for RAG-CLI.

This hook does something useful.

Metadata:
  priority: 65
  enabled: true
  triggers: ["UserPromptSubmit"]
"""

import sys
import os
import json

os.environ['CLAUDE_HOOK_CONTEXT'] = '1'

def main():
    try:
        # Read input from stdin
        input_data = json.loads(sys.stdin.read())

        # Process
        result = process_input(input_data)

        # Return result
        print(json.dumps(result))
        return 0
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        return 1

if __name__ == "__main__":
    sys.exit(main())
```

---

## Troubleshooting Hooks

### Hook not executing?
- Verify it's registered in `hooks.json`
- Check file is executable: `ls -l src/plugin/hooks/hook-name.py`
- Verify shebang is correct: `head -1 src/plugin/hooks/hook-name.py`
- Check logs for errors

### Hook causing performance issues?
- Review timeout values
- Check for blocking I/O operations
- Use async operations where possible
- Consider disabling less critical hooks

### Hook output not showing?
- Verify hook is not suppressing output incorrectly
- Check if `ResponsePost` hook is capturing output
- Review citation formatting settings

---

## Hook Status Summary

| Hook | Status | Priority | Trigger | Purpose |
|------|--------|----------|---------|---------|
| slash-command-blocker.py | [OK] Active | 150 | UserPromptSubmit | Block commentary |
| user-prompt-submit.py | [OK] Active | 100 | UserPromptSubmit | Query orchestration |
| response-post.py | [OK] Active | 80 | ResponsePost | Citation injection |
| plugin-state-change.py | [OK] Active | 60 | PluginStateChange | State persistence |
| document-indexing.py | [OK] Active | 50 | FileCreated/Modified | Auto-indexing |
| error-handler.py | [WARNING] Optional | 70 | (custom) | Error handling |
| update-rag-hook.py | [WARNING] Optional | (custom) | (custom) | File sync |

---

## Notes

- All active hooks are essential for plugin functionality
- Optional hooks can be enabled/disabled without breaking core features
- Hook execution is sequential by priority
- Hooks have access to full plugin context and configuration
- Most hooks suppress their console output to avoid cluttering Claude's interface

---

**Version**: 1.2.0
**Last Updated**: October 30, 2025

For more information, see the hook implementations in `src/plugin/hooks/`.
