# Known Issues and Workarounds

This document tracks known limitations and issues in RAG-CLI v2.0, along with their workarounds and resolution status.

## Active Issues

### PostToolUse Hook Disabled (Claude Code Framework Bug)

**Status:** Workaround implemented
**Severity:** Medium
**Affected Versions:** v2.0.0+
**Component:** `.claude-plugin/hooks.json` - PostToolUse hook

#### Issue Description

The PostToolUse hook (`response-post.py`) is designed to add inline citations to Claude's responses when RAG context is used. However, due to a JSON parsing bug in the Claude Code plugin framework, this hook causes response processing failures when enabled.

**Symptoms:**
- Responses fail to display correctly
- JSON parsing errors in hook output
- Citations not appearing in responses

#### Root Cause

The Claude Code plugin framework has a bug in how it handles JSON output from PostToolUse hooks. The framework expects a specific JSON structure but fails to properly parse the modified response object, causing the entire hook chain to fail.

**Technical Details:**
- Hook location: `src/rag_cli_plugin/hooks/response-post.py`
- Expected functionality: Inject citation markers `[1][2]` into responses
- Failure point: JSON serialization/deserialization in hook output processing
- Framework issue: Claude Code's hook result parsing does not handle nested metadata correctly

#### Workaround

The hook is currently **disabled** in `.claude-plugin/hooks.json`:

```json
{
  "PostToolUse": [
    {
      "hooks": [
        {
          "type": "command",
          "command": "python",
          "args": ["${CLAUDE_PLUGIN_ROOT}/src/rag_cli_plugin/hooks/response-post.py"],
          "name": "response-post",
          "priority": 80,
          "enabled": false,
          "description": "Adds inline citations to Claude responses when RAG context is used (DISABLED: Claude Code bug with JSON parsing)"
        }
      ]
    }
  ]
}
```

**Impact of Workaround:**
- RAG functionality works normally
- Context is properly retrieved and injected into prompts
- Claude generates accurate responses with RAG context
- **Missing:** Inline citation markers in responses (e.g., `[1][2]`)
- **Missing:** Source attribution at the end of responses

**Alternative for Citations:**
Users can manually request citation information by asking:
```
What sources did you use for that answer?
```

The UserPromptSubmit hook still logs retrieved documents, so citation information is available in logs at `logs/rag_cli.log`.

#### Resolution Plan

1. **Short-term (Current):** Keep hook disabled, system remains stable
2. **Medium-term:** Monitor Claude Code framework updates for JSON parsing fixes
3. **Long-term:** Re-enable hook once framework bug is resolved

**Potential Fix Approaches:**
- Wait for Claude Code framework update (recommended)
- Implement alternative citation method via UserPromptSubmit hook
- Use response metadata instead of modifying response text
- Create custom MCP server endpoint for citation retrieval

#### Testing

To test if the issue is resolved in future Claude Code versions:

1. Update Claude Code to latest version
2. Enable the hook in `.claude-plugin/hooks.json`:
   ```json
   "enabled": true
   ```
3. Restart Claude Code
4. Test with a RAG-enhanced query:
   ```
   /rag-retrieve --query "test query"
   ```
5. Check if citations appear correctly in the response
6. Monitor logs for JSON parsing errors

#### Related Files

- Hook implementation: `src/rag_cli_plugin/hooks/response-post.py`
- Configuration: `.claude-plugin/hooks.json`
- Test: `tests/test_hooks/test_response_post.py`
- Documentation: `docs/hooks/response-post.md`

#### Additional Notes

- The hook code is fully functional and tested in isolation
- Unit tests pass when run outside the Claude Code framework
- Issue is specific to the plugin framework's hook processing
- No data loss or corruption occurs with the workaround
- System stability and performance are unaffected

---

## Resolved Issues

### Import Structure Changes (v1.x to v2.0)

**Status:** Resolved in v2.0.0
**Resolution:** Complete restructuring to dual-package layout

The v1.x import structure was inconsistent. This was resolved in v2.0 by:
- Creating `rag_cli` core library package
- Creating `rag_cli_plugin` plugin package
- Updating all imports to use new structure
- Providing migration scripts in `scripts/utils/`

See `docs/V2_RESTRUCTURING_SUMMARY.md` for details.

---

## Future Enhancements

These are not bugs but planned features that will address limitations:

### 1. Advanced Citation System

Once PostToolUse hook is working:
- Sentence-level citation mapping using embeddings
- Smart citation placement (avoid over-citing)
- Interactive citation exploration in responses
- Clickable source references

### 2. Persistent Hook State Management

Improve hook state coordination:
- Shared state between hooks via Redis/file cache
- Better session context preservation
- Cross-hook data flow optimization

### 3. Hook Error Recovery

Enhanced error handling:
- Automatic retry logic for transient failures
- Graceful degradation when hooks fail
- Better error reporting to users

---

## Reporting New Issues

If you discover a new issue:

1. Check if it's already documented here
2. Check GitHub issues: https://github.com/ItMeDiaTech/rag-cli/issues
3. For security issues, see SECURITY.md
4. For general bugs, open a GitHub issue with:
   - RAG-CLI version (`rag --version`)
   - Claude Code version
   - Operating system
   - Steps to reproduce
   - Expected vs actual behavior
   - Relevant log excerpts from `logs/rag_cli.log`

---

## Issue History

| Date       | Issue                     | Status      | Version |
|------------|---------------------------|-------------|---------|
| 2025-11-04 | PostToolUse hook disabled | Workaround  | v2.0.0  |
| 2025-11-03 | Import structure v1.x     | Resolved    | v2.0.0  |
| 2025-11-02 | ChromaDB persistence      | Resolved    | v2.0.0  |
| 2025-11-01 | Duplicate detection       | Resolved    | v2.0.0  |

---

Last updated: 2025-11-04
