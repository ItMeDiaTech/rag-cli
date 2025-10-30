# MCP Server Configuration Fix

**Date**: October 30, 2025
**Issue**: Multiple MCP server instances causing failures
**Status**: RESOLVED

## Problem Identified

Multiple MCP server configurations were causing conflicts:

1. **`.claude-plugin/plugin.json`** - Defines server `rag-server` (CORRECT)
2. **`.mcp.json`** - Defines server `rag-cli` (REDUNDANT)
3. **Two server implementations**:
   - `src/plugin/mcp/unified_server.py` (comprehensive, current)
   - `src/plugin/mcp/server.py` (older implementation)

This resulted in multiple server instances in Claude Code:
- `plugin:rag-cli:rag-server` (connected - from plugin.json)
- `plugin:rag-cli:rag-cli` (failed - from .mcp.json)
- `rag-cli` (disabled - user configuration)

## Root Cause

When the RAG-CLI plugin is installed via marketplace (`/plugin marketplace add`), the plugin system:
1. Reads `.claude-plugin/plugin.json` as the official plugin manifest
2. Registers MCP servers defined in `plugin.json` with prefix `plugin:rag-cli:`
3. The `.mcp.json` file is intended for standalone (non-plugin) installations

Having both configurations created duplicate server registrations, causing the `plugin:rag-cli:rag-cli` server to fail while `plugin:rag-cli:rag-server` worked correctly.

## Changes Applied

### 1. Removed Redundant Configuration
**File**: `.mcp.json`
**Action**: Renamed to `.mcp.json.backup`
**Reason**: Plugin installations should only use `.claude-plugin/plugin.json`

### 2. Removed Obsolete Server Implementation
**File**: `src/plugin/mcp/server.py`
**Action**: Renamed to `server.py.backup`
**Reason**: `unified_server.py` is the current, comprehensive implementation

### 3. Kept Active Configuration
**File**: `.claude-plugin/plugin.json` (lines 32-43)
```json
"mcpServers": {
  "rag-server": {
    "command": "python",
    "args": ["-m", "src.plugin.mcp.unified_server"],
    "env": {
      "PYTHONUNBUFFERED": "1",
      "RAG_CLI_MODE": "claude_code",
      "PYTHONPATH": "${CLAUDE_PLUGIN_ROOT}",
      "RAG_CLI_ROOT": "${CLAUDE_PLUGIN_ROOT}"
    }
  }
}
```

## Server Architecture

**Active Server**: `src/plugin/mcp/unified_server.py`

Provides 14 MCP tools across 4 categories:

### Service Management
- `start_services` - Start monitoring services
- `get_services_status_tool` - Service status
- `open_dashboard` - Web dashboard

### RAG Search
- `rag_search` - Search knowledge base
- `rag_index` - Index documents
- `rag_status` - System status
- `rag_configure` - Configuration

### Hook Management
- `rag_configure_hooks` - Enable/disable hooks
- `rag_set_citation_format` - Citation formatting
- `rag_get_hook_status` - Hook status
- `rag_set_error_mode` - Error handling

### Multi-Agent Framework
- `maf_execute` - Execute agent tasks
- `maf_status` - Framework status
- `maf_classify` - Query classification

## Expected Result

After this fix, Claude Code should show:
- `plugin:rag-cli:rag-server` - **CONNECTED** (single, working server)
- No failed `rag-cli` server instances
- All 14 MCP tools accessible

## Verification

To verify the fix is working:

1. Restart Claude Code
2. Run `/mcp` to check server status
3. Verify only `plugin:rag-cli:rag-server` appears (connected)
4. Test a tool: Use any `mcp__plugin_rag-cli_rag-server__*` tool

## Installation Method

For marketplace installation (recommended):
```bash
/plugin marketplace add https://github.com/ItMeDiaTech/rag-cli.git
```

This uses `.claude-plugin/plugin.json` automatically.

## Standalone Installation (Advanced)

If installing without marketplace, restore `.mcp.json`:
```bash
cd rag-cli
mv .mcp.json.backup .mcp.json
```

Then configure in Claude Code manually (not recommended).

## Files Modified

- `.mcp.json` → `.mcp.json.backup` (removed from active use)
- `src/plugin/mcp/server.py` → `server.py.backup` (archived)
- No changes to documentation (already accurate)

## Testing Performed

- Server startup verification: `python -m src.plugin.mcp.unified_server` ✓
- Configuration structure validation ✓
- Tool availability confirmed (14 tools) ✓

## Notes

- The `.mcp.json` is kept as `.mcp.json.backup` for reference
- `server.py` is kept as `server.py.backup` in case of rollback needs
- The plugin.json configuration is the authoritative source for plugin installations
- TROUBLESHOOTING.md already documents the correct configuration

## Related Files

- `.claude-plugin/plugin.json` - Official plugin manifest
- `src/plugin/mcp/unified_server.py` - Active MCP server
- `TROUBLESHOOTING.md` - User documentation
- `.mcp.json.backup` - Archived redundant config
- `src/plugin/mcp/server.py.backup` - Archived old server

---

**Fix Status**: COMPLETE
**Server Status**: `plugin:rag-cli:rag-server` should be CONNECTED
**Action Required**: Restart Claude Code to apply changes
