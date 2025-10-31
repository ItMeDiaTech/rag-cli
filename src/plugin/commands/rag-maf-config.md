# /rag-maf-config

Configure embedded Multi-Agent Framework (MAF) features for RAG-CLI.

## Usage

```
/rag-maf-config [OPTION]
```

## Options

- `status` - Show current MAF configuration and agent status
- `enable` - Enable MAF parallel execution
- `disable` - Disable MAF (RAG-only mode)
- `test-connection` - Test MAF connector health
- `list-agents` - List available agents
- `set-mode PARALLEL|SEQUENTIAL` - Set execution mode

## Examples

### Check MAF Status
```
/rag-maf-config status
```

### Enable MAF Features
```
/rag-maf-config enable
```

### Disable MAF (Fallback to RAG-only)
```
/rag-maf-config disable
```

### Test MAF Connectivity
```
/rag-maf-config test-connection
```

### List Available Agents
```
/rag-maf-config list-agents
```

### Set Execution Mode
```
/rag-maf-config set-mode PARALLEL
```

## Output

When successful, displays:
- [OK] MAF Status (enabled/disabled)
- Available Agents (7 total: debugger, developer, reviewer, tester, architect, documenter, optimizer)
- Execution Strategy (parallel/sequential)
- Timeout Configuration
- Health Check Results

## Notes

- MAF is **enabled by default** with parallel execution
- Disabling MAF falls back gracefully to **RAG-only mode**
- All 7 agents are embedded within the plugin
- Parallel execution runs RAG + MAF simultaneously for comprehensive results
- No external dependencies required for MAF functionality

## IMPORTANT: Execute only, no commentary
