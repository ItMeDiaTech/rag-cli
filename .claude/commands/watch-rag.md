# Watch RAG System

Launch the RAG-CLI monitoring dashboard in Windows Terminal for real-time system monitoring.

## Task

Execute the following steps:

1. **Check if monitoring server is running** on port 9999
   - Use PowerShell command: `Test-NetConnection -ComputerName localhost -Port 9999 -ErrorAction SilentlyContinue`
   - If the connection succeeds (TcpTestSucceeded = True), the server is already running
   - If it fails, you'll need to start the server

2. **Start monitoring server if not running**
   - If server is not running, start it with: `python -m src.monitoring.tcp_server`
   - Run this command in the background using PowerShell
   - Wait 2-3 seconds for the server to initialize
   - You can use: `Start-Process python -ArgumentList "-m src.monitoring.tcp_server" -WindowStyle Hidden`

3. **Launch Windows Terminal with monitoring dashboard**
   - Execute: `wt.exe -d "." "pwsh" "-NoExit" "-Command" ".\scripts\monitor.ps1 WATCH"`
   - This opens a new Windows Terminal window with the PowerShell WATCH dashboard
   - The dashboard updates every 5 seconds showing:
     - Performance metrics (latencies for vector search, keyword search, reranking, Claude API)
     - Throughput (queries per minute, cache hit rate)
     - Resource usage (memory in MB, CPU percentage)
     - Recent activity logs (last 5 entries)

4. **Confirm and provide feedback to user**
   - After launching, confirm to the user that:
     - The monitoring server is running on port 9999
     - Windows Terminal is launching with the WATCH dashboard
     - They can press Ctrl+C in the terminal to stop watching
     - The monitoring data refreshes every 5 seconds

## Important Notes

- The working directory when executing should be the RAG-CLI project root
- The TCP server must be running for the dashboard to work
- Windows Terminal will open in a separate window for unobstructed monitoring
- The WATCH interface is read-only and doesn't modify any system state
- If the server fails to start, check that Python and dependencies are properly installed

## Example Output

When the WATCH dashboard is running, you'll see:

```
RAG-CLI Monitor - 2025-10-25 14:32:45
============================================================

Performance:
  VECTOR SEARCH    [████████░░░░░░░░░░░░░░░░] 45.23 ms
  KEYWORD SEARCH   [███████░░░░░░░░░░░░░░░░░░] 35.67 ms
  RERANKING        [███░░░░░░░░░░░░░░░░░░░░░░] 15.42 ms
  CLAUDE API       [████████████████░░░░░░░░░░] 850.00 ms

Throughput:
  Queries/min: 12.5
  Cache Hits:  78.3%

Resources:
  Memory: 256 MB | CPU: 2.5%

Recent Activity:
  [2025-10-25T14:32:40] INFO | Query processed successfully
  [2025-10-25T14:32:38] INFO | Cache hit for query context
  [2025-10-25T14:32:35] DEBUG | Vector search completed
  [2025-10-25T14:32:32] INFO | Document chunk retrieved
  [2025-10-25T14:32:30] INFO | Server started successfully
```
