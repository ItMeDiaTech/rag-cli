# Watch RAG System

Launch the RAG-CLI real-time monitoring dashboard in Chrome browser with live plugin activity tracking.

## Task

Execute the following steps in order:

1. **Check if monitoring services are running**
   - Check TCP server on port 9999: `Test-NetConnection -ComputerName localhost -Port 9999 -ErrorAction SilentlyContinue`
   - Check web dashboard on port 5000: `Test-NetConnection -ComputerName localhost -Port 5000 -ErrorAction SilentlyContinue`

2. **Start monitoring services if not running**
   - If TCP server (port 9999) is not running:
     - Start it: `Start-Process python -ArgumentList "-m src.monitoring.tcp_server" -WindowStyle Hidden`
     - Wait 2 seconds for initialization
   - If web dashboard (port 5000) is not running:
     - Start it: `Start-Process python -ArgumentList "-m src.monitoring.web_dashboard" -WindowStyle Hidden`
     - Wait 2 seconds for initialization

3. **Open Chrome browser with real-time dashboard**
   - Use Python to open the dashboard in Chrome: `python -c "from src.monitoring.service_manager import open_dashboard_in_browser; open_dashboard_in_browser()"`
   - This will:
     - Find Chrome on the system (Windows/Mac/Linux paths)
     - Open Chrome with security flags for local development (`--disable-web-security` for CORS)
     - Navigate to http://localhost:5000
     - Display the real-time monitoring dashboard with:
       - **Plugin Activity Timeline**: Live stream of query processing steps
       - **Plugin Reasoning Panel**: Decision explanations and document selection logic
       - **Query Enhancement Display**: Before/after query comparison with retrieved docs
       - **Performance Metrics**: Latencies, throughput, cache stats
       - **Real-time Logs**: SSE-powered instant log streaming

4. **Verify dashboard is live**
   - Confirm Chrome opened to http://localhost:5000
   - Check that the dashboard shows "Live (SSE)" status in the header
   - Verify event stream is connected (look for "Real-time" badges)

5. **Provide user feedback**
   - Confirm to the user that:
     - ✅ Monitoring server is running on port 9999
     - ✅ Web dashboard is running on port 5000
     - ✅ Chrome opened to http://localhost:5000
     - 📊 Real-time dashboard is streaming events via Server-Sent Events
     - 🔄 Plugin activity will appear instantly when RAG queries are processed
     - 💡 Plugin reasoning and decisions are visible in real-time
     - 🌐 If Chrome didn't open, they can manually navigate to http://localhost:5000

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
