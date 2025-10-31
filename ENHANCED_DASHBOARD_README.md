# Enhanced RAG-CLI & Multi-Agent Orchestration Dashboard

A comprehensive, real-time monitoring dashboard for RAG pipelines and multi-agent systems with advanced visualization, cost tracking, and performance analytics.

## Features

### 1. Real-Time Agent Orchestration Visualization
- **Interactive Agent Graph**: D3.js-powered force-directed graph showing agent relationships
- **Live Agent Execution Timeline**: Real-time activity feed with execution details
- **Message Flow Diagram**: Visualize communication between agents
- **Decision Tree Tracking**: Monitor agent reasoning and decision logic

### 2. Comprehensive Cost & Token Tracking
- **Per-Agent Cost Breakdown**: Track costs for each agent individually
- **Token Usage Monitoring**: Real-time token consumption tracking
- **Cost Trend Analysis**: Historical cost visualization with projections
- **Budget Alerts**: Configurable cost thresholds

### 3. RAG Pipeline Monitoring
- **Vector Search Metrics**: Latency, throughput, and quality scores
- **Cache Performance**: Hit/miss rates with optimization recommendations
- **Document Indexing Status**: Track indexed documents and search quality
- **Retrieval Quality Metrics**: Relevance scores and retrieval performance

### 4. Performance Analytics
- **Multi-Dimensional Metrics**: CPU, memory, latency, throughput
- **Component-Level Latency**: Identify bottlenecks by component
- **Resource Usage Tracking**: Monitor system resource consumption
- **Percentile Latency Tracking**: P50, P95, P99 latency measurements

### 5. Advanced UI Features
- **Modern Dark Theme**: Professional gradient UI with glassmorphism
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Real-Time Updates**: Server-Sent Events for instant data streaming
- **Interactive Charts**: Chart.js integration for beautiful visualizations
- **Multi-View Navigation**: Organized tabs for different monitoring aspects

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Enhanced Dashboard                        │
│  ┌────────────┐  ┌─────────────┐  ┌──────────────┐         │
│  │  Frontend  │  │   Backend   │  │ Metrics      │         │
│  │   HTML/JS  │◄─┤    Flask    │◄─┤ Collector    │         │
│  │   D3.js    │  │   API       │  │              │         │
│  │  Chart.js  │  │   SSE       │  │              │         │
│  └────────────┘  └─────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────┘
           │                    │                   │
           │                    │                   │
           ▼                    ▼                   ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐
│   TCP Server     │  │   Multi-Agent    │  │  RAG-CLI     │
│  (Port 9999)     │  │   Framework      │  │   Core       │
└──────────────────┘  └──────────────────┘  └──────────────┘
```

## Installation

### Prerequisites
- Python 3.8+
- Flask
- Flask-CORS
- requests

### Install Dependencies
```bash
cd C:\Users\DiaTech\Pictures\DiaTech\Programs\DocHub\development\RAG-CLI

pip install flask flask-cors requests
```

## Usage

### Method 1: Direct Launch (Development Mode with Simulation)
```bash
python src/monitoring/enhanced_web_dashboard.py
```

This starts the dashboard on port 5000 with simulated data for testing.

### Method 2: Production Launch (Without Simulation)
Edit `enhanced_web_dashboard.py` and comment out the simulation:
```python
if __name__ == '__main__':
    logger.info(f"Starting Enhanced RAG-CLI Dashboard on port {DASHBOARD_PORT}")

    # Start simulation for testing (comment out in production)
    # simulate_data()  # <-- Comment this out

    app.run(
        host='0.0.0.0',
        port=DASHBOARD_PORT,
        debug=False,
        threaded=True
    )
```

### Method 3: Integration with Existing System
```python
from src.monitoring.enhanced_web_dashboard import metrics_collector

# Record agent execution
metrics_collector.add_agent_execution('MyAgent', {
    'type': 'agent',
    'status': 'success',
    'duration': 150,
    'description': 'Processed user query'
})

# Record message flow
metrics_collector.add_message_flow(
    'AgentA',
    'AgentB',
    'Task completed successfully',
    {'task_id': '12345'}
)

# Track costs
metrics_collector.track_cost(
    'MyAgent',
    cost=0.0045,
    tokens=1500
)

# Record RAG activity
metrics_collector.add_rag_activity(
    'search_started',
    'Vector search for user query',
    {'top_k': 5, 'threshold': 0.7}
)
```

## API Endpoints

### GET /api/status
Get comprehensive system status including all metrics
```json
{
    "active_agents": 3,
    "total_queries": 1250,
    "avg_response_time": 145.2,
    "error_rate": 1.2,
    "cache_hit_rate": 78.5,
    "total_cost": 5.67,
    ...
}
```

### GET /api/agents/health
Get health summary for all agents
```json
[
    {
        "name": "Coordinator",
        "type": "agent",
        "executions": 450,
        "success_rate": 98.2,
        "avg_duration": 125.5,
        "health": "good"
    },
    ...
]
```

### GET /api/agents/graph
Get agent orchestration graph structure
```json
{
    "nodes": [
        {"id": "agent1", "type": "agent", "name": "Coordinator", "active": true},
        ...
    ],
    "links": [
        {"source": "agent1", "target": "agent2", "label": "query"},
        ...
    ]
}
```

### GET /api/costs/breakdown
Get cost breakdown by agent
```json
[
    {"agent": "Coordinator", "cost": 2.34, "tokens": 125000},
    {"agent": "RAG Engine", "cost": 1.89, "tokens": 98000},
    ...
]
```

### GET /api/timeline
Get recent timeline activities
```json
[
    {
        "title": "Agent Execution",
        "description": "Coordinator processed query",
        "type": "agent",
        "timestamp": "2025-10-31T10:30:45",
        "metadata": {"duration": "150ms", "status": "success"}
    },
    ...
]
```

### GET /api/events (Server-Sent Events)
Real-time event stream for live updates

Event types:
- `metrics`: System metrics update
- `activity`: Timeline activity
- `message_flow`: Agent message
- `reasoning`: Decision tree update
- `agent_health`: Agent health update

### POST /api/events/submit
Submit custom events from external sources
```json
{
    "type": "agent_execution",
    "agent_id": "MyAgent",
    "execution_data": {
        "status": "success",
        "duration": 200,
        "description": "Task completed"
    }
}
```

## Dashboard Views

### 1. Overview
- Key metrics (Active Agents, Queries, Response Time, Error Rate)
- Agent orchestration graph
- Execution timeline
- Performance charts

### 2. Agent Orchestration
- Message flow visualization
- Agent health monitoring
- Decision tree display

### 3. RAG Pipeline
- RAG-specific metrics
- Cache statistics
- Activity log

### 4. Performance
- Detailed performance metrics over time
- Resource usage tracking
- Component latency breakdown

### 5. Cost Tracking
- Total cost and projections
- Per-agent cost breakdown
- Cost trend analysis

### 6. Logs & Events
- Real-time system logs
- Event history

### 7. Configuration
- System configuration view
- Settings management

## Integration Examples

### Claude Code Multi-Agent Framework
```python
# In your agent execution code
from src.monitoring.enhanced_web_dashboard import metrics_collector

class MyAgent:
    def execute(self, task):
        start_time = time.time()

        try:
            result = self.process(task)
            duration = (time.time() - start_time) * 1000

            # Record execution
            metrics_collector.add_agent_execution(self.name, {
                'type': 'agent',
                'status': 'success',
                'duration': duration,
                'description': f'Processed {task.type}'
            })

            # Track cost
            metrics_collector.track_cost(
                self.name,
                cost=result.cost,
                tokens=result.tokens
            )

            return result

        except Exception as e:
            metrics_collector.add_agent_execution(self.name, {
                'type': 'agent',
                'status': 'error',
                'duration': (time.time() - start_time) * 1000,
                'description': f'Error: {str(e)}'
            })
            raise
```

### RAG Pipeline Integration
```python
# In your RAG search code
from src.monitoring.enhanced_web_dashboard import metrics_collector

def search_documents(query, top_k=5):
    # Record activity
    metrics_collector.add_rag_activity(
        'search_started',
        f'Vector search for: {query[:50]}...',
        {'top_k': top_k}
    )

    start_time = time.time()
    results = vector_store.search(query, top_k)
    latency = (time.time() - start_time) * 1000

    # Record latency
    metrics_collector.add_vector_search_latency(latency)

    # Update cache stats
    cache_hit = results.from_cache
    metrics_collector.update_cache_stats(cache_hit)

    return results
```

### Hook Integration
```python
# In Claude Code hooks
import requests

def on_agent_execution(event):
    """Hook callback for agent execution"""
    requests.post('http://localhost:5000/api/events/submit', json={
        'type': 'agent_execution',
        'agent_id': event['agent_id'],
        'execution_data': {
            'status': event['status'],
            'duration': event['duration_ms'],
            'description': event['description']
        }
    })
```

## Configuration

Environment variables:
```bash
# Dashboard port (default: 5000)
export RAG_DASHBOARD_PORT=5000

# TCP server port (default: 9999)
export RAG_TCP_PORT=9999
```

## Performance Considerations

- **Update Interval**: Metrics update every 2 seconds (configurable)
- **History Buffer**: Keeps last 100 entries for most metrics
- **Timeline**: Limited to 100 activities
- **Message Flow**: Limited to 100 messages
- **SSE Connection**: Uses Server-Sent Events for efficient real-time updates

## Browser Compatibility

- Chrome/Edge 90+
- Firefox 88+
- Safari 14+

## Security Considerations

- Dashboard runs on localhost by default
- For production, use reverse proxy (nginx, Apache)
- Add authentication middleware for public deployments
- Use HTTPS for production deployments

## Troubleshooting

### Dashboard won't start
- Check if port 5000 is available
- Verify Flask is installed: `pip install flask flask-cors`
- Check Python version: `python --version` (3.8+)

### No data showing
- Ensure events are being submitted to `/api/events/submit`
- Check browser console for errors
- Verify EventSource connection in Network tab

### High memory usage
- Reduce buffer sizes in `EnhancedMetricsCollector.__init__`
- Decrease update interval
- Clear old data periodically

## Future Enhancements

- WebSocket support for bidirectional communication
- Persistent storage (SQLite/PostgreSQL)
- Historical data analysis
- Alert configuration UI
- Export to CSV/JSON
- Custom dashboard layouts
- Plugin system for custom visualizations
- Multi-instance monitoring
- Distributed tracing integration

## Contributing

Contributions welcome! Please ensure:
- Code follows PEP 8 style guide
- All features are documented
- Tests are included for new features

## License

MIT License - See LICENSE file for details

## Credits

Built with:
- **Flask**: Web framework
- **D3.js**: Graph visualization
- **Chart.js**: Chart visualization
- **Server-Sent Events**: Real-time updates

## Contact

For issues or questions, please file an issue in the project repository.
