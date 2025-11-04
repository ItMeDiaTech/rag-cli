"""
Enhanced Web Dashboard with Multi-Agent Orchestration Support
Combines RAG monitoring with agent execution tracking, cost analysis, and real-time visualization
"""

import os
import sys
import json
import time
import logging
import threading
from datetime import datetime, timedelta
from collections import deque, defaultdict
from typing import Dict, List, Any, Optional
from flask import Flask, render_template, jsonify, Response, stream_with_context, request
from flask_cors import CORS
import requests

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
TCP_SERVER_PORT = int(os.environ.get('RAG_TCP_PORT', 9999))
DASHBOARD_PORT = int(os.environ.get('RAG_DASHBOARD_PORT', 5000))
UPDATE_INTERVAL = 2  # seconds

# Enhanced Metrics Storage
class EnhancedMetricsCollector:
    """Collects and aggregates metrics for multi-agent orchestration and RAG pipeline"""

    def __init__(self):
        # Agent orchestration metrics
        self.agent_executions = defaultdict(list)  # agent_id -> list of executions
        self.agent_messages = deque(maxlen=100)  # Message flow between agents
        self.agent_graph_nodes = []  # Current active agents
        self.agent_graph_links = []  # Connections between agents
        self.decision_tree = deque(maxlen=50)  # Decision tree nodes

        # Cost tracking
        self.cost_by_agent = defaultdict(float)  # agent_id -> total cost
        self.token_usage_by_agent = defaultdict(int)  # agent_id -> total tokens
        self.cost_history = deque(maxlen=100)  # Historical cost data

        # RAG pipeline metrics
        self.rag_activities = deque(maxlen=100)
        self.cache_stats = {'hits': 0, 'misses': 0}
        self.vector_search_latencies = deque(maxlen=100)
        self.document_count = 0

        # Performance metrics
        self.latency_history = deque(maxlen=100)
        self.throughput_history = deque(maxlen=100)
        self.cpu_history = deque(maxlen=100)
        self.memory_history = deque(maxlen=100)

        # Timeline activities
        self.timeline_activities = deque(maxlen=100)

        # System stats
        self.total_queries = 0
        self.error_count = 0
        self.active_agents_count = 0

        # Lock for thread safety
        self.lock = threading.Lock()

    def add_agent_execution(self, agent_id: str, execution_data: Dict[str, Any]):
        """Record an agent execution"""
        with self.lock:
            execution_data['timestamp'] = datetime.now().isoformat()
            self.agent_executions[agent_id].append(execution_data)

            # Update timeline
            self.timeline_activities.append({
                'title': f'{agent_id} Execution',
                'description': execution_data.get('description', 'Agent executed'),
                'type': 'agent',
                'timestamp': datetime.now().isoformat(),
                'metadata': {
                    'duration': f"{execution_data.get('duration', 0)}ms",
                    'status': execution_data.get('status', 'success')
                }
            })

    def add_message_flow(self, from_agent: str, to_agent: str, content: str, metadata: Optional[Dict] = None):
        """Record message flow between agents"""
        with self.lock:
            self.agent_messages.append({
                'from': from_agent,
                'to': to_agent,
                'content': content,
                'timestamp': datetime.now().isoformat(),
                'metadata': metadata or {}
            })

    def update_agent_graph(self, nodes: List[Dict], links: List[Dict]):
        """Update the agent orchestration graph"""
        with self.lock:
            self.agent_graph_nodes = nodes
            self.agent_graph_links = links
            self.active_agents_count = len([n for n in nodes if n.get('active', False)])

    def add_decision(self, decision_type: str, title: str, content: str, children: Optional[List] = None):
        """Add a decision tree node"""
        with self.lock:
            self.decision_tree.append({
                'type': decision_type,
                'title': title,
                'content': content,
                'children': children or [],
                'timestamp': datetime.now().isoformat()
            })

    def track_cost(self, agent_id: str, cost: float, tokens: int):
        """Track cost and token usage for an agent"""
        with self.lock:
            self.cost_by_agent[agent_id] += cost
            self.token_usage_by_agent[agent_id] += tokens

            self.cost_history.append({
                'timestamp': datetime.now().isoformat(),
                'agent_id': agent_id,
                'cost': cost,
                'tokens': tokens
            })

    def add_rag_activity(self, activity_type: str, description: str, metadata: Optional[Dict] = None):
        """Record RAG pipeline activity"""
        with self.lock:
            self.rag_activities.append({
                'type': activity_type,
                'description': description,
                'timestamp': datetime.now().isoformat(),
                'metadata': metadata or {}
            })

            # Also add to timeline
            self.timeline_activities.append({
                'title': activity_type.replace('_', ' ').title(),
                'description': description,
                'type': 'rag',
                'timestamp': datetime.now().isoformat(),
                'metadata': metadata or {}
            })

    def update_cache_stats(self, hit: bool):
        """Update cache hit/miss statistics"""
        with self.lock:
            if hit:
                self.cache_stats['hits'] += 1
            else:
                self.cache_stats['misses'] += 1

    def add_vector_search_latency(self, latency_ms: float):
        """Record vector search latency"""
        with self.lock:
            self.vector_search_latencies.append(latency_ms)

    def record_query(self, success: bool = True):
        """Record a query execution"""
        with self.lock:
            self.total_queries += 1
            if not success:
                self.error_count += 1

    def add_latency(self, latency_ms: float):
        """Add latency measurement"""
        with self.lock:
            self.latency_history.append({
                'timestamp': datetime.now().isoformat(),
                'value': latency_ms
            })

    def add_throughput(self, qps: float):
        """Add throughput measurement"""
        with self.lock:
            self.throughput_history.append({
                'timestamp': datetime.now().isoformat(),
                'value': qps
            })

    def add_resource_usage(self, cpu: float, memory: float):
        """Add resource usage measurement"""
        with self.lock:
            timestamp = datetime.now().isoformat()
            self.cpu_history.append({'timestamp': timestamp, 'value': cpu})
            self.memory_history.append({'timestamp': timestamp, 'value': memory})

    def get_agent_health_summary(self) -> List[Dict]:
        """Get health summary for all agents"""
        with self.lock:
            health_data = []

            for agent_id, executions in self.agent_executions.items():
                if not executions:
                    continue

                total = len(executions)
                successful = sum(1 for e in executions if e.get('status') == 'success')
                success_rate = (successful / total * 100) if total > 0 else 0

                durations = [e.get('duration', 0) for e in executions]
                avg_duration = sum(durations) / len(durations) if durations else 0

                # Determine health
                health = 'good' if success_rate >= 95 else 'warning' if success_rate >= 80 else 'error'

                health_data.append({
                    'name': agent_id,
                    'type': executions[0].get('type', 'agent'),
                    'executions': total,
                    'success_rate': round(success_rate, 1),
                    'avg_duration': round(avg_duration, 1),
                    'health': health
                })

            return health_data

    def get_cost_breakdown(self) -> List[Dict]:
        """Get cost breakdown by agent"""
        with self.lock:
            breakdown = []

            for agent_id in self.cost_by_agent.keys():
                breakdown.append({
                    'agent': agent_id,
                    'cost': round(self.cost_by_agent[agent_id], 4),
                    'tokens': self.token_usage_by_agent[agent_id]
                })

            # Sort by cost descending
            breakdown.sort(key=lambda x: x['cost'], reverse=True)
            return breakdown

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary"""
        with self.lock:
            # Calculate cache hit rate
            total_cache = self.cache_stats['hits'] + self.cache_stats['misses']
            cache_hit_rate = (self.cache_stats['hits'] / total_cache * 100) if total_cache > 0 else 0

            # Calculate error rate
            error_rate = (self.error_count / self.total_queries * 100) if self.total_queries > 0 else 0

            # Get latest latency
            latest_latency = self.latency_history[-1]['value'] if self.latency_history else 0

            # Calculate average vector search latency
            avg_vector_latency = (sum(self.vector_search_latencies) / len(self.vector_search_latencies)) if self.vector_search_latencies else 0

            # Get latest resource usage
            latest_cpu = self.cpu_history[-1]['value'] if self.cpu_history else 0
            latest_memory = self.memory_history[-1]['value'] if self.memory_history else 0

            # Calculate total cost
            total_cost = sum(self.cost_by_agent.values())
            avg_cost_per_query = (total_cost / self.total_queries) if self.total_queries > 0 else 0

            # Prepare chart data
            latency_labels = [d['timestamp'][-8:-3] for d in list(self.latency_history)[-20:]]
            latency_values = [d['value'] for d in list(self.latency_history)[-20:]]

            throughput_labels = [d['timestamp'][-8:-3] for d in list(self.throughput_history)[-20:]]
            throughput_values = [d['value'] for d in list(self.throughput_history)[-20:]]

            cpu_labels = [d['timestamp'][-8:-3] for d in list(self.cpu_history)[-20:]]
            cpu_values = [d['value'] for d in list(self.cpu_history)[-20:]]
            memory_values = [d['value'] for d in list(self.memory_history)[-20:]]

            return {
                # Key metrics
                'active_agents': self.active_agents_count,
                'total_queries': self.total_queries,
                'avg_response_time': round(latest_latency, 1),
                'error_rate': round(error_rate, 2),

                # RAG metrics
                'cache_hit_rate': round(cache_hit_rate, 1),
                'vector_search_latency': round(avg_vector_latency, 1),
                'documents_indexed': self.document_count,

                # Resource usage
                'cpu_usage': round(latest_cpu, 1),
                'memory_usage': round(latest_memory, 1),
                'active_threads': 0,  # To be populated

                # Cost metrics
                'total_cost': round(total_cost, 4),
                'avg_cost_query': round(avg_cost_per_query, 6),
                'total_tokens': sum(self.token_usage_by_agent.values()),

                # Chart data
                'latency_history': {
                    'labels': latency_labels,
                    'values': latency_values
                },
                'throughput_history': {
                    'labels': throughput_labels,
                    'values': throughput_values
                },
                'performance_history': {
                    'labels': cpu_labels,
                    'cpu': cpu_values,
                    'memory': memory_values
                }
            }


# Global metrics collector
metrics_collector = EnhancedMetricsCollector()


# Routes
@app.route('/')
def index():
    """Serve enhanced dashboard"""
    return render_template('enhanced_dashboard.html')


@app.route('/api/status')
def get_status():
    """Get comprehensive system status"""
    try:
        # Try to get data from TCP server
        try:
            response = requests.get(f'http://localhost:{TCP_SERVER_PORT}/status', timeout=2)
            tcp_data = response.json() if response.status_code == 200 else {}
        except (requests.RequestException, ConnectionError, TimeoutError, OSError) as e:
            logger.debug(f"TCP server not available: {e}")
            tcp_data = {}

        # Merge with enhanced metrics
        status = metrics_collector.get_metrics_summary()
        status.update(tcp_data)

        return jsonify(status)
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/metrics')
def get_metrics():
    """Get current metrics"""
    return jsonify(metrics_collector.get_metrics_summary())


@app.route('/api/agents/health')
def get_agent_health():
    """Get agent health summary"""
    return jsonify(metrics_collector.get_agent_health_summary())


@app.route('/api/agents/graph')
def get_agent_graph():
    """Get agent orchestration graph"""
    return jsonify({
        'nodes': metrics_collector.agent_graph_nodes,
        'links': metrics_collector.agent_graph_links
    })


@app.route('/api/costs/breakdown')
def get_cost_breakdown():
    """Get cost breakdown by agent"""
    return jsonify(metrics_collector.get_cost_breakdown())


@app.route('/api/timeline')
def get_timeline():
    """Get timeline activities"""
    return jsonify(list(metrics_collector.timeline_activities))


@app.route('/api/messages')
def get_messages():
    """Get message flow"""
    return jsonify(list(metrics_collector.agent_messages))


@app.route('/api/decisions')
def get_decisions():
    """Get decision tree"""
    return jsonify(list(metrics_collector.decision_tree))


@app.route('/api/rag/activities')
def get_rag_activities():
    """Get RAG pipeline activities"""
    return jsonify(list(metrics_collector.rag_activities))


@app.route('/api/events')
def event_stream():
    """Server-Sent Events stream for real-time updates"""

    def generate():
        last_update = time.time()

        while True:
            current_time = time.time()

            # Send metrics update every UPDATE_INTERVAL seconds
            if current_time - last_update >= UPDATE_INTERVAL:
                metrics = metrics_collector.get_metrics_summary()
                yield f"event: metrics\ndata: {json.dumps(metrics)}\n\n"

                # Send agent health update
                agent_health = metrics_collector.get_agent_health_summary()
                yield f"event: agent_health\ndata: {json.dumps(agent_health)}\n\n"

                last_update = current_time

            # Send recent activities
            if metrics_collector.timeline_activities:
                activity = metrics_collector.timeline_activities[-1]
                yield f"event: activity\ndata: {json.dumps(activity)}\n\n"

            # Send recent messages
            if metrics_collector.agent_messages:
                message = metrics_collector.agent_messages[-1]
                yield f"event: message_flow\ndata: {json.dumps(message)}\n\n"

            # Send recent decisions
            if metrics_collector.decision_tree:
                decision = metrics_collector.decision_tree[-1]
                yield f"event: reasoning\ndata: {json.dumps(decision)}\n\n"

            time.sleep(0.5)

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'
        }
    )


@app.route('/api/events/submit', methods=['POST'])
def submit_event():
    """Accept events from external sources (e.g., agents, hooks)"""
    try:
        data = request.json
        event_type = data.get('type')

        if event_type == 'agent_execution':
            metrics_collector.add_agent_execution(
                data.get('agent_id'),
                data.get('execution_data', {})
            )
        elif event_type == 'message_flow':
            metrics_collector.add_message_flow(
                data.get('from'),
                data.get('to'),
                data.get('content'),
                data.get('metadata')
            )
        elif event_type == 'decision':
            metrics_collector.add_decision(
                data.get('decision_type'),
                data.get('title'),
                data.get('content'),
                data.get('children')
            )
        elif event_type == 'cost':
            metrics_collector.track_cost(
                data.get('agent_id'),
                data.get('cost', 0.0),
                data.get('tokens', 0)
            )
        elif event_type == 'rag_activity':
            metrics_collector.add_rag_activity(
                data.get('activity_type'),
                data.get('description'),
                data.get('metadata')
            )
        elif event_type == 'query':
            metrics_collector.record_query(data.get('success', True))
            if data.get('latency'):
                metrics_collector.add_latency(data['latency'])

        return jsonify({'status': 'success'})
    except Exception as e:
        logger.error(f"Error submitting event: {e}")
        return jsonify({'error': str(e)}), 400


@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'uptime': time.time(),
        'version': '2.0.0-enhanced'
    })


# Simulation for testing (remove in production)
def simulate_data():
    """Simulate agent activity for testing"""
    import random
    import threading

    def simulation_loop():
        agents = ['Coordinator', 'RAG Engine', 'Responder', 'Analyzer']

        while True:
            # Simulate agent execution
            agent = random.choice(agents)
            metrics_collector.add_agent_execution(agent, {
                'type': 'agent' if agent != 'RAG Engine' else 'rag',
                'status': 'success' if random.random() > 0.1 else 'error',
                'duration': random.randint(100, 500),
                'description': f'{agent} processed request'
            })

            # Simulate message flow
            if len(agents) > 1:
                from_agent = random.choice(agents)
                to_agent = random.choice([a for a in agents if a != from_agent])
                metrics_collector.add_message_flow(
                    from_agent,
                    to_agent,
                    f"Message from {from_agent} to {to_agent}",
                    {'size': random.randint(100, 1000)}
                )

            # Simulate cost
            metrics_collector.track_cost(
                agent,
                random.uniform(0.0001, 0.01),
                random.randint(100, 5000)
            )

            # Simulate RAG activity
            if agent == 'RAG Engine':
                metrics_collector.add_rag_activity(
                    'search_started',
                    f'Vector search for query {random.randint(1000, 9999)}',
                    {'top_k': 5}
                )
                metrics_collector.add_vector_search_latency(random.uniform(10, 100))
                metrics_collector.update_cache_stats(random.random() > 0.3)

            # Simulate metrics
            metrics_collector.add_latency(random.uniform(50, 300))
            metrics_collector.add_throughput(random.uniform(10, 50))
            metrics_collector.add_resource_usage(
                random.uniform(20, 80),
                random.uniform(100, 500)
            )

            # Update agent graph
            metrics_collector.update_agent_graph(
                [
                    {'id': 'user', 'type': 'user', 'name': 'User Query', 'active': True},
                    {'id': 'coordinator', 'type': 'agent', 'name': 'Coordinator', 'active': True},
                    {'id': 'rag', 'type': 'rag', 'name': 'RAG Engine', 'active': True},
                    {'id': 'responder', 'type': 'agent', 'name': 'Responder', 'active': False}
                ],
                [
                    {'source': 'user', 'target': 'coordinator', 'label': 'query'},
                    {'source': 'coordinator', 'target': 'rag', 'label': 'search'},
                    {'source': 'rag', 'target': 'responder', 'label': 'context'},
                    {'source': 'responder', 'target': 'user', 'label': 'response'}
                ]
            )

            time.sleep(random.uniform(1, 3))

    thread = threading.Thread(target=simulation_loop, daemon=True)
    thread.start()


if __name__ == '__main__':
    logger.info(f"Starting Enhanced RAG-CLI Dashboard on port {DASHBOARD_PORT}")

    # Start simulation for testing (comment out in production)
    simulate_data()

    # SECURITY: Bind to localhost only to prevent network access
    app.run(
        host='127.0.0.1',
        port=DASHBOARD_PORT,
        debug=False,
        threaded=True
    )
