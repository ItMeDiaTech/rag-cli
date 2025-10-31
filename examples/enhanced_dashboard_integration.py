"""
Enhanced Dashboard Integration Example
Demonstrates how to integrate the enhanced dashboard with your multi-agent framework
"""

import time
import random
from datetime import datetime
import requests

# Dashboard API endpoint
DASHBOARD_API = "http://localhost:5000/api/events/submit"


class DashboardIntegration:
    """Helper class for integrating with enhanced dashboard"""

    def __init__(self, api_url=DASHBOARD_API):
        self.api_url = api_url

    def record_agent_execution(self, agent_id, status='success', duration_ms=0, description='', metadata=None):
        """Record an agent execution event"""
        data = {
            'type': 'agent_execution',
            'agent_id': agent_id,
            'execution_data': {
                'status': status,
                'duration': duration_ms,
                'description': description,
                'type': 'agent',
                'metadata': metadata or {}
            }
        }

        try:
            requests.post(self.api_url, json=data, timeout=1)
        except Exception as e:
            print(f"Warning: Could not send event to dashboard: {e}")

    def record_message_flow(self, from_agent, to_agent, content, metadata=None):
        """Record message flow between agents"""
        data = {
            'type': 'message_flow',
            'from': from_agent,
            'to': to_agent,
            'content': content,
            'metadata': metadata or {}
        }

        try:
            requests.post(self.api_url, json=data, timeout=1)
        except Exception as e:
            print(f"Warning: Could not send event to dashboard: {e}")

    def record_decision(self, decision_type, title, content, children=None):
        """Record a decision tree node"""
        data = {
            'type': 'decision',
            'decision_type': decision_type,
            'title': title,
            'content': content,
            'children': children or []
        }

        try:
            requests.post(self.api_url, json=data, timeout=1)
        except Exception as e:
            print(f"Warning: Could not send event to dashboard: {e}")

    def track_cost(self, agent_id, cost, tokens):
        """Track cost and token usage for an agent"""
        data = {
            'type': 'cost',
            'agent_id': agent_id,
            'cost': cost,
            'tokens': tokens
        }

        try:
            requests.post(self.api_url, json=data, timeout=1)
        except Exception as e:
            print(f"Warning: Could not send event to dashboard: {e}")

    def record_rag_activity(self, activity_type, description, metadata=None):
        """Record RAG pipeline activity"""
        data = {
            'type': 'rag_activity',
            'activity_type': activity_type,
            'description': description,
            'metadata': metadata or {}
        }

        try:
            requests.post(self.api_url, json=data, timeout=1)
        except Exception as e:
            print(f"Warning: Could not send event to dashboard: {e}")

    def record_query(self, success=True, latency_ms=0):
        """Record a query execution"""
        data = {
            'type': 'query',
            'success': success,
            'latency': latency_ms
        }

        try:
            requests.post(self.api_url, json=data, timeout=1)
        except Exception as e:
            print(f"Warning: Could not send event to dashboard: {e}")


# Example Multi-Agent System
class ExampleAgent:
    """Example agent with dashboard integration"""

    def __init__(self, name, agent_type='agent'):
        self.name = name
        self.agent_type = agent_type
        self.dashboard = DashboardIntegration()

    def execute(self, task):
        """Execute a task with full dashboard tracking"""
        print(f"[{self.name}] Starting task: {task}")

        start_time = time.time()

        try:
            # Simulate processing
            processing_time = random.uniform(0.1, 0.5)
            time.sleep(processing_time)

            # Simulate cost (example: $0.002 per 1000 tokens)
            tokens = random.randint(500, 2000)
            cost = (tokens / 1000) * 0.002

            duration_ms = (time.time() - start_time) * 1000

            # Record successful execution
            self.dashboard.record_agent_execution(
                agent_id=self.name,
                status='success',
                duration_ms=duration_ms,
                description=f'Processed task: {task}',
                metadata={'tokens': tokens}
            )

            # Track cost
            self.dashboard.track_cost(self.name, cost, tokens)

            # Record query
            self.dashboard.record_query(success=True, latency_ms=duration_ms)

            result = f"Result from {self.name} for {task}"
            print(f"[{self.name}] Completed in {duration_ms:.1f}ms (cost: ${cost:.6f}, tokens: {tokens})")

            return result

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000

            # Record failed execution
            self.dashboard.record_agent_execution(
                agent_id=self.name,
                status='error',
                duration_ms=duration_ms,
                description=f'Error processing task: {str(e)}'
            )

            # Record failed query
            self.dashboard.record_query(success=False, latency_ms=duration_ms)

            raise

    def send_message(self, to_agent, message):
        """Send message to another agent"""
        print(f"[{self.name}] Sending message to {to_agent}: {message}")

        self.dashboard.record_message_flow(
            from_agent=self.name,
            to_agent=to_agent,
            content=message,
            metadata={'timestamp': datetime.now().isoformat()}
        )

    def make_decision(self, decision_type, title, content):
        """Record a decision"""
        print(f"[{self.name}] Making decision: {title}")

        self.dashboard.record_decision(
            decision_type=decision_type,
            title=title,
            content=content
        )


class RAGAgent(ExampleAgent):
    """Example RAG agent"""

    def __init__(self, name):
        super().__init__(name, agent_type='rag')

    def search(self, query, top_k=5):
        """Perform RAG search with tracking"""
        print(f"[{self.name}] Searching for: {query}")

        # Record search start
        self.dashboard.record_rag_activity(
            'search_started',
            f'Vector search for: {query}',
            {'top_k': top_k}
        )

        # Simulate search
        time.sleep(random.uniform(0.05, 0.15))

        # Record search completion
        self.dashboard.record_rag_activity(
            'documents_retrieved',
            f'Retrieved {top_k} documents',
            {'query': query, 'count': top_k}
        )

        return [f"Document {i}" for i in range(top_k)]


def example_workflow():
    """Example multi-agent workflow with dashboard integration"""

    print("\n" + "="*70)
    print("Enhanced Dashboard Integration Example")
    print("="*70)
    print("\nThis example demonstrates a multi-agent workflow with full")
    print("dashboard integration. Open http://localhost:5000 to see real-time")
    print("visualization of agent execution, costs, and performance.\n")

    # Create agents
    coordinator = ExampleAgent("Coordinator", "agent")
    rag_engine = RAGAgent("RAG Engine")
    analyzer = ExampleAgent("Analyzer", "agent")
    responder = ExampleAgent("Responder", "agent")

    print("Starting workflow...")
    print("-" * 70)

    # Step 1: Coordinator receives query
    query = "What are the best practices for multi-agent systems?"

    # Record decision
    coordinator.make_decision(
        'decision',
        'Route Query',
        f'Routing query to RAG engine: {query}'
    )

    # Step 2: Coordinator sends to RAG
    coordinator.send_message("RAG Engine", query)
    documents = rag_engine.search(query, top_k=5)

    # Step 3: RAG sends context to Analyzer
    context = {"query": query, "documents": documents}
    rag_engine.send_message("Analyzer", str(context))

    # Step 4: Analyzer processes
    analyzer.make_decision(
        'condition',
        'Evaluate Context Quality',
        'Context quality is sufficient for response generation'
    )
    analysis_result = analyzer.execute(f"analyze context for: {query}")

    # Step 5: Analyzer sends to Responder
    analyzer.send_message("Responder", analysis_result)

    # Step 6: Responder generates response
    responder.make_decision(
        'action',
        'Generate Response',
        'Generating user-friendly response based on analysis'
    )
    final_response = responder.execute(f"generate response for: {query}")

    # Step 7: Send response back to user
    responder.send_message("User", final_response)

    print("-" * 70)
    print("Workflow complete!")
    print("\nCheck the dashboard to see:")
    print("  • Agent orchestration graph")
    print("  • Execution timeline")
    print("  • Message flow diagram")
    print("  • Decision tree")
    print("  • Cost breakdown")
    print("  • Performance metrics")


def continuous_monitoring_example():
    """Example of continuous monitoring with multiple workflows"""

    print("\n" + "="*70)
    print("Continuous Monitoring Example")
    print("="*70)
    print("\nRunning continuous workflows to demonstrate dashboard capabilities.")
    print("Press Ctrl+C to stop.\n")

    agents = {
        'coordinator': ExampleAgent("Coordinator"),
        'rag': RAGAgent("RAG Engine"),
        'analyzer': ExampleAgent("Analyzer"),
        'responder': ExampleAgent("Responder")
    }

    queries = [
        "How do I implement RAG?",
        "Best practices for agent orchestration",
        "Cost optimization strategies",
        "Performance monitoring techniques",
        "Multi-agent communication patterns"
    ]

    try:
        iteration = 0
        while True:
            iteration += 1
            query = random.choice(queries)

            print(f"\n[Iteration {iteration}] Processing: {query}")

            # Execute workflow
            agents['coordinator'].make_decision('decision', 'Route Query', f'Processing: {query}')
            agents['coordinator'].send_message('RAG Engine', query)

            documents = agents['rag'].search(query)
            agents['rag'].send_message('Analyzer', str(documents))

            agents['analyzer'].execute(f"analyze: {query}")
            agents['analyzer'].send_message('Responder', "analysis complete")

            agents['responder'].execute(f"respond to: {query}")
            agents['responder'].send_message('User', "response generated")

            # Wait before next iteration
            time.sleep(random.uniform(2, 5))

    except KeyboardInterrupt:
        print("\n\nStopping continuous monitoring...")


if __name__ == '__main__':
    import sys

    print("\nEnhanced Dashboard Integration Examples")
    print("Make sure the dashboard is running: python launch_enhanced_dashboard.py\n")

    if len(sys.argv) > 1 and sys.argv[1] == '--continuous':
        continuous_monitoring_example()
    else:
        example_workflow()

        print("\n" + "="*70)
        print("\nTo run continuous monitoring:")
        print("  python examples/enhanced_dashboard_integration.py --continuous")
        print("\nTo start the dashboard:")
        print("  python launch_enhanced_dashboard.py")
        print("="*70 + "\n")
