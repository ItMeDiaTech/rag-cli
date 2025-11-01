"""Test data generator for RAG-CLI monitoring dashboard.

This module provides utilities to populate the metrics collector with sample data
for testing and development purposes.
"""

import random
import time
from datetime import datetime, timedelta
import requests
from typing import List, Dict, Any

from rag_cli_plugin.services.logger import get_logger

logger = get_logger(__name__)


class TestDataGenerator:
    """Generates realistic test data for the monitoring dashboard."""

    def __init__(self, server_url: str = "http://localhost:9999"):
        """Initialize test data generator.

        Args:
            server_url: URL of the TCP server to send events to
        """
        self.server_url = server_url
        self.sample_queries = [
            "How do I configure the RAG system?",
            "What is the best chunk size for documents?",
            "How to optimize vector search performance?",
            "Explain semantic caching in RAG",
            "What embedding models are supported?",
            "How to implement hybrid search?",
            "Best practices for document preprocessing",
            "How to tune reranking parameters?",
            "What is the difference between FAISS and Pinecone?",
            "How to handle multi-language documents?",
        ]

        self.sample_components = [
            "vector_store",
            "embedding_generator",
            "document_processor",
            "retrieval_pipeline",
            "claude_client",
        ]

    def generate_sample_queries(self, count: int = 20) -> List[Dict[str, Any]]:
        """Generate sample query events.

        Args:
            count: Number of queries to generate

        Returns:
            List of query event dictionaries
        """
        events = []
        base_time = datetime.now() - timedelta(minutes=count * 2)

        for i in range(count):
            query_time = base_time + timedelta(minutes=i * 2)

            # Simulate realistic latencies
            vector_latency = random.uniform(10, 100)
            keyword_latency = random.uniform(5, 30)
            reranking_latency = random.uniform(20, 150)
            claude_latency = random.uniform(500, 2000)

            event = {
                "event_type": "query",
                "timestamp": query_time.isoformat(),
                "data": {
                    "query": random.choice(self.sample_queries),
                    "latency": {
                        "vector_search": vector_latency,
                        "keyword_search": keyword_latency,
                        "reranking": reranking_latency,
                        "claude_api": claude_latency,
                        "total": vector_latency + keyword_latency + reranking_latency + claude_latency,
                    },
                    "results_count": random.randint(3, 10),
                    "cache_hit": random.random() < 0.3,  # 30% cache hit rate
                },
            }
            events.append(event)

        return events

    def generate_sample_logs(self, count: int = 30) -> List[Dict[str, Any]]:
        """Generate sample log entries.

        Args:
            count: Number of log entries to generate

        Returns:
            List of log event dictionaries
        """
        events = []
        base_time = datetime.now() - timedelta(minutes=count)

        log_messages = [
            ("INFO", "Vector store initialized successfully"),
            ("INFO", "Embedding model loaded"),
            ("INFO", "Document processor ready"),
            ("DEBUG", "Processing query"),
            ("DEBUG", "Retrieved 5 documents from vector store"),
            ("INFO", "Query completed successfully"),
            ("WARNING", "High memory usage detected"),
            ("DEBUG", "Cache hit for query"),
            ("INFO", "Reranking results"),
            ("ERROR", "Failed to connect to Claude API"),
        ]

        for i in range(count):
            log_time = base_time + timedelta(minutes=i)
            level, message = random.choice(log_messages)

            event = {
                "event_type": "log",
                "timestamp": log_time.isoformat(),
                "data": {
                    "level": level,
                    "message": message,
                    "component": random.choice(self.sample_components),
                },
            }
            events.append(event)

        return events

    def generate_component_status(self) -> List[Dict[str, Any]]:
        """Generate component status updates.

        Returns:
            List of component status event dictionaries
        """
        events = []
        statuses = ["operational", "operational", "operational", "ready", "degraded"]

        for component in self.sample_components:
            event = {
                "event_type": "component_status",
                "timestamp": datetime.now().isoformat(),
                "data": {
                    "component": component,
                    "status": random.choice(statuses),
                },
            }
            events.append(event)

        return events

    def send_event(self, event: Dict[str, Any]) -> bool:
        """Send a single event to the monitoring server.

        Args:
            event: Event dictionary to send

        Returns:
            True if successful, False otherwise
        """
        try:
            response = requests.post(
                f"{self.server_url}/api/events/submit",
                json=event,
                timeout=5,
            )
            return response.status_code == 200
        except requests.RequestException as e:
            logger.error(f"Failed to send event: {e}")
            return False

    def populate_dashboard(self, queries: int = 20, logs: int = 30, delay: float = 0.1):
        """Populate the dashboard with test data.

        Args:
            queries: Number of sample queries to generate
            logs: Number of sample logs to generate
            delay: Delay between sending events (seconds)
        """
        logger.info("Starting test data generation...")

        # Generate all events
        all_events = []
        all_events.extend(self.generate_sample_queries(queries))
        all_events.extend(self.generate_sample_logs(logs))
        all_events.extend(self.generate_component_status())

        # Sort by timestamp
        all_events.sort(key=lambda x: x["timestamp"])

        # Send events
        success_count = 0
        for event in all_events:
            if self.send_event(event):
                success_count += 1
            time.sleep(delay)

        logger.info(
            f"Test data generation complete: {success_count}/{len(all_events)} events sent successfully"
        )

        return success_count, len(all_events)


def populate_test_data():
    """Main function to populate the dashboard with test data."""
    generator = TestDataGenerator()

    print("=" * 60)
    print("RAG-CLI Dashboard Test Data Generator")
    print("=" * 60)
    print()
    print("This will populate the monitoring dashboard with sample data.")
    print("Make sure the monitoring server is running on localhost:9999")
    print()

    input("Press Enter to continue...")

    print("\nGenerating test data...")
    success, total = generator.populate_dashboard(queries=20, logs=30, delay=0.05)

    print(f"\nComplete! Sent {success}/{total} events successfully.")
    print()
    print("Open the dashboard at http://localhost:5000 to see the results.")
    print()


if __name__ == "__main__":
    populate_test_data()
