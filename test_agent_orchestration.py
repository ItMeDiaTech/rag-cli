"""End-to-end orchestration testing for RAG-CLI agent system.

This script tests the complete multi-agent orchestration system including:
- Query decomposition (Phase 3)
- Result synthesis (Phase 3)
- Agent coordination (Phase 5)
- Message passing (Phase 5)
- Agent monitoring (Phase 5)

USAGE:
    python test_agent_orchestration.py

REQUIREMENTS:
    - Indexed documents in vector store
    - All agent components installed
"""

import sys
import asyncio
import time
from pathlib import Path
from typing import List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from agents import (
    get_query_decomposer,
    get_result_synthesizer,
    get_agent_coordinator,
    BaseAgent,
    AgentMessage,
    MessageType
)
from monitoring.agent_monitor import get_agent_monitor
from core.retrieval_pipeline import get_retriever


class OrchestrationTest:
    """Comprehensive orchestration test suite."""

    def __init__(self):
        """Initialize test suite."""
        self.decomposer = get_query_decomposer()
        self.synthesizer = get_result_synthesizer()
        self.coordinator = get_agent_coordinator()
        self.monitor = get_agent_monitor()
        self.retriever = get_retriever()

        # Clear any previous state
        self.monitor.clear()

        print("Orchestration test suite initialized")

    async def test_query_decomposition(self) -> bool:
        """Test Phase 3: Query decomposition.

        Returns:
            True if test passes
        """
        print("\n" + "=" * 70)
        print("TEST 1: Query Decomposition")
        print("=" * 70)

        # Test complex query
        complex_query = "How to build a production-ready FastAPI microservices architecture that handles real-time WebSocket connections for live updates, integrates with RabbitMQ for async task processing and event-driven communication, implements JWT-based authentication with role-based access control and refresh tokens, uses PostgreSQL with async SQLAlchemy for data persistence including migrations and connection pooling, and deploys to Kubernetes with proper health checks, horizontal pod autoscaling, and monitoring using Prometheus?"

        print(f"\nComplex query: {complex_query}")
        print("\nDecomposing query...")

        trace_id = "decomp_001"
        self.monitor.start_agent_execution(
            trace_id=trace_id,
            agent_id=self.decomposer.__class__.__name__,
            agent_type="QueryDecomposer"
        )

        start_time = time.time()
        result = await self.decomposer.decompose(complex_query)
        duration = time.time() - start_time

        self.monitor.complete_agent_execution(
            trace_id=trace_id,
            success=result.is_complex,
            input_size=len(complex_query),
            output_size=len(result.sub_queries)
        )

        print(f"\nDecomposition result:")
        print(f"  Complex: {result.is_complex}")
        print(f"  Strategy: {result.strategy_used.value}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Sub-queries: {len(result.sub_queries)}")
        print(f"  Duration: {duration:.3f}s")

        if result.is_complex:
            print(f"\n  Sub-queries:")
            for sq in result.sub_queries:
                print(f"    [{sq.index + 1}] {sq.text}")

        # Validation
        passed = result.sub_queries is not None and len(result.sub_queries) > 0

        status = "PASS" if passed else "FAIL"
        print(f"\nTest Status: {status}")

        return passed

    async def test_result_synthesis(self) -> bool:
        """Test Phase 3: Result synthesis.

        Returns:
            True if test passes
        """
        print("\n" + "=" * 70)
        print("TEST 2: Result Synthesis")
        print("=" * 70)

        # Create mock sub-queries and results
        from agents.query_decomposer import SubQuery
        from core.retrieval_pipeline import RetrievalResult

        sub_queries = [
            SubQuery(text="How to implement FastAPI?", index=0, original_context="...",
                    dependencies=[], priority=0, metadata={}),
            SubQuery(text="FastAPI async database?", index=1, original_context="...",
                    dependencies=[], priority=1, metadata={}),
            SubQuery(text="FastAPI CORS?", index=2, original_context="...",
                    dependencies=[], priority=2, metadata={})
        ]

        # Mock retrieval results (would normally come from retriever)
        mock_results = [
            [  # Results for sub-query 0
                RetrievalResult("id1", "FastAPI is a modern web framework...", 0.9, "doc1.md", {}, "hybrid", 1),
                RetrievalResult("id2", "To implement FastAPI, install it first...", 0.85, "doc2.md", {}, "vector", 2),
            ],
            [  # Results for sub-query 1
                RetrievalResult("id3", "Async database connections in FastAPI use SQLAlchemy...", 0.88, "doc3.md", {}, "hybrid", 1),
                RetrievalResult("id1", "FastAPI is a modern web framework...", 0.87, "doc1.md", {}, "vector", 2),  # Duplicate
            ],
            [  # Results for sub-query 2
                RetrievalResult("id4", "CORS middleware in FastAPI is configured...", 0.91, "doc4.md", {}, "hybrid", 1),
            ]
        ]

        print(f"\nSynthesizing results from {len(sub_queries)} sub-queries...")
        print(f"Total input results: {sum(len(results) for results in mock_results)}")

        trace_id = "synth_001"
        self.monitor.start_agent_execution(
            trace_id=trace_id,
            agent_id=self.synthesizer.__class__.__name__,
            agent_type="ResultSynthesizer"
        )

        start_time = time.time()
        synthesis = await self.synthesizer.synthesize(sub_queries, mock_results, top_k=5)
        duration = time.time() - start_time

        self.monitor.complete_agent_execution(
            trace_id=trace_id,
            success=True,
            input_size=sum(len(results) for results in mock_results),
            output_size=len(synthesis.merged_results)
        )

        print(f"\nSynthesis result:")
        print(f"  Total input results: {synthesis.total_input_results}")
        print(f"  Duplicates removed: {synthesis.duplicates_removed}")
        print(f"  Unique results: {len(synthesis.merged_results)}")
        print(f"  Confidence: {synthesis.confidence:.2%}")
        print(f"  Duration: {duration:.3f}s")

        print(f"\n  Top {min(3, len(synthesis.merged_results))} results:")
        for i, result in enumerate(synthesis.merged_results[:3], 1):
            print(f"    [{i}] Score: {result.score:.2f} | {result.source}")

        # Validation
        passed = (
            len(synthesis.merged_results) > 0 and
            synthesis.duplicates_removed >= 1 and  # Should have removed at least 1 duplicate
            synthesis.confidence > 0
        )

        status = "PASS" if passed else "FAIL"
        print(f"\nTest Status: {status}")

        return passed

    async def test_agent_coordination(self) -> bool:
        """Test Phase 5: Agent coordination and message passing.

        Returns:
            True if test passes
        """
        print("\n" + "=" * 70)
        print("TEST 3: Agent Coordination & Message Passing")
        print("=" * 70)

        # Create a simple test agent
        class TestAgent(BaseAgent):
            """Test agent for coordination testing."""

            async def process(self, message: AgentMessage) -> AgentMessage:
                """Process message and return response."""
                await asyncio.sleep(0.1)  # Simulate processing

                response_payload = {
                    'result': f"Processed: {message.payload.get('input', 'N/A')}",
                    'agent': self.agent_id
                }

                return AgentMessage.create_response(
                    sender_id=self.agent_id,
                    receiver_id=message.sender_id,
                    payload=response_payload,
                    parent_message=message
                )

        # Register test agents
        agent1 = TestAgent(agent_id="test_agent_1", agent_type="TestAgent")
        agent2 = TestAgent(agent_id="test_agent_2", agent_type="TestAgent")

        self.coordinator.register_agent(agent1)
        self.coordinator.register_agent(agent2)

        print(f"\nRegistered agents: {list(self.coordinator.agents.keys())}")

        # Test 1: Single agent execution
        print("\n[1/2] Testing single agent execution...")
        trace_id_1 = "coord_001"
        self.monitor.start_agent_execution(trace_id_1, "test_agent_1", "TestAgent")

        response1 = await self.coordinator.execute_agent(
            "test_agent_1",
            payload={'input': 'Hello Agent 1'},
            timeout=5.0
        )

        self.monitor.complete_agent_execution(trace_id_1, success=True)

        print(f"  Response: {response1.payload.get('result')}")

        # Test 2: Parallel agent execution
        print("\n[2/2] Testing parallel agent execution...")
        trace_id_2 = "coord_002"
        trace_id_3 = "coord_003"

        self.monitor.start_agent_execution(trace_id_2, "test_agent_1", "TestAgent")
        self.monitor.start_agent_execution(trace_id_3, "test_agent_2", "TestAgent")

        responses = await self.coordinator.execute_parallel([
            ("test_agent_1", {'input': 'Parallel task 1'}),
            ("test_agent_2", {'input': 'Parallel task 2'})
        ], timeout=5.0)

        self.monitor.complete_agent_execution(trace_id_2, success=True)
        self.monitor.complete_agent_execution(trace_id_3, success=True)

        print(f"  Responses received: {len(responses)}")
        for i, resp in enumerate(responses, 1):
            if isinstance(resp, AgentMessage):
                print(f"    [{i}] {resp.payload.get('result')}")

        # Validation
        passed = (
            response1 is not None and
            response1.message_type == MessageType.RESPONSE and
            len(responses) == 2 and
            all(isinstance(r, AgentMessage) for r in responses)
        )

        # Cleanup
        self.coordinator.unregister_agent("test_agent_1")
        self.coordinator.unregister_agent("test_agent_2")

        status = "PASS" if passed else "FAIL"
        print(f"\nTest Status: {status}")

        return passed

    def test_agent_monitoring(self) -> bool:
        """Test Phase 5: Agent monitoring and metrics.

        Returns:
            True if test passes
        """
        print("\n" + "=" * 70)
        print("TEST 4: Agent Monitoring & Metrics")
        print("=" * 70)

        # Generate monitoring report
        report = self.monitor.generate_report()

        print(f"\nMonitoring Report:")
        print(f"  Uptime: {report['uptime_seconds']:.2f}s")
        print(f"  Total Agents Tracked: {report['total_agents']}")
        print(f"  Total Executions: {report['total_executions']}")
        print(f"  Success Rate: {report['success_rate_percent']:.1f}%")
        print(f"  Total Messages: {report['total_messages']}")

        print(f"\n  Agent Statistics:")
        for agent_id, stats in report['agent_statistics'].items():
            print(f"    {agent_id}:")
            print(f"      Executions: {stats['executions']}")
            print(f"      Success Rate: {stats['success_rate']:.1f}%")
            if stats['avg_duration'] > 0:
                print(f"      Avg Duration: {stats['avg_duration']:.3f}s")

        # Validation
        passed = (
            report['total_executions'] >= 3 and  # At least from previous tests
            report['success_rate_percent'] > 0
        )

        status = "PASS" if passed else "FAIL"
        print(f"\nTest Status: {status}")

        return passed

    async def test_full_pipeline(self) -> bool:
        """Test full end-to-end orchestration pipeline.

        Returns:
            True if test passes
        """
        print("\n" + "=" * 70)
        print("TEST 5: Full End-to-End Pipeline")
        print("=" * 70)

        query = "How to implement a secure multi-tenant SaaS application using FastAPI that includes OAuth2 and JWT authentication with refresh token rotation, role-based and attribute-based access control across tenant boundaries, rate limiting and API throttling per tenant, async database operations with PostgreSQL using connection pooling and read replicas, Redis-based caching and session management, background task processing with Celery and RabbitMQ, comprehensive logging and monitoring with OpenTelemetry, and automated CI/CD deployment to AWS ECS with auto-scaling?"

        print(f"\nQuery: {query}")
        print("\nExecuting full orchestration pipeline...")

        start_time = time.time()

        # Step 1: Decompose query
        print("\n[1/3] Decomposing query...")
        decomp_result = await self.decomposer.decompose(query)
        print(f"  Sub-queries: {len(decomp_result.sub_queries)}")

        # Step 2: Execute sub-queries in parallel (simulated with mock results)
        print("\n[2/3] Executing sub-queries in parallel...")

        # For demonstration, create mock results
        # In production, this would call retriever for each sub-query
        from core.retrieval_pipeline import RetrievalResult
        mock_sub_results = [
            [RetrievalResult(f"id_{i}_{j}", f"Result {j} for sub-query {i}", 0.8, f"doc{j}.md", {}, "hybrid", j)
             for j in range(3)]
            for i in range(len(decomp_result.sub_queries))
        ]

        # Step 3: Synthesize results
        print("\n[3/3] Synthesizing results...")
        synthesis = await self.synthesizer.synthesize(
            decomp_result.sub_queries,
            mock_sub_results,
            top_k=10
        )

        duration = time.time() - start_time

        print(f"\n=== Pipeline Results ===")
        print(f"Total execution time: {duration:.3f}s")
        print(f"Sub-queries executed: {len(decomp_result.sub_queries)}")
        print(f"Total results retrieved: {synthesis.total_input_results}")
        print(f"Final results: {len(synthesis.merged_results)}")
        print(f"Confidence: {synthesis.confidence:.2%}")

        # Validation
        passed = (
            len(decomp_result.sub_queries) >= 1 and  # At least one sub-query
            len(synthesis.merged_results) > 0 and
            synthesis.confidence > 0 and
            duration < 15.0  # Allow more time for complex query processing
        )

        status = "PASS" if passed else "FAIL"
        print(f"\nTest Status: {status}")

        return passed


async def main():
    """Run all orchestration tests."""
    print("=" * 70)
    print("RAG-CLI AGENT ORCHESTRATION TEST SUITE")
    print("=" * 70)
    print("\nTesting multi-agent orchestration system:")
    print("- Query decomposition (Phase 3)")
    print("- Result synthesis (Phase 3)")
    print("- Agent coordination (Phase 5)")
    print("- Message passing (Phase 5)")
    print("- Agent monitoring (Phase 5)")
    print("=" * 70)

    # Initialize test suite
    test_suite = OrchestrationTest()

    # Run all tests
    results = {}

    try:
        results['query_decomposition'] = await test_suite.test_query_decomposition()
        results['result_synthesis'] = await test_suite.test_result_synthesis()
        results['agent_coordination'] = await test_suite.test_agent_coordination()
        results['agent_monitoring'] = test_suite.test_agent_monitoring()
        results['full_pipeline'] = await test_suite.test_full_pipeline()

    except Exception as e:
        print(f"\nERROR during testing: {e}")
        import traceback
        traceback.print_exc()
        results['error'] = str(e)

    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    for test_name, passed in results.items():
        if test_name != 'error':
            status = "PASS" if passed else "FAIL"
            print(f"{test_name}: {status}")

    if 'error' in results:
        print(f"\nERROR: {results['error']}")

    all_passed = all(v for k, v in results.items() if k != 'error')

    print("\n" + "=" * 70)
    if all_passed:
        print("ALL TESTS PASSED")
        print("Agent orchestration system is fully operational!")
    else:
        print("SOME TESTS FAILED")
        print("Review test output for details.")
    print("=" * 70)

    return all_passed


if __name__ == "__main__":
    asyncio.run(main())
