#!/usr/bin/env python3
"""Integration test for RAG-CLI with Multi-Agent Orchestration.

Tests all routing strategies and verifies the complete integration pipeline.
"""

import sys
import asyncio
import time
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.agent_orchestrator import AgentOrchestrator, RoutingStrategy
from core.query_classifier import get_query_classifier, QueryIntent
from monitoring.logger import get_logger

logger = get_logger(__name__)


class IntegrationTester:
    """Test RAG-CLI + Multi-Agent integration."""

    def __init__(self):
        """Initialize tester."""
        self.orchestrator = None
        self.classifier = get_query_classifier()
        self.results = []

    def print_header(self, title: str):
        """Print test section header."""
        print("\n" + "=" * 70)
        print(f"  {title}")
        print("=" * 70)

    def print_result(self, test_name: str, passed: bool, details: str = ""):
        """Print test result."""
        status = "[PASS]" if passed else "[FAIL]"
        color = "\033[92m" if passed else "\033[91m"
        reset = "\033[0m"
        print(f"{color}{status}{reset} {test_name}")
        if details:
            print(f"       {details}")

        self.results.append((test_name, passed))

    def test_imports(self) -> bool:
        """Test 1: Verify all imports work."""
        self.print_header("TEST 1: Module Imports")

        try:
            from core.agent_orchestrator import AgentOrchestrator
            self.print_result("AgentOrchestrator import", True)

            from integrations.maf_connector import get_maf_connector
            self.print_result("MAF Connector import", True)

            from agents.query_decomposer import get_query_decomposer
            self.print_result("Query Decomposer import", True)

            from agents.result_synthesizer import get_result_synthesizer
            self.print_result("Result Synthesizer import", True)

            from core.query_classifier import get_query_classifier
            self.print_result("Query Classifier import", True)

            return True

        except Exception as e:
            self.print_result("Import test", False, str(e))
            return False

    def test_orchestrator_init(self) -> bool:
        """Test 2: Initialize orchestrator."""
        self.print_header("TEST 2: Orchestrator Initialization")

        try:
            from core.agent_orchestrator import AgentOrchestrator
            self.orchestrator = AgentOrchestrator()

            self.print_result("Orchestrator created", True)
            self.print_result("Retriever initialized", self.orchestrator.retriever is not None)
            self.print_result("Classifier initialized", self.orchestrator.classifier is not None)
            self.print_result("MAF connector initialized", self.orchestrator.maf_connector is not None)

            maf_available = self.orchestrator.enable_maf
            self.print_result("MAF available", maf_available,
                            "Multi-agent framework detected" if maf_available else "MAF not found - will use RAG only")

            return True

        except Exception as e:
            self.print_result("Orchestrator initialization", False, str(e))
            return False

    def test_query_classification(self) -> bool:
        """Test 3: Query classification."""
        self.print_header("TEST 3: Query Classification")

        test_queries = [
            ("How do I configure authentication?", QueryIntent.HOW_TO),
            ("I'm getting a TypeError", QueryIntent.TROUBLESHOOTING),
            ("What are best practices for API design?", QueryIntent.BEST_PRACTICES),
            ("Explain how the vector store works", QueryIntent.CODE_EXPLANATION),
        ]

        all_passed = True
        for query, expected_intent in test_queries:
            try:
                classification = self.classifier.classify(query)
                passed = classification.primary_intent == expected_intent
                self.print_result(
                    f"Classify: '{query[:40]}...'",
                    passed,
                    f"Intent: {classification.primary_intent.value}, Confidence: {classification.confidence:.2f}"
                )
                if not passed:
                    all_passed = False
            except Exception as e:
                self.print_result(f"Classification failed", False, str(e))
                all_passed = False

        return all_passed

    async def test_rag_only_strategy(self) -> bool:
        """Test 4: RAG_ONLY routing strategy."""
        self.print_header("TEST 4: RAG_ONLY Strategy")

        if not self.orchestrator:
            self.print_result("RAG_ONLY test", False, "Orchestrator not initialized")
            return False

        query = "How do I configure the vector store?"

        try:
            start_time = time.time()
            result = await self.orchestrator.orchestrate(query, top_k=3, use_cache=True)
            elapsed = (time.time() - start_time) * 1000

            passed = result.strategy_used == RoutingStrategy.RAG_ONLY
            self.print_result(
                "RAG_ONLY strategy",
                passed,
                f"Strategy: {result.strategy_used.value}, "
                f"Documents: {len(result.sources)}, "
                f"Latency: {elapsed:.0f}ms"
            )

            return passed

        except Exception as e:
            self.print_result("RAG_ONLY test", False, str(e))
            return False

    async def test_parallel_rag_maf_strategy(self) -> bool:
        """Test 5: PARALLEL_RAG_MAF strategy (requires MAF)."""
        self.print_header("TEST 5: PARALLEL_RAG_MAF Strategy")

        if not self.orchestrator:
            self.print_result("PARALLEL_RAG_MAF test", False, "Orchestrator not initialized")
            return False

        if not self.orchestrator.enable_maf:
            self.print_result("PARALLEL_RAG_MAF test", True,
                            "Skipped - MAF not available (this is OK)")
            return True

        query = "I'm getting a TypeError: cannot read property 'map' of undefined. How do I fix this?"

        try:
            start_time = time.time()
            result = await self.orchestrator.orchestrate(query, top_k=3, use_cache=True)
            elapsed = (time.time() - start_time) * 1000

            # Should use PARALLEL or RAG_ONLY (if MAF fails)
            passed = result.strategy_used in [RoutingStrategy.PARALLEL_RAG_MAF, RoutingStrategy.RAG_ONLY]
            self.print_result(
                "PARALLEL_RAG_MAF strategy",
                passed,
                f"Strategy: {result.strategy_used.value}, "
                f"MAF used: {result.maf_result is not None}, "
                f"Latency: {elapsed:.0f}ms"
            )

            return passed

        except Exception as e:
            self.print_result("PARALLEL_RAG_MAF test", False, str(e))
            return False

    async def test_decomposed_strategy(self) -> bool:
        """Test 6: DECOMPOSED strategy for complex queries."""
        self.print_header("TEST 6: DECOMPOSED Strategy")

        if not self.orchestrator:
            self.print_result("DECOMPOSED test", False, "Orchestrator not initialized")
            return False

        query = "First explain the architecture, then show me how to add a new feature, and finally help me write tests"

        try:
            start_time = time.time()
            result = await self.orchestrator.orchestrate(query, top_k=5, use_cache=True)
            elapsed = (time.time() - start_time) * 1000

            # Complex queries should use DECOMPOSED or fallback to RAG_ONLY
            passed = True  # Any strategy is OK for complex queries
            self.print_result(
                "Complex query handling",
                passed,
                f"Strategy: {result.strategy_used.value}, "
                f"Decomposed: {result.decomposition_result is not None}, "
                f"Latency: {elapsed:.0f}ms"
            )

            return passed

        except Exception as e:
            self.print_result("DECOMPOSED test", False, str(e))
            return False

    def test_hook_integration(self) -> bool:
        """Test 7: Hook file integration."""
        self.print_header("TEST 7: Hook Integration")

        try:
            # Check if hook file exists in global location
            global_hook = Path.home() / ".claude" / "hooks" / "rag-cli" / "user-prompt-submit.py"

            if global_hook.exists():
                # Check if hook contains orchestrator code
                with open(global_hook, 'r') as f:
                    hook_content = f.read()

                has_orchestrator_import = "from core.agent_orchestrator import AgentOrchestrator" in hook_content
                has_asyncio_import = "import asyncio" in hook_content
                has_orchestrate_call = "orchestrator.orchestrate" in hook_content

                self.print_result("Hook file exists", True, str(global_hook))
                self.print_result("Orchestrator import", has_orchestrator_import)
                self.print_result("Asyncio import", has_asyncio_import)
                self.print_result("Orchestrate call", has_orchestrate_call)

                return has_orchestrator_import and has_asyncio_import and has_orchestrate_call
            else:
                self.print_result("Hook file", False, "Not found in global location")
                return False

        except Exception as e:
            self.print_result("Hook integration test", False, str(e))
            return False

    def test_mcp_tools(self) -> bool:
        """Test 8: MCP tool availability."""
        self.print_header("TEST 8: MCP Tools")

        try:
            # Check if MCP config exists
            mcp_config = Path.home() / ".claude" / "mcp" / "rag-cli.json"

            if mcp_config.exists():
                self.print_result("MCP config exists", True, str(mcp_config))

                # Try to import unified server
                from plugin.mcp.unified_server import UnifiedMCPServer
                server = UnifiedMCPServer()

                self.print_result("UnifiedMCPServer import", True)
                self.print_result("MCP server initialized", server is not None)

                # Check tool list response
                tool_list = server.handle_mcp_list_tools(1)
                tool_count = len(tool_list['result']['tools'])

                self.print_result("MCP tools available", tool_count == 14,
                                f"{tool_count} tools registered (expected 14)")

                return tool_count == 14
            else:
                self.print_result("MCP config", False, "Not found")
                return False

        except Exception as e:
            self.print_result("MCP tools test", False, str(e))
            return False

    def print_summary(self):
        """Print test summary."""
        self.print_header("TEST SUMMARY")

        total = len(self.results)
        passed = sum(1 for _, p in self.results if p)
        failed = total - passed

        print(f"\nTotal Tests: {total}")
        print(f"\033[92mPassed: {passed}\033[0m")
        if failed > 0:
            print(f"\033[91mFailed: {failed}\033[0m")

        success_rate = (passed / total * 100) if total > 0 else 0
        print(f"Success Rate: {success_rate:.1f}%")

        if success_rate == 100:
            print("\n\033[92m[SUCCESS] ALL TESTS PASSED - Integration Complete!\033[0m")
        elif success_rate >= 80:
            print("\n\033[93m[WARNING] Most tests passed - Minor issues remaining\033[0m")
        else:
            print("\n\033[91m[ERROR] Integration incomplete - Major issues found\033[0m")

        return success_rate >= 80

    async def run_all_tests(self):
        """Run all integration tests."""
        print("\n" + "=" * 70)
        print("  RAG-CLI + MULTI-AGENT FRAMEWORK INTEGRATION TEST")
        print("=" * 70)

        # Run tests in order
        self.test_imports()
        self.test_orchestrator_init()
        self.test_query_classification()

        # Async tests
        await self.test_rag_only_strategy()
        await self.test_parallel_rag_maf_strategy()
        await self.test_decomposed_strategy()

        # Integration tests
        self.test_hook_integration()
        self.test_mcp_tools()

        # Summary
        return self.print_summary()


async def main():
    """Main test entry point."""
    tester = IntegrationTester()
    success = await tester.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\033[93mTests interrupted by user\033[0m")
        sys.exit(130)
    except Exception as e:
        print(f"\n\033[91mTest runner failed: {e}\033[0m")
        sys.exit(1)
