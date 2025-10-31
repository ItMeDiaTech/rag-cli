"""Integration tests for embedded Multi-Agent Framework.

Verifies that MAF v1.2.0 embedded at src/agents/maf/ works correctly
without requiring an external multi-agent-framework directory.

These tests ensure marketplace installations work correctly.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestEmbeddedMAFImports:
    """Test that embedded MAF can be imported from src/agents/maf/."""

    def test_import_debugger_agent(self):
        """Verify DebuggerAgent can be imported from embedded location."""
        from agents.maf.agents.debugger import DebuggerAgent
        assert DebuggerAgent is not None

    def test_import_developer_agent(self):
        """Verify DeveloperAgent can be imported from embedded location."""
        from agents.maf.agents.developer import DeveloperAgent
        assert DeveloperAgent is not None

    def test_import_reviewer_agent(self):
        """Verify ReviewerAgent can be imported from embedded location."""
        from agents.maf.agents.reviewer import ReviewerAgent
        assert ReviewerAgent is not None

    def test_import_tester_agent(self):
        """Verify TesterAgent can be imported from embedded location."""
        from agents.maf.agents.tester import TesterAgent
        assert TesterAgent is not None

    def test_import_architect_agent(self):
        """Verify ArchitectAgent can be imported from embedded location."""
        from agents.maf.agents.architect import ArchitectAgent
        assert ArchitectAgent is not None

    def test_import_documenter_agent(self):
        """Verify DocumenterAgent can be imported from embedded location."""
        from agents.maf.agents.documenter import DocumenterAgent
        assert DocumenterAgent is not None

    def test_import_optimizer_agent(self):
        """Verify OptimizerAgent can be imported from embedded location."""
        from agents.maf.agents.optimizer import OptimizerAgent
        assert OptimizerAgent is not None

    def test_import_orchestrator(self):
        """Verify Orchestrator can be imported from embedded location."""
        from agents.maf.core.orchestrator import Orchestrator
        assert Orchestrator is not None

    def test_import_task_classifier(self):
        """Verify task classifier can be imported from embedded location."""
        from agents.maf.core.task_classifier import IntelligentTaskClassifier
        assert IntelligentTaskClassifier is not None


class TestMAFConnector:
    """Test that maf_connector uses embedded MAF correctly."""

    def test_maf_connector_initialization(self):
        """Verify maf_connector initializes with embedded MAF."""
        from integrations.maf_connector import get_maf_connector

        connector = get_maf_connector()
        assert connector is not None
        assert hasattr(connector, 'maf_available')

    def test_maf_connector_is_available(self):
        """Verify maf_connector reports availability."""
        from integrations.maf_connector import get_maf_connector

        connector = get_maf_connector()
        # Should be available since MAF is embedded
        assert connector.is_available() is True

    def test_maf_connector_available_agents(self):
        """Verify all 7 agents are available."""
        from integrations.maf_connector import get_maf_connector

        connector = get_maf_connector()
        agents = connector.get_available_agents()

        expected_agents = [
            'debugger', 'architect', 'developer',
            'reviewer', 'tester', 'documenter', 'optimizer'
        ]

        assert len(agents) == 7
        for agent in expected_agents:
            assert agent in agents

    @pytest.mark.asyncio
    async def test_maf_connector_health_check(self):
        """Verify health check returns embedded MAF status."""
        from integrations.maf_connector import get_maf_connector

        connector = get_maf_connector()
        health = await connector.health_check()

        assert health is not None
        assert health['maf_available'] is True
        assert health['maf_type'] == 'embedded'
        assert health['maf_location'] == 'src/agents/maf/'
        assert 'available_agents' in health
        assert len(health['available_agents']) == 7


class TestMAFConnectorNoExternalDependency:
    """Test that MAF works without external multi-agent-framework directory."""

    def test_no_external_path_reference(self):
        """Verify maf_connector doesn't reference external paths."""
        from integrations.maf_connector import MAFConnector
        import inspect

        # Get source code
        source = inspect.getsource(MAFConnector)

        # Should NOT contain references to parent directories or external MAF
        assert 'parent.parent' not in source
        assert '../multi-agent-framework' not in source

        # SHOULD contain embedded path references
        assert 'src.agents.maf' in source

    @pytest.mark.asyncio
    async def test_execute_agent_without_external_maf(self):
        """Verify agent execution works with only embedded MAF."""
        from integrations.maf_connector import get_maf_connector

        connector = get_maf_connector()

        # Mock the agent execution to avoid actual Claude API calls
        with patch.object(connector, '_execute_agent_task', new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = "Mock response from embedded agent"

            task_data = {"task": "test task", "workflow": "code_generation"}
            result = await connector.execute_agent('developer', task_data, timeout=5.0)

            # Should successfully create result even without external MAF
            assert result is not None
            # The mock should have been called
            assert mock_exec.called


class TestUnifiedServerMAFHandlers:
    """Test that unified_server MCP tools use embedded MAF."""

    @pytest.mark.asyncio
    async def test_handle_maf_status_uses_embedded(self):
        """Verify handle_maf_status uses embedded MAF via maf_connector."""
        from plugin.mcp.unified_server import UnifiedMCPServer

        server = UnifiedMCPServer()

        # Mock maf_connector to verify it's called
        with patch('src.plugin.mcp.unified_server.get_maf_connector') as mock_get_connector:
            mock_connector = AsyncMock()
            mock_connector.health_check.return_value = {
                'maf_available': True,
                'maf_type': 'embedded',
                'maf_location': 'src/agents/maf/',
                'available_agents': ['debugger', 'developer']
            }
            mock_get_connector.return_value = mock_connector

            response = await server.handle_maf_status(1, {})

            # Verify maf_connector was used
            assert mock_get_connector.called
            assert mock_connector.health_check.called

            # Verify response structure
            assert response['jsonrpc'] == '2.0'
            assert response['id'] == 1
            assert 'result' in response

    @pytest.mark.asyncio
    async def test_handle_maf_execute_uses_embedded(self):
        """Verify handle_maf_execute uses embedded MAF via maf_connector."""
        from plugin.mcp.unified_server import UnifiedMCPServer
        from integrations.maf_connector import MAFResult
        from datetime import datetime

        server = UnifiedMCPServer()

        # Mock successful execution
        mock_result = MAFResult(
            status='completed',
            content='Mock execution result',
            confidence=0.9,
            agent_name='developer',
            execution_time=1.5,
            metadata={},
            timestamp=datetime.now()
        )

        with patch('src.plugin.mcp.unified_server.get_maf_connector') as mock_get_connector:
            mock_connector = AsyncMock()
            mock_connector.is_available.return_value = True
            mock_connector.execute_agent.return_value = mock_result
            mock_get_connector.return_value = mock_connector

            arguments = {
                "task": "test task",
                "workflow": "code_generation"
            }
            response = await server.handle_maf_execute(1, arguments)

            # Verify maf_connector was used
            assert mock_get_connector.called
            assert mock_connector.is_available.called
            assert mock_connector.execute_agent.called

            # Verify response structure
            assert response['jsonrpc'] == '2.0'
            assert response['id'] == 1
            assert 'result' in response

    @pytest.mark.asyncio
    async def test_handle_maf_classify_no_external_dependency(self):
        """Verify handle_maf_classify works without external MAF."""
        from plugin.mcp.unified_server import UnifiedMCPServer

        server = UnifiedMCPServer()

        # Mock the query classifier to avoid dependencies
        with patch('src.plugin.mcp.unified_server.get_query_classifier') as mock_get_classifier:
            mock_classifier = AsyncMock()
            mock_classifier.classify.return_value = {
                'intent': 'CODE_GENERATION',
                'agents': ['developer'],
                'confidence': 0.85,
                'use_rag': False
            }
            mock_get_classifier.return_value = mock_classifier

            arguments = {"query": "test query"}
            response = await server.handle_maf_classify(1, arguments)

            # Verify response structure
            assert response['jsonrpc'] == '2.0'
            assert response['id'] == 1
            assert 'result' in response


class TestMarketplaceInstallationScenario:
    """Test scenarios specific to marketplace installation."""

    def test_embedded_maf_path_exists(self):
        """Verify embedded MAF directory structure exists."""
        maf_root = project_root / "src" / "agents" / "maf"

        assert maf_root.exists(), "Embedded MAF directory must exist"
        assert (maf_root / "agents").exists(), "MAF agents directory must exist"
        assert (maf_root / "core").exists(), "MAF core directory must exist"

        # Check all 7 agent files exist
        agent_names = [
            'debugger', 'developer', 'reviewer',
            'tester', 'architect', 'documenter', 'optimizer'
        ]

        for agent in agent_names:
            agent_file = maf_root / "agents" / f"{agent}.py"
            assert agent_file.exists(), f"Agent file {agent}.py must exist"

    def test_no_external_maf_required(self):
        """Verify system works without external multi-agent-framework directory."""
        # This test verifies that no code attempts to access parent.parent/multi-agent-framework

        from integrations.maf_connector import get_maf_connector

        # Should work without external MAF
        connector = get_maf_connector()
        assert connector.is_available() is True

        # Verify embedded location is used
        health = connector.maf_available
        assert health is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
