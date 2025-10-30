"""MAF (Multi-Agent Framework) Connector for RAG-CLI.

This module provides integration with the Multi-Agent Framework in the parent DocHub directory.
Enables routing queries to specialized MAF agents for enhanced processing:
- Debugger: Error analysis and troubleshooting
- Architect: Query planning and decomposition
- Developer: Code generation and implementation
- Reviewer: Result validation and quality checks

USAGE:
    connector = get_maf_connector()
    result = await connector.execute_agent('debugger', {
        'error_message': 'ValueError: invalid query',
        'context': 'User query processing'
    })
"""

import sys
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime

# Add MAF to path (parent directory: DocHub/multi-agent-framework)
maf_path = Path(__file__).parent.parent.parent.parent.parent / "multi-agent-framework"
if maf_path.exists():
    sys.path.insert(0, str(maf_path))

from src.monitoring.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MAFResult:
    """Result from MAF agent execution."""
    status: str  # 'completed', 'partial', 'error'
    content: str
    confidence: float
    agent_name: str
    execution_time: float
    metadata: Dict[str, Any]
    timestamp: datetime


class MAFConnector:
    """Connector to Multi-Agent Framework."""

    def __init__(self):
        """Initialize MAF connector."""
        self.maf_available = False
        self.maf_runner = None

        # Try to import MAF components
        try:
            from maf_simple import ImprovedMAFRunner
            self.MAFRunner = ImprovedMAFRunner
            self.maf_available = True
            logger.info("MAF connector initialized successfully")
        except ImportError as e:
            logger.warning(f"MAF not available: {e}")
            self.maf_available = False

    async def execute_agent(
        self,
        agent_name: str,
        task_data: Dict[str, Any],
        timeout: float = 30.0
    ) -> Optional[MAFResult]:
        """Execute a specific MAF agent with given task data.

        Args:
            agent_name: Name of agent to execute ('debugger', 'architect', etc.)
            task_data: Task data for the agent
            timeout: Execution timeout in seconds

        Returns:
            MAFResult if successful, None if failed or unavailable
        """
        if not self.maf_available:
            logger.warning("MAF not available, skipping agent execution")
            return None

        try:
            logger.info(f"Executing MAF agent: {agent_name}")
            start_time = asyncio.get_event_loop().time()

            # Create task description for MAF
            task_description = self._format_task_for_agent(agent_name, task_data)

            # Execute with timeout
            runner = self.MAFRunner()
            result_code = await asyncio.wait_for(
                runner.execute_task(task_description),
                timeout=timeout
            )

            execution_time = asyncio.get_event_loop().time() - start_time

            # Parse result
            status = 'completed' if result_code == 0 else 'error'

            result = MAFResult(
                status=status,
                content=task_description,  # Would be actual result from MAF
                confidence=0.8 if status == 'completed' else 0.3,
                agent_name=agent_name,
                execution_time=execution_time,
                metadata={
                    'return_code': result_code,
                    'task_data': task_data
                },
                timestamp=datetime.now()
            )

            logger.info(
                f"MAF agent execution complete",
                agent=agent_name,
                status=status,
                elapsed=execution_time
            )

            return result

        except asyncio.TimeoutError:
            logger.warning(f"MAF agent '{agent_name}' timed out after {timeout}s")
            return MAFResult(
                status='error',
                content='',
                confidence=0.0,
                agent_name=agent_name,
                execution_time=timeout,
                metadata={'error': 'timeout'},
                timestamp=datetime.now()
            )
        except Exception as e:
            logger.error(f"MAF agent execution failed", agent=agent_name, error=str(e))
            return None

    async def classify_task(self, query: str) -> Optional[Dict[str, Any]]:
        """Classify a query using MAF's task classifier.

        Args:
            query: Query to classify

        Returns:
            Classification result with task type, confidence, and agent sequence
        """
        if not self.maf_available:
            return None

        try:
            from core.task_classifier import IntelligentTaskClassifier

            classifier = IntelligentTaskClassifier()
            classification = classifier.classify_task(query)

            return {
                'task_type': classification.task_type,
                'confidence': classification.confidence,
                'primary_workflow': classification.primary_workflow,
                'agent_sequence': classification.agent_sequence,
                'requirements': classification.suggested_requirements
            }

        except Exception as e:
            logger.error(f"MAF task classification failed: {e}")
            return None

    async def execute_debugger(
        self,
        error_message: str,
        context: str,
        stack_trace: Optional[str] = None
    ) -> Optional[MAFResult]:
        """Execute MAF Debugger agent for error analysis.

        Args:
            error_message: Error message to analyze
            context: Context where error occurred
            stack_trace: Optional stack trace

        Returns:
            MAFResult with debugging analysis
        """
        task_data = {
            'error_message': error_message,
            'context': context,
            'stack_trace': stack_trace or 'No stack trace available'
        }

        return await self.execute_agent('debugger', task_data, timeout=30.0)

    async def execute_architect(
        self,
        query: str,
        complexity: str = 'medium'
    ) -> Optional[MAFResult]:
        """Execute MAF Architect agent for query planning.

        Args:
            query: Complex query to plan
            complexity: Complexity level ('simple', 'medium', 'complex')

        Returns:
            MAFResult with query decomposition plan
        """
        task_data = {
            'query': query,
            'complexity': complexity,
            'objective': 'Decompose query into sub-tasks for parallel execution'
        }

        return await self.execute_agent('architect', task_data, timeout=20.0)

    def _format_task_for_agent(self, agent_name: str, task_data: Dict[str, Any]) -> str:
        """Format task data into natural language description for MAF.

        Args:
            agent_name: Target agent name
            task_data: Task data dictionary

        Returns:
            Formatted task description
        """
        # Convert task_data to natural language based on agent type
        if agent_name == 'debugger':
            return (
                f"Debug the following error: {task_data.get('error_message', 'Unknown error')}. "
                f"Context: {task_data.get('context', 'Unknown context')}. "
                f"Stack trace: {task_data.get('stack_trace', 'Not available')}"
            )
        elif agent_name == 'architect':
            return (
                f"Plan and decompose this complex query: {task_data.get('query', '')}. "
                f"Complexity level: {task_data.get('complexity', 'medium')}. "
                f"Break it into executable sub-tasks for parallel processing."
            )
        elif agent_name == 'developer':
            return (
                f"Implement the following requirement: {task_data.get('requirement', '')}. "
                f"Specifications: {task_data.get('specifications', '')}"
            )
        else:
            # Generic format
            description = task_data.get('description', '')
            if not description:
                # Build description from all task_data fields
                parts = [f"{k}: {v}" for k, v in task_data.items()]
                description = "; ".join(parts)
            return description

    def is_available(self) -> bool:
        """Check if MAF is available.

        Returns:
            True if MAF can be used
        """
        return self.maf_available

    def get_available_agents(self) -> List[str]:
        """Get list of available MAF agents.

        Returns:
            List of agent names
        """
        if not self.maf_available:
            return []

        # Standard MAF agents
        return [
            'debugger',
            'architect',
            'developer',
            'reviewer',
            'tester',
            'documenter',
            'optimizer'
        ]

    async def health_check(self) -> Dict[str, Any]:
        """Check MAF connector health.

        Returns:
            Health status dictionary
        """
        health = {
            'status': 'healthy' if self.maf_available else 'unavailable',
            'maf_available': self.maf_available,
            'maf_path': str(maf_path) if maf_path.exists() else 'not found',
            'available_agents': self.get_available_agents()
        }

        # Try to import and get version info
        if self.maf_available:
            try:
                import core
                health['maf_version'] = getattr(core, '__version__', 'unknown')
            except:
                health['maf_version'] = 'unknown'

        return health


# Singleton instance
_maf_connector: Optional[MAFConnector] = None


def get_maf_connector() -> MAFConnector:
    """Get or create the global MAF connector instance.

    Returns:
        MAF connector instance
    """
    global _maf_connector

    if _maf_connector is None:
        _maf_connector = MAFConnector()

    return _maf_connector


async def test_maf_connection():
    """Test MAF connection and availability."""
    print("Testing MAF Connector...")
    print("-" * 60)

    connector = get_maf_connector()

    # Health check
    health = await connector.health_check()
    print(f"MAF Available: {health['maf_available']}")
    print(f"MAF Path: {health['maf_path']}")
    print(f"Available Agents: {', '.join(health['available_agents'])}")

    if not connector.is_available():
        print("\nMAF is not available. Make sure the multi-agent-framework")
        print("is installed in the parent DocHub directory.")
        return

    # Test classification
    print("\nTesting task classification...")
    classification = await connector.classify_task(
        "Debug this ValueError: invalid literal for int() with base 10"
    )
    if classification:
        print(f"Task Type: {classification['task_type']}")
        print(f"Confidence: {classification['confidence']:.2f}")
        print(f"Suggested Agents: {' -> '.join(classification['agent_sequence'])}")

    print("\nMAF Connector test complete!")


if __name__ == "__main__":
    asyncio.run(test_maf_connection())
