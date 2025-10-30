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

import asyncio
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime

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
    """Connector to Embedded Multi-Agent Framework.

    Uses the embedded MAF framework in src/agents/maf/ rather than external reference.
    Falls back gracefully if MAF components are unavailable.
    """

    def __init__(self):
        """Initialize MAF connector with embedded framework."""
        self.maf_available = False
        self.orchestrator = None
        self.agents = {}

        # Try to import embedded MAF components
        try:
            from src.agents.maf.core.orchestrator import Orchestrator
            from src.agents.maf.agents.debugger import DebuggerAgent
            from src.agents.maf.agents.developer import DeveloperAgent
            from src.agents.maf.agents.reviewer import ReviewerAgent
            from src.agents.maf.agents.tester import TesterAgent
            from src.agents.maf.agents.architect import ArchitectAgent
            from src.agents.maf.agents.documenter import DocumenterAgent
            from src.agents.maf.agents.optimizer import OptimizerAgent
            from src.agents.maf.core.agent import AgentConfig

            self.Orchestrator = Orchestrator
            self.AgentConfig = AgentConfig
            self.agents_map = {
                'debugger': DebuggerAgent,
                'developer': DeveloperAgent,
                'reviewer': ReviewerAgent,
                'tester': TesterAgent,
                'architect': ArchitectAgent,
                'documenter': DocumenterAgent,
                'optimizer': OptimizerAgent
            }
            self.maf_available = True
            logger.info("Embedded MAF framework initialized successfully", agents=list(self.agents_map.keys()))
        except ImportError as e:
            logger.warning(f"Embedded MAF not available - continuing with RAG-only mode: {e}",
                           fallback="RAG-only retrieval enabled")
            self.maf_available = False

    async def execute_agent(
        self,
        agent_name: str,
        task_data: Dict[str, Any],
        timeout: float = 30.0
    ) -> Optional[MAFResult]:
        """Execute a specific embedded MAF agent with given task data.

        Args:
            agent_name: Name of agent to execute ('debugger', 'architect', etc.)
            task_data: Task data for the agent
            timeout: Execution timeout in seconds

        Returns:
            MAFResult if successful, None if failed or unavailable
        """
        if not self.maf_available:
            logger.warning("Embedded MAF not available, falling back to RAG-only mode")
            return None

        try:
            logger.info(f"Executing embedded MAF agent: {agent_name}", task_keys=list(task_data.keys()))
            start_time = asyncio.get_event_loop().time()

            # Get the agent class
            agent_class = self.agents_map.get(agent_name)
            if not agent_class:
                logger.error(f"Agent '{agent_name}' not found in agent map")
                return None

            # Create agent instance
            config = self.AgentConfig(
                name=agent_name.capitalize(),
                role=f"Execute {agent_name} analysis",
                capabilities=[agent_name],
                max_retries=2,
                timeout=timeout
            )
            agent = agent_class(config)

            # Format task description
            task_description = self._format_task_for_agent(agent_name, task_data)

            # Execute with timeout
            result_content = await asyncio.wait_for(
                self._execute_agent_task(agent, task_description, task_data),
                timeout=timeout
            )

            execution_time = asyncio.get_event_loop().time() - start_time

            result = MAFResult(
                status='completed',
                content=result_content,
                confidence=0.8,
                agent_name=agent_name,
                execution_time=execution_time,
                metadata={
                    'agent_class': agent_class.__name__,
                    'task_data': task_data,
                    'timeout_seconds': timeout
                },
                timestamp=datetime.now()
            )

            logger.info(
                "Embedded MAF agent execution complete",
                agent=agent_name,
                execution_time=f"{execution_time:.2f}s"
            )

            return result

        except asyncio.TimeoutError:
            logger.warning(f"Embedded MAF agent '{agent_name}' timed out after {timeout}s",
                           fallback="returning RAG-only response")
            return MAFResult(
                status='error',
                content=f"Agent '{agent_name}' execution timed out after {timeout}s",
                confidence=0.0,
                agent_name=agent_name,
                execution_time=timeout,
                metadata={'error': 'timeout'},
                timestamp=datetime.now()
            )
        except Exception as e:
            logger.error("Embedded MAF agent execution failed", agent=agent_name, error=str(e),
                         fallback="returning RAG-only response")
            return None

    async def _execute_agent_task(self, agent: Any, task_description: str, task_data: Dict[str, Any]) -> str:
        """Execute a task using an embedded MAF agent.

        Args:
            agent: Instantiated agent
            task_description: Natural language task description
            task_data: Structured task data

        Returns:
            Agent's response/analysis
        """
        # Call agent's process method if available
        if hasattr(agent, 'process'):
            response = await agent.process(task_description) if asyncio.iscoroutinefunction(agent.process) \
                else agent.process(task_description)
            if isinstance(response, dict) and 'result' in response:
                return str(response['result'])
            return str(response)
        else:
            # Fallback: return formatted task description
            return task_description

    async def classify_task(self, query: str) -> Optional[Dict[str, Any]]:
        """Classify a query using embedded MAF's task classifier.

        Args:
            query: Query to classify

        Returns:
            Classification result with task type, confidence, and agent sequence
        """
        if not self.maf_available:
            logger.debug("Embedded MAF not available for task classification")
            return None

        try:
            from src.agents.maf.core.task_classifier import IntelligentTaskClassifier

            classifier = IntelligentTaskClassifier()
            classification = classifier.classify_task(query)

            result = {
                'task_type': classification.task_type,
                'confidence': classification.confidence,
                'primary_workflow': classification.primary_workflow,
                'agent_sequence': classification.agent_sequence,
                'requirements': classification.suggested_requirements
            }

            logger.debug("Task classification complete", workflow=classification.primary_workflow,
                         confidence=f"{classification.confidence:.2f}")
            return result

        except Exception as e:
            logger.warning(f"Embedded MAF task classification failed, falling back: {e}")
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
                "Break it into executable sub-tasks for parallel processing."
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
        """Check embedded MAF connector health.

        Returns:
            Health status dictionary
        """
        health = {
            'status': 'healthy' if self.maf_available else 'unavailable',
            'maf_available': self.maf_available,
            'maf_type': 'embedded',
            'maf_location': 'src/agents/maf/',
            'available_agents': self.get_available_agents()
        }

        # Try to get version info from embedded MAF
        if self.maf_available:
            try:
                from src.agents.maf.core import agent
                health['maf_version'] = getattr(agent, '__version__', '1.2.0')
            except Exception:
                health['maf_version'] = '1.2.0 (embedded)'

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
