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
import time
import threading
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime

from rag_cli.utils.logger import get_logger

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
            from rag_cli.agents.maf.agents.debugger import DebuggerAgent
            from rag_cli.agents.maf.agents.developer import DeveloperAgent
            from rag_cli.agents.maf.agents.reviewer import ReviewerAgent
            from rag_cli.agents.maf.agents.tester import TesterAgent
            from rag_cli.agents.maf.agents.architect import ArchitectAgent
            from rag_cli.agents.maf.agents.documenter import DocumenterAgent
            from rag_cli.agents.maf.agents.optimizer import OptimizerAgent
            from rag_cli.agents.maf.core.agent import AgentConfig

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

        Raises:
            ValueError: If agent_name is invalid or task_data is malformed
            TypeError: If arguments are of wrong type
        """
        # Validate agent_name
        if not agent_name:
            raise ValueError("agent_name cannot be empty")

        if not isinstance(agent_name, str):
            raise TypeError(f"agent_name must be str, got {type(agent_name).__name__}")

        # Validate task_data
        if not isinstance(task_data, dict):
            raise TypeError(f"task_data must be dict, got {type(task_data).__name__}")

        # Validate timeout
        if not isinstance(timeout, (int, float)):
            raise TypeError(f"timeout must be numeric, got {type(timeout).__name__}")

        if timeout <= 0:
            raise ValueError(f"timeout must be positive, got {timeout}")

        if not self.maf_available:
            logger.warning("Embedded MAF not available, falling back to RAG-only mode")
            return None

        try:
            logger.info(f"Executing embedded MAF agent: {agent_name}", task_keys=list(task_data.keys()))
            start_time = asyncio.get_event_loop().time()

            # Get the agent class
            agent_class = self.agents_map.get(agent_name)
            if not agent_class:
                available = ', '.join(self.agents_map.keys())
                raise ValueError(
                    f"Unknown agent: '{agent_name}'. "
                    f"Available agents: {available}"
                )

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
        except (ValueError, TypeError, AttributeError) as e:
            # Expected errors - agent not found, wrong types, etc.
            logger.error("Embedded MAF agent execution failed", agent=agent_name, error=str(e),
                         fallback="returning RAG-only response")
            return None
        except Exception as e:
            # Unexpected errors - log with traceback and return None
            logger.exception("Unexpected error in MAF agent execution",
                            agent=agent_name, exc_info=True)
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
        # Generate task ID
        task_id = f"task_{int(time.time())}"

        # Format task dictionary
        task = {
            'type': task_data.get('type', 'general'),
            'description': task_description,
            **task_data
        }

        # Call agent's process method with correct signature
        if hasattr(agent, 'process'):
            response = await agent.process(task_id, task, context={})

            # Extract result from AgentResponse
            if hasattr(response, 'result'):
                return str(response.result)
            elif isinstance(response, dict) and 'result' in response:
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
            from rag_cli.agents.maf.core.task_classifier import IntelligentTaskClassifier

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

        except (ImportError, AttributeError) as e:
            # Expected errors - missing classifier or wrong attributes
            logger.warning(f"Embedded MAF task classification failed, falling back: {e}")
            return None
        except Exception as e:
            # Unexpected errors - log with traceback
            logger.exception("Unexpected error in task classification", exc_info=True)
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

    async def execute_multiple_agents(
        self,
        agents: List[str],
        task_data: Dict[str, Any],
        strategy: str = 'parallel',
        timeout: float = 45.0
    ) -> Dict[str, Optional[MAFResult]]:
        """Execute multiple MAF agents concurrently or sequentially.

        Args:
            agents: List of agent names to execute
            task_data: Task data shared across agents
            strategy: Execution strategy ('parallel', 'sequential')
            timeout: Overall timeout for all agents

        Returns:
            Dictionary mapping agent names to their MAFResults
        """
        if not self.maf_available:
            logger.warning("MAF not available for multi-agent execution")
            return {agent: None for agent in agents}

        if not agents:
            logger.warning("No agents specified for multi-agent execution")
            return {}

        logger.info(f"Executing {len(agents)} MAF agents with {strategy} strategy",
                   agents=agents)

        results = {}
        per_agent_timeout = timeout / len(agents)

        if strategy == 'parallel':
            # Execute all agents concurrently
            tasks = [
                self.execute_agent(agent, task_data.copy(), timeout=per_agent_timeout)
                for agent in agents
            ]

            try:
                agent_results = await asyncio.gather(*tasks, return_exceptions=True)

                for agent, result in zip(agents, agent_results):
                    if isinstance(result, Exception):
                        logger.error(f"Agent {agent} failed: {result}")
                        results[agent] = None
                    else:
                        results[agent] = result
                        logger.debug(f"Agent {agent} completed successfully")

            except (asyncio.CancelledError, RuntimeError) as e:
                # Expected errors - task cancellation, event loop issues
                logger.error(f"Multi-agent parallel execution failed: {e}")
                for agent in agents:
                    results[agent] = None
            except Exception as e:
                # Unexpected errors - log with traceback
                logger.exception("Unexpected error in multi-agent parallel execution", exc_info=True)
                for agent in agents:
                    results[agent] = None

        else:  # sequential
            # Execute agents one at a time
            for agent in agents:
                try:
                    result = await self.execute_agent(agent, task_data.copy(),
                                                     timeout=per_agent_timeout)
                    results[agent] = result
                    logger.debug(f"Agent {agent} completed")
                except (ValueError, TypeError) as e:
                    # Expected errors - validation failures
                    logger.error(f"Agent {agent} failed: {e}")
                    results[agent] = None
                except Exception as e:
                    # Unexpected errors - log with traceback
                    logger.exception(f"Unexpected error executing agent {agent}", exc_info=True)
                    results[agent] = None

        # Log summary
        successful = sum(1 for r in results.values() if r is not None)
        logger.info(f"Multi-agent execution complete: {successful}/{len(agents)} succeeded")

        return results

    async def execute_workflow(
        self,
        workflow_name: str,
        task_data: Dict[str, Any],
        timeout: float = 60.0
    ) -> Optional[Dict[str, Any]]:
        """Execute a predefined MAF workflow with multiple agents.

        Workflows define which agents to run and how to combine their results.

        Args:
            workflow_name: Name of workflow ('code_review', 'debugging', 'architecture')
            task_data: Input data for the workflow
            timeout: Workflow execution timeout

        Returns:
            Combined workflow results or None if failed
        """
        if not self.maf_available:
            logger.warning(f"MAF not available for workflow '{workflow_name}'")
            return None

        # Define workflow patterns
        workflows = {
            'code_review': {
                'agents': ['developer', 'reviewer', 'tester'],
                'strategy': 'parallel',
                'description': 'Comprehensive code review with development, review, and testing'
            },
            'debugging': {
                'agents': ['debugger', 'developer'],
                'strategy': 'sequential',
                'description': 'Debug error, then provide fix implementation'
            },
            'architecture': {
                'agents': ['architect', 'developer', 'reviewer'],
                'strategy': 'sequential',
                'description': 'Design architecture, implement, then review'
            },
            'optimization': {
                'agents': ['optimizer', 'reviewer', 'tester'],
                'strategy': 'parallel',
                'description': 'Optimize code and validate improvements'
            },
            'comprehensive': {
                'agents': ['architect', 'developer', 'reviewer', 'tester', 'documenter'],
                'strategy': 'sequential',
                'description': 'Full development lifecycle'
            }
        }

        if workflow_name not in workflows:
            logger.error(f"Unknown workflow: {workflow_name}")
            return None

        workflow = workflows[workflow_name]
        logger.info(
            f"Executing workflow '{workflow_name}'",
            agents=workflow['agents'],
            strategy=workflow['strategy']
        )

        # Execute agents
        results = await self.execute_multiple_agents(
            agents=workflow['agents'],
            task_data=task_data,
            strategy=workflow['strategy'],
            timeout=timeout
        )

        # Combine results
        workflow_result = {
            'workflow_name': workflow_name,
            'description': workflow['description'],
            'strategy': workflow['strategy'],
            'agents_executed': list(results.keys()),
            'agent_results': {},
            'summary': self._synthesize_workflow_results(results),
            'success_rate': sum(1 for r in results.values() if r) / len(results) if results else 0.0
        }

        for agent, result in results.items():
            if result:
                workflow_result['agent_results'][agent] = {
                    'status': result.status,
                    'content': result.content,
                    'confidence': result.confidence,
                    'execution_time': result.execution_time
                }

        logger.info(
            f"Workflow '{workflow_name}' completed",
            success_rate=f"{workflow_result['success_rate']:.0%}"
        )

        return workflow_result

    def _synthesize_workflow_results(self, results: Dict[str, Optional[MAFResult]]) -> str:
        """Synthesize results from multiple agents into a summary.

        Args:
            results: Dictionary of agent results

        Returns:
            Synthesized summary text
        """
        successful_results = {k: v for k, v in results.items() if v and v.status == 'completed'}

        if not successful_results:
            return "All agents failed to produce results."

        summary_parts = []
        for agent, result in successful_results.items():
            summary_parts.append(f"[{agent.upper()}] {result.content[:200]}...")

        return "\n\n".join(summary_parts)

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
                from rag_cli.agents.maf.core import agent
                health['maf_version'] = getattr(agent, '__version__', '1.2.2')
            except Exception:
                health['maf_version'] = '1.2.2 (embedded)'

        return health


# Singleton instance
_maf_connector: Optional[MAFConnector] = None
_maf_lock = threading.Lock()


def get_maf_connector() -> MAFConnector:
    """Get or create the global MAF connector instance with thread-safe initialization.

    Returns:
        MAF connector instance
    """
    global _maf_connector

    if _maf_connector is None:
        with _maf_lock:
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
    print(f"MAF Path: {health['maf_location']}")
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
