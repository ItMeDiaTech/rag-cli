#!/usr/bin/env python3
"""
Improved Orchestrator with Proper Lifecycle Management
Fixes the agent process and communication issues
"""

import asyncio
import logging
import signal
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

@dataclass
class OrchestrationResult:
    """Result of workflow orchestration"""
    status: str
    execution_time: float
    stages_completed: int
    stages_failed: List[str]
    results: Dict[str, Any]
    errors: List[str]

class ImprovedOrchestrator:
    """
    Improved orchestrator with proper agent lifecycle management

    Key improvements:
    1. Async context managers for resource management
    2. Proper task cancellation and cleanup
    3. Graceful shutdown handling
    4. Better error recovery and state management
    """

    def __init__(self, agents: Dict[str, Any], communication_hub: Any = None):
        self.agents = agents
        self.communication_hub = communication_hub
        self.logger = logging.getLogger('ImprovedOrchestrator')

        # Lifecycle management
        self._shutdown_event = asyncio.Event()
        self._active_workflows: Set[asyncio.Task] = set()
        self._setup_signal_handlers()

        self.logger.info("Improved orchestrator initialized")

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        try:
            loop = asyncio.get_running_loop()

            def shutdown_handler():
                self.logger.info("Received shutdown signal")
                self._shutdown_event.set()

            loop.add_signal_handler(signal.SIGINT, shutdown_handler)
            loop.add_signal_handler(signal.SIGTERM, shutdown_handler)
        except (RuntimeError, NotImplementedError):
            self.logger.debug("Signal handlers not available")

    @asynccontextmanager
    async def workflow_context(self, workflow_name: str):
        """Context manager for workflow execution with proper cleanup"""
        workflow_id = f"workflow_{workflow_name}_{asyncio.get_running_loop().time()}"

        try:
            self.logger.info(f"Starting workflow: {workflow_id}")

            # Initialize communication hub if available
            if self.communication_hub:
                await self.communication_hub.start()

            yield workflow_id

        except Exception as e:
            self.logger.error(f"Workflow {workflow_id} failed: {e}")
            raise
        finally:
            # Cleanup resources
            try:
                if self.communication_hub and hasattr(self.communication_hub, 'stop'):
                    await self.communication_hub.stop()
            except Exception as e:
                self.logger.error(f"Communication hub cleanup failed: {e}")

            self.logger.info(f"Workflow {workflow_id} cleanup completed")

    async def execute_workflow_improved(
        self,
        workflow_name: str,
        task_data: Dict[str, Any],
        workflow_config: Dict[str, Any] = None
    ) -> OrchestrationResult:
        """
        Execute workflow with improved lifecycle management
        """
        start_time = asyncio.get_running_loop().time()
        stages_completed = 0
        stages_failed = []
        results = {}
        errors = []

        # Default workflow configurations
        workflows = {
            'bug_fix': ['debugger', 'developer', 'tester'],
            'code_generation': ['architect', 'developer', 'reviewer', 'tester', 'documenter'],
            'code_review': ['reviewer', 'tester'],
            'testing': ['tester', 'developer'],
            'optimization': ['optimizer', 'developer', 'tester', 'reviewer'],
            'documentation': ['documenter', 'reviewer'],
            'system_design': ['architect', 'developer', 'reviewer']
        }

        agent_sequence = workflows.get(workflow_name, ['developer'])

        async with self.workflow_context(workflow_name) as workflow_id:
            try:
                # Set project context if communication hub available
                if self.communication_hub:
                    self.communication_hub.set_project_context(
                        goal=task_data.get('description', f'Execute {workflow_name} workflow'),
                        requirements=task_data.get('requirements', []),
                        constraints=task_data.get('constraints', [])
                    )

                # Execute each stage with proper error handling
                for stage_idx, agent_name in enumerate(agent_sequence):
                    if self._shutdown_event.is_set():
                        self.logger.warning(f"Workflow {workflow_id} cancelled due to shutdown")
                        break

                    try:
                        self.logger.info(f"[{workflow_id}] Stage {stage_idx + 1}: {agent_name}")

                        agent = self.agents.get(agent_name)
                        if not agent:
                            error_msg = f"Agent {agent_name} not found"
                            self.logger.error(f"[{workflow_id}] {error_msg}")
                            errors.append(error_msg)
                            stages_failed.append(agent_name)
                            continue

                        # Create stage-specific task
                        stage_task = task_data.copy()
                        stage_task['stage'] = agent_name
                        stage_task['workflow_id'] = workflow_id
                        stage_task['stage_number'] = stage_idx + 1

                        # Execute with timeout and cancellation support
                        try:
                            # Use improved processing if available, fallback to original
                            if hasattr(agent, 'process_with_lifecycle'):
                                result = await asyncio.wait_for(
                                    agent.process_with_lifecycle(stage_task),
                                    timeout=120  # 2 minute timeout per stage
                                )
                            else:
                                # Fallback to original method with wrapper
                                result = await asyncio.wait_for(
                                    agent.process_simple(stage_task),
                                    timeout=120
                                )

                            # Store stage result
                            results[agent_name] = result
                            stages_completed += 1

                            self.logger.info(f"[{workflow_id}] Stage {agent_name} completed successfully")

                        except asyncio.TimeoutError:
                            error_msg = f"Stage {agent_name} timed out"
                            self.logger.error(f"[{workflow_id}] {error_msg}")
                            errors.append(error_msg)
                            stages_failed.append(agent_name)

                        except Exception as stage_error:
                            error_msg = f"Stage {agent_name} failed: {stage_error}"
                            self.logger.error(f"[{workflow_id}] {error_msg}")
                            errors.append(error_msg)
                            stages_failed.append(agent_name)

                    except asyncio.CancelledError:
                        self.logger.warning(f"[{workflow_id}] Stage {agent_name} cancelled")
                        stages_failed.append(agent_name)
                        raise

                # Determine final status
                execution_time = asyncio.get_running_loop().time() - start_time

                if stages_failed:
                    if stages_completed > 0:
                        status = "partial"
                    else:
                        status = "failed"
                else:
                    status = "completed"

                self.logger.info(f"[{workflow_id}] Workflow {status}: {stages_completed}/{len(agent_sequence)} stages")

                return OrchestrationResult(
                    status=status,
                    execution_time=execution_time,
                    stages_completed=stages_completed,
                    stages_failed=stages_failed,
                    results=results,
                    errors=errors
                )

            except asyncio.CancelledError:
                self.logger.warning(f"Workflow {workflow_id} cancelled")
                return OrchestrationResult(
                    status="cancelled",
                    execution_time=asyncio.get_running_loop().time() - start_time,
                    stages_completed=stages_completed,
                    stages_failed=stages_failed + ["cancelled"],
                    results=results,
                    errors=errors + ["Workflow cancelled"]
                )

            except Exception as workflow_error:
                self.logger.error(f"Workflow {workflow_id} failed: {workflow_error}")
                return OrchestrationResult(
                    status="error",
                    execution_time=asyncio.get_running_loop().time() - start_time,
                    stages_completed=stages_completed,
                    stages_failed=stages_failed + ["workflow_error"],
                    results=results,
                    errors=errors + [str(workflow_error)]
                )

    async def shutdown_all_agents(self, timeout: int = 30):
        """Gracefully shutdown all agents"""
        self.logger.info("Starting graceful shutdown of all agents")

        shutdown_tasks = []
        for agent_name, agent in self.agents.items():
            if hasattr(agent, 'shutdown'):
                self.logger.info(f"Shutting down agent: {agent_name}")
                task = asyncio.create_task(agent.shutdown(timeout))
                shutdown_tasks.append(task)

        if shutdown_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*shutdown_tasks, return_exceptions=True),
                    timeout=timeout
                )
                self.logger.info("All agents shutdown completed")
            except asyncio.TimeoutError:
                self.logger.warning("Some agents did not shutdown within timeout")

    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics"""
        agent_stats = {}
        for name, agent in self.agents.items():
            if hasattr(agent, 'get_stats'):
                agent_stats[name] = agent.get_stats()
            else:
                agent_stats[name] = {'name': name, 'status': 'unknown'}

        return {
            'orchestrator': 'ImprovedOrchestrator',
            'agents': agent_stats,
            'active_workflows': len(self._active_workflows),
            'shutdown_requested': self._shutdown_event.is_set()
        }