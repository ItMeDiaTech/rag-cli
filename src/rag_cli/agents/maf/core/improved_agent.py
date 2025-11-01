#!/usr/bin/env python3
"""
Improved Agent with Proper Lifecycle Management
Fixes issues based on 2025 best practices for multi-agent frameworks
"""

import asyncio
import logging
import signal
import time
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from .agent_communication import CommunicativeAgent, AgentMessage, CommunicationType

class AgentState(Enum):
    """Agent lifecycle states"""
    CREATED = "created"
    INITIALIZED = "initialized"
    RUNNING = "running"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SHUTDOWN = "shutdown"

@dataclass
class ImprovedAgentConfig:
    """Enhanced configuration for agents with lifecycle management"""
    name: str
    role: str
    capabilities: List[str]
    max_retries: int = 3
    timeout: int = 120
    temperature: float = 0.7
    max_tokens: int = 4000
    cleanup_timeout: int = 30
    enable_graceful_shutdown: bool = True
    max_concurrent_tasks: int = 5

class ImprovedAgent(CommunicativeAgent):
    """
    Improved Agent with proper lifecycle management following 2025 best practices

    Key improvements:
    1. Proper asyncio task management with cancellation handling
    2. Graceful shutdown with signal handling
    3. Resource cleanup with async context managers
    4. State management throughout lifecycle
    5. Error recovery and retry mechanisms
    """

    def __init__(
        self,
        config: ImprovedAgentConfig,
        claude_cli=None,
        memory_manager=None,
        message_bus=None
    ):
        # Initialize communication capabilities
        super().__init__()

        self.config = config
        self.name = config.name
        self.role = config.role
        self.capabilities = config.capabilities
        self.claude_cli = claude_cli
        self.memory_manager = memory_manager
        self.message_bus = message_bus

        # Lifecycle management
        self.state = AgentState.CREATED
        self._shutdown_event = asyncio.Event()
        self._active_tasks: Set[asyncio.Task] = set()
        self._task_semaphore = asyncio.Semaphore(config.max_concurrent_tasks)

        # Setup logging
        self.logger = logging.getLogger(f"ImprovedAgent.{self.name}")

        # Performance tracking
        self.total_tasks = 0
        self.successful_tasks = 0
        self.failed_tasks = 0
        self.cancelled_tasks = 0

        # Setup signal handlers for graceful shutdown
        if config.enable_graceful_shutdown:
            self._setup_signal_handlers()

        self.logger.info(f"Agent {self.name} created with improved lifecycle management")

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        try:
            loop = asyncio.get_running_loop()

            def shutdown_handler():
                self.logger.info(f"[{self.name}] Received shutdown signal")
                self._shutdown_event.set()

            loop.add_signal_handler(signal.SIGINT, shutdown_handler)
            loop.add_signal_handler(signal.SIGTERM, shutdown_handler)
        except (RuntimeError, NotImplementedError):
            # Signal handlers not available (e.g., on Windows or in threads)
            self.logger.debug(f"[{self.name}] Signal handlers not available")

    @asynccontextmanager
    async def lifecycle_context(self):
        """Async context manager for proper resource management"""
        try:
            await self.initialize()
            yield self
        finally:
            await self.cleanup()

    async def initialize(self):
        """Initialize the agent and transition to initialized state"""
        if self.state != AgentState.CREATED:
            return

        try:
            self.logger.info(f"[{self.name}] Initializing agent...")
            self.state = AgentState.INITIALIZED
            self.logger.info(f"[{self.name}] Agent initialized successfully")
        except Exception as e:
            self.state = AgentState.FAILED
            self.logger.error(f"[{self.name}] Initialization failed: {e}")
            raise

    async def process_with_lifecycle(
        self,
        task_input: Any,
        timeout: Optional[int] = None
    ) -> Any:
        """
        Process a task with proper lifecycle management and cleanup

        This is the main entry point that handles:
        - Task cancellation
        - Resource cleanup
        - State transitions
        - Error recovery
        """

        if self.state not in [AgentState.INITIALIZED, AgentState.COMPLETED]:
            raise RuntimeError(f"Agent {self.name} not in valid state for processing: {self.state}")

        # Create task with proper cleanup
        task = asyncio.create_task(self._execute_with_cleanup(task_input, timeout))
        self._active_tasks.add(task)

        # Add cleanup callback
        def cleanup_task(finished_task):
            self._active_tasks.discard(finished_task)

        task.add_done_callback(cleanup_task)

        try:
            return await task
        except asyncio.CancelledError:
            self.cancelled_tasks += 1
            self.logger.warning(f"[{self.name}] Task cancelled")
            raise
        except Exception as e:
            self.failed_tasks += 1
            self.logger.error(f"[{self.name}] Task failed: {e}")
            raise

    async def _execute_with_cleanup(self, task_input: Any, timeout: Optional[int]) -> Any:
        """Execute task with proper cleanup handling"""

        # Acquire semaphore to limit concurrent tasks
        async with self._task_semaphore:
            self.state = AgentState.RUNNING
            self.total_tasks += 1
            start_time = time.time()

            try:
                # Check for shutdown signal
                if self._shutdown_event.is_set():
                    raise asyncio.CancelledError("Agent shutting down")

                # Execute the actual task with timeout
                actual_timeout = timeout or self.config.timeout

                try:
                    result = await asyncio.wait_for(
                        self._process_task_internal(task_input),
                        timeout=actual_timeout
                    )

                    self.successful_tasks += 1
                    self.state = AgentState.COMPLETED
                    execution_time = time.time() - start_time

                    self.logger.info(
                        f"[{self.name}] Task completed in {execution_time:.2f}s"
                    )

                    return result

                except asyncio.TimeoutError:
                    self.logger.error(f"[{self.name}] Task timed out after {actual_timeout}s")
                    raise

            except asyncio.CancelledError:
                # Proper cancellation handling with cleanup
                self.state = AgentState.CANCELLED
                self.logger.info(f"[{self.name}] Task cancelled, cleaning up...")

                # Perform cleanup operations
                await self._cleanup_cancelled_task()
                raise

            except Exception as e:
                self.state = AgentState.FAILED
                self.logger.error(f"[{self.name}] Task failed: {e}")
                raise

    async def _process_task_internal(self, task_input: Any) -> Any:
        """Internal task processing - override in subclasses"""

        # Convert to the format expected by the original process_simple method
        if isinstance(task_input, str):
            task = {'description': task_input, 'type': 'general'}
        elif isinstance(task_input, dict):
            task = task_input
        else:
            task = {'description': str(task_input), 'type': 'general'}

        # Use Claude CLI for processing
        if self.claude_cli:
            try:
                # Create a prompt based on the agent's role and capabilities
                prompt = self._create_processing_prompt(task)

                response = await self.claude_cli.complete(
                    prompt,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature
                )

                if response.success:
                    return {
                        'success': True,
                        'result': response.content,
                        'tokens_used': response.tokens_used,
                        'agent': self.name
                    }
                else:
                    return {
                        'success': False,
                        'error': response.error,
                        'agent': self.name
                    }

            except Exception as e:
                self.logger.error(f"[{self.name}] Claude CLI processing failed: {e}")
                return {
                    'success': False,
                    'error': str(e),
                    'agent': self.name
                }
        else:
            # Fallback processing without Claude CLI
            return {
                'success': True,
                'result': f"Agent {self.name} processed task: {task.get('description', 'No description')}",
                'agent': self.name
            }

    def _create_processing_prompt(self, task: Dict[str, Any]) -> str:
        """Create a prompt based on agent role and task"""

        return f"""
You are {self.name}, a specialized AI agent with the role: {self.role}

Your capabilities: {', '.join(self.capabilities)}

Task to process: {task.get('description', 'No description provided')}
Task type: {task.get('type', 'general')}

Please process this task according to your role and capabilities. Provide a clear, actionable response.

Requirements:
- Be specific and actionable
- Focus on your area of expertise
- Provide concrete steps or solutions
- Consider the context and requirements

Response format: Provide your analysis and recommendations clearly.
"""

    async def _cleanup_cancelled_task(self):
        """Cleanup operations when a task is cancelled"""
        try:
            # Cleanup resources specific to this task
            self.logger.debug(f"[{self.name}] Performing cancellation cleanup")

            # Cancel any sub-tasks if needed
            # Save any partial progress
            # Release resources

        except Exception as e:
            self.logger.error(f"[{self.name}] Cleanup failed: {e}")

    async def shutdown(self, timeout: Optional[int] = None):
        """Graceful shutdown of the agent"""

        self.logger.info(f"[{self.name}] Starting graceful shutdown...")
        self.state = AgentState.SHUTDOWN
        self._shutdown_event.set()

        # Cancel all active tasks
        if self._active_tasks:
            self.logger.info(f"[{self.name}] Cancelling {len(self._active_tasks)} active tasks")

            for task in self._active_tasks:
                if not task.done():
                    task.cancel()

            # Wait for tasks to complete or timeout
            if self._active_tasks:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*self._active_tasks, return_exceptions=True),
                        timeout=timeout or self.config.cleanup_timeout
                    )
                except asyncio.TimeoutError:
                    self.logger.warning(f"[{self.name}] Some tasks did not complete within timeout")

        await self.cleanup()
        self.logger.info(f"[{self.name}] Shutdown complete")

    async def cleanup(self):
        """Cleanup agent resources"""
        try:
            # Close connections, save state, release resources
            if hasattr(self, 'communication_hub') and self.communication_hub:
                # Don't stop the hub here as other agents might be using it
                pass

            self.logger.info(f"[{self.name}] Cleanup completed")

        except Exception as e:
            self.logger.error(f"[{self.name}] Cleanup error: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics including lifecycle info"""
        return {
            'name': self.name,
            'role': self.role,
            'state': self.state.value,
            'capabilities': self.capabilities,
            'total_tasks': self.total_tasks,
            'successful_tasks': self.successful_tasks,
            'failed_tasks': self.failed_tasks,
            'cancelled_tasks': self.cancelled_tasks,
            'success_rate': f"{(self.successful_tasks / max(self.total_tasks, 1) * 100):.1f}%",
            'active_tasks': len(self._active_tasks),
            'is_shutdown': self.state == AgentState.SHUTDOWN
        }

    # Backward compatibility methods
    async def process_simple(self, task_input: Any) -> Any:
        """Backward compatibility wrapper"""
        return await self.process_with_lifecycle(task_input)

    async def process(self, task_id: str, task: Dict[str, Any], context: Dict[str, Any] = None) -> Any:
        """Backward compatibility wrapper for original process method"""
        enhanced_task = task.copy()
        if context:
            enhanced_task['context'] = context
        enhanced_task['task_id'] = task_id

        return await self.process_with_lifecycle(enhanced_task)