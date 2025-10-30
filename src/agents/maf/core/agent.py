"""
Base Agent Class for Multi-Agent Framework
"""

import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .agent_communication import CommunicativeAgent, AgentCommunicationHub, AgentMessage, CommunicationType

@dataclass
class AgentConfig:
    """Configuration for an agent"""
    name: str
    role: str
    capabilities: List[str]
    max_retries: int = 3
    timeout: int = 120
    temperature: float = 0.7
    max_tokens: int = 4000

@dataclass
class AgentResponse:
    """Response from an agent"""
    agent_name: str
    task_id: str
    success: bool
    result: Any
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    tokens_used: int = 0
    execution_time: float = 0
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()
        if self.metadata is None:
            self.metadata = {}

class Agent(CommunicativeAgent):
    """Base Agent class for all specialized agents"""

    def __init__(
        self,
        config: AgentConfig,
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

        # Setup logging
        self.logger = logging.getLogger(f"Agent.{self.name}")
        self.logger.setLevel(logging.DEBUG)

        # Performance tracking
        self.total_tasks = 0
        self.successful_tasks = 0
        self.failed_tasks = 0
        self.total_tokens = 0
        self.total_time = 0

        # Task history
        self.task_history = []

        self.logger.info(
            "Agent %s initialized with capabilities: %s",
            self.name,
            self.capabilities
        )

    async def process_simple(self, task_input) -> AgentResponse:
        """
        Simplified process method for backward compatibility
        Handles various input formats from legacy runners
        """

        # Generate a task ID
        task_id = f"task_{int(time.time())}"

        # Normalize task input to expected format
        if isinstance(task_input, str):
            # String input - treat as description
            task = {
                'type': 'general',
                'description': task_input
            }
        elif isinstance(task_input, dict):
            # Dictionary input - use as-is or extract fields
            if 'description' in task_input:
                task = task_input.copy()
                if 'type' not in task:
                    task['type'] = 'general'
            else:
                # Old format - try to extract meaningful fields
                task = {
                    'type': 'general',
                    'description': str(task_input),
                    'requirements': task_input.get('requirements', {}),
                    'context': task_input.get('context', {}),
                    'goal': task_input.get('goal', '')
                }
        else:
            # Other formats - convert to string
            task = {
                'type': 'general',
                'description': str(task_input)
            }

        # Extract context if provided
        context = task.pop('context', {}) if 'context' in task else {}

        return await self.process(task_id, task, context)

    async def process(
        self,
        task_id: str,
        task: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """Process a task with comprehensive logging"""

        if context is None:
            context = {}

        start_time = time.time()
        self.total_tasks += 1

        # Log incoming task
        self.logger.info(
            "[%s] Received task: %s",
            task_id,
            task.get('type', 'unknown')
        )
        self.logger.debug(
            "[%s] Task details: %s",
            task_id,
            json.dumps(task, indent=2)
        )

        if context:
            self.logger.debug(
                "[%s] Context: %s",
                task_id,
                json.dumps(context, indent=2)
            )

        # Broadcast task receipt if message bus available
        if self.message_bus:
            await self.message_bus.publish({
                'type': 'task_received',
                'agent': self.name,
                'task_id': task_id,
                'task': task,
                'timestamp': datetime.now(timezone.utc).isoformat()
            })

        try:
            # Check capabilities
            if not self._can_handle_task(task):
                self.logger.warning(
                    "[%s] Agent %s cannot handle task type: %s",
                    task_id,
                    self.name,
                    task.get('type')
                )
                return AgentResponse(
                    agent_name=self.name,
                    task_id=task_id,
                    success=False,
                    result=None,
                    error=(
                        f"Task type {task.get('type')} not in capabilities"
                    )
                )

            # Retrieve relevant memories if memory manager available
            memories = []
            if self.memory_manager:
                memories = await self._retrieve_memories(task, context)
                self.logger.debug(
                    "[%s] Retrieved %d relevant memories",
                    task_id,
                    len(memories)
                )

            # Execute the task
            result = await self._execute_task(task_id, task, context, memories)

            # Store successful result in memory
            if self.memory_manager and result:
                await self._store_memory(task, result, context)

            # Calculate metrics
            execution_time = time.time() - start_time
            self.total_time += execution_time

            # Create response
            response = AgentResponse(
                agent_name=self.name,
                task_id=task_id,
                success=True,
                result=result,
                execution_time=execution_time,
                metadata={
                    'memories_used': len(memories),
                    'context_provided': context is not None
                }
            )

            self.successful_tasks += 1
            self.logger.info(
                "[%s] Task completed successfully in %.2fs",
                task_id,
                execution_time
            )

            # Broadcast success
            if self.message_bus:
                await self.message_bus.publish({
                    'type': 'task_completed',
                    'agent': self.name,
                    'task_id': task_id,
                    'success': True,
                    'execution_time': execution_time,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                })

            # Store in history
            self.task_history.append({
                'task_id': task_id,
                'task': task,
                'response': asdict(response),
                'timestamp': datetime.now(timezone.utc).isoformat()
            })

            return response

        except Exception as e:
            self.failed_tasks += 1
            execution_time = time.time() - start_time

            self.logger.error(
                "[%s] Task failed: %s",
                task_id,
                str(e),
                exc_info=True
            )

            # Broadcast failure
            if self.message_bus:
                await self.message_bus.publish({
                    'type': 'task_failed',
                    'agent': self.name,
                    'task_id': task_id,
                    'error': str(e),
                    'timestamp': datetime.now(timezone.utc).isoformat()
                })

            return AgentResponse(
                agent_name=self.name,
                task_id=task_id,
                success=False,
                result=None,
                error=str(e),
                execution_time=execution_time
            )
    def _can_handle_task(self, task: Dict[str, Any]) -> bool:
        """Check if agent can handle the task type"""
        task_type = task.get('type', '')
        required_capabilities = task.get('required_capabilities', [])

        # Check if task type matches any capability
        if task_type and not any(
            cap in task_type for cap in self.capabilities
        ):
            return False

        # Check required capabilities
        if required_capabilities:
            return all(
                cap in self.capabilities for cap in required_capabilities
            )

        return True

    async def _handle_task_assignment(self, message: AgentMessage):
        """Handle incoming task assignment"""
        self.logger.info(f"[{self.name}] Received task assignment: {message.subject}")

        task_content = message.content
        task_id = task_content.get('task_id', 'unknown')
        description = task_content.get('description', '')
        project_context = task_content.get('project_context', {})

        # Process the task using the enhanced context
        try:
            # Create task dict for processing
            task = {
                'description': description,
                'type': 'assignment',
                'project_context': project_context,
                'task_id': task_id
            }

            # Process using existing method
            response = await self.process_simple(task)

            # Send completion notification
            if self.communication_hub:
                await self.communication_hub.complete_task(
                    task_id, self.agent_id, {
                        'success': response.success,
                        'result': response.result,
                        'execution_time': response.execution_time
                    }
                )

            # Broadcast status
            await self.broadcast_status("task_completed", {
                'task_id': task_id,
                'success': response.success
            })

        except Exception as e:
            self.logger.error(f"[{self.name}] Failed to process assigned task: {e}")

            # Notify of failure
            await self.broadcast_status("task_failed", {
                'task_id': task_id,
                'error': str(e)
            })

    async def _handle_assistance_request(self, message: AgentMessage):
        """Handle assistance request from another agent"""
        self.logger.info(f"[{self.name}] Assistance requested: {message.subject}")

        request_content = message.content
        task_description = request_content.get('task_description', '')
        required_expertise = request_content.get('required_expertise', [])

        # Check if we can help
        can_assist = any(skill in self.capabilities for skill in required_expertise)

        if can_assist:
            # Provide assistance
            try:
                assistance_task = {
                    'description': f"Assist with: {task_description}",
                    'type': 'assistance',
                    'original_requester': message.sender
                }

                response = await self.process_simple(assistance_task)

                # Send assistance response
                await self.send_message(
                    recipient=message.sender,
                    subject=f"Assistance Provided: {task_description}",
                    content={
                        'assistance_provided': True,
                        'result': response.result,
                        'suggestions': response.result if isinstance(response.result, str) else str(response.result)
                    },
                    message_type=CommunicationType.PROVIDE_FEEDBACK
                )

            except Exception as e:
                self.logger.error(f"[{self.name}] Failed to provide assistance: {e}")

                await self.send_message(
                    recipient=message.sender,
                    subject=f"Assistance Failed: {task_description}",
                    content={
                        'assistance_provided': False,
                        'error': str(e)
                    },
                    message_type=CommunicationType.PROVIDE_FEEDBACK
                )
        else:
            # Cannot assist
            await self.send_message(
                recipient=message.sender,
                subject=f"Cannot Assist: {task_description}",
                content={
                    'assistance_provided': False,
                    'reason': f"Required expertise {required_expertise} not in my capabilities {self.capabilities}"
                },
                message_type=CommunicationType.PROVIDE_FEEDBACK
            )

    async def _handle_context_update(self, message: AgentMessage):
        """Handle project context updates"""
        self.logger.info(f"[{self.name}] Project context updated")

        # Store context for future tasks
        if hasattr(self, 'project_context'):
            self.project_context = message.content
        else:
            self.project_context = message.content

        # Log important context changes
        if 'goal' in message.content:
            self.logger.info(f"[{self.name}] New project goal: {message.content['goal']}")

    async def _handle_knowledge_share(self, message: AgentMessage):
        """Handle knowledge sharing from other agents"""
        knowledge_type = message.content.get('type', 'general')
        knowledge_data = message.content.get('data', {})

        self.logger.info(f"[{self.name}] Received knowledge share: {knowledge_type}")

        # Store knowledge for future reference
        if not hasattr(self, 'shared_knowledge'):
            self.shared_knowledge = {}

        if knowledge_type not in self.shared_knowledge:
            self.shared_knowledge[knowledge_type] = []

        self.shared_knowledge[knowledge_type].append({
            'source': message.sender,
            'data': knowledge_data,
            'received_at': datetime.now(timezone.utc).isoformat()
        })

    async def collaborate_with_agent(self, other_agent_id: str, task_description: str):
        """Initiate collaboration with another agent"""
        await self.send_message(
            recipient=other_agent_id,
            subject=f"Collaboration Request: {task_description}",
            content={
                'collaboration_type': 'joint_task',
                'task_description': task_description,
                'my_capabilities': self.capabilities
            },
            message_type=CommunicationType.COLLABORATION
        )

    async def share_work_artifact(self, artifact_type: str, artifact_data: Any):
        """Share a work artifact with other agents"""
        await self.share_knowledge(artifact_type, {
            'artifact': artifact_data,
            'created_by': self.name,
            'capabilities_used': self.capabilities
        })

    def get_enhanced_context(self) -> Dict[str, Any]:
        """Get enhanced context including project and communication info"""
        base_context = self.get_project_context()

        enhanced = {
            'agent_info': {
                'name': self.name,
                'role': self.role,
                'capabilities': self.capabilities
            },
            'communication_history': len(self.received_messages),
            'shared_knowledge': getattr(self, 'shared_knowledge', {}),
            'project_context': getattr(self, 'project_context', {})
        }

        enhanced.update(base_context)
        return enhanced

    async def _retrieve_memories(
        self, task: Dict[str, Any], context: Dict[str, Any]
    ) -> List[Dict]:
        """Retrieve relevant memories for the task"""
        try:
            if not self.memory_manager:
                return []
            query = self._build_memory_query(task, context)
            memories = await self.memory_manager.search(query, limit=10)
            return memories
        except Exception as e:
            self.logger.warning("Failed to retrieve memories: %s", e)
            return []

    def _build_memory_query(
        self, task: Dict[str, Any], context: Dict[str, Any]
    ) -> str:
        """Build a query for memory retrieval"""
        parts = []

        if task.get('type'):
            parts.append(f"task type: {task['type']}")

        if task.get('description'):
            parts.append(task['description'])

        if context and context.get('previous_result'):
            parts.append(
                f"previous: {str(context['previous_result'])[:100]}"
            )

        return " ".join(parts)

    async def _store_memory(
        self, task: Dict[str, Any], result: Any, context: Dict[str, Any]
    ):
        """Store task result in memory"""
        try:
            if self.memory_manager:
                memory_entry = {
                    'agent': self.name,
                    'task': task,
                    'result': result,
                    'context': context,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
                await self.memory_manager.store(memory_entry)
        except Exception as e:
            self.logger.warning("Failed to store memory: %s", e)

    async def _execute_task(
        self,
        task_id: str,
        task: Dict[str, Any],
        context: Dict[str, Any],
        memories: List[Dict]
    ) -> Any:
        """Execute the actual task - to be overridden by specialized agents"""

        # Build prompt for Claude
        prompt = self._build_prompt(task, context, memories)

        self.logger.debug(
            "[%s] Sending prompt to Claude CLI", task_id
        )

        # Call Claude CLI if available
        if self.claude_cli:
            response = await self.claude_cli.complete(
                prompt,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )

            # Track token usage
            if response.tokens_used:
                self.total_tokens += response.tokens_used

            return response.content
        # Fallback for testing
        return f"[{self.name}] Processed task: {task.get('type', 'unknown')}"

    def _build_prompt(
        self,
        task: Dict[str, Any],
        context: Dict[str, Any],
        memories: List[Dict]
    ) -> str:
        """Build prompt for Claude"""
        prompt_parts = [
            f"You are {self.name}, a specialized agent with the role: {self.role}",
            f"Your capabilities include: {', '.join(self.capabilities)}",
            "",
            "Task:"
        ]

        if task.get('type'):
            prompt_parts.append(f"Type: {task['type']}")

        if task.get('description'):
            prompt_parts.append(f"Description: {task['description']}")

        if task.get('requirements'):
            prompt_parts.append(
                f"Requirements: {json.dumps(task['requirements'], indent=2)}"
            )

        if context:
            prompt_parts.extend([
                "",
                "Context:",
                json.dumps(context, indent=2)
            ])

        if memories:
            prompt_parts.extend([
                "",
                "Relevant memories:",
                json.dumps(memories[:5], indent=2)
            ])

        prompt_parts.extend([
            "",
            "Please complete this task to the best of your ability.",
            "Provide detailed, production-ready output."
        ])

        return "\n".join(prompt_parts)

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        success_rate = (
            (self.successful_tasks / self.total_tasks * 100)
            if self.total_tasks > 0 else 0
        )
        avg_time = (
            (self.total_time / self.total_tasks)
            if self.total_tasks > 0 else 0
        )

        return {
            'name': self.name,
            'total_tasks': self.total_tasks,
            'successful_tasks': self.successful_tasks,
            'failed_tasks': self.failed_tasks,
            'success_rate': f"{success_rate:.2f}%",
            'total_tokens': self.total_tokens,
            'total_time': f"{self.total_time:.2f}s",
            'average_time': f"{avg_time:.2f}s"
        }

    async def reset_stats(self):
        """Reset agent statistics"""
        self.total_tasks = 0
        self.successful_tasks = 0
        self.failed_tasks = 0
        self.total_tokens = 0
        self.total_time = 0
        self.task_history = []
        self.logger.info("Agent %s statistics reset", self.name)