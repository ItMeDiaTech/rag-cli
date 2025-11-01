#!/usr/bin/env python3
"""
Enhanced Agent Communication System
Implements MCP-inspired patterns for proper inter-agent communication and context sharing
"""

import asyncio
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class CommunicationType(Enum):
    """Types of inter-agent communication"""
    TASK_ASSIGNMENT = "task_assignment"
    TASK_COMPLETION = "task_completion"
    CONTEXT_SHARE = "context_share"
    REQUEST_ASSISTANCE = "request_assistance"
    PROVIDE_FEEDBACK = "provide_feedback"
    STATUS_UPDATE = "status_update"
    KNOWLEDGE_SHARE = "knowledge_share"
    RESOURCE_REQUEST = "resource_request"
    COLLABORATION = "collaboration"
    BROADCAST = "broadcast"


@dataclass
class AgentContext:
    """Shared context between agents for project objectives"""
    project_goal: str
    current_phase: str
    requirements: List[str]
    constraints: List[str]
    shared_knowledge: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    decisions: List[Dict[str, Any]] = field(default_factory=list)
    dependencies: Dict[str, List[str]] = field(default_factory=dict)
    timeline: Dict[str, str] = field(default_factory=dict)
    success_criteria: List[str] = field(default_factory=list)


@dataclass
class AgentMessage:
    """Enhanced message structure for agent communication"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: CommunicationType = CommunicationType.STATUS_UPDATE
    sender: str = ""
    recipient: Optional[str] = None  # None for broadcast
    subject: str = ""
    content: Any = None
    context: Optional[AgentContext] = None
    priority: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    requires_response: bool = False
    parent_message_id: Optional[str] = None

    def __post_init__(self):
        if isinstance(self.type, str):
            try:
                self.type = CommunicationType(self.type)
            except ValueError:
                self.type = CommunicationType.STATUS_UPDATE


class AgentCommunicationHub:
    """Central hub for managing inter-agent communication and context"""

    def __init__(self):
        self.logger = logging.getLogger('AgentCommunicationHub')

        # Message routing
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.message_handlers: Dict[str, List[callable]] = {}
        self.active_agents: Dict[str, Any] = {}

        # Shared context management
        self.project_context = AgentContext(
            project_goal="",
            current_phase="initialization",
            requirements=[],
            constraints=[]
        )

        # Communication history
        self.message_history: List[AgentMessage] = []
        self.conversation_threads: Dict[str, List[str]] = {}

        # Task coordination
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        self.task_dependencies: Dict[str, Set[str]] = {}
        self.completed_tasks: Set[str] = set()

        # Knowledge management
        self.shared_memory: Dict[str, Any] = {}
        self.agent_expertise: Dict[str, List[str]] = {}

        self.running = False

    async def start(self):
        """Start the communication hub"""
        if not self.running:
            self.running = True
            asyncio.create_task(self._process_messages())
            self.logger.info("Agent Communication Hub started")

    async def stop(self):
        """Stop the communication hub"""
        self.running = False
        self.logger.info("Agent Communication Hub stopped")

    def register_agent(self, agent_id: str, agent: Any, expertise: List[str] = None):
        """Register an agent with the communication hub"""
        self.active_agents[agent_id] = agent
        if expertise:
            self.agent_expertise[agent_id] = expertise

        self.logger.info(f"Agent {agent_id} registered with expertise: {expertise}")

    def set_project_context(self, goal: str, requirements: List[str], constraints: List[str] = None):
        """Set the overall project context for all agents"""
        self.project_context.project_goal = goal
        self.project_context.requirements = requirements
        self.project_context.constraints = constraints or []

        # Broadcast context update to all agents
        asyncio.create_task(self.broadcast_message(
            sender="system",
            subject="Project Context Updated",
            content={
                "goal": goal,
                "requirements": requirements,
                "constraints": constraints
            },
            message_type=CommunicationType.CONTEXT_SHARE
        ))

    async def send_message(self, message: AgentMessage) -> bool:
        """Send a message through the communication hub"""
        try:
            # Add to queue for processing
            await self.message_queue.put(message)

            # Store in history
            self.message_history.append(message)

            # Update conversation threads
            if message.parent_message_id:
                if message.parent_message_id not in self.conversation_threads:
                    self.conversation_threads[message.parent_message_id] = []
                self.conversation_threads[message.parent_message_id].append(message.id)

            self.logger.debug(f"Message {message.id} queued for delivery")
            return True

        except Exception as e:
            self.logger.error(f"Failed to send message: {e}")
            return False

    async def broadcast_message(self, sender: str, subject: str, content: Any,
                                message_type: CommunicationType = CommunicationType.BROADCAST):
        """Broadcast a message to all active agents"""
        message = AgentMessage(
            type=message_type,
            sender=sender,
            recipient=None,  # Broadcast
            subject=subject,
            content=content,
            context=self.project_context
        )

        await self.send_message(message)

    async def request_assistance(self, requester: str, task_description: str,
                                 required_expertise: List[str]) -> Optional[str]:
        """Request assistance from agents with specific expertise"""

        # Find suitable agents
        suitable_agents = []
        for agent_id, expertise in self.agent_expertise.items():
            if agent_id != requester and any(skill in expertise for skill in required_expertise):
                suitable_agents.append(agent_id)

        if not suitable_agents:
            self.logger.warning(f"No agents found with required expertise: {required_expertise}")
            return None

        # Send assistance request to the most suitable agent
        selected_agent = suitable_agents[0]  # Simple selection strategy

        message = AgentMessage(
            type=CommunicationType.REQUEST_ASSISTANCE,
            sender=requester,
            recipient=selected_agent,
            subject=f"Assistance Request: {task_description}",
            content={
                "task_description": task_description,
                "required_expertise": required_expertise,
                "priority": "normal"
            },
            requires_response=True,
            context=self.project_context
        )

        await self.send_message(message)
        return selected_agent

    async def share_knowledge(self, agent_id: str, knowledge_type: str,
                              knowledge_data: Dict[str, Any]):
        """Share knowledge that other agents can benefit from"""

        # Store in shared memory
        if knowledge_type not in self.shared_memory:
            self.shared_memory[knowledge_type] = []

        self.shared_memory[knowledge_type].append({
            "source": agent_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": knowledge_data
        })

        # Broadcast to interested agents
        await self.broadcast_message(
            sender=agent_id,
            subject=f"Knowledge Share: {knowledge_type}",
            content={
                "type": knowledge_type,
                "data": knowledge_data
            },
            message_type=CommunicationType.KNOWLEDGE_SHARE
        )

    async def update_project_status(self, agent_id: str, phase: str,
                                    completed_work: Dict[str, Any],
                                    next_steps: List[str]):
        """Update project status and communicate progress"""

        self.project_context.current_phase = phase

        # Update shared context with completed work
        if "completed_work" not in self.project_context.shared_knowledge:
            self.project_context.shared_knowledge["completed_work"] = {}

        self.project_context.shared_knowledge["completed_work"][agent_id] = completed_work

        # Broadcast status update
        await self.broadcast_message(
            sender=agent_id,
            subject=f"Project Status Update: {phase}",
            content={
                "phase": phase,
                "completed_work": completed_work,
                "next_steps": next_steps
            },
            message_type=CommunicationType.STATUS_UPDATE
        )

    async def coordinate_task(self, task_id: str, description: str,
                              assigned_to: str, depends_on: List[str] = None):
        """Coordinate task assignment and dependencies"""

        # Check if dependencies are met
        if depends_on:
            unmet_dependencies = [dep for dep in depends_on if dep not in self.completed_tasks]
            if unmet_dependencies:
                self.logger.warning(f"Task {task_id} has unmet dependencies: {unmet_dependencies}")
                return False

        # Record task
        self.active_tasks[task_id] = {
            "description": description,
            "assigned_to": assigned_to,
            "dependencies": depends_on or [],
            "status": "assigned",
            "created_at": datetime.now(timezone.utc).isoformat()
        }

        # Send task assignment
        message = AgentMessage(
            type=CommunicationType.TASK_ASSIGNMENT,
            sender="coordinator",
            recipient=assigned_to,
            subject=f"Task Assignment: {task_id}",
            content={
                "task_id": task_id,
                "description": description,
                "dependencies": depends_on or [],
                "project_context": asdict(self.project_context)
            },
            requires_response=True,
            context=self.project_context
        )

        await self.send_message(message)
        return True

    async def complete_task(self, task_id: str, agent_id: str, results: Dict[str, Any]):
        """Mark a task as complete and share results"""

        if task_id in self.active_tasks:
            self.active_tasks[task_id]["status"] = "completed"
            self.active_tasks[task_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
            self.active_tasks[task_id]["results"] = results

            self.completed_tasks.add(task_id)

            # Broadcast completion
            await self.broadcast_message(
                sender=agent_id,
                subject=f"Task Completed: {task_id}",
                content={
                    "task_id": task_id,
                    "results": results,
                    "completed_by": agent_id
                },
                message_type=CommunicationType.TASK_COMPLETION
            )

            self.logger.info(f"Task {task_id} completed by {agent_id}")

    def get_context_for_agent(self, agent_id: str) -> Dict[str, Any]:
        """Get relevant context for a specific agent"""
        return {
            "project_context": asdict(self.project_context),
            "shared_memory": self.shared_memory,
            "active_tasks": {tid: task for tid, task in self.active_tasks.items()
                             if task.get("assigned_to") == agent_id or task.get("status") == "completed"},
            "agent_expertise": self.agent_expertise,
            "recent_messages": self.message_history[-10:]  # Last 10 messages
        }

    async def _process_messages(self):
        """Process messages in the queue"""
        while self.running:
            try:
                # Get message with timeout
                message = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)

                await self._deliver_message(message)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Error processing message: {e}")

    async def _deliver_message(self, message: AgentMessage):
        """Deliver a message to its recipient(s)"""
        try:
            if message.recipient:
                # Direct message
                if message.recipient in self.active_agents:
                    agent = self.active_agents[message.recipient]
                    if hasattr(agent, 'receive_message'):
                        await agent.receive_message(message)
                    else:
                        self.logger.warning(f"Agent {message.recipient} does not support message receiving")
                else:
                    self.logger.warning(f"Agent {message.recipient} not found")
            else:
                # Broadcast message
                for agent_id, agent in self.active_agents.items():
                    if agent_id != message.sender:  # Don't send to sender
                        if hasattr(agent, 'receive_message'):
                            await agent.receive_message(message)

            self.logger.debug(f"Message {message.id} delivered")

        except Exception as e:
            self.logger.error(f"Failed to deliver message {message.id}: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get communication hub statistics"""
        return {
            "active_agents": len(self.active_agents),
            "total_messages": len(self.message_history),
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.completed_tasks),
            "shared_knowledge_types": len(self.shared_memory),
            "conversation_threads": len(self.conversation_threads),
            "current_phase": self.project_context.current_phase
        }

# Mixin class for agents to support enhanced communication


class CommunicativeAgent:
    """Mixin class that adds communication capabilities to agents"""

    def __init__(self):
        self.communication_hub: Optional[AgentCommunicationHub] = None
        self.agent_id: str = ""
        self.received_messages: List[AgentMessage] = []

    def connect_to_hub(self, hub: AgentCommunicationHub, agent_id: str, expertise: List[str] = None):
        """Connect this agent to the communication hub"""
        self.communication_hub = hub
        self.agent_id = agent_id
        hub.register_agent(agent_id, self, expertise)

    async def receive_message(self, message: AgentMessage):
        """Receive and process a message"""
        self.received_messages.append(message)

        # Process different message types
        if message.type == CommunicationType.TASK_ASSIGNMENT:
            await self._handle_task_assignment(message)
        elif message.type == CommunicationType.REQUEST_ASSISTANCE:
            await self._handle_assistance_request(message)
        elif message.type == CommunicationType.CONTEXT_SHARE:
            await self._handle_context_update(message)
        elif message.type == CommunicationType.KNOWLEDGE_SHARE:
            await self._handle_knowledge_share(message)

    async def send_message(self, recipient: str, subject: str, content: Any,
                           message_type: CommunicationType = CommunicationType.STATUS_UPDATE):
        """Send a message to another agent"""
        if not self.communication_hub:
            return False

        message = AgentMessage(
            type=message_type,
            sender=self.agent_id,
            recipient=recipient,
            subject=subject,
            content=content,
            context=self.communication_hub.project_context
        )

        return await self.communication_hub.send_message(message)

    async def broadcast_status(self, status: str, details: Dict[str, Any] = None):
        """Broadcast status update to all agents"""
        if not self.communication_hub:
            return

        await self.communication_hub.broadcast_message(
            sender=self.agent_id,
            subject=f"Status Update from {self.agent_id}",
            content={"status": status, "details": details or {}},
            message_type=CommunicationType.STATUS_UPDATE
        )

    async def request_assistance(self, task_description: str, required_expertise: List[str]):
        """Request assistance from other agents"""
        if not self.communication_hub:
            return None

        return await self.communication_hub.request_assistance(
            self.agent_id, task_description, required_expertise
        )

    async def share_knowledge(self, knowledge_type: str, knowledge_data: Dict[str, Any]):
        """Share knowledge with other agents"""
        if not self.communication_hub:
            return

        await self.communication_hub.share_knowledge(
            self.agent_id, knowledge_type, knowledge_data
        )

    def get_project_context(self) -> Dict[str, Any]:
        """Get current project context"""
        if not self.communication_hub:
            return {}

        return self.communication_hub.get_context_for_agent(self.agent_id)

    # Override these methods in your agent implementation
    async def _handle_task_assignment(self, message: AgentMessage):
        """Handle task assignment - override in subclass"""

    async def _handle_assistance_request(self, message: AgentMessage):
        """Handle assistance request - override in subclass"""

    async def _handle_context_update(self, message: AgentMessage):
        """Handle context update - override in subclass"""

    async def _handle_knowledge_share(self, message: AgentMessage):
        """Handle knowledge sharing - override in subclass"""
