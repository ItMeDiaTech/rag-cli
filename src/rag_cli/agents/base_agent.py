"""Base Agent Protocol for RAG-CLI Agent System.

This module defines the core agent protocol and coordination system for
multi-agent orchestration in RAG-CLI. All specialized agents inherit from
the BaseAgent class and implement the agent protocol.

AGENT TYPES:
1. QueryDecomposer - Breaks complex queries into sub-queries
2. ResultSynthesizer - Merges and ranks results from multiple retrievals
3. DocumentationAgent - Auto-generates and updates project documentation
4. Future agents: ReRanker, QueryExpander, ResponseValidator, etc.

COORDINATION PATTERNS:
- Message passing: Agents communicate via typed messages
- Pipeline: Sequential agent execution (A -> B -> C)
- Parallel: Concurrent agent execution (A + B + C -> Synthesize)
- Hierarchical: Parent-child agent delegation

USAGE:
    class MyAgent(BaseAgent):
        async def process(self, message: AgentMessage) -> AgentMessage:
            # Agent implementation
            return AgentMessage(...)
"""

import asyncio
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import threading

from monitoring.logger import get_logger

logger = get_logger(__name__)


class AgentStatus(Enum):
    """Agent execution status."""
    IDLE = "idle"
    PROCESSING = "processing"
    WAITING = "waiting"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"


class MessageType(Enum):
    """Types of inter-agent messages."""
    REQUEST = "request"        # Request for agent to process data
    RESPONSE = "response"      # Response from agent with results
    ERROR = "error"            # Error notification
    PROGRESS = "progress"      # Progress update
    CANCEL = "cancel"          # Cancellation request
    COORDINATE = "coordinate"  # Coordination message between agents


@dataclass
class AgentMessage:
    """Message passed between agents.

    This is the primary communication primitive for the agent system.
    All inter-agent communication uses typed messages.
    """
    message_id: str
    message_type: MessageType
    sender_id: str
    receiver_id: Optional[str]  # None for broadcast
    payload: Dict[str, Any]
    timestamp: datetime
    parent_message_id: Optional[str] = None
    correlation_id: Optional[str] = None  # For tracking related messages
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create_request(
        cls,
        sender_id: str,
        receiver_id: str,
        payload: Dict[str, Any],
        correlation_id: Optional[str] = None
    ) -> 'AgentMessage':
        """Create a request message.

        Args:
            sender_id: ID of sending agent
            receiver_id: ID of receiving agent
            payload: Message payload
            correlation_id: Optional correlation ID for tracking

        Returns:
            AgentMessage instance
        """
        return cls(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.REQUEST,
            sender_id=sender_id,
            receiver_id=receiver_id,
            payload=payload,
            timestamp=datetime.now(),
            correlation_id=correlation_id or str(uuid.uuid4())
        )

    @classmethod
    def create_response(
        cls,
        sender_id: str,
        receiver_id: str,
        payload: Dict[str, Any],
        parent_message: 'AgentMessage'
    ) -> 'AgentMessage':
        """Create a response message.

        Args:
            sender_id: ID of sending agent
            receiver_id: ID of receiving agent
            payload: Response payload
            parent_message: Original request message

        Returns:
            AgentMessage instance
        """
        return cls(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.RESPONSE,
            sender_id=sender_id,
            receiver_id=receiver_id,
            payload=payload,
            timestamp=datetime.now(),
            parent_message_id=parent_message.message_id,
            correlation_id=parent_message.correlation_id
        )


@dataclass
class AgentMetrics:
    """Metrics for agent execution."""
    agent_id: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_processing_time: float = 0.0
    avg_processing_time: float = 0.0
    last_execution_time: Optional[datetime] = None
    errors: List[str] = field(default_factory=list)

    def record_execution(self, success: bool, duration: float, error: Optional[str] = None):
        """Record an execution.

        Args:
            success: Whether execution was successful
            duration: Execution duration in seconds
            error: Optional error message
        """
        self.total_requests += 1
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
            if error:
                self.errors.append(error)

        self.total_processing_time += duration
        self.avg_processing_time = self.total_processing_time / self.total_requests
        self.last_execution_time = datetime.now()


class BaseAgent(ABC):
    """Base class for all agents in the RAG-CLI system.

    All agents must inherit from this class and implement the process() method.
    This provides:
    - Unique agent identification
    - Message handling and routing
    - Status tracking
    - Metrics collection
    - Lifecycle management
    """

    def __init__(self, agent_id: Optional[str] = None, agent_type: Optional[str] = None):
        """Initialize base agent.

        Args:
            agent_id: Unique agent identifier (auto-generated if not provided)
            agent_type: Type of agent (defaults to class name)
        """
        self.agent_id = agent_id or f"{self.__class__.__name__}_{uuid.uuid4().hex[:8]}"
        self.agent_type = agent_type or self.__class__.__name__
        self.status = AgentStatus.IDLE
        self.metrics = AgentMetrics(agent_id=self.agent_id)
        self._message_queue = asyncio.Queue()
        self._running = False
        self._task = None

        logger.info("Agent initialized", agent_id=self.agent_id, agent_type=self.agent_type)

    @abstractmethod
    async def process(self, message: AgentMessage) -> AgentMessage:
        """Process a message and return a response.

        This is the core method that all agents must implement.
        It receives a message, processes it, and returns a response message.

        Args:
            message: Input message to process

        Returns:
            Response message with results

        Raises:
            AgentExecutionError: If processing fails
        """

    async def send_message(
        self,
        receiver_id: str,
        payload: Dict[str, Any],
        message_type: MessageType = MessageType.REQUEST
    ) -> AgentMessage:
        """Send a message to another agent.

        Args:
            receiver_id: ID of receiving agent
            payload: Message payload
            message_type: Type of message

        Returns:
            The sent message
        """
        message = AgentMessage(
            message_id=str(uuid.uuid4()),
            message_type=message_type,
            sender_id=self.agent_id,
            receiver_id=receiver_id,
            payload=payload,
            timestamp=datetime.now()
        )

        logger.debug(
            "Agent sending message",
            agent_id=self.agent_id,
            message_type=message_type.value,
            receiver=receiver_id
        )

        return message

    async def handle_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle an incoming message.

        This method wraps the process() method with error handling,
        metrics tracking, and status management.

        Args:
            message: Message to handle

        Returns:
            Response message or None if no response
        """
        if message.message_type == MessageType.CANCEL:
            self.status = AgentStatus.CANCELLED
            logger.info("Agent cancelled", agent_id=self.agent_id)
            return None

        self.status = AgentStatus.PROCESSING
        start_time = time.time()
        error_msg = None

        try:
            logger.debug(
                "Agent processing message",
                agent_id=self.agent_id,
                message_id=message.message_id
            )

            response = await self.process(message)
            self.status = AgentStatus.COMPLETED
            success = True

            logger.info(
                "Agent completed processing",
                agent_id=self.agent_id,
                duration_s=f"{time.time() - start_time:.3f}"
            )

            return response

        except Exception as e:
            error_msg = str(e)
            logger.error(
                "Agent processing error",
                agent_id=self.agent_id,
                error=error_msg
            )

            self.status = AgentStatus.ERROR
            success = False

            # Create error response
            error_response = AgentMessage.create_response(
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                payload={
                    'error': error_msg,
                    'original_request': message.payload
                },
                parent_message=message
            )
            error_response.message_type = MessageType.ERROR

            return error_response

        finally:
            duration = time.time() - start_time
            self.metrics.record_execution(success, duration, error_msg)
            self.status = AgentStatus.IDLE

    async def start(self):
        """Start the agent's message processing loop."""
        if self._running:
            logger.warning("Agent already running", agent_id=self.agent_id)
            return

        self._running = True
        logger.info("Agent started", agent_id=self.agent_id)

        while self._running:
            try:
                # Wait for message with timeout
                message = await asyncio.wait_for(
                    self._message_queue.get(),
                    timeout=1.0
                )

                # Handle message
                await self.handle_message(message)

            except asyncio.TimeoutError:
                # No message, continue loop
                continue
            except Exception as e:
                logger.error("Agent loop error", agent_id=self.agent_id, error=str(e))

    async def stop(self):
        """Stop the agent's message processing loop."""
        self._running = False
        logger.info("Agent stopped", agent_id=self.agent_id)

    def get_metrics(self) -> AgentMetrics:
        """Get agent execution metrics.

        Returns:
            AgentMetrics instance
        """
        return self.metrics

    def get_status(self) -> AgentStatus:
        """Get current agent status.

        Returns:
            AgentStatus enum value
        """
        return self.status

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.agent_type}(id={self.agent_id}, status={self.status.value})"


class AgentCoordinator:
    """Coordinates message passing and execution between multiple agents.

    This is the central coordination point for the agent system. It:
    - Registers and manages agent instances
    - Routes messages between agents
    - Orchestrates parallel agent execution
    - Collects system-wide metrics
    """

    def __init__(self):
        """Initialize agent coordinator."""
        self.agents: Dict[str, BaseAgent] = {}
        self._message_bus: Dict[str, asyncio.Queue] = {}
        self._lock = asyncio.Lock()

        logger.info("Agent coordinator initialized")

    def register_agent(self, agent: BaseAgent):
        """Register an agent with the coordinator.

        Args:
            agent: Agent instance to register
        """
        self.agents[agent.agent_id] = agent
        self._message_bus[agent.agent_id] = asyncio.Queue()

        logger.info(
            "Agent registered",
            agent_id=agent.agent_id,
            agent_type=agent.agent_type
        )

    def unregister_agent(self, agent_id: str):
        """Unregister an agent.

        Args:
            agent_id: ID of agent to unregister
        """
        if agent_id in self.agents:
            del self.agents[agent_id]
            del self._message_bus[agent_id]
            logger.info("Agent unregistered", agent_id=agent_id)

    async def send_message(self, message: AgentMessage):
        """Send a message to an agent via the coordinator.

        Args:
            message: Message to send
        """
        receiver_id = message.receiver_id

        if not receiver_id:
            # Broadcast message
            for agent_id in self.agents:
                await self._message_bus[agent_id].put(message)
        elif receiver_id in self._message_bus:
            # Send to specific agent
            await self._message_bus[receiver_id].put(message)
        else:
            logger.warning("Unknown receiver agent", receiver_id=receiver_id)

    async def execute_agent(
        self,
        agent_id: str,
        payload: Dict[str, Any],
        timeout: Optional[float] = None
    ) -> AgentMessage:
        """Execute a single agent with given payload.

        Args:
            agent_id: ID of agent to execute
            payload: Input payload
            timeout: Optional timeout in seconds

        Returns:
            Agent response message

        Raises:
            KeyError: If agent not found
            asyncio.TimeoutError: If execution exceeds timeout
        """
        if agent_id not in self.agents:
            raise KeyError(f"Agent not found: {agent_id}")

        agent = self.agents[agent_id]

        # Create request message
        message = AgentMessage.create_request(
            sender_id="coordinator",
            receiver_id=agent_id,
            payload=payload
        )

        # Execute agent
        if timeout:
            response = await asyncio.wait_for(
                agent.handle_message(message),
                timeout=timeout
            )
        else:
            response = await agent.handle_message(message)

        return response

    async def execute_parallel(
        self,
        agent_payloads: List[Tuple[str, Dict[str, Any]]],
        timeout: Optional[float] = None
    ) -> List[AgentMessage]:
        """Execute multiple agents in parallel.

        Args:
            agent_payloads: List of (agent_id, payload) tuples
            timeout: Optional timeout for all executions

        Returns:
            List of agent responses in same order as input

        Raises:
            KeyError: If any agent not found
            asyncio.TimeoutError: If any execution exceeds timeout
        """
        tasks = [
            self.execute_agent(agent_id, payload, timeout)
            for agent_id, payload in agent_payloads
        ]

        responses = await asyncio.gather(*tasks, return_exceptions=True)

        return responses

    def get_all_metrics(self) -> Dict[str, AgentMetrics]:
        """Get metrics for all registered agents.

        Returns:
            Dictionary mapping agent_id to AgentMetrics
        """
        return {
            agent_id: agent.get_metrics()
            for agent_id, agent in self.agents.items()
        }

    def get_agent_status(self, agent_id: str) -> Optional[AgentStatus]:
        """Get status of a specific agent.

        Args:
            agent_id: ID of agent

        Returns:
            AgentStatus or None if agent not found
        """
        agent = self.agents.get(agent_id)
        return agent.get_status() if agent else None


# Global coordinator instance
_coordinator: Optional[AgentCoordinator] = None
_coordinator_lock = threading.Lock()


def get_agent_coordinator() -> AgentCoordinator:
    """Get or create the global agent coordinator (thread-safe).

    Returns:
        AgentCoordinator instance
    """
    global _coordinator

    if _coordinator is None:
        with _coordinator_lock:
            if _coordinator is None:
                _coordinator = AgentCoordinator()

    return _coordinator
