"""Agent-level monitoring and tracing for RAG-CLI agent system.

This module provides comprehensive monitoring, tracing, and metrics collection
for the multi-agent orchestration system. It tracks:
- Individual agent execution metrics
- Inter-agent message flows
- Agent coordination patterns
- System-wide agent performance

USAGE:
    from monitoring.agent_monitor import get_agent_monitor

    monitor = get_agent_monitor()
    monitor.trace_agent_execution(agent_id, duration, success)
    monitor.trace_message_flow(message)
    report = monitor.generate_report()
"""

import time
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any

from monitoring.logger import get_logger

logger = get_logger(__name__)


@dataclass
class AgentExecutionTrace:
    """Trace of a single agent execution."""
    trace_id: str
    agent_id: str
    agent_type: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None
    input_size: Optional[int] = None  # Size of input data
    output_size: Optional[int] = None  # Size of output data
    metadata: Dict[str, Any] = field(default_factory=dict)

    def complete(self, success: bool = True, error: Optional[str] = None):
        """Mark execution as complete.

        Args:
            success: Whether execution was successful
            error: Optional error message
        """
        self.end_time = datetime.now()
        self.duration_seconds = (self.end_time - self.start_time).total_seconds()
        self.success = success
        self.error_message = error


@dataclass
class MessageFlowTrace:
    """Trace of a message flow between agents."""
    message_id: str
    correlation_id: str
    sender_id: str
    receiver_id: str
    message_type: str
    timestamp: datetime
    payload_size: int = 0
    processing_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentPerformanceMetrics:
    """Performance metrics for a single agent."""
    agent_id: str
    agent_type: str
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    total_duration: float = 0.0
    avg_duration: float = 0.0
    min_duration: Optional[float] = None
    max_duration: Optional[float] = None
    messages_sent: int = 0
    messages_received: int = 0
    error_count: int = 0
    last_execution: Optional[datetime] = None

    def update(self, trace: AgentExecutionTrace):
        """Update metrics with a new execution trace.

        Args:
            trace: Execution trace to incorporate
        """
        self.total_executions += 1
        self.last_execution = trace.end_time

        if trace.success:
            self.successful_executions += 1
        else:
            self.failed_executions += 1
            self.error_count += 1

        if trace.duration_seconds is not None:
            self.total_duration += trace.duration_seconds

            if self.min_duration is None or trace.duration_seconds < self.min_duration:
                self.min_duration = trace.duration_seconds

            if self.max_duration is None or trace.duration_seconds > self.max_duration:
                self.max_duration = trace.duration_seconds

            self.avg_duration = self.total_duration / self.total_executions


class AgentMonitor:
    """Monitors and tracks agent execution and coordination.

    This class provides centralized monitoring for the multi-agent system,
    collecting traces, metrics, and generating performance reports.
    """

    def __init__(self):
        """Initialize agent monitor."""
        self.execution_traces: List[AgentExecutionTrace] = []
        self.message_traces: List[MessageFlowTrace] = []
        self.agent_metrics: Dict[str, AgentPerformanceMetrics] = {}
        self._active_traces: Dict[str, AgentExecutionTrace] = {}
        self._lock = threading.Lock()
        self._start_time = datetime.now()

        logger.info("Agent monitor initialized")

    def start_agent_execution(
        self,
        trace_id: str,
        agent_id: str,
        agent_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AgentExecutionTrace:
        """Start tracking an agent execution.

        Args:
            trace_id: Unique trace identifier
            agent_id: ID of agent being executed
            agent_type: Type of agent
            metadata: Optional metadata

        Returns:
            AgentExecutionTrace instance
        """
        with self._lock:
            trace = AgentExecutionTrace(
                trace_id=trace_id,
                agent_id=agent_id,
                agent_type=agent_type,
                start_time=datetime.now(),
                metadata=metadata or {}
            )

            self._active_traces[trace_id] = trace

            logger.debug(
                "Started agent execution trace",
                trace_id=trace_id,
                agent_id=agent_id
            )

            return trace

    def complete_agent_execution(
        self,
        trace_id: str,
        success: bool = True,
        error: Optional[str] = None,
        input_size: Optional[int] = None,
        output_size: Optional[int] = None
    ):
        """Complete tracking an agent execution.

        Args:
            trace_id: Trace identifier
            success: Whether execution was successful
            error: Optional error message
            input_size: Size of input data
            output_size: Size of output data
        """
        with self._lock:
            if trace_id not in self._active_traces:
                logger.warning("Unknown trace ID", trace_id=trace_id)
                return

            trace = self._active_traces.pop(trace_id)
            trace.complete(success=success, error=error)
            trace.input_size = input_size
            trace.output_size = output_size

            # Store completed trace
            self.execution_traces.append(trace)

            # Update agent metrics
            self._update_agent_metrics(trace)

            logger.debug(
                "Completed agent execution trace",
                trace_id=trace_id,
                agent_id=trace.agent_id,
                duration=f"{trace.duration_seconds:.3f}s",
                success=success
            )

    def trace_message_flow(
        self,
        message_id: str,
        correlation_id: str,
        sender_id: str,
        receiver_id: str,
        message_type: str,
        payload_size: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Trace a message flow between agents.

        Args:
            message_id: Unique message identifier
            correlation_id: Correlation ID for related messages
            sender_id: Sending agent ID
            receiver_id: Receiving agent ID
            message_type: Type of message
            payload_size: Size of payload in bytes
            metadata: Optional metadata
        """
        with self._lock:
            trace = MessageFlowTrace(
                message_id=message_id,
                correlation_id=correlation_id,
                sender_id=sender_id,
                receiver_id=receiver_id,
                message_type=message_type,
                timestamp=datetime.now(),
                payload_size=payload_size,
                metadata=metadata or {}
            )

            self.message_traces.append(trace)

            # Update message counters
            if sender_id in self.agent_metrics:
                self.agent_metrics[sender_id].messages_sent += 1
            if receiver_id in self.agent_metrics:
                self.agent_metrics[receiver_id].messages_received += 1

            logger.debug(
                "Traced message flow",
                message_id=message_id,
                sender=sender_id,
                receiver=receiver_id
            )

    def _update_agent_metrics(self, trace: AgentExecutionTrace):
        """Update agent performance metrics.

        Args:
            trace: Completed execution trace
        """
        agent_id = trace.agent_id

        if agent_id not in self.agent_metrics:
            self.agent_metrics[agent_id] = AgentPerformanceMetrics(
                agent_id=agent_id,
                agent_type=trace.agent_type
            )

        self.agent_metrics[agent_id].update(trace)

    def get_agent_metrics(self, agent_id: str) -> Optional[AgentPerformanceMetrics]:
        """Get performance metrics for a specific agent.

        Args:
            agent_id: Agent identifier

        Returns:
            AgentPerformanceMetrics or None if agent not found
        """
        return self.agent_metrics.get(agent_id)

    def get_all_metrics(self) -> Dict[str, AgentPerformanceMetrics]:
        """Get performance metrics for all agents.

        Returns:
            Dictionary mapping agent_id to metrics
        """
        return self.agent_metrics.copy()

    def get_execution_traces(
        self,
        agent_id: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[AgentExecutionTrace]:
        """Get execution traces with optional filtering.

        Args:
            agent_id: Filter by agent ID
            since: Filter traces after this time
            limit: Maximum number of traces to return

        Returns:
            List of execution traces
        """
        traces = self.execution_traces

        # Filter by agent_id
        if agent_id:
            traces = [t for t in traces if t.agent_id == agent_id]

        # Filter by time
        if since:
            traces = [t for t in traces if t.start_time >= since]

        # Apply limit
        if limit:
            traces = traces[-limit:]

        return traces

    def get_message_traces(
        self,
        correlation_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[MessageFlowTrace]:
        """Get message flow traces with optional filtering.

        Args:
            correlation_id: Filter by correlation ID
            agent_id: Filter by sender or receiver agent ID
            limit: Maximum number of traces to return

        Returns:
            List of message flow traces
        """
        traces = self.message_traces

        # Filter by correlation_id
        if correlation_id:
            traces = [t for t in traces if t.correlation_id == correlation_id]

        # Filter by agent_id (sender or receiver)
        if agent_id:
            traces = [t for t in traces if t.sender_id == agent_id or t.receiver_id == agent_id]

        # Apply limit
        if limit:
            traces = traces[-limit:]

        return traces

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive monitoring report.

        Returns:
            Dictionary with monitoring statistics
        """
        with self._lock:
            uptime = (datetime.now() - self._start_time).total_seconds()

            # Overall statistics
            total_executions = len(self.execution_traces)
            successful_executions = sum(1 for t in self.execution_traces if t.success)
            failed_executions = total_executions - successful_executions

            success_rate = (successful_executions / total_executions * 100) if total_executions > 0 else 0

            # Per-agent statistics
            agent_stats = {
                agent_id: {
                    'type': metrics.agent_type,
                    'executions': metrics.total_executions,
                    'success_rate': (metrics.successful_executions / metrics.total_executions * 100)
                    if metrics.total_executions > 0 else 0,
                    'avg_duration': metrics.avg_duration,
                    'messages_sent': metrics.messages_sent,
                    'messages_received': metrics.messages_received
                }
                for agent_id, metrics in self.agent_metrics.items()
            }

            # Message statistics
            total_messages = len(self.message_traces)
            message_types = defaultdict(int)
            for trace in self.message_traces:
                message_types[trace.message_type] += 1

            report = {
                'uptime_seconds': uptime,
                'total_agents': len(self.agent_metrics),
                'total_executions': total_executions,
                'successful_executions': successful_executions,
                'failed_executions': failed_executions,
                'success_rate_percent': success_rate,
                'total_messages': total_messages,
                'message_types': dict(message_types),
                'agent_statistics': agent_stats,
                'active_traces': len(self._active_traces)
            }

            return report

    def clear(self):
        """Clear all traces and metrics."""
        with self._lock:
            self.execution_traces.clear()
            self.message_traces.clear()
            self.agent_metrics.clear()
            self._active_traces.clear()
            self._start_time = datetime.now()

            logger.info("Agent monitor cleared")


# Global monitor instance
_monitor: Optional[AgentMonitor] = None
_monitor_lock = threading.Lock()


def get_agent_monitor() -> AgentMonitor:
    """Get or create the global agent monitor (thread-safe).

    Returns:
        AgentMonitor instance
    """
    global _monitor

    if _monitor is None:
        with _monitor_lock:
            if _monitor is None:
                _monitor = AgentMonitor()

    return _monitor


if __name__ == "__main__":
    # Test agent monitor
    print("Testing Agent Monitor...")

    monitor = get_agent_monitor()

    # Simulate agent executions
    print("\nSimulating agent executions...")

    # Agent 1: QueryDecomposer
    trace_id_1 = "trace_001"
    monitor.start_agent_execution(trace_id_1, "query_decomposer_1", "QueryDecomposer")
    time.sleep(0.1)  # Simulate processing
    monitor.complete_agent_execution(trace_id_1, success=True, input_size=100, output_size=300)

    # Agent 2: ResultSynthesizer
    trace_id_2 = "trace_002"
    monitor.start_agent_execution(trace_id_2, "result_synthesizer_1", "ResultSynthesizer")
    time.sleep(0.05)  # Simulate processing
    monitor.complete_agent_execution(trace_id_2, success=True, input_size=300, output_size=150)

    # Agent 3: QueryDecomposer (error case)
    trace_id_3 = "trace_003"
    monitor.start_agent_execution(trace_id_3, "query_decomposer_1", "QueryDecomposer")
    time.sleep(0.02)  # Simulate processing
    monitor.complete_agent_execution(trace_id_3, success=False, error="Invalid query format")

    # Simulate message flows
    print("Simulating message flows...")
    monitor.trace_message_flow(
        message_id="msg_001",
        correlation_id="corr_001",
        sender_id="coordinator",
        receiver_id="query_decomposer_1",
        message_type="REQUEST",
        payload_size=100
    )
    monitor.trace_message_flow(
        message_id="msg_002",
        correlation_id="corr_001",
        sender_id="query_decomposer_1",
        receiver_id="result_synthesizer_1",
        message_type="REQUEST",
        payload_size=300
    )

    # Generate and print report
    print("\n" + "=" * 70)
    print("AGENT MONITOR REPORT")
    print("=" * 70)

    report = monitor.generate_report()

    print("\nSystem Statistics:")
    print(f"  Uptime: {report['uptime_seconds']:.2f}s")
    print(f"  Total Agents: {report['total_agents']}")
    print(f"  Total Executions: {report['total_executions']}")
    print(f"  Success Rate: {report['success_rate_percent']:.1f}%")
    print(f"  Total Messages: {report['total_messages']}")

    print("\nAgent Statistics:")
    for agent_id, stats in report['agent_statistics'].items():
        print(f"\n  {agent_id} ({stats['type']}):")
        print(f"    Executions: {stats['executions']}")
        print(f"    Success Rate: {stats['success_rate']:.1f}%")
        print(f"    Avg Duration: {stats['avg_duration']:.3f}s")
        print(f"    Messages Sent: {stats['messages_sent']}")
        print(f"    Messages Received: {stats['messages_received']}")

    print("\nAgent monitor test complete!")
