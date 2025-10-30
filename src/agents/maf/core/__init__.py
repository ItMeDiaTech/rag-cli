"""Multi-Agent Framework Core Module - v1.3.0+

Includes improved components with multi-instance support and enhanced CLI output.
"""

from .agent import Agent, AgentConfig, AgentResponse
from .orchestrator import WorkflowOrchestrator
from .agent_communication import AgentCommunicationHub
from .memory import MemoryManager
from .task_classifier import IntelligentTaskClassifier

# Alias for backward compatibility
Orchestrator = WorkflowOrchestrator

# Import improved components (v1.3.0+)
try:
    from .improved_agent import ImprovedAgent, ImprovedAgentConfig
    from .improved_orchestrator import ImprovedOrchestrator
    from .cli_output_formatter import CliOutputFormatter, create_formatter, OutputLevel
    IMPROVED_AVAILABLE = True
except ImportError:
    IMPROVED_AVAILABLE = False
    # Fallback to None if improved components not available
    ImprovedAgent = None
    ImprovedAgentConfig = None
    ImprovedOrchestrator = None
    CliOutputFormatter = None
    create_formatter = None
    OutputLevel = None

__all__ = [
    # Core components
    'Agent',
    'AgentConfig',
    'AgentResponse',
    'Orchestrator',
    'AgentCommunicationHub',
    'MemoryManager',
    'IntelligentTaskClassifier',
    # Improved components (v1.3.0+)
    'ImprovedAgent',
    'ImprovedAgentConfig',
    'ImprovedOrchestrator',
    'CliOutputFormatter',
    'create_formatter',
    'OutputLevel',
    # Availability flag
    'IMPROVED_AVAILABLE'
]
