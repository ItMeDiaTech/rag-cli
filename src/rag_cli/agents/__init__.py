"""Agent modules for RAG-CLI orchestration.

This package contains specialized agents for complex query processing:
- BaseAgent: Core agent protocol and coordination system
- QueryDecomposer: Breaks complex queries into atomic sub-queries
- ResultSynthesizer: Merges and synthesizes multi-query results
- AgentCoordinator: Central coordination for multi-agent execution
- DocumentationAgent: Maintains project documentation (future)
"""

from rag_cli.agents.base_agent import (
    BaseAgent,
    AgentCoordinator,
    AgentMessage,
    AgentStatus,
    MessageType,
    AgentMetrics,
    get_agent_coordinator
)
from rag_cli.agents.query_decomposer import (
    QueryDecomposer,
    SubQuery,
    DecompositionResult,
    get_query_decomposer
)
from rag_cli.agents.result_synthesizer import (
    ResultSynthesizer,
    SynthesisResult,
    get_result_synthesizer
)

__all__ = [
    # Base agent protocol
    'BaseAgent',
    'AgentCoordinator',
    'AgentMessage',
    'AgentStatus',
    'MessageType',
    'AgentMetrics',
    'get_agent_coordinator',

    # Query decomposition
    'QueryDecomposer',
    'SubQuery',
    'DecompositionResult',
    'get_query_decomposer',

    # Result synthesis
    'ResultSynthesizer',
    'SynthesisResult',
    'get_result_synthesizer'
]
