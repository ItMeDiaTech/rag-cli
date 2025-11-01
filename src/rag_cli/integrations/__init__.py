"""Integration modules for RAG-CLI.

This package contains connectors to external frameworks and services:
- Multi-Agent Framework (MAF) integration
- ArXiv academic paper search (free, rate-limited)
- Tavily AI-optimized web search (free tier with quota tracking)
- External API connectors
- Third-party service integrations
"""

__all__ = ['maf_connector', 'arxiv_connector', 'tavily_connector']
