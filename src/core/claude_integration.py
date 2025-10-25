"""Claude API integration for RAG-CLI.

This module handles response generation using Claude with streaming support,
retry logic, and proper context assembly from retrieved documents.
"""

import os
import time
import json
from typing import List, Dict, Any, Optional, Generator, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

import anthropic
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from anthropic.types import MessageStreamEvent

from src.core.config import get_config
from src.core.retrieval_pipeline import RetrievalResult
from src.monitoring.logger import get_logger, get_metrics_logger, log_api_call


logger = get_logger(__name__)
metrics = get_metrics_logger()


@dataclass
class ClaudeResponse:
    """Response from Claude API."""
    answer: str
    sources: List[str]
    token_usage: Dict[str, int]
    latency_seconds: float
    model: str
    cached: bool = False


class ClaudeIntegration:
    """Handles Claude API interactions for RAG responses."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize Claude integration.

        Args:
            api_key: Optional API key override
        """
        config = get_config()

        # API configuration
        self.model = config.claude.model
        self.max_tokens = config.claude.max_tokens
        self.temperature = config.claude.temperature
        self.stream = config.claude.stream
        self.timeout = config.claude.timeout_seconds

        # Retry configuration
        self.max_retries = config.claude.max_retries
        self.retry_delay = config.claude.retry_delay
        self.exponential_backoff = config.claude.exponential_backoff

        # Response configuration
        self.include_citations = config.claude.include_citations
        self.citation_format = config.claude.citation_format
        self.system_prompt = config.claude.system_prompt

        # Cost tracking
        self.track_usage = config.claude.track_usage
        self.warn_cost_threshold = config.claude.warn_cost_threshold
        self.total_tokens_used = {"input": 0, "output": 0}
        self.total_cost = 0.0

        # Get API key
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = os.environ.get(config.claude.api_key_env)

        if not self.api_key:
            logger.warning("No API key found, Claude integration will not work")
            self.client = None
        else:
            # Initialize Anthropic client
            self.client = Anthropic(api_key=self.api_key)
            logger.info(f"Claude integration initialized", model=self.model)

        # Response cache (simple in-memory cache)
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

    def _build_context(self, retrieval_results: List[RetrievalResult]) -> str:
        """Build context from retrieval results.

        Args:
            retrieval_results: Retrieved document chunks

        Returns:
            Formatted context string
        """
        if not retrieval_results:
            return "No relevant context found."

        context_parts = []
        sources_seen = set()

        for i, result in enumerate(retrieval_results, 1):
            # Format each chunk with source
            source_name = os.path.basename(result.source) if result.source else f"Document {i}"

            # Track unique sources
            sources_seen.add(source_name)

            # Add chunk to context
            context_parts.append(f"[{i}] From {source_name}:\n{result.text}\n")

        context = "\n".join(context_parts)

        # Add source summary
        context = f"Context from {len(sources_seen)} source(s):\n\n{context}"

        logger.debug(f"Context built", chunks=len(retrieval_results), sources=len(sources_seen))

        return context

    def _build_prompt(self, query: str, context: str) -> str:
        """Build the complete prompt for Claude.

        Args:
            query: User's question
            context: Retrieved context

        Returns:
            Complete prompt
        """
        # Use configured system prompt or default
        system_prompt = self.system_prompt or """You are a helpful assistant with access to retrieved documentation.
Answer questions based only on the provided context.
Always cite your sources using the format [Source: filename].
If the context doesn't contain enough information, clearly state this."""

        # Build user message
        user_message = f"""Context:
{context}

Question: {query}

Please provide a comprehensive answer based on the context above."""

        if self.include_citations:
            user_message += f"\nRemember to cite sources using the format: {self.citation_format}"

        return system_prompt, user_message

    def _extract_sources(self, text: str) -> List[str]:
        """Extract source citations from response text.

        Args:
            text: Response text

        Returns:
            List of cited sources
        """
        sources = []

        # Look for citations in the configured format
        import re

        # Default pattern for [Source: filename]
        pattern = r'\[Source:\s*([^\]]+)\]'

        # Adjust pattern based on citation format
        if "{filename}" in self.citation_format:
            # Create regex from format string
            escaped = re.escape(self.citation_format)
            pattern = escaped.replace(r'\{filename\}', r'([^\\]]+)')

        matches = re.findall(pattern, text, re.IGNORECASE)
        sources = list(set(matches))  # Unique sources

        return sources

    def _retry_with_backoff(self, func, *args, **kwargs):
        """Execute function with retry logic and exponential backoff.

        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            Exception: If all retries fail
        """
        last_exception = None
        delay = self.retry_delay

        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e

                if attempt < self.max_retries:
                    logger.warning(
                        f"API call failed, retrying",
                        attempt=attempt + 1,
                        max_retries=self.max_retries,
                        error=str(e)
                    )

                    time.sleep(delay)

                    if self.exponential_backoff:
                        delay *= 2  # Double the delay for next attempt

        logger.error(f"All retries exhausted", error=str(last_exception))
        raise last_exception

    @log_api_call("claude")
    def generate_response(
        self,
        query: str,
        retrieval_results: List[RetrievalResult],
        stream: Optional[bool] = None,
        use_cache: bool = True
    ) -> ClaudeResponse:
        """Generate response using Claude API.

        Args:
            query: User's question
            retrieval_results: Retrieved context chunks
            stream: Whether to stream response
            use_cache: Whether to use response cache

        Returns:
            Claude's response with metadata
        """
        if not self.client:
            logger.error("Claude client not initialized (no API key)")
            return ClaudeResponse(
                answer="Claude API is not configured. Please set your API key.",
                sources=[],
                token_usage={"input": 0, "output": 0},
                latency_seconds=0,
                model=self.model,
                cached=False
            )

        # Check cache
        cache_key = f"{query}:{len(retrieval_results)}"
        if use_cache and cache_key in self.cache:
            self.cache_hits += 1
            logger.debug("Response cache hit", query_length=len(query))
            metrics.record_success("response_cache_hit")
            cached_response = self.cache[cache_key]
            cached_response.cached = True
            return cached_response

        self.cache_misses += 1
        start_time = time.time()

        # Build context and prompt
        context = self._build_context(retrieval_results)
        system_prompt, user_message = self._build_prompt(query, context)

        # Determine if streaming
        should_stream = stream if stream is not None else self.stream

        try:
            # Make API call with retry logic
            if should_stream:
                response_text, token_usage = self._generate_streaming(system_prompt, user_message)
            else:
                response_text, token_usage = self._generate_standard(system_prompt, user_message)

            # Extract sources from response
            sources = self._extract_sources(response_text)

            # Add sources from retrieval if not cited
            if not sources and retrieval_results:
                sources = list(set(os.path.basename(r.source) for r in retrieval_results[:3]))

            # Calculate latency
            latency = time.time() - start_time

            # Track usage
            if self.track_usage:
                self._track_usage(token_usage)

            # Create response object
            response = ClaudeResponse(
                answer=response_text,
                sources=sources,
                token_usage=token_usage,
                latency_seconds=latency,
                model=self.model,
                cached=False
            )

            # Cache response
            if use_cache:
                self.cache[cache_key] = response

            # Log metrics
            logger.info(
                f"Response generated",
                query_length=len(query),
                context_chunks=len(retrieval_results),
                response_length=len(response_text),
                latency=latency,
                tokens=token_usage
            )
            metrics.record_latency("claude_response", latency * 1000)
            metrics.record_count("tokens_used", token_usage.get("total", 0))

            return response

        except Exception as e:
            logger.error(f"Failed to generate response", error=str(e))
            metrics.record_failure("claude_response", str(e))

            return ClaudeResponse(
                answer=f"Error generating response: {str(e)}",
                sources=[],
                token_usage={"input": 0, "output": 0},
                latency_seconds=time.time() - start_time,
                model=self.model,
                cached=False
            )

    def _generate_standard(
        self,
        system_prompt: str,
        user_message: str
    ) -> Tuple[str, Dict[str, int]]:
        """Generate response without streaming.

        Args:
            system_prompt: System prompt
            user_message: User message

        Returns:
            Tuple of (response text, token usage)
        """
        def api_call():
            return self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_message}
                ],
                timeout=self.timeout
            )

        # Make API call with retry
        message = self._retry_with_backoff(api_call)

        # Extract response and usage
        response_text = message.content[0].text if message.content else ""

        token_usage = {
            "input": message.usage.input_tokens if hasattr(message, 'usage') else 0,
            "output": message.usage.output_tokens if hasattr(message, 'usage') else 0,
            "total": (message.usage.input_tokens + message.usage.output_tokens) if hasattr(message, 'usage') else 0
        }

        return response_text, token_usage

    def _generate_streaming(
        self,
        system_prompt: str,
        user_message: str
    ) -> Tuple[str, Dict[str, int]]:
        """Generate response with streaming.

        Args:
            system_prompt: System prompt
            user_message: User message

        Returns:
            Tuple of (response text, token usage)
        """
        def api_call():
            return self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_message}
                ],
                timeout=self.timeout,
                stream=True
            )

        # Make API call with retry
        stream = self._retry_with_backoff(api_call)

        # Collect streamed response
        response_parts = []
        final_message = None

        for event in stream:
            if event.type == "content_block_delta":
                text = event.delta.text
                response_parts.append(text)
                # Could yield here for real-time streaming
            elif event.type == "message_stop":
                final_message = event.message

        response_text = "".join(response_parts)

        # Get token usage from final message
        if final_message and hasattr(final_message, 'usage'):
            token_usage = {
                "input": final_message.usage.input_tokens,
                "output": final_message.usage.output_tokens,
                "total": final_message.usage.input_tokens + final_message.usage.output_tokens
            }
        else:
            # Estimate if not available
            token_usage = {
                "input": len(user_message) // 4,
                "output": len(response_text) // 4,
                "total": (len(user_message) + len(response_text)) // 4
            }

        return response_text, token_usage

    def _track_usage(self, token_usage: Dict[str, int]):
        """Track token usage and costs.

        Args:
            token_usage: Token usage dictionary
        """
        self.total_tokens_used["input"] += token_usage.get("input", 0)
        self.total_tokens_used["output"] += token_usage.get("output", 0)

        # Calculate cost (example rates for Haiku)
        input_cost = token_usage.get("input", 0) * 0.00000025  # $0.25 per 1M tokens
        output_cost = token_usage.get("output", 0) * 0.00000125  # $1.25 per 1M tokens
        query_cost = input_cost + output_cost

        self.total_cost += query_cost

        # Log usage
        logger.debug(
            f"Token usage tracked",
            input_tokens=token_usage.get("input", 0),
            output_tokens=token_usage.get("output", 0),
            query_cost=query_cost,
            total_cost=self.total_cost
        )

        # Warn if exceeding threshold
        if self.total_cost > self.warn_cost_threshold:
            logger.warning(
                f"Cost threshold exceeded",
                total_cost=self.total_cost,
                threshold=self.warn_cost_threshold
            )

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics.

        Returns:
            Dictionary with usage stats
        """
        return {
            "total_tokens": self.total_tokens_used,
            "total_cost": self.total_cost,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": self.cache_hits / max(1, self.cache_hits + self.cache_misses),
            "model": self.model
        }

    def clear_cache(self):
        """Clear response cache."""
        self.cache.clear()
        logger.info("Response cache cleared")


# Singleton instance
_claude_integration: Optional[ClaudeIntegration] = None


def get_claude_integration(api_key: Optional[str] = None) -> ClaudeIntegration:
    """Get or create the global Claude integration.

    Args:
        api_key: Optional API key override

    Returns:
        Claude integration instance
    """
    global _claude_integration

    if _claude_integration is None or api_key:
        _claude_integration = ClaudeIntegration(api_key)

    return _claude_integration


if __name__ == "__main__":
    # Test Claude integration
    print("Testing Claude Integration...")

    # Set up test environment
    import os
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Warning: ANTHROPIC_API_KEY not set, using mock mode")

    # Initialize integration
    claude = get_claude_integration()

    # Create mock retrieval results
    mock_results = [
        RetrievalResult(
            chunk_id="chunk_1",
            text="RAG stands for Retrieval-Augmented Generation. It's a technique that combines information retrieval with language generation.",
            score=0.95,
            source="rag_basics.md",
            metadata={"section": "Introduction"},
            retrieval_method="hybrid",
            rank_position=1
        ),
        RetrievalResult(
            chunk_id="chunk_2",
            text="The key advantage of RAG is that it grounds language model responses in retrieved factual information.",
            score=0.87,
            source="rag_benefits.md",
            metadata={"section": "Benefits"},
            retrieval_method="vector",
            rank_position=2
        )
    ]

    # Test response generation
    query = "What is RAG and why is it useful?"

    if claude.client:
        print(f"\nGenerating response for: '{query}'")
        response = claude.generate_response(query, mock_results, stream=False)

        print(f"\nResponse:")
        print(response.answer)
        print(f"\nSources: {response.sources}")
        print(f"Tokens used: {response.token_usage}")
        print(f"Latency: {response.latency_seconds:.2f}s")

        # Get usage stats
        stats = claude.get_usage_stats()
        print(f"\nUsage Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    else:
        print("Claude client not initialized (no API key)")

    print("\nClaude integration test completed!")