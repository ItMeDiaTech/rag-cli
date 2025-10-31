"""Claude API integration for RAG-CLI.

This module handles response generation using Claude with streaming support,
retry logic, and proper context assembly from retrieved documents.
"""

import os
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

try:
    import anthropic
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    anthropic = None
    Anthropic = None

from core.config import get_config
from core.constants import RESPONSE_CACHE_MAX_SIZE
from core.retrieval_pipeline import RetrievalResult
from core.claude_code_adapter import get_adapter, is_claude_code_mode
from core.prompt_templates import get_prompt_manager
from core.query_classifier import QueryClassification
from monitoring.logger import get_logger, get_metrics_logger, log_api_call


logger = get_logger(__name__)
metrics = get_metrics_logger()


class CostLimitExceededError(Exception):
    """Raised when cost limit is exceeded."""


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

        # Get adapter for mode detection
        self.adapter = get_adapter()
        self.is_claude_code = is_claude_code_mode()

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

        # Prompt templates
        try:
            self.prompt_manager = get_prompt_manager()
            self.use_structured_prompts = getattr(config.claude, 'use_structured_prompts', True)
            logger.info("Structured prompt templates enabled")
        except Exception as e:
            logger.warning(f"Failed to initialize prompt manager: {e}")
            self.prompt_manager = None
            self.use_structured_prompts = False

        # Cost tracking
        self.track_usage = config.claude.track_usage
        self.warn_cost_threshold = config.claude.warn_cost_threshold
        self.max_cost_limit = config.claude.max_cost_limit
        self.enable_cost_limiting = config.claude.enable_cost_limiting
        self.total_tokens_used = {"input": 0, "output": 0}
        self.total_cost = 0.0

        # Get API key (only needed in standalone mode)
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = os.environ.get(config.claude.api_key_env)

        # Initialize client based on mode
        if self.is_claude_code:
            logger.info("Claude Code mode detected - API calls will be skipped")
            self.client = None
        elif not self.api_key:
            logger.warning("No API key found - operating in context-only mode")
            self.client = None
        elif not ANTHROPIC_AVAILABLE:
            logger.warning("Anthropic package not installed - operating in context-only mode")
            self.client = None
        else:
            # Initialize Anthropic client for standalone mode
            self.client = Anthropic(api_key=self.api_key)
            logger.info("Claude integration initialized in standalone mode", model=self.model)

        # Response cache with LRU eviction
        self.cache = {}
        self.cache_access_order = []
        self.cache_max_size = RESPONSE_CACHE_MAX_SIZE
        self.cache_hits = 0
        self.cache_misses = 0

    def _build_context(
        self,
        retrieval_results: List[RetrievalResult],
        classification: Optional['QueryClassification'] = None,
        include_metadata: bool = True
    ) -> str:
        """Build context from retrieval results with optional intent metadata.

        Args:
            retrieval_results: Retrieved document chunks
            classification: Optional query classification for metadata
            include_metadata: Whether to include confidence and authority indicators

        Returns:
            Formatted context string
        """
        if not retrieval_results:
            return "No relevant context found."

        context_parts = []
        sources_seen = set()

        # Add intent metadata if available
        if classification and include_metadata:
            intent_info = f"Query Intent: {classification.primary_intent.value} (confidence: {classification.confidence:.2f})\n"
            if classification.technical_depth:
                intent_info += f"Technical Depth: {classification.technical_depth.value}\n"
            if classification.entities:
                entities_str = ", ".join([e.name for e in classification.entities])
                intent_info += f"Detected Technologies: {entities_str}\n"
            context_parts.append(intent_info)

        for i, result in enumerate(retrieval_results, 1):
            # Format each chunk with source
            source_name = os.path.basename(result.source) if result.source else f"Document {i}"

            # Track unique sources
            sources_seen.add(source_name)

            # Determine source authority
            authority_indicator = ""
            if include_metadata and result.metadata:
                doc_type = result.metadata.get('doc_type', '').lower()
                if 'official' in doc_type:
                    authority_indicator = " [Official Documentation]"
                elif 'tutorial' in doc_type:
                    authority_indicator = " [Tutorial]"
                elif 'community' in doc_type:
                    authority_indicator = " [Community]"

            # Build chunk entry with confidence score and authority
            chunk_header = f"[{i}] From {source_name}{authority_indicator}"
            if include_metadata and result.score:
                chunk_header += f" (relevance: {result.score:.2f})"

            context_parts.append(f"{chunk_header}:\n{result.text}\n")

        context = "\n".join(context_parts)

        # Add source summary
        context = f"Context from {len(sources_seen)} source(s):\n\n{context}"

        logger.debug(
            "Context built with metadata",
            chunks=len(retrieval_results),
            sources=len(sources_seen),
            has_intent=classification is not None
        )

        return context

    def _build_prompt(self, query: str, context: str) -> str:
        """Build the complete prompt for Claude using structured templates.

        Args:
            query: User's question
            context: Retrieved context

        Returns:
            Tuple of (system_prompt, user_message)
        """
        # Use structured prompts if available
        if self.use_structured_prompts and self.prompt_manager:
            try:
                prompt_dict = self.prompt_manager.format_prompt(query, context)

                logger.debug("Using structured prompt template",
                             prompt_type=prompt_dict.get("type", "unknown"))

                return prompt_dict["system"], prompt_dict["user"]

            except Exception as e:
                logger.warning(f"Failed to use structured prompt, falling back to basic: {e}")
                # Fall through to basic prompt

        # Fallback to basic prompt (backward compatibility)
        system_prompt = self.system_prompt or """You are a helpful assistant with access to retrieved documentation.
Answer questions based only on the provided context.
Always cite your sources using the format [Source: filename].
If the context doesn't contain enough information, clearly state this."""

        # Build user message
        user_message = """Context:
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
                        "API call failed, retrying",
                        attempt=attempt + 1,
                        max_retries=self.max_retries,
                        error=str(e)
                    )

                    time.sleep(delay)

                    if self.exponential_backoff:
                        delay *= 2  # Double the delay for next attempt

        logger.error("All retries exhausted", error=str(last_exception))
        raise last_exception

    @log_api_call("claude")
    def generate_response(
        self,
        query: str,
        retrieval_results: List[RetrievalResult],
        stream: Optional[bool] = None,
        use_cache: bool = True,
        classification: Optional['QueryClassification'] = None
    ) -> ClaudeResponse:
        """Generate response using Claude API with optional intent metadata.

        Args:
            query: User's question
            retrieval_results: Retrieved context chunks
            stream: Whether to stream response
            use_cache: Whether to use response cache
            classification: Optional query classification for context metadata

        Returns:
            Claude's response with metadata
        """
        # Check if we're in Claude Code mode
        if self.is_claude_code:
            logger.debug("Claude Code mode - returning formatted context instead of API call")

            # Format context for Claude Code
            context_response = self.adapter.format_context_for_claude(
                documents=[{"content": r.text, "source": r.source, "score": r.score}
                           for r in retrieval_results],
                query=query
            )

            return ClaudeResponse(
                answer=context_response.context,
                sources=context_response.sources,
                token_usage={"input": 0, "output": 0},
                latency_seconds=0,
                model="claude-code-internal",
                cached=False
            )

        # Check if client is available for standalone mode
        if not self.client:
            logger.warning("No Claude client available - returning context only")

            # Fall back to context-only response
            context = self._build_context(retrieval_results, classification=classification)
            return ClaudeResponse(
                answer=f"### Retrieved Context\n\n{context}\n\n### Note\nClaude API is not configured. The above context has been retrieved for your query: {query}",
                sources=[os.path.basename(r.source) for r in retrieval_results if r.source],
                token_usage={"input": 0, "output": 0},
                latency_seconds=0,
                model=self.model,
                cached=False
            )

        # Check cache
        cache_key = f"{query}:{len(retrieval_results)}"
        if use_cache and cache_key in self.cache:
            self.cache_hits += 1
            # Update LRU order
            if cache_key in self.cache_access_order:
                self.cache_access_order.remove(cache_key)
            self.cache_access_order.append(cache_key)

            logger.debug("Response cache hit", query_length=len(query))
            metrics.record_success("response_cache_hit")
            cached_response = self.cache[cache_key]
            cached_response.cached = True
            return cached_response

        self.cache_misses += 1

        # Check cost limit before making API call
        if self.enable_cost_limiting and self.total_cost >= self.max_cost_limit:
            error_msg = f"Cost limit exceeded: ${self.total_cost:.2f} / ${self.max_cost_limit:.2f}"
            logger.error(error_msg)
            raise CostLimitExceededError(error_msg)

        start_time = time.time()

        # Build context and prompt with optional classification metadata
        context = self._build_context(retrieval_results, classification=classification)
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

            # Cache response with LRU eviction
            if use_cache:
                # Evict if at capacity
                if len(self.cache) >= self.cache_max_size and cache_key not in self.cache:
                    if self.cache_access_order:
                        lru_key = self.cache_access_order.pop(0)
                        if lru_key in self.cache:
                            del self.cache[lru_key]
                        logger.debug("Evicted from response cache", key=lru_key[:50])

                self.cache[cache_key] = response

                # Update LRU order
                if cache_key in self.cache_access_order:
                    self.cache_access_order.remove(cache_key)
                self.cache_access_order.append(cache_key)

            # Log metrics
            logger.info(
                "Response generated",
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
            logger.error("Failed to generate response", error=str(e))
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
            "Token usage tracked",
            input_tokens=token_usage.get("input", 0),
            output_tokens=token_usage.get("output", 0),
            query_cost=query_cost,
            total_cost=self.total_cost
        )

        # Warn if exceeding threshold
        if self.total_cost > self.warn_cost_threshold:
            logger.warning(
                "Cost threshold exceeded",
                total_cost=self.total_cost,
                threshold=self.warn_cost_threshold,
                remaining=self.max_cost_limit - self.total_cost
            )

        # Check if approaching hard limit
        if self.enable_cost_limiting:
            remaining_budget = self.max_cost_limit - self.total_cost
            if remaining_budget < 0.5:  # Less than $0.50 remaining
                logger.warning(
                    "Approaching cost limit",
                    total_cost=self.total_cost,
                    limit=self.max_cost_limit,
                    remaining=remaining_budget
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
        self.cache_access_order.clear()
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


# Alias for backward compatibility
class ClaudeAssistant(ClaudeIntegration):
    """Backward compatibility alias for ClaudeIntegration."""


if __name__ == "__main__":
    # Test Claude integration
    print("Testing Claude Integration...")

    # Set up test environment
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

        print("\nResponse:")
        print(response.answer)
        print(f"\nSources: {response.sources}")
        print(f"Tokens used: {response.token_usage}")
        print(f"Latency: {response.latency_seconds:.2f}s")

        # Get usage stats
        stats = claude.get_usage_stats()
        print("\nUsage Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    else:
        print("Claude client not initialized (no API key)")

    print("\nClaude integration test completed!")
