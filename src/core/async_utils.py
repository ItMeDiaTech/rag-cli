#!/usr/bin/env python3
"""Utilities for safe async execution in mixed sync/async contexts.

This module provides helpers for running async code in environments where
an event loop may already be running (like Claude Code hooks).
"""

import asyncio
import concurrent.futures
from typing import TypeVar, Coroutine, Any, Optional
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')


def safe_asyncio_run(coro: Coroutine[Any, Any, T], timeout: Optional[float] = None) -> T:
    """Safely run async code in any context (sync or async).

    This function detects whether an event loop is already running and adapts:
    - If no loop is running: uses asyncio.run() directly
    - If loop is running: uses a thread pool executor to run in a separate thread

    This is essential for hooks in Claude Code which may run in async contexts.

    Args:
        coro: Coroutine to execute
        timeout: Optional timeout in seconds

    Returns:
        Result of coroutine execution

    Raises:
        TimeoutError: If timeout exceeded
        Exception: Any exception raised by the coroutine

    Example:
        >>> async def fetch_data():
        ...     await asyncio.sleep(0.1)
        ...     return "data"
        >>> result = safe_asyncio_run(fetch_data())
        >>> print(result)  # "data"
    """
    try:
        # Check if there's a running event loop in current thread
        asyncio.get_running_loop()
    except RuntimeError:
        # No running loop, safe to use asyncio.run()
        try:
            return asyncio.run(coro, debug=False)
        except TimeoutError:
            logger.error("Asyncio operation timed out")
            raise

    # Event loop is already running, execute in separate thread
    logger.debug("Event loop detected in current thread, using ThreadPoolExecutor")

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="async_") as executor:
            # Create new event loop in the executor thread
            future = executor.submit(_run_in_new_loop, coro, timeout)
            return future.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        logger.error("Async operation exceeded timeout")
        raise TimeoutError(f"Async operation exceeded timeout of {timeout} seconds")
    except Exception as e:
        logger.error(f"Failed to execute async operation: {e}", exc_info=True)
        raise


def _run_in_new_loop(coro: Coroutine[Any, Any, T], timeout: Optional[float] = None) -> T:
    """Run a coroutine in a new event loop (meant for executor threads).

    Args:
        coro: Coroutine to execute
        timeout: Optional timeout in seconds

    Returns:
        Result of coroutine
    """
    try:
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)

        try:
            if timeout:
                return new_loop.run_until_complete(asyncio.wait_for(coro, timeout=timeout))
            else:
                return new_loop.run_until_complete(coro)
        finally:
            new_loop.close()
    except asyncio.TimeoutError:
        raise TimeoutError(f"Async operation exceeded timeout of {timeout} seconds")


def is_event_loop_running() -> bool:
    """Check if an event loop is currently running in this thread.

    Returns:
        True if event loop is running, False otherwise

    Example:
        >>> is_event_loop_running()
        False
    """
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False


class AsyncIOAdapter:
    """Adapter class for running async functions in sync contexts.

    This class provides a clean API for code that needs to call async functions
    but may be running in a synchronous context (like hooks).

    Example:
        >>> async def get_data():
        ...     await asyncio.sleep(0.1)
        ...     return "result"
        >>> adapter = AsyncIOAdapter()
        >>> result = adapter.run(get_data())
    """

    def __init__(self, timeout: Optional[float] = None):
        """Initialize adapter with optional timeout.

        Args:
            timeout: Default timeout for all operations in seconds
        """
        self.timeout = timeout

    def run(self, coro: Coroutine[Any, Any, T], timeout: Optional[float] = None) -> T:
        """Run a coroutine and return the result.

        Args:
            coro: Coroutine to execute
            timeout: Override default timeout for this operation

        Returns:
            Result of coroutine
        """
        actual_timeout = timeout if timeout is not None else self.timeout
        return safe_asyncio_run(coro, timeout=actual_timeout)

    def run_gather(self, *coros: Coroutine[Any, Any, Any], timeout: Optional[float] = None) -> list:
        """Run multiple coroutines concurrently and wait for all to complete.

        Args:
            *coros: Coroutines to execute
            timeout: Override default timeout

        Returns:
            List of results in same order as coroutines
        """
        actual_timeout = timeout if timeout is not None else self.timeout
        return safe_asyncio_run(asyncio.gather(*coros), timeout=actual_timeout)


# Global adapter instance for convenient access
_async_adapter = AsyncIOAdapter()


def run_async(coro: Coroutine[Any, Any, T], timeout: Optional[float] = None) -> T:
    """Module-level convenience function for safe async execution.

    This is the recommended way to call async code from sync contexts:

    Args:
        coro: Coroutine to execute
        timeout: Optional timeout in seconds

    Returns:
        Result of coroutine

    Example:
        >>> from src.core.async_utils import run_async
        >>> async def main():
        ...     return "hello"
        >>> result = run_async(main())
    """
    return _async_adapter.run(coro, timeout=timeout)
