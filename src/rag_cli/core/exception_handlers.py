#!/usr/bin/env python3
"""Specific exception handlers for better error recovery and debugging.

This module provides context managers and utilities for handling different
error types with appropriate recovery strategies.
"""

from typing import Callable, Any, Type, Union
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)


@contextmanager
def safe_operation(
    operation_name: str,
    expected_errors: Union[Type[Exception], tuple] = Exception,
    fallback_value: Any = None,
    log_level: str = "warning",
    reraise: bool = False,
):
    """Context manager for operations with specific error handling.

    Args:
        operation_name: Name of operation for logging
        expected_errors: Exception class(es) to catch
        fallback_value: Value to return if exception occurs
        log_level: Logging level for caught exceptions ('warning', 'error', 'debug')
        reraise: If True, re-raise after logging

    Example:
        >>> with safe_operation("file_read", expected_errors=(IOError, FileNotFoundError)):
        ...     with open("missing.txt") as f:
        ...         return f.read()
        >>> # Returns None, logs warning about failed file_read
    """
    try:
        yield
    except expected_errors as e:
        log_func = getattr(logger, log_level, logger.warning)
        log_func(f"{operation_name} failed: {type(e).__name__}: {e}")
        if reraise:
            raise
        return fallback_value


class ErrorRecoveryChain:
    """Chain multiple error recovery strategies.

    Tries recovery strategies in order until one succeeds.

    Example:
        >>> recovery = ErrorRecoveryChain()
        >>> recovery.add_strategy(load_from_cache, "cache fallback")
        >>> recovery.add_strategy(lambda: get_default_value(), "default")
        >>> result = recovery.execute(primary_operation)
    """

    def __init__(self):
        self.strategies = []

    def add_strategy(
        self,
        operation: Callable[[], Any],
        name: str,
        catch_errors: Union[Type[Exception], tuple] = Exception,
    ):
        """Add a recovery strategy.

        Args:
            operation: Callable that returns value or raises exception
            name: Name of strategy for logging
            catch_errors: Exceptions this strategy handles
        """
        self.strategies.append({
            "operation": operation,
            "name": name,
            "catch_errors": catch_errors,
        })
        return self

    def execute(self, primary_operation: Callable[[], Any], operation_name: str = "operation") -> Any:
        """Execute primary operation with fallback strategies.

        Args:
            primary_operation: Main operation to attempt
            operation_name: Name for logging

        Returns:
            Result from first successful operation

        Raises:
            Exception: If all strategies fail
        """
        # Try primary operation
        try:
            return primary_operation()
        except Exception as e:
            logger.warning(f"{operation_name} failed with {type(e).__name__}: {e}")

        # Try recovery strategies in order
        for strategy in self.strategies:
            try:
                logger.info(f"Attempting {operation_name} recovery: {strategy['name']}")
                result = strategy["operation"]()
                logger.info(f"Successfully recovered using: {strategy['name']}")
                return result
            except strategy["catch_errors"] as e:
                logger.debug(f"Strategy '{strategy['name']}' failed: {e}")
                continue

        # All strategies failed
        raise RuntimeError(
            f"All recovery strategies failed for {operation_name}. "
            f"Tried: {', '.join(s['name'] for s in self.strategies)}"
        )


class SpecificExceptionHandler:
    """Handle different exception types with specific strategies.

    Example:
        >>> handler = SpecificExceptionHandler()
        >>> handler.on(ValueError, lambda e: "invalid input")
        >>> handler.on(IOError, lambda e: "file error", reraise=True)
        >>> try:
        ...     risky_operation()
        ... except Exception as e:
        ...     result = handler.handle(e)
    """

    def __init__(self):
        self.handlers = {}
        self.default_handler = None

    def on(
        self,
        error_type: Type[Exception],
        handler: Callable[[Exception], Any],
        reraise: bool = False,
    ):
        """Register handler for specific exception type.

        Args:
            error_type: Exception class to handle
            handler: Function that receives exception and returns value
            reraise: If True, re-raise exception after handling

        Returns:
            self for chaining
        """
        self.handlers[error_type] = {
            "handler": handler,
            "reraise": reraise,
        }
        return self

    def default(self, handler: Callable[[Exception], Any], reraise: bool = False):
        """Register handler for unmatched exceptions.

        Args:
            handler: Function that receives exception
            reraise: If True, re-raise after handling

        Returns:
            self for chaining
        """
        self.default_handler = {
            "handler": handler,
            "reraise": reraise,
        }
        return self

    def handle(self, exception: Exception) -> Any:
        """Handle exception using registered handlers.

        Args:
            exception: Exception to handle

        Returns:
            Result from handler

        Raises:
            Exception: If handler has reraise=True
        """
        # Try exact match first
        handler_info = self.handlers.get(type(exception))

        # Try parent classes if no exact match
        if not handler_info:
            for error_type, info in self.handlers.items():
                if isinstance(exception, error_type):
                    handler_info = info
                    break

        # Use default if no specific handler found
        if not handler_info:
            if self.default_handler:
                handler_info = self.default_handler
            else:
                # No handler found, re-raise original exception
                raise exception

        # Call handler
        try:
            result = handler_info["handler"](exception)
            if handler_info["reraise"]:
                raise exception
            return result
        except Exception:
            if handler_info["reraise"]:
                raise exception
            raise


def validate_not_none(value: Any, field_name: str, context: str = "") -> Any:
    """Validate that value is not None.

    Args:
        value: Value to check
        field_name: Name of field for error message
        context: Additional context for error message

    Returns:
        value if valid

    Raises:
        ValueError: If value is None
    """
    if value is None:
        context_str = f" ({context})" if context else ""
        raise ValueError(f"Required field '{field_name}' is None{context_str}")
    return value


def validate_type(value: Any, expected_type: Type, field_name: str) -> Any:
    """Validate that value matches expected type.

    Args:
        value: Value to check
        expected_type: Type or tuple of types
        field_name: Name of field for error message

    Returns:
        value if valid

    Raises:
        TypeError: If type doesn't match
    """
    if not isinstance(value, expected_type):
        type_name = expected_type.__name__ if hasattr(expected_type, '__name__') else str(expected_type)
        raise TypeError(
            f"Field '{field_name}' must be {type_name}, got {type(value).__name__}"
        )
    return value


def validate_range(value: float, min_val: float, max_val: float, field_name: str) -> float:
    """Validate that numeric value is within range.

    Args:
        value: Value to check
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)
        field_name: Name of field for error message

    Returns:
        value if valid

    Raises:
        ValueError: If value outside range
    """
    if not (min_val <= value <= max_val):
        raise ValueError(
            f"Field '{field_name}' must be between {min_val} and {max_val}, got {value}"
        )
    return value
