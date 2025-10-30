#!/usr/bin/env python3
"""Singleton factory pattern for managing global component instances.

This module provides a thread-safe factory for creating and managing
singleton instances that properly handle parameter variants.
"""

import threading
from typing import TypeVar, Generic, Dict, Tuple, Any, Callable, Optional
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')


class SingletonFactory(Generic[T], ABC):
    """Generic thread-safe singleton factory with parameter support.

    Handles parameter variants properly - instances with different
    parameters are stored separately.

    Example:
        >>> class EmbeddingFactory(SingletonFactory):
        ...     def create(self, model_name="default"):
        ...         return EmbeddingGenerator(model_name)
        >>>
        >>> factory = EmbeddingFactory()
        >>> gen1 = factory.get(model_name="model-a")
        >>> gen2 = factory.get(model_name="model-a")  # Same instance
        >>> gen3 = factory.get(model_name="model-b")  # Different instance
        >>> assert gen1 is gen2 and gen1 is not gen3
    """

    def __init__(self):
        self._instances: Dict[Tuple, T] = {}
        self._lock = threading.RLock()

    @abstractmethod
    def create(self, *args, **kwargs) -> T:
        """Create a new instance. Must be implemented by subclass.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            New instance of type T
        """

    def get(self, *args, **kwargs) -> T:
        """Get or create singleton instance with given parameters.

        Thread-safe: Returns same instance for same parameters across threads.

        Args:
            *args: Positional arguments for create()
            **kwargs: Keyword arguments for create()

        Returns:
            Singleton instance
        """
        # Create hashable key from arguments
        key = self._make_key(*args, **kwargs)

        # Double-checked locking pattern
        if key not in self._instances:
            with self._lock:
                # Check again inside lock
                if key not in self._instances:
                    try:
                        self._instances[key] = self.create(*args, **kwargs)
                        logger.debug(f"Created new instance with key: {key}")
                    except Exception as e:
                        logger.error(f"Failed to create instance with key {key}: {e}")
                        raise

        return self._instances[key]

    def _make_key(self, *args, **kwargs) -> Tuple:
        """Create hashable key from function arguments.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Hashable tuple suitable for dictionary key
        """
        # Convert kwargs to sorted tuple for consistent hashing
        kwargs_tuple = tuple(sorted(kwargs.items()))
        return (args, kwargs_tuple)

    def clear(self, *args, **kwargs) -> None:
        """Remove cached instance(s).

        Args:
            *args: Positional arguments of instance to remove
            **kwargs: Keyword arguments of instance to remove

        If no arguments provided, clears all instances.
        """
        with self._lock:
            if not args and not kwargs:
                # Clear all instances
                self._instances.clear()
                logger.info("Cleared all cached instances")
            else:
                key = self._make_key(*args, **kwargs)
                if key in self._instances:
                    del self._instances[key]
                    logger.debug(f"Removed instance with key: {key}")

    def exists(self, *args, **kwargs) -> bool:
        """Check if instance exists without creating.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            True if instance exists, False otherwise
        """
        key = self._make_key(*args, **kwargs)
        return key in self._instances

    def count(self) -> int:
        """Get number of cached instances.

        Returns:
            Number of instances currently cached
        """
        return len(self._instances)

    def info(self) -> Dict[str, Any]:
        """Get information about cached instances.

        Returns:
            Dictionary with instance count and key info
        """
        return {
            "count": self.count(),
            "keys": [str(k) for k in self._instances.keys()],
        }


class ParameterizedSingletonFactory(SingletonFactory[T]):
    """Singleton factory using a factory function.

    Simpler alternative for using existing factory functions.

    Example:
        >>> def make_gen(model_name="default"):
        ...     return EmbeddingGenerator(model_name)
        >>>
        >>> factory = ParameterizedSingletonFactory(make_gen)
        >>> gen = factory.get(model_name="bert")
    """

    def __init__(self, factory_func: Callable[..., T]):
        """Initialize with factory function.

        Args:
            factory_func: Callable that creates instances
        """
        super().__init__()
        self.factory_func = factory_func

    def create(self, *args, **kwargs) -> T:
        """Create instance using factory function.

        Args:
            *args: Arguments to pass to factory function
            **kwargs: Keyword arguments to pass to factory function

        Returns:
            Instance created by factory function
        """
        return self.factory_func(*args, **kwargs)


class SingletonRegistry:
    """Registry of multiple singleton factories.

    Useful for managing multiple component types from one place.

    Example:
        >>> registry = SingletonRegistry()
        >>> registry.register("embeddings", EmbeddingFactory())
        >>> registry.register("vector_store", VectorStoreFactory())
        >>>
        >>> embeddings = registry.get("embeddings", model_name="bert")
        >>> vector_store = registry.get("vector_store", backend="faiss")
    """

    def __init__(self):
        self._factories: Dict[str, SingletonFactory] = {}
        self._lock = threading.RLock()

    def register(self, name: str, factory: SingletonFactory) -> None:
        """Register a singleton factory.

        Args:
            name: Name to register factory under
            factory: SingletonFactory instance
        """
        with self._lock:
            self._factories[name] = factory
            logger.info(f"Registered factory: {name}")

    def get(self, name: str, *args, **kwargs) -> Any:
        """Get instance from registered factory.

        Args:
            name: Name of registered factory
            *args: Arguments to pass to factory
            **kwargs: Keyword arguments to pass to factory

        Returns:
            Instance from factory

        Raises:
            KeyError: If factory not registered
        """
        if name not in self._factories:
            raise KeyError(f"Factory '{name}' not registered. Available: {list(self._factories.keys())}")

        return self._factories[name].get(*args, **kwargs)

    def clear(self, name: Optional[str] = None) -> None:
        """Clear instances from one or all factories.

        Args:
            name: Name of factory to clear. If None, clears all.
        """
        with self._lock:
            if name:
                if name in self._factories:
                    self._factories[name].clear()
                    logger.info(f"Cleared instances from factory: {name}")
            else:
                for factory in self._factories.values():
                    factory.clear()
                logger.info("Cleared all instances from all factories")

    def info(self) -> Dict[str, Any]:
        """Get information about all registered factories.

        Returns:
            Dictionary with factory names and their instance info
        """
        return {
            name: factory.info()
            for name, factory in self._factories.items()
        }
