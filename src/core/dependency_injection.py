"""Lightweight Dependency Injection container for RAG-CLI.

This module provides a simple yet powerful DI framework to replace singleton
patterns with proper dependency injection for better testability and maintainability.

Key Features:
- Simple registration and resolution
- Lifecycle management (singleton, transient, scoped)
- Factory functions and class-based providers
- Thread-safe operations
- Easy testing with mock injection
"""

import threading
from typing import TypeVar, Generic, Dict, Any, Callable, Optional, Type, Union
from enum import Enum
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')


class Lifecycle(Enum):
    """Component lifecycle strategies."""
    SINGLETON = "singleton"  # One instance per container
    TRANSIENT = "transient"  # New instance per resolution
    SCOPED = "scoped"  # One instance per scope (e.g., per request)


class Provider(Generic[T]):
    """Provider for creating and managing component instances.

    Args:
        factory: Function that creates instances of T
        lifecycle: Lifecycle strategy for this provider
    """

    def __init__(
        self,
        factory: Callable[..., T],
        lifecycle: Lifecycle = Lifecycle.SINGLETON,
    ):
        self.factory = factory
        self.lifecycle = lifecycle
        self._instance: Optional[T] = None
        self._lock = threading.RLock()

    def resolve(self, container: 'DIContainer', **kwargs) -> T:
        """Resolve and return an instance based on lifecycle strategy.

        Args:
            container: Parent container for resolving dependencies
            **kwargs: Additional arguments to pass to factory

        Returns:
            Instance of type T
        """
        if self.lifecycle == Lifecycle.SINGLETON:
            return self._get_or_create_singleton(container, **kwargs)
        elif self.lifecycle == Lifecycle.TRANSIENT:
            return self._create_transient(container, **kwargs)
        else:  # SCOPED
            # For now, scoped behaves like singleton
            # Future: implement proper scoping mechanism
            return self._get_or_create_singleton(container, **kwargs)

    def _get_or_create_singleton(self, container: 'DIContainer', **kwargs) -> T:
        """Get existing or create new singleton instance."""
        if self._instance is None:
            with self._lock:
                if self._instance is None:
                    self._instance = self._create_transient(container, **kwargs)
                    logger.debug(f"Created singleton instance: {type(self._instance).__name__}")
        return self._instance

    def _create_transient(self, container: 'DIContainer', **kwargs) -> T:
        """Create new transient instance."""
        try:
            instance = self.factory(container, **kwargs)
            logger.debug(f"Created transient instance: {type(instance).__name__}")
            return instance
        except Exception as e:
            logger.error(f"Failed to create instance with factory {self.factory}: {e}")
            raise

    def clear(self) -> None:
        """Clear cached singleton instance."""
        with self._lock:
            if self._instance is not None:
                logger.debug(f"Cleared singleton instance: {type(self._instance).__name__}")
                self._instance = None


class DIContainer:
    """Dependency Injection container for managing components.

    Example:
        >>> container = DIContainer()
        >>>
        >>> # Register with factory function
        >>> def create_config(c):
        ...     return Config(path="/etc/config.yaml")
        >>> container.register(Config, create_config, Lifecycle.SINGLETON)
        >>>
        >>> # Register with lambda
        >>> container.register(
        ...     VectorStore,
        ...     lambda c: VectorStore(c.resolve(Config)),
        ...     Lifecycle.SINGLETON
        ... )
        >>>
        >>> # Resolve dependencies
        >>> vector_store = container.resolve(VectorStore)
    """

    def __init__(self):
        self._providers: Dict[Type, Provider] = {}
        self._instances: Dict[Type, Any] = {}
        self._lock = threading.RLock()

    def register(
        self,
        interface: Type[T],
        factory: Union[Callable[[Any], T], Type[T]],
        lifecycle: Lifecycle = Lifecycle.SINGLETON,
    ) -> None:
        """Register a component with the container.

        Args:
            interface: Type to register (used as key for resolution)
            factory: Factory function or class to create instances
            lifecycle: Lifecycle strategy for this component
        """
        with self._lock:
            # If factory is a class, wrap it in a lambda
            if isinstance(factory, type):
                factory_func = lambda c: factory()
            else:
                factory_func = factory

            provider = Provider(factory_func, lifecycle)
            self._providers[interface] = provider
            logger.info(f"Registered {interface.__name__} with {lifecycle.value} lifecycle")

    def register_instance(self, interface: Type[T], instance: T) -> None:
        """Register an existing instance (always singleton).

        Args:
            interface: Type to register
            instance: Existing instance to register
        """
        with self._lock:
            self._instances[interface] = instance
            logger.info(f"Registered instance of {interface.__name__}")

    def resolve(self, interface: Type[T], **kwargs) -> T:
        """Resolve and return an instance of the requested type.

        Args:
            interface: Type to resolve
            **kwargs: Additional arguments to pass to factory

        Returns:
            Instance of requested type

        Raises:
            KeyError: If type is not registered
        """
        # Check for direct instance registration
        if interface in self._instances:
            return self._instances[interface]

        # Check for provider registration
        if interface not in self._providers:
            raise KeyError(
                f"Type '{interface.__name__}' not registered. "
                f"Available types: {[t.__name__ for t in self._providers.keys()]}"
            )

        provider = self._providers[interface]
        return provider.resolve(self, **kwargs)

    def is_registered(self, interface: Type) -> bool:
        """Check if a type is registered.

        Args:
            interface: Type to check

        Returns:
            True if registered, False otherwise
        """
        return interface in self._providers or interface in self._instances

    def clear(self, interface: Optional[Type] = None) -> None:
        """Clear cached instances.

        Args:
            interface: Type to clear. If None, clears all.
        """
        with self._lock:
            if interface:
                if interface in self._providers:
                    self._providers[interface].clear()
                    logger.info(f"Cleared instances of {interface.__name__}")
                if interface in self._instances:
                    del self._instances[interface]
                    logger.info(f"Removed instance of {interface.__name__}")
            else:
                for provider in self._providers.values():
                    provider.clear()
                self._instances.clear()
                logger.info("Cleared all cached instances")

    def info(self) -> Dict[str, Any]:
        """Get container information.

        Returns:
            Dictionary with registration info
        """
        return {
            "providers": {
                t.__name__: p.lifecycle.value
                for t, p in self._providers.items()
            },
            "instances": [t.__name__ for t in self._instances.keys()],
            "total_registered": len(self._providers) + len(self._instances),
        }


# Global container instance
_global_container: Optional[DIContainer] = None
_container_lock = threading.RLock()


def get_container() -> DIContainer:
    """Get the global DI container instance.

    Returns:
        Global DIContainer instance
    """
    global _global_container

    if _global_container is None:
        with _container_lock:
            if _global_container is None:
                _global_container = DIContainer()
                logger.info("Created global DI container")

    return _global_container


def configure_container(container: DIContainer) -> None:
    """Configure the global container with custom instance.

    Useful for testing to inject mock container.

    Args:
        container: Container instance to use as global
    """
    global _global_container

    with _container_lock:
        _global_container = container
        logger.info("Configured custom global container")


def reset_container() -> None:
    """Reset the global container (useful for testing).

    Clears all registrations and creates a fresh container.
    """
    global _global_container

    with _container_lock:
        if _global_container:
            _global_container.clear()
        _global_container = DIContainer()
        logger.info("Reset global container")


# Helper decorators for registration
def injectable(
    interface: Optional[Type] = None,
    lifecycle: Lifecycle = Lifecycle.SINGLETON
):
    """Decorator to mark a class as injectable.

    Example:
        >>> @injectable(lifecycle=Lifecycle.SINGLETON)
        ... class MyService:
        ...     def __init__(self, config: Config):
        ...         self.config = config
        >>>
        >>> # Auto-registers the class
        >>> container = get_container()
        >>> service = container.resolve(MyService)

    Args:
        interface: Type to register as (defaults to decorated class)
        lifecycle: Lifecycle strategy
    """
    def decorator(cls):
        target_interface = interface or cls
        container = get_container()

        # Create factory that auto-resolves constructor dependencies
        def factory(c):
            # For now, simple factory without auto-wiring
            # Future: implement constructor injection with annotations
            return cls()

        container.register(target_interface, factory, lifecycle)
        return cls

    return decorator
