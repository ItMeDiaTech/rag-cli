"""Tests for dependency injection framework."""

import pytest
from rag_cli.core.dependency_injection import (
    DIContainer,
    Provider,
    Lifecycle,
    get_container,
    configure_container,
    reset_container,
    injectable,
)


# Test classes
class MockConfig:
    """Mock configuration class."""
    def __init__(self, value: str = "default"):
        self.value = value


class MockService:
    """Mock service class."""
    def __init__(self, config: MockConfig):
        self.config = config

    def get_value(self) -> str:
        return self.config.value


class MockRepository:
    """Mock repository class."""
    def __init__(self):
        self.data = []

    def add(self, item):
        self.data.append(item)


class TestProvider:
    """Tests for Provider class."""

    def test_singleton_lifecycle(self):
        """Test singleton provider returns same instance."""
        container = DIContainer()
        factory = lambda c: MockConfig("test")
        provider = Provider(factory, Lifecycle.SINGLETON)

        # Resolve twice
        instance1 = provider.resolve(container)
        instance2 = provider.resolve(container)

        # Should be same instance
        assert instance1 is instance2
        assert instance1.value == "test"

    def test_transient_lifecycle(self):
        """Test transient provider returns new instances."""
        container = DIContainer()
        factory = lambda c: MockConfig("test")
        provider = Provider(factory, Lifecycle.TRANSIENT)

        # Resolve twice
        instance1 = provider.resolve(container)
        instance2 = provider.resolve(container)

        # Should be different instances
        assert instance1 is not instance2
        assert instance1.value == instance2.value

    def test_provider_clear(self):
        """Test clearing singleton instance."""
        container = DIContainer()
        factory = lambda c: MockConfig("test")
        provider = Provider(factory, Lifecycle.SINGLETON)

        # Get instance
        instance1 = provider.resolve(container)

        # Clear and get again
        provider.clear()
        instance2 = provider.resolve(container)

        # Should be different instances
        assert instance1 is not instance2

    def test_scoped_lifecycle(self):
        """Test scoped provider (currently same as singleton)."""
        container = DIContainer()
        factory = lambda c: MockConfig("test")
        provider = Provider(factory, Lifecycle.SCOPED)

        instance1 = provider.resolve(container)
        instance2 = provider.resolve(container)

        # Currently behaves like singleton
        assert instance1 is instance2


class TestDIContainer:
    """Tests for DIContainer class."""

    def setup_method(self):
        """Set up test container."""
        self.container = DIContainer()

    def test_register_and_resolve(self):
        """Test basic registration and resolution."""
        # Register config
        self.container.register(
            MockConfig,
            lambda c: MockConfig("configured"),
            Lifecycle.SINGLETON
        )

        # Resolve
        config = self.container.resolve(MockConfig)

        assert isinstance(config, MockConfig)
        assert config.value == "configured"

    def test_register_with_class(self):
        """Test registration with class (not factory)."""
        self.container.register(MockConfig, MockConfig, Lifecycle.SINGLETON)

        config = self.container.resolve(MockConfig)

        assert isinstance(config, MockConfig)
        assert config.value == "default"

    def test_register_instance(self):
        """Test registering existing instance."""
        instance = MockConfig("existing")
        self.container.register_instance(MockConfig, instance)

        resolved = self.container.resolve(MockConfig)

        assert resolved is instance
        assert resolved.value == "existing"

    def test_dependency_resolution(self):
        """Test resolving dependencies between components."""
        # Register config
        self.container.register(
            MockConfig,
            lambda c: MockConfig("test-config"),
            Lifecycle.SINGLETON
        )

        # Register service that depends on config
        self.container.register(
            MockService,
            lambda c: MockService(c.resolve(MockConfig)),
            Lifecycle.SINGLETON
        )

        # Resolve service
        service = self.container.resolve(MockService)

        assert isinstance(service, MockService)
        assert service.get_value() == "test-config"

    def test_resolve_unregistered_type(self):
        """Test resolving unregistered type raises error."""
        with pytest.raises(KeyError) as exc_info:
            self.container.resolve(MockRepository)

        assert "MockRepository" in str(exc_info.value)
        assert "not registered" in str(exc_info.value)

    def test_is_registered(self):
        """Test checking if type is registered."""
        assert not self.container.is_registered(MockConfig)

        self.container.register(MockConfig, MockConfig, Lifecycle.SINGLETON)

        assert self.container.is_registered(MockConfig)

    def test_clear_specific_type(self):
        """Test clearing specific type."""
        self.container.register(
            MockConfig,
            lambda c: MockConfig("test"),
            Lifecycle.SINGLETON
        )

        # Resolve to create instance
        config1 = self.container.resolve(MockConfig)

        # Clear
        self.container.clear(MockConfig)

        # Resolve again should create new instance
        config2 = self.container.resolve(MockConfig)

        assert config1 is not config2

    def test_clear_all(self):
        """Test clearing all types."""
        self.container.register(MockConfig, MockConfig, Lifecycle.SINGLETON)
        self.container.register(MockRepository, MockRepository, Lifecycle.SINGLETON)

        # Resolve both
        config = self.container.resolve(MockConfig)
        repo = self.container.resolve(MockRepository)

        # Clear all
        self.container.clear()

        # Resolve again should create new instances
        new_config = self.container.resolve(MockConfig)
        new_repo = self.container.resolve(MockRepository)

        assert config is not new_config
        assert repo is not new_repo

    def test_container_info(self):
        """Test getting container information."""
        self.container.register(MockConfig, MockConfig, Lifecycle.SINGLETON)
        self.container.register(MockRepository, MockRepository, Lifecycle.TRANSIENT)
        self.container.register_instance(MockService, MockService(MockConfig()))

        info = self.container.info()

        assert info["total_registered"] == 3
        assert "MockConfig" in info["providers"]
        assert info["providers"]["MockConfig"] == "singleton"
        assert "MockRepository" in info["providers"]
        assert info["providers"]["MockRepository"] == "transient"
        assert "MockService" in info["instances"]

    def test_singleton_lifecycle_in_container(self):
        """Test singleton lifecycle through container."""
        self.container.register(MockConfig, MockConfig, Lifecycle.SINGLETON)

        config1 = self.container.resolve(MockConfig)
        config2 = self.container.resolve(MockConfig)

        assert config1 is config2

    def test_transient_lifecycle_in_container(self):
        """Test transient lifecycle through container."""
        self.container.register(MockConfig, MockConfig, Lifecycle.TRANSIENT)

        config1 = self.container.resolve(MockConfig)
        config2 = self.container.resolve(MockConfig)

        assert config1 is not config2


class TestGlobalContainer:
    """Tests for global container functions."""

    def setup_method(self):
        """Reset global container before each test."""
        reset_container()

    def test_get_container(self):
        """Test getting global container."""
        container = get_container()

        assert isinstance(container, DIContainer)

        # Getting again should return same instance
        container2 = get_container()
        assert container is container2

    def test_configure_container(self):
        """Test configuring custom container."""
        custom_container = DIContainer()
        custom_container.register_instance(MockConfig, MockConfig("custom"))

        configure_container(custom_container)

        container = get_container()
        config = container.resolve(MockConfig)

        assert config.value == "custom"

    def test_reset_container(self):
        """Test resetting global container."""
        container1 = get_container()
        container1.register(MockConfig, MockConfig, Lifecycle.SINGLETON)

        reset_container()

        container2 = get_container()

        # Should be different instance
        assert container1 is not container2

        # Previous registration should be gone
        assert not container2.is_registered(MockConfig)


class TestInjectableDecorator:
    """Tests for injectable decorator."""

    def setup_method(self):
        """Reset container before each test."""
        reset_container()

    def test_injectable_decorator(self):
        """Test injectable decorator registers class."""
        @injectable(lifecycle=Lifecycle.SINGLETON)
        class DecoratedService:
            def __init__(self):
                self.value = "decorated"

        container = get_container()

        # Should be registered
        assert container.is_registered(DecoratedService)

        # Should be resolvable
        service = container.resolve(DecoratedService)
        assert isinstance(service, DecoratedService)
        assert service.value == "decorated"

    def test_injectable_with_custom_interface(self):
        """Test injectable with custom interface type."""
        class ServiceInterface:
            pass

        @injectable(interface=ServiceInterface, lifecycle=Lifecycle.SINGLETON)
        class ConcreteService(ServiceInterface):
            def __init__(self):
                self.value = "concrete"

        container = get_container()

        # Should be registered under interface
        assert container.is_registered(ServiceInterface)

        service = container.resolve(ServiceInterface)
        assert isinstance(service, ConcreteService)
        assert service.value == "concrete"

    def test_injectable_singleton(self):
        """Test injectable creates singletons by default."""
        @injectable()
        class SingletonService:
            pass

        container = get_container()

        service1 = container.resolve(SingletonService)
        service2 = container.resolve(SingletonService)

        assert service1 is service2

    def test_injectable_transient(self):
        """Test injectable with transient lifecycle."""
        @injectable(lifecycle=Lifecycle.TRANSIENT)
        class TransientService:
            pass

        container = get_container()

        service1 = container.resolve(TransientService)
        service2 = container.resolve(TransientService)

        assert service1 is not service2


class TestThreadSafety:
    """Tests for thread safety of DI container."""

    def test_concurrent_resolution(self):
        """Test concurrent resolution from multiple threads."""
        import threading
        import time

        container = DIContainer()
        results = []
        errors = []

        # Register slow-creating singleton
        def slow_factory(c):
            time.sleep(0.01)  # Simulate slow creation
            return MockConfig("slow")

        container.register(MockConfig, slow_factory, Lifecycle.SINGLETON)

        def resolve_in_thread():
            try:
                config = container.resolve(MockConfig)
                results.append(config)
            except Exception as e:
                errors.append(e)

        # Create multiple threads
        threads = [threading.Thread(target=resolve_in_thread) for _ in range(10)]

        # Start all threads
        for t in threads:
            t.start()

        # Wait for all to complete
        for t in threads:
            t.join()

        # Should have no errors
        assert len(errors) == 0

        # All should get same instance (singleton)
        assert len(results) == 10
        first_instance = results[0]
        assert all(r is first_instance for r in results)

    def test_concurrent_registration(self):
        """Test concurrent registration from multiple threads."""
        import threading

        container = DIContainer()
        errors = []

        # Create unique types for each thread
        service_types = [type(f"Service{i}", (), {}) for i in range(10)]

        def register_in_thread(index):
            try:
                # Each thread registers different type
                container.register(
                    service_types[index],
                    lambda c: f"service-{index}",
                    Lifecycle.SINGLETON
                )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=register_in_thread, args=(i,)) for i in range(10)]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        # Should have no errors
        assert len(errors) == 0


class TestErrorHandling:
    """Tests for error handling in DI container."""

    def test_factory_exception(self):
        """Test handling of factory exceptions."""
        container = DIContainer()

        def failing_factory(c):
            raise ValueError("Factory failed!")

        container.register(MockConfig, failing_factory, Lifecycle.SINGLETON)

        with pytest.raises(ValueError) as exc_info:
            container.resolve(MockConfig)

        assert "Factory failed!" in str(exc_info.value)

    def test_dependency_resolution_failure(self):
        """Test handling of dependency resolution failure."""
        container = DIContainer()

        # Register service that depends on unregistered config
        container.register(
            MockService,
            lambda c: MockService(c.resolve(MockConfig)),
            Lifecycle.SINGLETON
        )

        with pytest.raises(KeyError):
            container.resolve(MockService)
