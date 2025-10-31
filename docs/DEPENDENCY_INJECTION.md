# Dependency Injection Framework

## Overview

RAG-CLI now includes a lightweight Dependency Injection (DI) framework to replace singleton patterns with proper dependency injection for better testability and maintainability.

## Features

- **Simple registration and resolution**: Easy-to-use API for registering and resolving dependencies
- **Lifecycle management**: Support for singleton, transient, and scoped lifecycles
- **Factory functions and class-based providers**: Flexible component creation
- **Thread-safe operations**: Safe for concurrent access
- **Easy testing**: Mock injection for unit tests

## Quick Start

### Basic Usage

```python
from core.dependency_injection import DIContainer, Lifecycle

# Create container
container = DIContainer()

# Register components
def create_config():
    return Config(path="/etc/config.yaml")

container.register(Config, create_config, Lifecycle.SINGLETON)

# Resolve dependencies
config = container.resolve(Config)
```

### Using the Global Container

```python
from core.dependency_injection import get_container

# Get global container
container = get_container()

# Register and resolve
container.register(VectorStore, lambda c: VectorStore())
vector_store = container.resolve(VectorStore)
```

### Dependency Resolution

```python
# Register config
container.register(
    Config,
    lambda c: Config("/etc/config.yaml"),
    Lifecycle.SINGLETON
)

# Register service that depends on config
container.register(
    VectorStore,
    lambda c: VectorStore(c.resolve(Config)),
    Lifecycle.SINGLETON
)

# Resolve automatically resolves dependencies
vector_store = container.resolve(VectorStore)
```

## Lifecycle Strategies

### Singleton

One instance per container. Instance is created on first resolution and reused.

```python
container.register(Config, Config, Lifecycle.SINGLETON)

config1 = container.resolve(Config)
config2 = container.resolve(Config)
assert config1 is config2  # Same instance
```

### Transient

New instance created every time.

```python
container.register(Repository, Repository, Lifecycle.TRANSIENT)

repo1 = container.resolve(Repository)
repo2 = container.resolve(Repository)
assert repo1 is not repo2  # Different instances
```

### Scoped

One instance per scope (currently behaves like singleton, scoping to be implemented).

```python
container.register(Service, Service, Lifecycle.SCOPED)
```

## Advanced Usage

### Injectable Decorator

Mark classes as auto-registered:

```python
from core.dependency_injection import injectable, Lifecycle

@injectable(lifecycle=Lifecycle.SINGLETON)
class MyService:
    def __init__(self):
        self.value = "service"

# Automatically registered
container = get_container()
service = container.resolve(MyService)
```

### Instance Registration

Register existing instances directly:

```python
config = Config("/etc/config.yaml")
container.register_instance(Config, config)

resolved = container.resolve(Config)
assert resolved is config
```

### Container Management

```python
# Check if registered
if container.is_registered(Config):
    config = container.resolve(Config)

# Clear specific type
container.clear(Config)

# Clear all
container.clear()

# Get container info
info = container.info()
print(f"Registered providers: {info['providers']}")
print(f"Instances: {info['instances']}")
```

## Testing with DI

### Mock Injection

```python
import pytest
from core.dependency_injection import DIContainer

def test_service_with_mock_config():
    # Create test container
    container = DIContainer()

    # Register mock
    mock_config = MockConfig("test-value")
    container.register_instance(Config, mock_config)

    # Register service
    container.register(
        Service,
        lambda c: Service(c.resolve(Config)),
        Lifecycle.SINGLETON
    )

    # Test with mock
    service = container.resolve(Service)
    assert service.config is mock_config
```

### Container Reset

```python
from core.dependency_injection import reset_container

def test_with_clean_container():
    # Reset global container
    reset_container()

    # Configure for test
    container = get_container()
    container.register_instance(Config, test_config)

    # Run test
    ...
```

### Custom Test Container

```python
from core.dependency_injection import configure_container

def test_with_custom_container():
    # Create custom container
    test_container = DIContainer()
    test_container.register_instance(Config, test_config)

    # Use as global
    configure_container(test_container)

    # Run test
    ...
```

## Migration from Singletons

### Before (Singleton Pattern)

```python
class VectorStore:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

# Usage
vector_store = VectorStore.get_instance()
```

### After (Dependency Injection)

```python
# In setup/initialization
container = get_container()
container.register(VectorStore, VectorStore, Lifecycle.SINGLETON)

# Usage
vector_store = container.resolve(VectorStore)
```

## Best Practices

### 1. Register at Startup

Register all components in a central configuration function:

```python
def configure_dependencies():
    container = get_container()

    # Core services
    container.register(Config, lambda c: Config(), Lifecycle.SINGLETON)
    container.register(Logger, lambda c: Logger(), Lifecycle.SINGLETON)

    # Application services
    container.register(
        VectorStore,
        lambda c: VectorStore(c.resolve(Config)),
        Lifecycle.SINGLETON
    )
    container.register(
        RetrievalPipeline,
        lambda c: RetrievalPipeline(
            c.resolve(VectorStore),
            c.resolve(Config)
        ),
        Lifecycle.SINGLETON
    )
```

### 2. Use Constructor Injection

Pass dependencies through constructors:

```python
class Service:
    def __init__(self, config: Config, logger: Logger):
        self.config = config
        self.logger = logger

container.register(
    Service,
    lambda c: Service(
        c.resolve(Config),
        c.resolve(Logger)
    ),
    Lifecycle.SINGLETON
)
```

### 3. Avoid Service Locator Pattern

Don't pass the container around. Resolve dependencies at composition root:

```python
# Good
service = container.resolve(Service)
service.do_work()

# Bad - don't do this
class Service:
    def __init__(self, container):
        self.container = container

    def do_work(self):
        config = self.container.resolve(Config)  # Service locator anti-pattern
```

### 4. Use Interfaces for Abstraction

Register interfaces, not concrete implementations:

```python
from abc import ABC, abstractmethod

class IDataStore(ABC):
    @abstractmethod
    def save(self, data): pass

class FAISSDataStore(IDataStore):
    def save(self, data):
        # Implementation
        pass

container.register(IDataStore, FAISSDataStore, Lifecycle.SINGLETON)

# Depend on interface
class Service:
    def __init__(self, store: IDataStore):
        self.store = store
```

## Examples

### Complete Application Setup

```python
from core.dependency_injection import get_container, Lifecycle
from core.config import Config
from core.vector_store import VectorStore
from core.embeddings import EmbeddingGenerator
from core.retrieval_pipeline import RetrievalPipeline

def setup_application():
    container = get_container()

    # Configuration
    container.register(
        Config,
        lambda c: Config.load("config.yaml"),
        Lifecycle.SINGLETON
    )

    # Embeddings
    container.register(
        EmbeddingGenerator,
        lambda c: EmbeddingGenerator(c.resolve(Config)),
        Lifecycle.SINGLETON
    )

    # Vector Store
    container.register(
        VectorStore,
        lambda c: VectorStore(
            c.resolve(Config),
            c.resolve(EmbeddingGenerator)
        ),
        Lifecycle.SINGLETON
    )

    # Retrieval Pipeline
    container.register(
        RetrievalPipeline,
        lambda c: RetrievalPipeline(
            c.resolve(VectorStore),
            c.resolve(Config)
        ),
        Lifecycle.SINGLETON
    )

    return container

# Application entry point
def main():
    container = setup_application()

    # Resolve and use
    pipeline = container.resolve(RetrievalPipeline)
    results = pipeline.search("query")
```

### Testing Example

```python
import pytest
from core.dependency_injection import DIContainer

class TestRetrievalPipeline:
    def setup_method(self):
        """Set up test container with mocks."""
        self.container = DIContainer()

        # Mock dependencies
        self.mock_config = MockConfig()
        self.mock_vector_store = MockVectorStore()

        self.container.register_instance(Config, self.mock_config)
        self.container.register_instance(VectorStore, self.mock_vector_store)

        # Register system under test
        self.container.register(
            RetrievalPipeline,
            lambda c: RetrievalPipeline(
                c.resolve(VectorStore),
                c.resolve(Config)
            ),
            Lifecycle.SINGLETON
        )

    def test_search(self):
        """Test search with mocked dependencies."""
        pipeline = self.container.resolve(RetrievalPipeline)

        # Configure mock
        self.mock_vector_store.set_results(["doc1", "doc2"])

        # Test
        results = pipeline.search("test query")

        assert len(results) == 2
        assert "doc1" in results
```

## Performance Considerations

- **Singleton creation**: Lazy, thread-safe using double-checked locking
- **Resolution overhead**: Minimal dictionary lookup
- **Memory**: Only singleton instances are cached
- **Thread safety**: All operations are thread-safe with RLock

## Troubleshooting

### Common Issues

1. **KeyError: Type not registered**
   - Ensure type is registered before resolution
   - Check spelling and import paths

2. **Circular dependencies**
   - Use lazy initialization or factory methods
   - Refactor to break circular dependency

3. **Wrong instance returned**
   - Check lifecycle strategy
   - Ensure correct type is registered

### Debug Information

```python
# Print container state
info = container.info()
print(f"Total registered: {info['total_registered']}")
print(f"Providers: {info['providers']}")
print(f"Instances: {info['instances']}")

# Check registration
if not container.is_registered(MyService):
    print("MyService not registered!")
```

## API Reference

### DIContainer

- `register(interface, factory, lifecycle)`: Register a component
- `register_instance(interface, instance)`: Register existing instance
- `resolve(interface, **kwargs)`: Resolve and return instance
- `is_registered(interface)`: Check if type is registered
- `clear(interface=None)`: Clear cached instances
- `info()`: Get container information

### Lifecycle

- `Lifecycle.SINGLETON`: One instance per container
- `Lifecycle.TRANSIENT`: New instance per resolution
- `Lifecycle.SCOPED`: One instance per scope

### Global Functions

- `get_container()`: Get global container
- `configure_container(container)`: Set custom global container
- `reset_container()`: Reset global container

### Decorators

- `@injectable(interface, lifecycle)`: Mark class as auto-registered

## Future Enhancements

- Constructor auto-wiring with type hints
- Proper scoped lifecycle implementation
- Named registrations for multiple implementations
- Lazy resolution with proxies
- Validation of dependency graph
- Configuration-based registration

## Contributing

When adding new components to RAG-CLI, follow DI best practices:

1. Define clear dependencies in constructor
2. Register in central configuration
3. Use appropriate lifecycle
4. Add tests with mock injection

## See Also

- Test examples: `tests/test_dependency_injection.py`
- Implementation: `src/core/dependency_injection.py`
- Migration guide: `docs/MIGRATION_TO_DI.md` (to be created)
