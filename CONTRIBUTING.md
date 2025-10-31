# Contributing to RAG-CLI

Thank you for your interest in contributing to RAG-CLI! This document provides guidelines and instructions for contributing.

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on the code, not the person
- Help others learn and grow

## Getting Started

### Fork and Clone
```bash
git clone https://github.com/ItMeDiaTech/rag-cli.git
cd rag-cli
```

### Development Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install in development mode
pip install -e .
pip install -r requirements.txt

# Install development tools
pip install black isort flake8 mypy pytest pytest-cov
```

## Development Workflow

### 1. Create Feature Branch
```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes
- Follow the code style (see below)
- Add tests for new functionality
- Update documentation

### 3. Test
```bash
# Run tests
pytest tests/

# Run linters
black src/
isort src/
flake8 src/
mypy src/
```

### 4. Commit
```bash
git add .
git commit -m "type: brief description

Longer description if needed.
- Point 1
- Point 2
"
```

### Commit Types
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation
- `test:` - Test addition
- `refactor:` - Code improvement
- `perf:` - Performance optimization
- `chore:` - Maintenance

### 5. Push and Create PR
```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

## Code Style

### Python
- Use Black for formatting (line length: 120)
- Use isort for imports
- Follow PEP 8 for naming conventions
- Add docstrings to all functions

```python
def example_function(param: str) -> int:
    """Brief description.

    Longer description if needed.

    Args:
        param: Parameter description

    Returns:
        Return value description
    """
    return 0
```

### Pre-commit Hook
```bash
# Install pre-commit
pip install pre-commit

# Setup hook
pre-commit install

# Run manually
pre-commit run --all-files
```

## Testing

### Unit Tests
```bash
pytest tests/unit/ -v
```

### Integration Tests
```bash
pytest tests/integration/ -v
```

### Coverage
```bash
pytest --cov=src --cov-report=html
```

### Test Requirements
- All new features need tests
- Aim for >80% coverage
- Test both success and failure cases

## Documentation

### README Updates
- Keep README.md current
- Add examples for new features
- Update table of contents

### Code Comments
- Explain WHY, not WHAT
- Keep comments concise
- Update when code changes

### API Documentation
- Document public APIs
- Include examples
- List exceptions

## Architecture Guidelines

### Core Components
- Each component in `src/core/` should be independent
- Minimize cross-module dependencies
- Use dependency injection

### Plugin Components
- Commands in `src/plugin/commands/`
- Hooks in `src/plugin/hooks/`
- Skills in `src/plugin/skills/`

### Multi-Agent System
- Agents inherit from base Agent class
- Use agent communication hub
- Implement proper error handling

## Performance Considerations

- Target latency: <100ms for vector search
- Target end-to-end: <5 seconds
- Profile code for bottlenecks
- Use async for I/O operations
- Cache expensive computations

## Security Considerations

- Never commit API keys or secrets
- Use environment variables
- Validate user input
- Handle exceptions gracefully
- Review dependency updates

## Pull Request Process

1. **Before submitting**:
   - Run all tests: `pytest`
   - Check code style: `black`, `isort`, `flake8`
   - Update CHANGELOG.md
   - Update documentation

2. **PR Description**:
   - Describe changes
   - Explain motivation
   - Reference related issues
   - Include testing notes

3. **Review Process**:
   - Maintain responsiveness
   - Be open to feedback
   - Make requested changes
   - Squash commits if asked

4. **After Merge**:
   - Delete feature branch
   - Close related issues
   - Announce in discussions

## Areas for Contribution

### High Priority
- Bug fixes and security issues
- Performance improvements
- Documentation enhancements
- Test coverage

### Medium Priority
- New features
- CLI improvements
- Monitoring enhancements

### Future Features
- API server for remote access
- Database-backed vector store
- Custom agent creation
- Advanced caching strategies

## Reporting Issues

### Bug Reports
Include:
- Python version
- Installation method
- Steps to reproduce
- Expected vs actual behavior
- Logs and error messages

### Feature Requests
Include:
- Use case description
- Proposed solution
- Alternatives considered
- Expected benefits

## Questions and Discussions

- GitHub Issues: For bugs and features
- GitHub Discussions: For questions
- Documentation: Check before asking

## Release Process

Maintainers follow:
- Semantic versioning (MAJOR.MINOR.PATCH)
- CHANGELOG.md updates
- GitHub releases with notes
- PyPI package updates

## License

By contributing, you agree your code is licensed under the MIT License.

## Thank You!

Your contributions help make RAG-CLI better for everyone. We appreciate your effort!

---

For questions, open a GitHub Discussion or Issue.
Happy contributing! [LAUNCH]
