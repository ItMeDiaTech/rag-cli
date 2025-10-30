"""
Documenter Agent - Specialized in documentation generation and maintenance
"""

import json
import re
from typing import Any, Dict, List

from ..core.agent import Agent, AgentConfig

class DocumenterAgent(Agent):
    """Agent specialized in documentation creation and maintenance"""

    def __init__(
        self,
        config: AgentConfig,
        claude_cli=None,
        memory_manager=None,
        message_bus=None
    ):
        config.capabilities = [
            "api_documentation", "code_comments", "readme_generation",
            "technical_writing", "user_guides", "docstring_generation",
            "architecture_documentation", "tutorial_creation", "changelog_generation", "document", "write", "describe"
        ]

        super().__init__(config, claude_cli, memory_manager, message_bus)

        self.doc_formats = ['Markdown', 'reStructuredText', 'HTML', 'DocString', 'OpenAPI']
        self.doc_sections = [
            'Overview', 'Installation', 'Usage', 'API Reference',
            'Configuration', 'Examples', 'Troubleshooting', 'Contributing'
        ]

    async def _execute_task(
        self,
        task_id: str,
        task: Dict[str, Any],
        context: Dict[str, Any],
        memories: List[Dict]
    ) -> Any:
        """Execute documenter-specific tasks"""

        task_type = task.get('type', '').lower()

        if 'document' in task_type or 'docs' in task_type:
            return await self._generate_documentation(
                task_id, task, context, memories
            )
        if 'readme' in task_type:
            return await self._generate_readme(task_id, task, context, memories)
        if 'api' in task_type:
            return await self._generate_api_docs(task_id, task, context, memories)
        if 'comment' in task_type:
            return await self._add_code_comments(
                task_id, task, context, memories
            )
        return await self._generate_documentation(
            task_id, task, context, memories
        )

    async def _generate_documentation(
        self,
        task_id: str,
        task: Dict[str, Any],
        _context: Dict[str, Any],
        _memories: List[Dict]
    ) -> Dict[str, Any]:
        """Generate comprehensive documentation"""

        self.logger.info("[%s] Generating documentation", task_id)

        # Get code from context
        code = ''
        if _context.get('results', {}).get('Developer'):
            dev_result = _context['results']['Developer']
            if isinstance(dev_result, dict) and 'code' in dev_result:
                code_blocks = dev_result['code']
                if code_blocks and isinstance(code_blocks, list):
                    code = '\n'.join([block.get('content', '') for block in code_blocks])

        if not code:
            code = task.get('code', '')

        architecture = _context.get('results', {}).get('Architect', {})

        prompt = f"""
Generate comprehensive documentation for:

Code:
```
{code}
```

Architecture:
{json.dumps(architecture, indent=2) if architecture else 'Not provided'}

Include:
1. Project overview and purpose
2. Installation instructions
3. Usage examples
4. API documentation
5. Configuration options
6. Best practices
7. Troubleshooting guide
8. Contributing guidelines
"""

        if self.claude_cli:
            response = await self.claude_cli.complete(
                prompt,
                max_tokens=self.config.max_tokens,
                temperature=0.6,
                system=(
                    "You are an expert technical writer. Create clear, "
                    "comprehensive, and user-friendly documentation."
                )
            )

            if response.success:
                return self._parse_documentation_response(response.content)
            raise Exception(
                f"Documentation generation failed: {response.error}"
            )
        return self._mock_documentation_response()

    async def _generate_readme(
        self,
        task_id: str,
        task: Dict[str, Any],
        _context: Dict[str, Any],
        _memories: List[Dict]
    ) -> Dict[str, Any]:
        """Generate README file"""

        self.logger.info("[%s] Generating README", task_id)

        project_info = task.get('project_info', {})

        prompt = f"""
Generate a comprehensive README.md file for:

Project: {project_info.get('name', 'Project Name')}
Description: {project_info.get('description', 'Project description')}
Features: {json.dumps(project_info.get('features', []), indent=2)}

Include:
- Project title and badges
- Description
- Features
- Installation
- Quick start
- Usage examples
- API overview
- Configuration
- Contributing
- License
"""

        if self.claude_cli:
            response = await self.claude_cli.complete(prompt, temperature=0.6)

            if response.success:
                return {
                    'format': 'markdown',
                    'content': response.content,
                    'sections': self._extract_sections(response.content),
                    'word_count': len(response.content.split())
                }
            raise Exception(f"README generation failed: {response.error}")
        return {
            'format': 'markdown',
            'content': self._generate_mock_readme(),
            'sections': [
                'Overview', 'Installation', 'Usage', 'API', 'Contributing'
            ],
            'word_count': 500
        }

    async def _generate_api_docs(
        self,
        task_id: str,
        task: Dict[str, Any],
        _context: Dict[str, Any],
        _memories: List[Dict]
    ) -> Dict[str, Any]:
        """Generate API documentation"""

        self.logger.info("[%s] Generating API documentation", task_id)

        endpoints = task.get('endpoints', [])
        if not endpoints and _context.get('results', {}).get('Architect'):
            arch_result = _context['results']['Architect']
            if isinstance(arch_result, dict):
                endpoints = arch_result.get('endpoints', [])

        prompt = f"""
Generate API documentation for:

Endpoints:
{json.dumps(endpoints, indent=2)}

For each endpoint, document:
1. Purpose and description
2. HTTP method and path
3. Request parameters (query, path, body)
4. Request/response examples
5. Status codes
6. Error responses
7. Rate limiting
8. Authentication requirements
"""

        if self.claude_cli:
            response = await self.claude_cli.complete(prompt, temperature=0.5)

            if response.success:
                return self._parse_api_documentation(response.content)
            raise Exception(f"API documentation failed: {response.error}")
        return {
            'format': 'OpenAPI',
            'endpoints_documented': len(endpoints),
            'content': '# API Documentation\n\n## Endpoints\n\n...',
            'examples_included': True
        }

    async def _add_code_comments(
        self,
        task_id: str,
        task: Dict[str, Any],
        _context: Dict[str, Any],
        _memories: List[Dict]
    ) -> Dict[str, Any]:
        """Add comments to code"""

        self.logger.info("[%s] Adding code comments", task_id)

        code = task.get('code', '')
        language = task.get('language', 'python')

        prompt = f"""
Add comprehensive comments to this code:

```{language}
{code}
```

Add:
1. File/module docstring
2. Class docstrings
3. Function/method docstrings
4. Inline comments for complex logic
5. TODO comments where improvements are needed
6. Type hints where applicable
"""

        if self.claude_cli:
            response = await self.claude_cli.complete(prompt, temperature=0.4)

            if response.success:
                return {
                    'commented_code': self._extract_code(response.content),
                    'comments_added': self._count_comments(response.content),
                    'docstrings_added': self._count_docstrings(response.content)
                }
            raise Exception(f"Comment generation failed: {response.error}")
        return {
            'commented_code': f'"""\nModule documentation\n"""\n\n{code}',
            'comments_added': 10,
            'docstrings_added': 5
        }

    def _parse_documentation_response(self, content: str) -> Dict[str, Any]:
        """Parse documentation response"""

        sections = self._extract_sections(content)

        # Calculate metrics
        word_count = len(content.split())
        code_examples = len(re.findall(r'```', content)) // 2

        return {
            'format': 'markdown',
            'content': content,
            'sections': sections,
            'word_count': word_count,
            'code_examples': code_examples,
            'has_toc': (
                '## Table of Contents' in content or '## Contents' in content
            ),
            'has_examples': code_examples > 0,
            'completeness_score': min(
                100, len(sections) * 12.5
            )  # 8 sections = 100%
        }

    def _extract_sections(self, content: str) -> List[str]:
        """Extract section headers from markdown"""

        sections = []

        # Find markdown headers
        headers = re.findall(r'^#{1,3}\s+(.+)$', content, re.MULTILINE)

        for header in headers:
            # Clean header text
            clean_header = header.strip('#').strip()
            if clean_header and clean_header not in sections:
                sections.append(clean_header)

        return sections

    def _parse_api_documentation(self, content: str) -> Dict[str, Any]:
        """Parse API documentation response"""

        # Count documented endpoints
        endpoint_count = len(
            re.findall(r'(?:GET|POST|PUT|DELETE|PATCH)\s+/', content)
        )

        # Check for examples
        has_examples = '```json' in content or 'Example:' in content

        # Extract status codes
        status_codes = re.findall(r'\b[1-5]\d{2}\b', content)

        return {
            'format': (
                'OpenAPI' if 'openapi' in content.lower() else 'Markdown'
            ),
            'endpoints_documented': endpoint_count,
            'content': content,
            'examples_included': has_examples,
            'status_codes_documented': list(set(status_codes)),
            'has_authentication': (
                'auth' in content.lower() or 'token' in content.lower()
            )
        }

    def _extract_code(self, content: str) -> str:
        """Extract code from response"""

        code_blocks = re.findall(
            r'```(?:\w+)?\n(.*?)```', content, re.DOTALL
        )

        if code_blocks:
            return code_blocks[0].strip()
        return content

    def _count_comments(self, content: str) -> int:
        """Count number of comments in code"""

        single_line_comments = len(
            re.findall(r'^\s*#.*$', content, re.MULTILINE)
        )
        multi_line_comments = len(
            re.findall(r'""".*?"""', content, re.DOTALL)
        )

        return single_line_comments + multi_line_comments

    def _count_docstrings(self, content: str) -> int:
        """Count number of docstrings"""

        return len(
            re.findall(
                r'^\s*""".*?"""',
                content,
                re.MULTILINE | re.DOTALL
            )
        )

    def _generate_mock_readme(self) -> str:
        """Generate mock README content"""

        return """# Project Name

[![Build Status](https://img.shields.io/badge/build-passing-green)]()
[![License](https://img.shields.io/badge/license-MIT-blue)]()

## Overview

This project provides a comprehensive multi-agent framework for collaborative AI development.

## Features

- Multi-agent orchestration
- Intelligent memory management
- Self-healing capabilities
- Comprehensive logging

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from framework import MultiAgentFramework

framework = MultiAgentFramework()
result = await framework.execute_workflow('code_generation', task)
```

## Configuration

Edit `config.yaml` to customize agent behavior and system parameters.

## API Reference

See `/docs/api.md` for complete API documentation.

## Contributing

Please read CONTRIBUTING.md for details on our code of conduct and submission process.

## License

This project is licensed under the MIT License - see LICENSE file for details.
"""

    def _mock_documentation_response(self) -> Dict[str, Any]:
        """Generate mock documentation response"""

        content = """# Documentation

## Overview

This system provides a multi-agent framework for automated software development.

## Architecture

The system uses a microservices architecture with the following components:
- Orchestrator: Coordinates agent workflows
- Agents: Specialized workers for different tasks
- Memory Manager: Handles persistent storage and retrieval
- Message Bus: Enables inter-agent communication

## Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Configure settings in `config.yaml`
4. Run: `python main.py`

## Usage

### Basic Example

```python
from framework import MultiAgentFramework

framework = MultiAgentFramework()
result = await framework.execute_workflow('code_generation', {
    'description': 'Create a REST API',
    'requirements': ['FastAPI', 'PostgreSQL']
})
```

## API Reference

### Workflows

- `code_generation`: Generate new code
- `bug_fix`: Fix existing code issues
- `optimization`: Optimize performance

### Agents

- Developer: Code implementation
- Reviewer: Code review and quality checks
- Tester: Test generation and execution
- Debugger: Error analysis and fixing

## Configuration

Key configuration options in `config.yaml`:
- `max_parallel_agents`: Number of concurrent agents
- `memory.cache_size`: Memory cache size
- `logging.level`: Log verbosity

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure all dependencies are installed
2. **Memory errors**: Increase cache size or enable consolidation
3. **Timeout errors**: Adjust timeout values in configuration

## Best Practices

- Always define clear task requirements
- Use appropriate workflow strategies
- Monitor agent performance metrics
- Regular memory consolidation

## Contributing

We welcome contributions! Please see CONTRIBUTING.md for guidelines.
"""

        return {
            'format': 'markdown',
            'content': content,
            'sections': ['Overview', 'Architecture', 'Installation', 'Usage', 'API Reference',
                        'Configuration', 'Troubleshooting', 'Best Practices', 'Contributing'],
            'word_count': len(content.split()),
            'code_examples': 2,
            'has_toc': False,
            'has_examples': True,
            'completeness_score': 90
        }
