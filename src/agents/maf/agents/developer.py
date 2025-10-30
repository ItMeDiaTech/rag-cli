"""
Developer Agent - Specialized in code implementation and generation
"""

import json
import re
from typing import Any, Dict, List

from ..core.agent import Agent, AgentConfig


class DeveloperAgent(Agent):
    """Agent specialized in code development and implementation"""

    def __init__(self, config: AgentConfig, claude_cli=None, memory_manager=None, message_bus=None):
        # Set developer-specific capabilities
        config.capabilities = [
            "python", "javascript", "typescript", "java", "c++", "go", "rust",
            "sql", "html", "css", "react", "vue", "angular",
            "api_implementation", "algorithm_design", "data_structures",
            "database_design", "microservices", "rest_api", "graphql",
            "testing", "debugging", "refactoring", "optimization", "implement", "develop", "code", "fix"
        ]

        super().__init__(config, claude_cli, memory_manager, message_bus)

        # Developer-specific templates
        self.code_templates = {
            'python_class': self._get_python_class_template(),
            'api_endpoint': self._get_api_endpoint_template(),
            'test_function': self._get_test_template()
        }

    async def _execute_task(self, task_id: str, task: Dict[str, Any], context: Dict[str, Any], memories: List[Dict]) -> Any:
        """Execute developer-specific tasks"""

        task_type = task.get('type', '').lower()

        if 'implement' in task_type or 'code' in task_type or 'develop' in task_type:
            return await self._implement_code(task_id, task, context, memories)
        elif 'fix' in task_type or 'patch' in task_type:
            return await self._fix_code(task_id, task, context, memories)
        elif 'refactor' in task_type:
            return await self._refactor_code(task_id, task, context, memories)
        elif 'optimize' in task_type:
            return await self._optimize_code(task_id, task, context, memories)
        else:
            # Default to general code implementation
            return await self._implement_code(task_id, task, context, memories)

    async def _implement_code(self, task_id: str, task: Dict[str, Any], context: Dict[str, Any], memories: List[Dict]) -> Dict[str, Any]:
        """Implement new code based on requirements"""

        self.logger.info("[%s] Implementing code for: %s", task_id, task.get('description', 'unspecified task'))

        # Build implementation prompt
        prompt = self._build_implementation_prompt(task, context, memories)

        if self.claude_cli:
            response = await self.claude_cli.complete(
                prompt,
                max_tokens=self.config.max_tokens,
                temperature=0.7,
                system="You are an expert software developer. Write clean, efficient, production-ready code with proper error handling and documentation."
            )

            if response.success:
                # Extract code from response
                code_blocks = self._extract_code_blocks(response.content)

                result = {
                    'code': code_blocks,
                    'explanation': self._extract_explanation(response.content),
                    'language': self._detect_language(code_blocks),
                    'dependencies': self._extract_dependencies(response.content),
                    'tokens_used': response.tokens_used
                }

                # Store successful implementation pattern
                if self.memory_manager:
                    await self.memory_manager.store({
                        'agent': self.name,
                        'task_type': 'implementation',
                        'description': task.get('description', ''),
                        'result': result,
                        'importance': 0.7
                    })

                return result
            raise Exception(f"Code implementation failed: {response.error}")
        else:
            # Mock implementation for testing
            return {
                'code': [{'language': 'python', 'content': 'def example():\n    return "Mock implementation"'}],
                'explanation': 'Mock code implementation',
                'language': 'python',
                'dependencies': []
            }

    async def _fix_code(self, task_id: str, task: Dict[str, Any], context: Dict[str, Any], memories: List[Dict]) -> Dict[str, Any]:
        """Fix bugs in existing code"""

        self.logger.info("[%s] Fixing code issues", task_id)

        prompt = self._build_fix_prompt(task, context, memories)

        if self.claude_cli:
            response = await self.claude_cli.complete(
                prompt,
                max_tokens=self.config.max_tokens,
                temperature=0.5,
                system="You are an expert debugger. Identify and fix bugs while preserving existing functionality."
            )

            if response.success:
                return {
                    'fixed_code': self._extract_code_blocks(response.content),
                    'changes': self._extract_changes(response.content),
                    'explanation': self._extract_explanation(response.content),
                    'tokens_used': response.tokens_used
                }
            raise Exception(f"Code fix failed: {response.error}")
        else:
            return {
                'fixed_code': [{'language': 'python', 'content': 'def fixed():\n    return "Fixed"'}],
                'changes': ['Fixed bug in line 42'],
                'explanation': 'Mock fix applied'
            }

    async def _refactor_code(self, task_id: str, task: Dict[str, Any], context: Dict[str, Any], memories: List[Dict]) -> Dict[str, Any]:
        """Refactor code for better structure and maintainability"""

        self.logger.info("[%s] Refactoring code", task_id)

        prompt = self._build_refactor_prompt(task, context, memories)

        if self.claude_cli:
            response = await self.claude_cli.complete(
                prompt,
                max_tokens=self.config.max_tokens,
                temperature=0.6,
                system="You are an expert at code refactoring. Improve code structure, readability, and maintainability while preserving functionality."
            )

            if response.success:
                return {
                    'refactored_code': self._extract_code_blocks(response.content),
                    'improvements': self._extract_improvements(response.content),
                    'explanation': self._extract_explanation(response.content),
                    'tokens_used': response.tokens_used
                }
            raise Exception(f"Code refactoring failed: {response.error}")
        else:
            return {
                'refactored_code': [{'language': 'python', 'content': 'def refactored():\n    return "Clean"'}],
                'improvements': ['Improved naming', 'Added type hints'],
                'explanation': 'Mock refactoring complete'
            }

    async def _optimize_code(self, task_id: str, task: Dict[str, Any], context: Dict[str, Any], memories: List[Dict]) -> Dict[str, Any]:
        """Optimize code for performance"""

        self.logger.info("[%s] Optimizing code performance", task_id)

        prompt = self._build_optimization_prompt(task, context, memories)

        if self.claude_cli:
            response = await self.claude_cli.complete(
                prompt,
                max_tokens=self.config.max_tokens,
                temperature=0.6,
                system="You are a performance optimization expert. Optimize code for speed and efficiency while maintaining correctness."
            )

            if response.success:
                return {
                    'optimized_code': self._extract_code_blocks(response.content),
                    'performance_gains': self._extract_performance_info(response.content),
                    'explanation': self._extract_explanation(response.content),
                    'tokens_used': response.tokens_used
                }
            raise Exception(f"Code optimization failed: {response.error}")
        else:
            return {
                'optimized_code': [{'language': 'python', 'content': 'def optimized():\n    return "Fast"'}],
                'performance_gains': '30% faster execution',
                'explanation': 'Mock optimization complete'
            }

    def _build_implementation_prompt(self, task: Dict[str, Any], context: Dict[str, Any], memories: List[Dict]) -> str:
        """Build prompt for code implementation"""

        prompt_parts = [
            "Task: Implement the following functionality",
            "",
            f"Description: {task.get('description', 'Implement the requested functionality')}",
            ""
        ]

        if task.get('requirements'):
            prompt_parts.extend([
                "Requirements:",
                json.dumps(task['requirements'], indent=2),
                ""
            ])

        if task.get('language'):
            prompt_parts.append(f"Language: {task['language']}")

        if task.get('framework'):
            prompt_parts.append(f"Framework: {task['framework']}")

        if context.get('architecture'):
            prompt_parts.extend([
                "",
                "Architecture Context:",
                str(context['architecture'])
            ])

        if memories:
            prompt_parts.extend([
                "",
                "Relevant past implementations:",
                self._format_memories(memories[:3])
            ])

        prompt_parts.extend([
            "",
            "Please provide:",
            "1. Complete, production-ready code implementation",
            "2. Proper error handling and validation",
            "3. Clear documentation and comments",
            "4. List any dependencies required",
            "5. Brief explanation of the implementation approach"
        ])

        return "\n".join(prompt_parts)

    def _build_fix_prompt(self, task: Dict[str, Any], context: Dict[str, Any], memories: List[Dict]) -> str:
        """Build prompt for bug fixing"""

        prompt_parts = [
            "Task: Fix the following code issue",
            "",
            f"Error: {task.get('error', 'Unknown error')}",
            "",
            "Current Code:",
            "```",
            task.get('code', ''),
            "```",
            ""
        ]

        if task.get('stack_trace'):
            prompt_parts.extend([
                "Stack Trace:",
                task['stack_trace'],
                ""
            ])

        if context.get('test_results'):
            prompt_parts.extend([
                "Test Results:",
                str(context['test_results']),
                ""
            ])

        prompt_parts.extend([
            "Please provide:",
            "1. Fixed code that resolves the issue",
            "2. Explanation of what was wrong",
            "3. Description of the fix applied",
            "4. Any additional improvements made"
        ])

        return "\n".join(prompt_parts)

    def _build_refactor_prompt(self, task: Dict[str, Any], context: Dict[str, Any], memories: List[Dict]) -> str:
        """Build prompt for code refactoring"""

        return """
Task: Refactor the following code for better structure and maintainability

Current Code:
```
{task.get('code', '')}
```

Refactoring Goals:
{json.dumps(task.get('goals', ['Improve readability', 'Reduce complexity', 'Follow best practices']), indent=2)}

Please provide:
1. Refactored code with improvements
2. List of specific improvements made
3. Explanation of refactoring decisions
4. Any pattern or architecture changes
"""

    def _build_optimization_prompt(self, task: Dict[str, Any], context: Dict[str, Any], memories: List[Dict]) -> str:
        """Build prompt for code optimization"""

        return """
Task: Optimize the following code for performance

Current Code:
```
{task.get('code', '')}
```

Performance Issues:
{task.get('issues', 'General optimization needed')}

Optimization Goals:
{json.dumps(task.get('goals', ['Reduce time complexity', 'Minimize memory usage', 'Improve efficiency']), indent=2)}

Please provide:
1. Optimized code implementation
2. Expected performance improvements
3. Explanation of optimizations applied
4. Any trade-offs made
"""

    def _extract_code_blocks(self, content: str) -> List[Dict[str, str]]:
        """Extract code blocks from response"""

        code_blocks = []

        # Find all code blocks with optional language specification
        pattern = r'```(\w+)?\n(.*?)```'
        matches = re.findall(pattern, content, re.DOTALL)

        for language, code in matches:
            code_blocks.append({
                'language': language or 'plain',
                'content': code.strip()
            })

        # If no code blocks found, treat entire content as code
        if not code_blocks and content.strip():
            code_blocks.append({
                'language': 'plain',
                'content': content.strip()
            })

        return code_blocks

    def _extract_explanation(self, content: str) -> str:
        """Extract explanation from response"""

        # Remove code blocks
        content_no_code = re.sub(r'```.*?```', '', content, flags=re.DOTALL)

        # Look for explanation section
        if 'Explanation:' in content_no_code:
            parts = content_no_code.split('Explanation:')
            return parts[1].strip()

        # Return cleaned content
        return content_no_code.strip()

    def _extract_dependencies(self, content: str) -> List[str]:
        """Extract dependencies from response"""

        dependencies = []

        # Look for import statements
        import_pattern = r'(?:import|from|require|use)\s+([a-zA-Z0-9_\-\.]+)'
        imports = re.findall(import_pattern, content)
        dependencies.extend(imports)

        # Look for package mentions
        if 'Dependencies:' in content or 'Requirements:' in content:
            deps_section = re.search(r'(?:Dependencies|Requirements):(.*?)(?:\n\n|$)', content, re.DOTALL)
            if deps_section:
                deps_text = deps_section.group(1)
                # Extract package names
                package_pattern = r'[a-zA-Z0-9_\-\.]+'
                packages = re.findall(package_pattern, deps_text)
                dependencies.extend(packages)

        # Remove duplicates
        return list(set(dependencies))

    def _detect_language(self, code_blocks: List[Dict[str, str]]) -> str:
        """Detect primary programming language"""

        if not code_blocks:
            return 'unknown'

        # Check specified languages
        languages = [block['language'] for block in code_blocks if block['language'] != 'plain']
        if languages:
            return languages[0]

        # Try to detect from content
        first_block = code_blocks[0]['content']

        if 'def ' in first_block or 'import ' in first_block:
            return 'python'
        if 'function ' in first_block or 'const ' in first_block or 'let ' in first_block:
            return 'javascript'
        if 'public class' in first_block or 'private ' in first_block:
            return 'java'
        if '#include' in first_block:
            return 'cpp'
        if 'func ' in first_block and 'package ' in first_block:
            return 'go'

        return 'unknown'

    def _extract_changes(self, content: str) -> List[str]:
        """Extract list of changes made"""

        changes = []

        # Look for changes section
        if 'Changes:' in content:
            changes_section = content.split('Changes:')[1].split('\n\n')[0]
            # Extract bullet points
            changes = re.findall(r'[-*]\s+(.*)', changes_section)

        return changes

    def _extract_improvements(self, content: str) -> List[str]:
        """Extract list of improvements"""

        improvements = []

        # Look for improvements section
        if 'Improvements:' in content:
            improvements_section = content.split('Improvements:')[1].split('\n\n')[0]
            improvements = re.findall(r'[-*]\s+(.*)', improvements_section)

        return improvements

    def _extract_performance_info(self, content: str) -> str:
        """Extract performance improvement information"""

        # Look for performance metrics
        perf_pattern = r'(\d+%?\s*(?:faster|slower|improvement|reduction))'
        matches = re.findall(perf_pattern, content, re.IGNORECASE)

        if matches:
            return ', '.join(matches)

        return 'Performance improvements applied'

    def _format_memories(self, memories: List[Dict]) -> str:
        """Format memories for prompt"""

        formatted = []
        for memory in memories:
            if isinstance(memory, dict):
                formatted.append(f"- {memory.get('content', str(memory))[:200]}")

        return "\n".join(formatted)

    def _get_python_class_template(self) -> str:
        """Get Python class template"""
        return '''class {ClassName}:
    """
    {Description}
    """

    def __init__(self, {parameters}):
        """Initialize {ClassName}"""
        {initialization}

    def {method_name}(self, {method_params}):
        """
        {Method description}

        Args:
            {args_description}

        Returns:
            {return_description}
        """
        {method_implementation}
'''

    def _get_api_endpoint_template(self) -> str:
        """Get API endpoint template"""
        return '''@app.route('/{endpoint}', methods=['{method}'])
def {function_name}():
    """
    {Description}

    Returns:
        JSON response
    """
    try:
        # Validate request
        {validation}

        # Process request
        {processing}

        # Return response
        return jsonify({
            'success': True,
            'data': {data}
        }), 200

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
'''

    def _get_test_template(self) -> str:
        """Get test function template"""
        return '''def test_{function_name}():
    """Test {description}"""

    # Arrange
    {setup}

    # Act
    {action}

    # Assert
    {assertions}
'''
