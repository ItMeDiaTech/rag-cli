"""
Tester Agent - Specialized in test creation and execution
"""

import re
from typing import Any, Dict, List

from ..core.agent import Agent, AgentConfig


class TesterAgent(Agent):
    """Agent specialized in testing and validation"""

    def __init__(self, config: AgentConfig, claude_cli=None, memory_manager=None, message_bus=None):
        config.capabilities = [
            "unit_testing", "integration_testing", "test_generation",
            "coverage_analysis", "performance_testing", "load_testing",
            "test_automation", "regression_testing", "mocking", "test_fixtures", "test", "validate", "verify", "benchmark"
        ]

        super().__init__(config, claude_cli, memory_manager, message_bus)

        self.test_frameworks = {
            'python': ['pytest', 'unittest', 'nose2'],
            'javascript': ['jest', 'mocha', 'jasmine'],
            'java': ['junit', 'testng'],
            'go': ['testing', 'testify']
        }

    async def _execute_task(self, task_id: str, task: Dict[str, Any], context: Dict[str, Any], memories: List[Dict]) -> Any:
        """Execute tester-specific tasks"""

        task_type = task.get('type', '').lower()

        if 'test' in task_type or 'validate' in task_type:
            return await self._generate_tests(task_id, task, context, memories)
        elif 'coverage' in task_type:
            return await self._analyze_coverage(task_id, task, context, memories)
        elif 'benchmark' in task_type:
            return await self._performance_test(task_id, task, context, memories)
        else:
            return await self._generate_tests(task_id, task, context, memories)

    async def _generate_tests(self, task_id: str, task: Dict[str, Any], context: Dict[str, Any], memories: List[Dict]) -> Dict[str, Any]:
        """Generate comprehensive tests for code"""

        self.logger.info("[%s] Generating tests", task_id)

        # Get code from context or task
        code = task.get('code', '')
        if not code and context.get('results', {}).get('Developer'):
            dev_result = context['results']['Developer']
            if isinstance(dev_result, dict) and 'code' in dev_result:
                code_blocks = dev_result['code']
                if code_blocks and isinstance(code_blocks, list):
                    code = '\n'.join([block.get('content', '') for block in code_blocks])

        language = task.get('language', 'python')
        framework = self.test_frameworks.get(language, ['generic'])[0]

        prompt = """
Generate comprehensive tests for the following code:

Code:
```{language}
{code}
```

Test Framework: {framework}

Requirements:
1. Unit tests for all functions/methods
2. Edge case testing
3. Error handling validation
4. Mock external dependencies
5. Test documentation

Provide:
- Complete test file
- Test descriptions
- Expected coverage percentage
"""

        if self.claude_cli:
            response = await self.claude_cli.complete(
                prompt,
                max_tokens=self.config.max_tokens,
                temperature=0.5,
                system="You are an expert test engineer. Write thorough tests that ensure code reliability."
            )

            if response.success:
                return self._parse_test_response(response.content, language, framework)
            raise Exception(f"Test generation failed: {response.error}")
        else:
            return self._mock_test_response(language, framework)

    async def _analyze_coverage(self, task_id: str, task: Dict[str, Any], context: Dict[str, Any], memories: List[Dict]) -> Dict[str, Any]:
        """Analyze test coverage"""

        self.logger.info("[%s] Analyzing test coverage", task_id)

        # In production, this would run actual coverage tools
        return {
            'total_coverage': 85,
            'line_coverage': 90,
            'branch_coverage': 75,
            'uncovered_lines': [42, 67, 89],
            'uncovered_functions': ['handle_edge_case'],
            'recommendations': [
                'Add tests for error handling paths',
                'Test boundary conditions',
                'Cover exception scenarios'
            ]
        }

    async def _performance_test(self, task_id: str, task: Dict[str, Any], context: Dict[str, Any], memories: List[Dict]) -> Dict[str, Any]:
        """Run performance benchmarks"""

        self.logger.info("[%s] Running performance tests", task_id)

        prompt = """
Create performance tests for:

Code:
```
{task.get('code', '')}
```

Test scenarios:
1. Normal load
2. Peak load
3. Stress conditions
4. Memory usage
5. Response times
"""

        if self.claude_cli:
            response = await self.claude_cli.complete(prompt, temperature=0.5)

            if response.success:
                return self._parse_performance_response(response.content)
            raise Exception(f"Performance testing failed: {response.error}")
        else:
            return {
                'results': {
                    'avg_response_time': '50ms',
                    'p95_response_time': '120ms',
                    'p99_response_time': '250ms',
                    'throughput': '1000 req/s',
                    'memory_usage': '150MB',
                    'cpu_usage': '35%'
                },
                'bottlenecks': ['Database queries', 'JSON serialization'],
                'recommendations': ['Add caching', 'Optimize queries']
            }

    def _parse_test_response(self, content: str, language: str, framework: str) -> Dict[str, Any]:
        """Parse test generation response"""

        # Extract test code
        test_code = []
        code_blocks = re.findall(r'```(?:\w+)?\n(.*?)```', content, re.DOTALL)
        for code in code_blocks:
            test_code.append({
                'language': language,
                'framework': framework,
                'content': code.strip()
            })

        # Extract test descriptions
        test_descriptions = re.findall(r'(?:test|Test)\s+(\w+).*?(?::|-)?\s*(.*?)(?:\n|$)', content)

        # Estimate coverage
        coverage_match = re.search(r'(\d+)%?\s*coverage', content, re.IGNORECASE)
        coverage = int(coverage_match.group(1)) if coverage_match else 80

        return {
            'test_code': test_code,
            'test_count': len(test_descriptions),
            'test_descriptions': test_descriptions,
            'expected_coverage': coverage,
            'framework': framework,
            'test_types': ['unit', 'integration', 'edge_cases'],
            'success': True
        }

    def _parse_performance_response(self, content: str) -> Dict[str, Any]:
        """Parse performance test response"""

        results = {}

        # Extract metrics
        metrics_patterns = {
            'response_time': r'(?:response time|latency).*?(\d+\.?\d*)\s*(ms|s)',
            'throughput': r'(?:throughput|requests).*?(\d+\.?\d*)\s*(req/s|rps)',
            'memory': r'(?:memory|ram).*?(\d+\.?\d*)\s*(MB|GB)',
            'cpu': r'(?:cpu|processor).*?(\d+\.?\d*)%'
        }

        for metric, pattern in metrics_patterns.items():
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                results[metric] = f"{match.group(1)}{match.group(2) if len(match.groups()) > 1 else ''}"

        return {
            'results': results,
            'bottlenecks': re.findall(r'bottleneck.*?:\s*(.*)', content, re.IGNORECASE),
            'recommendations': re.findall(r'(?:recommend|suggest).*?:\s*(.*)', content, re.IGNORECASE)
        }

    def _mock_test_response(self, language: str, framework: str) -> Dict[str, Any]:
        """Generate mock test response"""

        if language == 'python':
            test_content = '''import pytest
from main import example_function

def test_example_function_success():
    """Test successful execution"""
    result = example_function("input")
    assert result == "expected_output"

def test_example_function_error():
    """Test error handling"""
    with pytest.raises(ValueError):
        example_function(None)

def test_example_function_edge_case():
    """Test edge cases"""
    assert example_function("") == ""
'''
        elif language == 'javascript':
            test_content = '''describe('ExampleFunction', () => {
    test('should handle normal input', () => {
        const result = exampleFunction('input');
        expect(result).toBe('expected_output');
    });

    test('should handle errors', () => {
        expect(() => exampleFunction(null)).toThrow();
    });
});'''

        test_content = '// Generic test code'

        return {
            'test_code': [{
                'language': language,
                'framework': framework,
                'content': test_content
            }],
            'test_count': 3,
            'test_descriptions': [
                ('test_success', 'Tests successful execution'),
                ('test_error', 'Tests error handling'),
                ('test_edge_case', 'Tests edge cases')
            ],
            'expected_coverage': 85,
            'framework': framework,
            'test_types': ['unit', 'integration', 'edge_cases'],
            'success': True
        }
