"""
Debugger Agent - Specialized in error analysis and bug fixing
"""

import ast
import re
from typing import Any, Dict, List

from ..core.agent import Agent, AgentConfig


class DebuggerAgent(Agent):
    """Agent specialized in debugging and error fixing"""

    def __init__(self, config: AgentConfig, claude_cli=None, memory_manager=None, message_bus=None):
        config.capabilities = [
            "error_analysis", "bug_fixing", "stack_trace_analysis",
            "performance_debugging", "memory_leak_detection", "deadlock_detection",
            "root_cause_analysis", "exception_handling", "logging_analysis", "analyze", "debug", "fix", "investigate"
        ]

        super().__init__(config, claude_cli, memory_manager, message_bus)

        self.error_patterns = {
            'syntax': ['SyntaxError', 'IndentationError', 'TabError'],
            'runtime': ['RuntimeError', 'TypeError', 'ValueError', 'AttributeError'],
            'logic': ['AssertionError', 'LogicError'],
            'memory': ['MemoryError', 'RecursionError'],
            'io': ['IOError', 'FileNotFoundError', 'PermissionError']
        }

        self.fix_strategies = {
            'syntax': self._fix_syntax_error,
            'runtime': self._fix_runtime_error,
            'logic': self._fix_logic_error,
            'memory': self._fix_memory_error,
            'io': self._fix_io_error
        }

    async def _execute_task(self, task_id: str, task: Dict[str, Any], context: Dict[str, Any], memories: List[Dict]) -> Any:
        """Execute debugger-specific tasks"""

        task_type = task.get('type', '').lower()

        if 'debug' in task_type or 'fix' in task_type or 'analyze' in task_type:
            return await self._debug_and_fix(task_id, task, context, memories)
        elif 'trace' in task_type:
            return await self._analyze_stack_trace(task_id, task, context, memories)
        elif 'performance' in task_type:
            return await self._debug_performance(task_id, task, context, memories)
        else:
            return await self._debug_and_fix(task_id, task, context, memories)

    async def _debug_and_fix(self, task_id: str, task: Dict[str, Any], context: Dict[str, Any], memories: List[Dict]) -> Dict[str, Any]:
        """Debug and fix code issues"""

        self.logger.info("[%s] Debugging and fixing code", task_id)

        error = task.get('error', '')
        code = task.get('code', '')
        stack_trace = task.get('stack_trace', '')

        # Classify error type
        error_type = self._classify_error(error, stack_trace)

        self.logger.debug("[%s] Error classified as: %s", task_id, error_type)

        # Build debugging prompt
        prompt = """
Debug and fix the following code issue:

Error: {error}

Stack Trace:
```
{stack_trace}
```

Code:
```
{code}
```

Please:
1. Identify the root cause of the error
2. Provide the fixed code
3. Explain what was wrong and how it was fixed
4. Add error prevention measures
5. Suggest additional improvements
"""

        if self.claude_cli:
            response = await self.claude_cli.complete(
                prompt,
                max_tokens=self.config.max_tokens,
                temperature=0.3,
                system="You are an expert debugger. Identify and fix bugs systematically and thoroughly."
            )

            if response.success:
                result = self._parse_debug_response(response.content, error_type)

                # Store successful fix pattern
                if self.memory_manager and result['success']:
                    await self.memory_manager.store({
                        'agent': self.name,
                        'error_type': error_type,
                        'error': error[:200],
                        'fix_applied': result['fix_explanation'],
                        'importance': 0.8
                    })

                return result
            else:
                raise Exception(f"Debugging failed: {response.error}")
        else:
            # Try automated fix strategy
            return await self._apply_fix_strategy(error_type, code, error, stack_trace)

    async def _analyze_stack_trace(self, task_id: str, task: Dict[str, Any], context: Dict[str, Any], memories: List[Dict]) -> Dict[str, Any]:
        """Analyze stack trace to identify root cause"""

        self.logger.info("[%s] Analyzing stack trace", task_id)

        stack_trace = task.get('stack_trace', '')

        # Parse stack trace
        frames = self._parse_stack_trace(stack_trace)

        # Identify problematic frame
        problem_frame = self._identify_problem_frame(frames)

        return {
            'root_cause': problem_frame.get('issue', 'Unknown'),
            'location': {
                'file': problem_frame.get('file', 'unknown'),
                'line': problem_frame.get('line', -1),
                'function': problem_frame.get('function', 'unknown')
            },
            'analysis': f"Error originated in {problem_frame.get('function', 'unknown function')}",
            'frames': frames,
            'suggestions': self._generate_fix_suggestions(problem_frame)
        }

    async def _debug_performance(self, task_id: str, task: Dict[str, Any], context: Dict[str, Any], memories: List[Dict]) -> Dict[str, Any]:
        """Debug performance issues"""

        self.logger.info("[%s] Debugging performance issues", task_id)

        prompt = """
Analyze performance issues in this code:

Code:
```
{code}
```

Performance Data:
{json.dumps(performance_data, indent=2)}

Identify:
1. Performance bottlenecks
2. Inefficient algorithms or data structures
3. Memory leaks or excessive allocations
4. I/O blocking operations
5. Optimization opportunities
"""

        if self.claude_cli:
            response = await self.claude_cli.complete(prompt, temperature=0.4)

            if response.success:
                return self._parse_performance_response(response.content)
            else:
                raise Exception(f"Performance debugging failed: {response.error}")
        else:
            return {
                'bottlenecks': ['Nested loops with O(n²) complexity', 'Synchronous I/O operations'],
                'optimizations': ['Use dictionary lookup instead of list search', 'Implement caching'],
                'memory_issues': ['Large list kept in memory', 'No garbage collection'],
                'recommendations': ['Refactor algorithm', 'Use async I/O', 'Implement pagination']
            }

    def _classify_error(self, error: str, stack_trace: str) -> str:
        """Classify error type"""

        error_text = f"{error} {stack_trace}".lower()

        for error_type, patterns in self.error_patterns.items():
            for pattern in patterns:
                if pattern.lower() in error_text:
                    return error_type

        return 'unknown'

    async def _apply_fix_strategy(self, error_type: str, code: str, error: str, stack_trace: str) -> Dict[str, Any]:
        """Apply automated fix strategy based on error type"""

        if error_type in self.fix_strategies:
            return await self.fix_strategies[error_type](code, error, stack_trace)

        return {
            'success': False,
            'fixed_code': code,
            'root_cause': 'Unable to automatically determine root cause',
            'fix_explanation': 'Manual debugging required',
            'changes': [],
            'prevention_measures': ['Add error handling', 'Implement validation']
        }

    async def _fix_syntax_error(self, code: str, error: str, stack_trace: str) -> Dict[str, Any]:
        """Fix syntax errors"""

        # Try to parse and identify syntax issue
        try:
            ast.parse(code)
            return {
                'success': False,
                'fixed_code': code,
                'root_cause': 'No syntax error detected',
                'fix_explanation': 'Code appears syntactically correct',
                'changes': []
            }
        except SyntaxError as e:
            # Common syntax fixes
            fixed_code = code
            changes = []

            if 'unexpected indent' in str(e):
                # Fix indentation
                lines = code.split('\n')
                fixed_lines = []
                for line in lines:
                    # Normalize indentation to 4 spaces
                    stripped = line.lstrip()
                    indent_level = (len(line) - len(stripped)) // 4
                    fixed_lines.append(' ' * (indent_level * 4) + stripped)
                fixed_code = '\n'.join(fixed_lines)
                changes.append('Fixed indentation to use 4 spaces')

            elif 'invalid syntax' in str(e):
                # Common syntax fixes
                if '=' in error and '==' not in error:
                    fixed_code = re.sub(r'if\s+(\w+)\s*=\s*', r'if \1 == ', code)
                    changes.append('Fixed assignment in conditional to comparison')

            return {
                'success': len(changes) > 0,
                'fixed_code': fixed_code,
                'root_cause': str(e),
                'fix_explanation': ' '.join(changes) if changes else 'Syntax error identified but automatic fix not available',
                'changes': changes,
                'prevention_measures': ['Use a linter', 'Enable IDE syntax checking']
            }

    async def _fix_runtime_error(self, code: str, error: str, stack_trace: str) -> Dict[str, Any]:
        """Fix runtime errors"""

        fixed_code = code
        changes = []

        if 'NoneType' in error:
            # Add None checks
            fixed_code = re.sub(
                r'(\w+)\.(\w+)',
                r'\1.\2 if \1 is not None else None',
                code,
                count=1
            )
            changes.append('Added None check')

        elif 'division by zero' in error.lower():
            # Add zero check
            fixed_code = re.sub(
                r'(\w+)\s*/\s*(\w+)',
                r'\1 / \2 if \2 != 0 else 0',
                code,
                count=1
            )
            changes.append('Added division by zero check')

        return {
            'success': len(changes) > 0,
            'fixed_code': fixed_code,
            'root_cause': error,
            'fix_explanation': ' '.join(changes) if changes else 'Runtime error requires manual fix',
            'changes': changes,
            'prevention_measures': ['Add input validation', 'Implement error handling']
        }

    async def _fix_logic_error(self, code: str, error: str, stack_trace: str) -> Dict[str, Any]:
        """Fix logic errors"""

        return {
            'success': False,
            'fixed_code': code,
            'root_cause': 'Logic error detected',
            'fix_explanation': 'Logic errors require manual review and correction',
            'changes': [],
            'prevention_measures': ['Add unit tests', 'Implement assertions', 'Use type hints']
        }

    async def _fix_memory_error(self, code: str, error: str, stack_trace: str) -> Dict[str, Any]:
        """Fix memory errors"""

        fixed_code = code
        changes = []

        if 'recursion' in error.lower():
            # Add recursion limit or convert to iteration
            changes.append('Consider converting recursion to iteration')
            prevention = ['Set recursion limit', 'Use iterative approach']
        else:
            changes.append('Optimize memory usage')
            prevention = ['Use generators', 'Process data in chunks', 'Clear unused references']

        return {
            'success': False,
            'fixed_code': fixed_code,
            'root_cause': 'Memory/recursion limit exceeded',
            'fix_explanation': 'Memory optimization required',
            'changes': changes,
            'prevention_measures': prevention
        }

    async def _fix_io_error(self, code: str, error: str, stack_trace: str) -> Dict[str, Any]:
        """Fix I/O errors"""

        fixed_code = code
        changes = []

        # Add try-except for file operations
        if 'open(' in code:
            fixed_code = re.sub(
                r'(.*?)open\((.*?)\)(.*)',
                r'try:\n    \1open(\2)\3\nexcept IOError as e:\n    print(f"File error: {e}")',
                code
            )
            changes.append('Added error handling for file operations')

        return {
            'success': len(changes) > 0,
            'fixed_code': fixed_code,
            'root_cause': 'I/O operation failed',
            'fix_explanation': ' '.join(changes) if changes else 'I/O error requires manual fix',
            'changes': changes,
            'prevention_measures': ['Check file existence', 'Handle permissions', 'Use context managers']
        }

    def _parse_debug_response(self, content: str, error_type: str) -> Dict[str, Any]:
        """Parse debugging response from Claude"""

        # Extract fixed code
        code_blocks = re.findall(r'```(?:\w+)?\n(.*?)```', content, re.DOTALL)
        fixed_code = code_blocks[0] if code_blocks else ''

        # Extract root cause
        root_cause = 'Unknown'
        cause_match = re.search(r'(?:root cause|problem|issue).*?:(.*?)(?:\n|$)', content, re.IGNORECASE)
        if cause_match:
            root_cause = cause_match.group(1).strip()

        # Extract changes
        changes = re.findall(r'(?:change|fix|update).*?:(.*?)(?:\n|$)', content, re.IGNORECASE)

        return {
            'success': bool(fixed_code),
            'fixed_code': fixed_code,
            'root_cause': root_cause,
            'fix_explanation': self._extract_explanation(content),
            'error_type': error_type,
            'changes': changes,
            'prevention_measures': re.findall(r'(?:prevent|avoid|measure).*?:(.*?)(?:\n|$)', content, re.IGNORECASE)
        }

    def _parse_performance_response(self, content: str) -> Dict[str, Any]:
        """Parse performance debugging response"""

        return {
            'bottlenecks': re.findall(r'bottleneck.*?:(.*?)(?:\n|$)', content, re.IGNORECASE),
            'optimizations': re.findall(r'optimi.*?:(.*?)(?:\n|$)', content, re.IGNORECASE),
            'memory_issues': re.findall(r'memory.*?:(.*?)(?:\n|$)', content, re.IGNORECASE),
            'recommendations': re.findall(r'recommend.*?:(.*?)(?:\n|$)', content, re.IGNORECASE)
        }

    def _parse_stack_trace(self, stack_trace: str) -> List[Dict[str, Any]]:
        """Parse stack trace into frames"""

        frames = []
        frame_pattern = r'File "(.+?)", line (\d+), in (.+?)\n\s*(.*?)$'

        for match in re.finditer(frame_pattern, stack_trace, re.MULTILINE):
            frames.append({
                'file': match.group(1),
                'line': int(match.group(2)),
                'function': match.group(3),
                'code': match.group(4)
            })

        return frames

    def _identify_problem_frame(self, frames: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Identify the most likely problem frame"""

        if not frames:
            return {'issue': 'No stack frames found'}

        # Usually the last frame in user code
        for frame in reversed(frames):
            if not frame['file'].startswith('<'):
                return frame

        return frames[-1] if frames else {}

    def _generate_fix_suggestions(self, frame: Dict[str, Any]) -> List[str]:
        """Generate fix suggestions based on frame analysis"""

        suggestions = []
        code = frame.get('code', '')

        if 'None' in code:
            suggestions.append('Add None check before accessing attributes')
        if '/' in code:
            suggestions.append('Check for division by zero')
        if 'open' in code:
            suggestions.append('Add file existence check and error handling')
        if not suggestions:
            suggestions.append('Review logic and add error handling')

        return suggestions

    def _extract_explanation(self, content: str) -> str:
        """Extract explanation from response"""

        # Remove code blocks
        content_no_code = re.sub(r'```.*?```', '', content, flags=re.DOTALL)

        # Look for explanation section
        if 'explanation' in content_no_code.lower():
            parts = re.split(r'explanation:?', content_no_code, flags=re.IGNORECASE)
            if len(parts) > 1:
                return parts[1].strip().split('\n')[0]

        # Return first paragraph
        paragraphs = content_no_code.strip().split('\n\n')
        return paragraphs[0] if paragraphs else 'Fix applied'
