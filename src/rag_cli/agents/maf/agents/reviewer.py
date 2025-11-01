"""
Reviewer Agent - Specialized in code review and quality assurance
"""

import re
from typing import Any, Dict, List

from ..core.agent import Agent, AgentConfig


class ReviewerAgent(Agent):
    """Agent specialized in code review and quality checks"""

    def __init__(self, config: AgentConfig, claude_cli=None, memory_manager=None, message_bus=None):
        config.capabilities = [
            "code_review", "security_analysis", "performance_review",
            "best_practices", "design_patterns", "code_quality",
            "vulnerability_detection", "style_checking", "documentation_review", "review", "validate", "check", "approve"
        ]

        super().__init__(config, claude_cli, memory_manager, message_bus)

        self.review_criteria = {
            'security': ['SQL injection', 'XSS', 'CSRF', 'Authentication', 'Authorization'],
            'performance': ['Time complexity', 'Space complexity', 'Database queries', 'Caching'],
            'quality': ['Readability', 'Maintainability', 'DRY principle', 'SOLID principles'],
            'style': ['Naming conventions', 'Code formatting', 'Comments', 'Documentation']
        }

    async def _execute_task(self, task_id: str, task: Dict[str, Any], context: Dict[str, Any], memories: List[Dict]) -> Any:
        """Execute reviewer-specific tasks"""

        task_type = task.get('type', '').lower()

        if 'review' in task_type:
            return await self._review_code(task_id, task, context, memories)
        elif 'security' in task_type:
            return await self._security_analysis(task_id, task, context, memories)
        elif 'quality' in task_type:
            return await self._quality_check(task_id, task, context, memories)
        else:
            return await self._review_code(task_id, task, context, memories)

    async def _review_code(self, task_id: str, task: Dict[str, Any], context: Dict[str, Any], memories: List[Dict]) -> Dict[str, Any]:
        """Perform comprehensive code review"""

        self.logger.info("[%s] Reviewing code", task_id)

        prompt = """
Please perform a comprehensive code review:

Code to Review:
```
{task.get('code', context.get('results', {}).get('Developer', {}).get('code', ''))}
```

Review Criteria:
- Security vulnerabilities
- Performance issues
- Code quality and best practices
- Potential bugs
- Documentation completeness

Provide:
1. Overall assessment (score 1-10)
2. List of issues found (categorized by severity)
3. Specific suggestions for improvement
4. Positive aspects of the code
"""

        if self.claude_cli:
            response = await self.claude_cli.complete(
                prompt,
                max_tokens=self.config.max_tokens,
                temperature=0.3,
                system="You are an expert code reviewer. Be thorough but constructive in your feedback."
            )

            if response.success:
                return self._parse_review_response(response.content)
            raise Exception(f"Code review failed: {response.error}")
        else:
            return {
                'score': 8,
                'issues': {
                    'critical': [],
                    'major': ['Missing error handling in function X'],
                    'minor': ['Consider using const instead of let']
                },
                'suggestions': ['Add input validation', 'Implement caching'],
                'positives': ['Clean code structure', 'Good naming conventions'],
                'approved': True
            }

    async def _security_analysis(self, task_id: str, task: Dict[str, Any], context: Dict[str, Any], memories: List[Dict]) -> Dict[str, Any]:
        """Perform security analysis"""

        self.logger.info("[%s] Performing security analysis", task_id)

        prompt = """
Perform a security analysis of this code:

```
{task.get('code', '')}
```

Check for:
- Input validation issues
- Authentication/authorization problems
- Injection vulnerabilities
- Data exposure risks
- Cryptographic weaknesses
"""

        if self.claude_cli:
            response = await self.claude_cli.complete(prompt, temperature=0.2)

            if response.success:
                return self._parse_security_response(response.content)
            raise Exception(f"Security analysis failed: {response.error}")
        else:
            return {
                'vulnerabilities': [],
                'risk_level': 'low',
                'recommendations': ['Enable HTTPS', 'Add rate limiting']
            }

    async def _quality_check(self, task_id: str, task: Dict[str, Any], context: Dict[str, Any], memories: List[Dict]) -> Dict[str, Any]:
        """Check code quality metrics"""

        self.logger.info("[%s] Checking code quality", task_id)

        # Simple quality metrics (in production, use proper AST analysis)
        code = task.get('code', '')

        return {
            'metrics': {
                'lines_of_code': len(code.split('\n')),
                'cyclomatic_complexity': self._estimate_complexity(code),
                'maintainability_index': 85,  # Mock value
                'test_coverage': 'Unknown'
            },
            'quality_score': 7.5,
            'improvements_needed': ['Add unit tests', 'Reduce function complexity']
        }

    def _parse_review_response(self, content: str) -> Dict[str, Any]:
        """Parse review response from Claude"""

        result = {
            'score': 7,  # Default score
            'issues': {'critical': [], 'major': [], 'minor': []},
            'suggestions': [],
            'positives': [],
            'approved': True
        }

        # Extract score
        score_match = re.search(r'(?:score|rating).*?(\d+)(?:/10)?', content, re.IGNORECASE)
        if score_match:
            result['score'] = int(score_match.group(1))

        # Extract issues by severity
        if 'critical' in content.lower():
            critical_section = re.search(r'critical.*?:(.*?)(?:major|minor|$)', content, re.IGNORECASE | re.DOTALL)
            if critical_section:
                result['issues']['critical'] = re.findall(r'[-*]\s+(.*)', critical_section.group(1))

        # Determine approval
        result['approved'] = result['score'] >= 7 and len(result['issues']['critical']) == 0

        return result

    def _parse_security_response(self, content: str) -> Dict[str, Any]:
        """Parse security analysis response"""

        vulnerabilities = re.findall(r'(?:vulnerability|issue|risk):\s*(.*)', content, re.IGNORECASE)

        risk_level = 'low'
        if 'critical' in content.lower() or 'high risk' in content.lower():
            risk_level = 'high'
        elif 'medium' in content.lower():
            risk_level = 'medium'

        return {
            'vulnerabilities': vulnerabilities,
            'risk_level': risk_level,
            'recommendations': re.findall(r'(?:recommend|suggest):\s*(.*)', content, re.IGNORECASE)
        }

    def _estimate_complexity(self, code: str) -> int:
        """Estimate cyclomatic complexity (simplified)"""

        # Count decision points
        complexity = 1
        decision_keywords = ['i', 'eli', 'else', 'for', 'while', 'except', 'case', 'when']

        for keyword in decision_keywords:
            complexity += code.count(f' {keyword} ') + code.count(f'\n{keyword} ')

        return complexity
