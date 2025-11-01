"""
Optimizer Agent - Specialized in performance and efficiency optimization
"""

import re
from typing import Any, Dict, List

from ..core.agent import Agent, AgentConfig


class OptimizerAgent(Agent):
    """Agent specialized in code and system optimization"""

    def __init__(self, config: AgentConfig, claude_cli=None, memory_manager=None, message_bus=None):
        config.capabilities = [
            "performance_analysis", "algorithm_optimization", "memory_optimization",
            "code_refactoring", "database_optimization", "caching_strategies",
            "async_optimization", "parallel_processing", "resource_optimization", "analyze", "optimize", "improve", "enhance"
        ]

        super().__init__(config, claude_cli, memory_manager, message_bus)

        self.optimization_techniques = {
            'algorithmic': ['Dynamic programming', 'Memoization', 'Binary search', 'Divide and conquer'],
            'memory': ['Object pooling', 'Lazy loading', 'Memory streaming', 'Garbage collection tuning'],
            'io': ['Async I/O', 'Batch processing', 'Connection pooling', 'Caching'],
            'database': ['Query optimization', 'Indexing', 'Denormalization', 'Partitioning'],
            'concurrency': ['Threading', 'Multiprocessing', 'Async/await', 'Worker pools']
        }

    async def _execute_task(self, task_id: str, task: Dict[str, Any], context: Dict[str, Any], memories: List[Dict]) -> Any:
        """Execute optimizer-specific tasks"""

        task_type = task.get('type', '').lower()

        if 'optimize' in task_type or 'performance' in task_type:
            return await self._optimize_code(task_id, task, context, memories)
        elif 'analyze' in task_type:
            return await self._analyze_performance(task_id, task, context, memories)
        elif 'refactor' in task_type:
            return await self._refactor_for_performance(task_id, task, context, memories)
        elif 'database' in task_type:
            return await self._optimize_database(task_id, task, context, memories)
        else:
            return await self._optimize_code(task_id, task, context, memories)

    async def _optimize_code(self, task_id: str, task: Dict[str, Any], context: Dict[str, Any], memories: List[Dict]) -> Dict[str, Any]:
        """Optimize code for performance"""

        self.logger.info("[%s] Optimizing code performance", task_id)

        code = task.get('code', '')
        if not code and context.get('results', {}).get('Developer'):
            dev_result = context['results']['Developer']
            if isinstance(dev_result, dict) and 'code' in dev_result:
                code_blocks = dev_result['code']
                if code_blocks and isinstance(code_blocks, list):
                    code = code_blocks[0].get('content', '')

        prompt = """
Optimize the following code for better performance:

Code:
```
{code}
```

Known Issues:
{json.dumps(performance_issues, indent=2) if performance_issues else 'General optimization needed'}

Focus on:
1. Algorithm complexity reduction
2. Memory usage optimization
3. I/O operation improvements
4. Caching opportunities
5. Parallel processing potential
6. Database query optimization

Provide:
- Optimized code
- Performance improvements achieved
- Complexity analysis (before/after)
- Trade-offs made
"""

        if self.claude_cli:
            response = await self.claude_cli.complete(
                prompt,
                max_tokens=self.config.max_tokens,
                temperature=0.5,
                system="You are a performance optimization expert. Focus on measurable improvements while maintaining code correctness."
            )

            if response.success:
                return self._parse_optimization_response(response.content)
            raise Exception(f"Code optimization failed: {response.error}")
        else:
            return self._mock_optimization_response()

    async def _analyze_performance(self, task_id: str, task: Dict[str, Any], context: Dict[str, Any], memories: List[Dict]) -> Dict[str, Any]:
        """Analyze code performance characteristics"""

        self.logger.info("[%s] Analyzing performance", task_id)

        prompt = """
Analyze the performance characteristics of this code:

```
{code}
```

Analyze:
1. Time complexity (Big O notation)
2. Space complexity
3. Bottlenecks and hotspots
4. Resource usage patterns
5. Scalability limitations
6. Optimization opportunities
"""

        if self.claude_cli:
            response = await self.claude_cli.complete(prompt, temperature=0.4)

            if response.success:
                return self._parse_analysis_response(response.content)
            raise Exception(f"Performance analysis failed: {response.error}")
        else:
            return {
                'time_complexity': 'O(n²)',
                'space_complexity': 'O(n)',
                'bottlenecks': ['Nested loops', 'Multiple database queries'],
                'resource_usage': {
                    'cpu': 'High',
                    'memory': 'Moderate',
                    'io': 'Heavy'
                },
                'scalability_issues': ['Linear scaling with data size'],
                'optimization_opportunities': [
                    'Use hash tables for lookups',
                    'Implement caching',
                    'Batch database operations'
                ]
            }

    async def _refactor_for_performance(self, task_id: str, task: Dict[str, Any], context: Dict[str, Any], memories: List[Dict]) -> Dict[str, Any]:
        """Refactor code for better performance"""

        self.logger.info("[%s] Refactoring for performance", task_id)

        prompt = """
Refactor this code for optimal performance:

```
{code}
```

Refactoring goals:
1. Reduce algorithmic complexity
2. Improve data structure usage
3. Optimize memory allocation
4. Enable better parallelization
5. Reduce I/O operations
6. Improve cache locality

Maintain functionality while improving performance.
"""

        if self.claude_cli:
            response = await self.claude_cli.complete(prompt, temperature=0.5)

            if response.success:
                return {
                    'refactored_code': self._extract_code(response.content),
                    'improvements': self._extract_improvements(response.content),
                    'performance_gains': self._extract_performance_metrics(response.content)
                }
            raise Exception(f"Performance refactoring failed: {response.error}")
        else:
            return {
                'refactored_code': '# Refactored code with optimizations',
                'improvements': ['Used dictionary for O(1) lookups', 'Implemented caching'],
                'performance_gains': '40% faster execution'
            }

    async def _optimize_database(self, task_id: str, task: Dict[str, Any], context: Dict[str, Any], memories: List[Dict]) -> Dict[str, Any]:
        """Optimize database queries and schema"""

        self.logger.info("[%s] Optimizing database", task_id)

        prompt = """
Optimize database performance:

Queries:
{json.dumps(queries, indent=2) if queries else 'No queries provided'}

Schema:
{json.dumps(schema, indent=2) if schema else 'No schema provided'}

Optimize:
1. Query execution plans
2. Index recommendations
3. Query rewriting for efficiency
4. Schema optimizations
5. Caching strategies
6. Partitioning recommendations
"""

        if self.claude_cli:
            response = await self.claude_cli.complete(prompt, temperature=0.4)

            if response.success:
                return self._parse_database_optimization(response.content)
            raise Exception(f"Database optimization failed: {response.error}")
        else:
            return {
                'optimized_queries': ['SELECT * FROM users WHERE indexed_column = ?'],
                'index_recommendations': [
                    'CREATE INDEX idx_users_email ON users(email)',
                    'CREATE INDEX idx_orders_user_date ON orders(user_id, order_date)'
                ],
                'schema_changes': ['Add denormalized column for frequent joins'],
                'caching_strategy': 'Implement Redis cache for frequent queries',
                'expected_improvement': '60% query time reduction'
            }

    def _parse_optimization_response(self, content: str) -> Dict[str, Any]:
        """Parse optimization response"""

        # Extract optimized code
        code_blocks = re.findall(r'```(?:\w+)?\n(.*?)```', content, re.DOTALL)
        optimized_code = code_blocks[0] if code_blocks else ''

        # Extract performance metrics
        performance_gains = self._extract_performance_metrics(content)

        # Extract complexity analysis
        complexity_before = re.search(r'before.*?O\(([^)]+)\)', content, re.IGNORECASE)
        complexity_after = re.search(r'after.*?O\(([^)]+)\)', content, re.IGNORECASE)

        complexity = {
            'before': complexity_before.group(1) if complexity_before else 'unknown',
            'after': complexity_after.group(1) if complexity_after else 'unknown'
        }

        # Extract improvements
        improvements = self._extract_improvements(content)

        # Extract trade-offs
        tradeoffs = re.findall(r'trade-?off.*?:(.*?)(?:\n|$)', content, re.IGNORECASE)

        return {
            'optimized_code': optimized_code,
            'performance_gains': performance_gains,
            'complexity': complexity,
            'improvements': improvements,
            'tradeoffs': tradeoffs,
            'techniques_applied': self._identify_techniques(content)
        }

    def _parse_analysis_response(self, content: str) -> Dict[str, Any]:
        """Parse performance analysis response"""

        # Extract complexity
        time_complexity = re.search(r'time complexity.*?O\(([^)]+)\)', content, re.IGNORECASE)
        space_complexity = re.search(r'space complexity.*?O\(([^)]+)\)', content, re.IGNORECASE)

        # Extract bottlenecks
        bottlenecks = re.findall(r'bottleneck.*?:(.*?)(?:\n|$)', content, re.IGNORECASE)

        # Extract optimization opportunities
        opportunities = re.findall(r'(?:opportunity|optimize|improve).*?:(.*?)(?:\n|$)', content, re.IGNORECASE)

        return {
            'time_complexity': time_complexity.group(1) if time_complexity else 'unknown',
            'space_complexity': space_complexity.group(1) if space_complexity else 'unknown',
            'bottlenecks': bottlenecks[:5],
            'resource_usage': {
                'cpu': 'High' if 'cpu' in content.lower() else 'Moderate',
                'memory': 'High' if 'memory' in content.lower() else 'Moderate',
                'io': 'Heavy' if 'i/o' in content.lower() or 'disk' in content.lower() else 'Light'
            },
            'scalability_issues': re.findall(r'scalability.*?:(.*?)(?:\n|$)', content, re.IGNORECASE),
            'optimization_opportunities': opportunities[:5]
        }

    def _parse_database_optimization(self, content: str) -> Dict[str, Any]:
        """Parse database optimization response"""

        # Extract optimized queries
        queries = re.findall(r'(?:SELECT|INSERT|UPDATE|DELETE).*?;', content, re.IGNORECASE)

        # Extract index recommendations
        indexes = re.findall(r'CREATE INDEX.*?;', content, re.IGNORECASE)

        # Extract schema changes
        schema_changes = re.findall(r'ALTER TABLE.*?;', content, re.IGNORECASE)

        # Extract caching recommendations
        caching = 'Redis' if 'redis' in content.lower() else 'Memcached' if 'memcached' in content.lower() else 'Query cache'

        # Extract performance improvement
        improvement = re.search(r'(\d+)%?\s*(?:improvement|faster|reduction)', content)

        return {
            'optimized_queries': queries[:5],
            'index_recommendations': indexes[:5],
            'schema_changes': schema_changes[:3],
            'caching_strategy': caching,
            'expected_improvement': f"{improvement.group(1)}%" if improvement else 'Significant improvement expected'
        }

    def _extract_code(self, content: str) -> str:
        """Extract code from response"""

        code_blocks = re.findall(r'```(?:\w+)?\n(.*?)```', content, re.DOTALL)
        return code_blocks[0] if code_blocks else ''

    def _extract_improvements(self, content: str) -> List[str]:
        """Extract list of improvements"""

        improvements = []

        # Look for bullet points
        bullets = re.findall(r'[-*]\s+(.*?)(?:\n|$)', content)

        # Filter for improvement-related items
        for bullet in bullets:
            if any(word in bullet.lower() for word in ['optimiz', 'improv', 'reduc', 'faster', 'efficient']):
                improvements.append(bullet)

        return improvements[:10]

    def _extract_performance_metrics(self, content: str) -> str:
        """Extract performance improvement metrics"""

        # Look for percentage improvements
        percentages = re.findall(r'(\d+)%?\s*(?:faster|improvement|reduction|speedup)', content, re.IGNORECASE)

        if percentages:
            return f"{percentages[0]}% performance improvement"

        # Look for complexity improvements
        if 'O(n)' in content and 'O(n²)' in content:
            return 'Reduced from O(n²) to O(n)'
        if 'O(1)' in content:
            return 'Achieved O(1) complexity'

        return 'Performance improved'

    def _identify_techniques(self, content: str) -> List[str]:
        """Identify optimization techniques used"""

        techniques_found = []

        # Check for each technique category
        for category, techniques in self.optimization_techniques.items():
            for technique in techniques:
                if technique.lower() in content.lower():
                    techniques_found.append(technique)

        # Check for common optimization patterns
        common_patterns = {
            'caching': ['cache', 'memoiz'],
            'async': ['async', 'await', 'concurrent'],
            'batch': ['batch', 'bulk'],
            'index': ['index', 'btree', 'hash'],
            'pool': ['pool', 'connection pool', 'thread pool']
        }

        for pattern, keywords in common_patterns.items():
            if any(keyword in content.lower() for keyword in keywords):
                if pattern not in techniques_found:
                    techniques_found.append(pattern)

        return techniques_found[:5]

    def _mock_optimization_response(self) -> Dict[str, Any]:
        """Generate mock optimization response"""

        return {
            'optimized_code': '''# Optimized implementation
def optimized_function(data):
    # Use dictionary for O(1) lookups
    lookup = {item.id: item for item in data}

    # Process in batches for better memory usage
    batch_size = 1000
    results = []

    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        # Process batch with optimized algorithm
        batch_results = process_batch(batch, lookup)
        results.extend(batch_results)

    return results''',
            'performance_gains': '65% faster execution',
            'complexity': {
                'before': 'n²',
                'after': 'n log n'
            },
            'improvements': [
                'Replaced nested loops with hash table lookups',
                'Implemented batch processing for memory efficiency',
                'Added caching for repeated computations',
                'Used generator expressions to reduce memory usage'
            ],
            'tradeoffs': [
                'Slightly higher memory usage for hash tables',
                'More complex code structure'
            ],
            'techniques_applied': ['Memoization', 'Batch processing', 'Caching', 'Hash tables']
        }
