"""
Architect Agent - Specialized in system design and architecture
"""

from typing import Any, Dict, List
from ..core.agent import Agent, AgentConfig


class ArchitectAgent(Agent):
    """Agent specialized in system architecture and design"""

    def __init__(self, config: AgentConfig, claude_cli=None, memory_manager=None, message_bus=None):
        config.capabilities = [
            "system_design", "api_design", "database_schema",
            "architecture_patterns", "microservices", "cloud_architecture",
            "scalability_planning", "security_architecture", "design_patterns",
            "technology_selection", "integration_design",
            # Common action aliases - ADD THESE LINES
            "design", "architect", "plan"
        ]

        super().__init__(config, claude_cli, memory_manager, message_bus)

        self.design_patterns = [
            'Singleton', 'Factory', 'Observer', 'Strategy', 'Adapter',
            'MVC', 'MVP', 'MVVM', 'Repository', 'Unit of Work'
        ]

        self.architecture_styles = [
            'Monolithic', 'Microservices', 'Serverless', 'Event-driven',
            'Layered', 'Hexagonal', 'Clean Architecture', 'CQRS'
        ]

    async def _execute_task(self, task_id: str, task: Dict[str, Any], context: Dict[str, Any], memories: List[Dict]) -> Any:
        """Execute architect-specific tasks"""

        task_type = task.get('type', '').lower()

        if 'design' in task_type or 'architect' in task_type:
            return await self._design_system(task_id, task, context, memories)
        elif 'api' in task_type:
            return await self._design_api(task_id, task, context, memories)
        elif 'database' in task_type or 'schema' in task_type:
            return await self._design_database(task_id, task, context, memories)
        elif 'review' in task_type:
            return await self._review_architecture(task_id, task, context, memories)
        else:
            return await self._design_system(task_id, task, context, memories)

    async def _design_system(self, task_id: str, task: Dict[str, Any], context: Dict[str, Any], memories: List[Dict]) -> Dict[str, Any]:
        """Design system architecture"""

        self.logger.info("[%s] Designing system architecture", task_id)

        prompt = """
Design a comprehensive system architecture for:

Requirements:
{json.dumps(requirements, indent=2)}

Constraints:
{json.dumps(constraints, indent=2)}

Please provide:
1. High-level architecture diagram description
2. Component breakdown and responsibilities
3. Technology stack recommendations
4. Data flow and storage design
5. Security considerations
6. Scalability approach
7. Integration points
8. Deployment strategy
"""

        if self.claude_cli:
            response = await self.claude_cli.complete(
                prompt,
                max_tokens=self.config.max_tokens,
                temperature=0.7,
                system="You are an expert system architect. Design robust, scalable, and maintainable systems."
            )

            if response.success:
                return self._parse_architecture_response(response.content)
            else:
                raise Exception(f"System design failed: {response.error}")
        else:
            return self._mock_architecture_response()

    async def _design_api(self, task_id: str, task: Dict[str, Any], context: Dict[str, Any], memories: List[Dict]) -> Dict[str, Any]:
        """Design API structure"""

        self.logger.info("[%s] Designing API", task_id)

        prompt = """
Design a RESTful API for:

Functionality: {task.get('description', 'API design needed')}

Requirements:
{json.dumps(task.get('requirements', {}), indent=2)}

Provide:
1. Endpoint definitions (paths, methods, parameters)
2. Request/response schemas
3. Authentication/authorization approach
4. Error handling strategy
5. Versioning approach
6. Rate limiting and throttling
7. OpenAPI/Swagger specification outline
"""

        if self.claude_cli:
            response = await self.claude_cli.complete(prompt, temperature=0.6)

            if response.success:
                return self._parse_api_response(response.content)
            else:
                raise Exception(f"API design failed: {response.error}")
        else:
            return {
                'endpoints': [
                    {'path': '/api/v1/users', 'method': 'GET', 'description': 'List users'},
                    {'path': '/api/v1/users/{id}', 'method': 'GET', 'description': 'Get user'},
                    {'path': '/api/v1/users', 'method': 'POST', 'description': 'Create user'}
                ],
                'authentication': 'JWT Bearer token',
                'versioning': 'URL path versioning (/v1, /v2)',
                'rate_limiting': '100 requests per minute per API key',
                'error_format': 'Standard HTTP status codes with JSON error details'
            }

    async def _design_database(self, task_id: str, task: Dict[str, Any], context: Dict[str, Any], memories: List[Dict]) -> Dict[str, Any]:
        """Design database schema"""

        self.logger.info("[%s] Designing database schema", task_id)

        prompt = """
Design a database schema for:

Entities: {json.dumps(entities, indent=2)}
Relationships: {json.dumps(relationships, indent=2)}

Requirements:
{json.dumps(task.get('requirements', {}), indent=2)}

Provide:
1. Table definitions with columns and data types
2. Primary and foreign keys
3. Indexes for performance
4. Constraints and validations
5. Normalization level and rationale
6. Sample SQL DDL statements
7. Data migration strategy
"""

        if self.claude_cli:
            response = await self.claude_cli.complete(prompt, temperature=0.5)

            if response.success:
                return self._parse_database_response(response.content)
            else:
                raise Exception(f"Database design failed: {response.error}")
        else:
            return {
                'tables': [
                    {
                        'name': 'users',
                        'columns': [
                            {'name': 'id', 'type': 'UUID', 'primary_key': True},
                            {'name': 'email', 'type': 'VARCHAR(255)', 'unique': True},
                            {'name': 'created_at', 'type': 'TIMESTAMP'}
                        ]
                    }
                ],
                'indexes': ['CREATE INDEX idx_users_email ON users(email)'],
                'constraints': ['ALTER TABLE orders ADD CONSTRAINT fk_user FOREIGN KEY (user_id) REFERENCES users(id)'],
                'normalization': '3NF',
                'database_type': 'PostgreSQL'
            }

    async def _review_architecture(self, task_id: str, task: Dict[str, Any], context: Dict[str, Any], memories: List[Dict]) -> Dict[str, Any]:
        """Review existing architecture"""

        self.logger.info("[%s] Reviewing architecture", task_id)

        prompt = """
Review the following architecture:

{json.dumps(architecture, indent=2)}

Evaluate:
1. Scalability potential
2. Security vulnerabilities
3. Performance bottlenecks
4. Maintainability concerns
5. Technology choices
6. Design pattern usage
7. Best practices adherence
8. Improvement recommendations
"""

        if self.claude_cli:
            response = await self.claude_cli.complete(prompt, temperature=0.4)

            if response.success:
                return self._parse_review_response(response.content)
            else:
                raise Exception(f"Architecture review failed: {response.error}")
        else:
            return {
                'score': 7.5,
                'strengths': ['Good separation of concerns', 'Scalable design'],
                'weaknesses': ['Single point of failure in auth service', 'No caching layer'],
                'recommendations': ['Add Redis cache', 'Implement circuit breakers', 'Use message queue for async operations'],
                'security_issues': ['API keys in environment variables need better management'],
                'performance_concerns': ['Database queries not optimized', 'No CDN for static assets']
            }

    def _parse_architecture_response(self, content: str) -> Dict[str, Any]:
        """Parse architecture design response"""

        result = {
            'architecture_style': 'microservices',  # Default
            'components': [],
            'technology_stack': {},
            'data_flow': [],
            'security_measures': [],
            'scalability_approach': '',
            'deployment_strategy': ''
        }

        # Extract architecture style
        for style in self.architecture_styles:
            if style.lower() in content.lower():
                result['architecture_style'] = style
                break

        # Extract components
        component_section = content.split('Component')[1].split('\n\n')[0] if 'Component' in content else ''
        result['components'] = [line.strip('- ') for line in component_section.split('\n') if line.strip().startswith('-')]

        # Extract technology mentions
        tech_keywords = ['Python', 'Node.js', 'React', 'PostgreSQL', 'MongoDB', 'Redis', 'Docker', 'Kubernetes']
        result['technology_stack'] = {
            'mentioned_technologies': [tech for tech in tech_keywords if tech.lower() in content.lower()]
        }

        return result

    def _parse_api_response(self, content: str) -> Dict[str, Any]:
        """Parse API design response"""

        import re

        # Extract endpoints
        endpoint_pattern = r'(GET|POST|PUT|DELETE|PATCH)\s+(/[\w/{}]+)'
        endpoints = []
        for match in re.finditer(endpoint_pattern, content):
            endpoints.append({
                'method': match.group(1),
                'path': match.group(2),
                'description': ''
            })

        # Extract authentication method
        auth = 'JWT' if 'jwt' in content.lower() else 'API Key' if 'api key' in content.lower() else 'OAuth2'

        return {
            'endpoints': endpoints,
            'authentication': auth,
            'versioning': 'URL path versioning',
            'rate_limiting': '100 requests per minute',
            'error_format': 'JSON with status codes'
        }

    def _parse_database_response(self, content: str) -> Dict[str, Any]:
        """Parse database design response"""

        import re

        tables = []

        # Extract CREATE TABLE statements
        table_pattern = r'CREATE TABLE (\w+)\s*\((.*?)\);'
        for match in re.finditer(table_pattern, content, re.DOTALL):
            table_name = match.group(1)
            columns_str = match.group(2)

            columns = []
            for col_line in columns_str.split(','):
                col_parts = col_line.strip().split()
                if col_parts:
                    columns.append({
                        'name': col_parts[0],
                        'type': col_parts[1] if len(col_parts) > 1 else 'VARCHAR',
                        'primary_key': 'PRIMARY KEY' in col_line
                    })

            tables.append({
                'name': table_name,
                'columns': columns
            })

        # Determine database type
        db_type = 'PostgreSQL'
        if 'mysql' in content.lower():
            db_type = 'MySQL'
        elif 'mongodb' in content.lower():
            db_type = 'MongoDB'

        return {
            'tables': tables,
            'indexes': re.findall(r'CREATE INDEX.*?;', content),
            'constraints': re.findall(r'ALTER TABLE.*?;', content),
            'normalization': '3NF' if '3n' in content.lower() else '2NF',
            'database_type': db_type
        }

    def _parse_review_response(self, content: str) -> Dict[str, Any]:
        """Parse architecture review response"""

        import re

        # Extract score
        score_match = re.search(r'(\d+(?:\.\d+)?)/10', content)
        score = float(score_match.group(1)) if score_match else 7.0

        # Extract lists
        strengths = re.findall(r'(?:strength|positive|good).*?:\s*(.*)', content, re.IGNORECASE)
        weaknesses = re.findall(r'(?:weakness|issue|concern).*?:\s*(.*)', content, re.IGNORECASE)
        recommendations = re.findall(r'(?:recommend|suggest|improve).*?:\s*(.*)', content, re.IGNORECASE)

        return {
            'score': score,
            'strengths': strengths[:3] if strengths else ['Well-structured design'],
            'weaknesses': weaknesses[:3] if weaknesses else ['Needs optimization'],
            'recommendations': recommendations[:5] if recommendations else ['Consider caching'],
            'security_issues': ['Review authentication flow'],
            'performance_concerns': ['Monitor database queries']
        }

    def _mock_architecture_response(self) -> Dict[str, Any]:
        """Generate mock architecture response"""

        return {
            'architecture_style': 'Microservices',
            'components': [
                'API Gateway',
                'Authentication Service',
                'User Service',
                'Data Processing Service',
                'Notification Service',
                'Database Layer'
            ],
            'technology_stack': {
                'backend': 'Python/FastAPI',
                'frontend': 'React/TypeScript',
                'database': 'PostgreSQL',
                'cache': 'Redis',
                'message_queue': 'RabbitMQ',
                'container': 'Docker',
                'orchestration': 'Kubernetes'
            },
            'data_flow': [
                'Client -> API Gateway -> Service',
                'Service -> Message Queue -> Worker',
                'Service -> Cache -> Database'
            ],
            'security_measures': [
                'JWT authentication',
                'Rate limiting',
                'Input validation',
                'HTTPS encryption',
                'API key management'
            ],
            'scalability_approach': 'Horizontal scaling with load balancing',
            'deployment_strategy': 'Blue-green deployment with rolling updates'
        }
