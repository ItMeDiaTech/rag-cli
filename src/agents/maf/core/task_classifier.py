#!/usr/bin/env python3
"""
Intelligent Task Classification System
Automatically determines the appropriate workflow and agents based on task description
"""

from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class TaskClassification:
    """Result of task classification"""
    primary_workflow: str
    agent_sequence: List[str]
    confidence: float
    task_type: str
    keywords_matched: List[str]
    suggested_requirements: List[str]
    needs_claude_classification: bool = False


class IntelligentTaskClassifier:
    """Classifies tasks and determines appropriate workflows and agents"""

    def __init__(self):
        # Define keyword patterns for different task types
        self.patterns = {
            'debug': {
                'keywords': ['debug', 'error', 'bug', 'fix', 'problem', 'issue', 'broken', 'crash',
                             'exception', 'fail', 'traceback', 'warning', 'vs code problems'],
                'workflow': 'bug_fix',
                'agents': ['debugger', 'developer', 'tester'],
                'requirements': ['error_analysis', 'debugging', 'testing']
            },
            'implement': {
                'keywords': ['implement', 'create', 'build', 'develop', 'code', 'write', 'make',
                             'add', 'new', 'feature', 'function', 'class', 'module', 'api'],
                'workflow': 'code_generation',
                'agents': ['architect', 'developer', 'reviewer', 'tester', 'documenter'],
                'requirements': ['implementation', 'design', 'testing', 'documentation']
            },
            'review': {
                'keywords': ['review', 'check', 'analyze', 'audit', 'inspect', 'examine',
                             'quality', 'standards', 'best practices'],
                'workflow': 'code_review',
                'agents': ['reviewer', 'tester'],
                'requirements': ['code_review', 'quality_assurance']
            },
            'test': {
                'keywords': ['test', 'testing', 'unit test', 'integration test', 'coverage',
                             'validate', 'verification', 'qa', 'quality assurance'],
                'workflow': 'testing',
                'agents': ['tester', 'developer'],
                'requirements': ['testing', 'validation']
            },
            'optimize': {
                'keywords': ['optimize', 'performance', 'speed', 'efficiency', 'improve',
                             'refactor', 'enhance', 'faster', 'memory', 'cpu'],
                'workflow': 'optimization',
                'agents': ['optimizer', 'developer', 'tester', 'reviewer'],
                'requirements': ['performance_optimization', 'refactoring']
            },
            'document': {
                'keywords': ['document', 'docs', 'readme', 'comment', 'explain', 'description',
                             'api doc', 'help', 'guide', 'manual'],
                'workflow': 'documentation',
                'agents': ['documenter', 'reviewer'],
                'requirements': ['documentation', 'technical_writing']
            },
            'design': {
                'keywords': ['design', 'architecture', 'plan', 'structure', 'schema',
                             'pattern', 'framework', 'system', 'blueprint'],
                'workflow': 'system_design',
                'agents': ['architect', 'developer', 'reviewer'],
                'requirements': ['system_design', 'architecture']
            }
        }

        # VS Code specific patterns
        self.vscode_patterns = {
            'vscode_problems': {
                'keywords': ['vs code', 'vscode', 'problems', 'warnings', 'errors', 'diagnostics',
                             'problems panel', 'intellisense', 'linting'],
                'workflow': 'bug_fix',
                'agents': ['debugger', 'developer'],
                'requirements': ['vscode_integration', 'error_analysis']
            }
        }

        # Confidence thresholds
        self.high_confidence_threshold = 0.8
        self.medium_confidence_threshold = 0.5
        self.low_confidence_threshold = 0.3

    def classify_task(self, task_description: str) -> TaskClassification:
        """Classify a task based on its description"""

        task_lower = task_description.lower()

        # Score each task type
        scores = {}
        matched_keywords = {}

        # Check main patterns
        all_patterns = {**self.patterns, **self.vscode_patterns}

        for task_type, config in all_patterns.items():
            score = 0
            keywords_found = []

            for keyword in config['keywords']:
                if keyword in task_lower:
                    # Weight longer keywords more heavily
                    weight = len(keyword.split()) * 1.5
                    score += weight
                    keywords_found.append(keyword)

            # Bonus for multiple keyword matches
            if len(keywords_found) > 1:
                score *= 1.2

            scores[task_type] = score
            matched_keywords[task_type] = keywords_found

        # Find the best match
        if not scores:
            return self._create_fallback_classification(task_description)

        best_match = max(scores.items(), key=lambda x: x[1])
        task_type, best_score = best_match

        # Calculate confidence
        total_words = len(task_lower.split())
        confidence = min(best_score / max(total_words, 1), 1.0)

        # Get configuration for best match
        if task_type in self.patterns:
            config = self.patterns[task_type]
        else:
            config = self.vscode_patterns[task_type]

        # Determine if we need Claude's help for classification
        needs_claude = confidence < self.medium_confidence_threshold

        return TaskClassification(
            primary_workflow=config['workflow'],
            agent_sequence=config['agents'].copy(),
            confidence=confidence,
            task_type=task_type,
            keywords_matched=matched_keywords[task_type],
            suggested_requirements=config['requirements'].copy(),
            needs_claude_classification=needs_claude
        )

    def _create_fallback_classification(self, task_description: str) -> TaskClassification:
        """Create a fallback classification for unmatched tasks"""

        return TaskClassification(
            primary_workflow='bug_fix',  # Default to most common workflow
            agent_sequence=['developer', 'reviewer'],
            confidence=0.1,
            task_type='general',
            keywords_matched=[],
            suggested_requirements=['general'],
            needs_claude_classification=True
        )

    async def get_claude_classification(self, task_description: str, claude_cli) -> Optional[TaskClassification]:
        """Use Claude to classify ambiguous tasks"""

        classification_prompt = """
Analyze this task description and classify it into one of these categories:

Task: "{task_description}"

Categories:
1. debug - Fix errors, bugs, problems, issues
2. implement - Create new code, features, functions
3. review - Check code quality, analyze existing code
4. test - Write tests, validate functionality
5. optimize - Improve performance, refactor code
6. document - Write documentation, comments, guides
7. design - Plan architecture, system design

Please respond with JSON in this exact format:
{{
    "category": "one_of_the_categories_above",
    "confidence": 0.95,
    "reasoning": "brief explanation of why you chose this category",
    "suggested_agents": ["agent1", "agent2"],
    "workflow": "suggested_workflow_name"
}}

Focus on the main intent of the task.
"""

        try:
            response = await claude_cli.complete(classification_prompt, max_tokens=200)

            if response.success:
                # Try to extract JSON from response
                import json
                import re

                json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
                if json_match:
                    claude_result = json.loads(json_match.group())

                    # Convert Claude's classification to our format
                    category = claude_result.get('category', 'debug')
                    confidence = claude_result.get('confidence', 0.5)

                    # Map to our workflow system
                    workflow_map = {
                        'debug': 'bug_fix',
                        'implement': 'code_generation',
                        'review': 'code_review',
                        'test': 'testing',
                        'optimize': 'optimization',
                        'document': 'documentation',
                        'design': 'system_design'
                    }

                    agent_map = {
                        'debug': ['debugger', 'developer', 'tester'],
                        'implement': ['architect', 'developer', 'reviewer', 'tester'],
                        'review': ['reviewer', 'tester'],
                        'test': ['tester', 'developer'],
                        'optimize': ['optimizer', 'developer', 'reviewer'],
                        'document': ['documenter', 'reviewer'],
                        'design': ['architect', 'developer', 'reviewer']
                    }

                    return TaskClassification(
                        primary_workflow=workflow_map.get(category, 'bug_fix'),
                        agent_sequence=agent_map.get(category, ['developer']),
                        confidence=confidence,
                        task_type=category,
                        keywords_matched=['claude_classified'],
                        suggested_requirements=[category, 'claude_analyzed'],
                        needs_claude_classification=False
                    )

        except Exception as e:
            print(f"Claude classification failed: {e}")

        return None

    def get_task_summary(self, classification: TaskClassification, task_description: str) -> Dict[str, any]:
        """Generate a summary of the task classification"""

        return {
            'original_task': task_description,
            'classification': {
                'type': classification.task_type,
                'workflow': classification.primary_workflow,
                'confidence': f"{classification.confidence:.2f}",
                'confidence_level': self._get_confidence_level(classification.confidence)
            },
            'execution_plan': {
                'agents': classification.agent_sequence,
                'estimated_steps': len(classification.agent_sequence),
                'requirements': classification.suggested_requirements
            },
            'matched_keywords': classification.keywords_matched,
            'needs_clarification': classification.needs_claude_classification
        }

    def _get_confidence_level(self, confidence: float) -> str:
        """Convert confidence score to human readable level"""
        if confidence >= self.high_confidence_threshold:
            return "High"
        elif confidence >= self.medium_confidence_threshold:
            return "Medium"
        elif confidence >= self.low_confidence_threshold:
            return "Low"
        else:
            return "Very Low"

    def suggest_task_refinements(self, task_description: str, classification: TaskClassification) -> List[str]:
        """Suggest ways to improve task description for better classification"""

        suggestions = []

        if classification.confidence < self.medium_confidence_threshold:
            suggestions.append("Consider being more specific about what you want to accomplish")

        if not classification.keywords_matched:
            suggestions.append("Try including keywords like: fix, create, test, review, optimize, document")

        if classification.task_type == 'general':
            suggestions.append("Specify whether you want to: debug issues, implement features, or review code")

        if len(task_description.split()) < 3:
            suggestions.append("Provide more details about the task context")

        return suggestions
