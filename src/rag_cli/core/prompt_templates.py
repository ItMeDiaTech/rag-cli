"""Structured prompt templates for RAG-CLI.

This module provides standardized, well-structured prompts for consistent
and high-quality responses from Claude.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

from rag_cli_plugin.services.logger import get_logger

logger = get_logger(__name__)


class PromptType(Enum):
    """Types of prompts for different use cases."""
    GENERAL_QA = "general_qa"
    TECHNICAL_DOCS = "technical_docs"
    CODE_EXPLANATION = "code_explanation"
    TROUBLESHOOTING = "troubleshooting"
    HOW_TO = "how_to"
    COMPARISON = "comparison"
    BEST_PRACTICES = "best_practices"


@dataclass
class PromptTemplate:
    """Structured prompt template."""
    system_prompt: str
    user_template: str
    response_format: str
    few_shot_examples: List[Dict[str, str]]
    metadata: Dict[str, Any]


class PromptTemplateManager:
    """Manages structured prompt templates for RAG responses."""

    def __init__(self):
        """Initialize prompt template manager."""
        self.templates = self._initialize_templates()
        logger.info("Prompt template manager initialized",
                    template_count=len(self.templates))

    def _initialize_templates(self) -> Dict[PromptType, PromptTemplate]:
        """Initialize all prompt templates.

        Returns:
            Dictionary of prompt templates
        """
        return {
            PromptType.GENERAL_QA: self._create_general_qa_template(),
            PromptType.TECHNICAL_DOCS: self._create_technical_docs_template(),
            PromptType.CODE_EXPLANATION: self._create_code_explanation_template(),
            PromptType.TROUBLESHOOTING: self._create_troubleshooting_template(),
            PromptType.HOW_TO: self._create_how_to_template(),
            PromptType.COMPARISON: self._create_comparison_template(),
            PromptType.BEST_PRACTICES: self._create_best_practices_template(),
        }

    def _create_general_qa_template(self) -> PromptTemplate:
        """Create general Q&A template."""
        system_prompt = """You are a helpful technical documentation assistant with access to a knowledge base.

TASK: Answer the user's question using ONLY the provided context from the knowledge base.

GUIDELINES:
1. Extract relevant information from the provided context
2. Synthesize a clear, accurate answer
3. Cite sources using [Source: filename] format
4. If context is insufficient, explicitly state what information is missing
5. Be concise but comprehensive

RESPONSE FORMAT:
- Direct answer to the question
- Supporting details from context
- Source citations
- Related information (if applicable)"""

        user_template = """### Retrieved Context from Knowledge Base

{context}

### User Question
{query}

Please answer the question based on the context provided above."""

        response_format = """Answer: [Direct answer here]

Details: [Supporting details from context]

Sources: [List of sources cited]"""

        few_shot_examples = [
            {
                "query": "What is FAISS used for?",
                "response": "Answer: FAISS is a library for efficient similarity search and clustering of dense vectors.\n\nDetails: According to the documentation, FAISS provides algorithms for searching in sets of vectors of any size, up to ones that possibly do not fit in RAM. It supports both exact and approximate nearest neighbor search.\n\nSources: [Source: faiss_overview.md]"
            }
        ]

        return PromptTemplate(
            system_prompt=system_prompt,
            user_template=user_template,
            response_format=response_format,
            few_shot_examples=few_shot_examples,
            metadata={"type": "general_qa", "priority": 1}
        )

    def _create_technical_docs_template(self) -> PromptTemplate:
        """Create technical documentation template."""
        system_prompt = """You are a technical documentation expert with access to official documentation.

TASK: Provide accurate technical information based on the documentation in the knowledge base.

GUIDELINES:
1. Use precise technical terminology
2. Include code examples when available in context
3. Cite specific documentation sections
4. Highlight important notes, warnings, or requirements
5. Structure information hierarchically

RESPONSE FORMAT:
- Overview
- Key concepts
- Code examples (if applicable)
- Requirements/Prerequisites
- Additional notes
- Documentation references"""

        user_template = """### Documentation Context

{context}

### Technical Query
{query}

Please provide technical information based on the documentation above."""

        response_format = """## Overview
[Brief overview]

## Key Concepts
- [Concept 1]
- [Concept 2]

## Code Examples
```
[Code if available]
```

## Requirements
- [Requirement 1]

## Documentation References
[Source: file.md, Section: X]"""

        few_shot_examples = []

        return PromptTemplate(
            system_prompt=system_prompt,
            user_template=user_template,
            response_format=response_format,
            few_shot_examples=few_shot_examples,
            metadata={"type": "technical_docs", "priority": 2}
        )

    def _create_code_explanation_template(self) -> PromptTemplate:
        """Create code explanation template."""
        system_prompt = """You are a code explanation expert with access to code documentation and examples.

TASK: Explain code functionality, patterns, and best practices based on the provided context.

GUIDELINES:
1. Break down code into understandable components
2. Explain the purpose and flow
3. Highlight important patterns or idioms
4. Note potential issues or improvements
5. Reference related code or documentation

RESPONSE FORMAT:
- What the code does (high-level)
- How it works (step-by-step)
- Key components explained
- Important notes
- Related resources"""

        user_template = """### Code Context

{context}

### Question about Code
{query}

Please explain the code based on the context provided."""

        response_format = """## What It Does
[High-level purpose]

## How It Works
1. [Step 1]
2. [Step 2]

## Key Components
- [Component 1]: [Explanation]

## Important Notes
- [Note 1]

## Related Resources
[Source: file.py, Line: X]"""

        few_shot_examples = []

        return PromptTemplate(
            system_prompt=system_prompt,
            user_template=user_template,
            response_format=response_format,
            few_shot_examples=few_shot_examples,
            metadata={"type": "code_explanation", "priority": 3}
        )

    def _create_troubleshooting_template(self) -> PromptTemplate:
        """Create troubleshooting template."""
        system_prompt = """You are a troubleshooting expert with access to error documentation and solutions.

TASK: Help diagnose and resolve issues based on error documentation in the knowledge base.

GUIDELINES:
1. Identify the root cause based on context
2. Provide step-by-step solutions
3. List common pitfalls to avoid
4. Suggest verification steps
5. Reference official troubleshooting guides

RESPONSE FORMAT:
- Problem diagnosis
- Root cause
- Solution steps (numbered)
- Verification
- Prevention tips
- Additional resources"""

        user_template = """### Error Documentation & Solutions

{context}

### Issue/Error
{query}

Please help troubleshoot this issue based on the documentation above."""

        response_format = """## Diagnosis
[What's wrong]

## Root Cause
[Why it's happening]

## Solution
1. [Step 1]
2. [Step 2]

## Verify Solution
- [Verification step 1]

## Prevention
- [Prevention tip 1]

## Additional Resources
[Source: troubleshooting.md]"""

        few_shot_examples = []

        return PromptTemplate(
            system_prompt=system_prompt,
            user_template=user_template,
            response_format=response_format,
            few_shot_examples=few_shot_examples,
            metadata={"type": "troubleshooting", "priority": 4}
        )

    def _create_how_to_template(self) -> PromptTemplate:
        """Create how-to guide template."""
        system_prompt = """You are a tutorial writer with access to implementation guides and examples.

TASK: Create clear, actionable how-to instructions based on documentation in the knowledge base.

GUIDELINES:
1. Break down into clear, sequential steps
2. Include prerequisites upfront
3. Provide code examples or commands
4. Add tips and warnings
5. Reference official guides

RESPONSE FORMAT:
- Prerequisites
- Step-by-step instructions
- Code/Command examples
- Tips & Warnings
- Next steps
- Documentation links"""

        user_template = """### Implementation Guides & Examples

{context}

### How-To Request
{query}

Please provide step-by-step instructions based on the guides above."""

        response_format = """## Prerequisites
- [Prerequisite 1]

## Steps
1. [Step 1]
   ```
   [Command/Code]
   ```

2. [Step 2]

## Tips & Warnings
  [WARNING] [Warning]
[TIP] [Tip]

## Next Steps
- [Next step 1]

## Documentation
[Source: guide.md]"""

        few_shot_examples = []

        return PromptTemplate(
            system_prompt=system_prompt,
            user_template=user_template,
            response_format=response_format,
            few_shot_examples=few_shot_examples,
            metadata={"type": "how_to", "priority": 5}
        )

    def _create_comparison_template(self) -> PromptTemplate:
        """Create comparison template."""
        system_prompt = """You are a technical comparison expert with access to documentation for multiple options.

TASK: Provide objective comparisons based on documentation in the knowledge base.

GUIDELINES:
1. Compare key characteristics side-by-side
2. List pros and cons for each option
3. Provide use case recommendations
4. Note compatibility or migration considerations
5. Reference official comparison documentation

RESPONSE FORMAT:
- Overview of options
- Feature comparison table
- Pros & Cons for each
- Use case recommendations
- Migration notes (if applicable)
- Documentation references"""

        user_template = """### Documentation for Comparison

{context}

### Comparison Request
{query}

Please provide a comparison based on the documentation above."""

        response_format = """## Options Overview
- Option 1: [Brief description]
- Option 2: [Brief description]

## Feature Comparison
| Feature | Option 1 | Option 2 |
|---------|----------|----------|
| [Feature 1] | [Value] | [Value] |

## Pros & Cons
**Option 1:**
+ [Pro 1]
- [Con 1]

**Option 2:**
+ [Pro 1]
- [Con 1]

## Recommendations
- Use Option 1 when: [Use case]
- Use Option 2 when: [Use case]

## Documentation
[Source: comparison.md]"""

        few_shot_examples = []

        return PromptTemplate(
            system_prompt=system_prompt,
            user_template=user_template,
            response_format=response_format,
            few_shot_examples=few_shot_examples,
            metadata={"type": "comparison", "priority": 6}
        )

    def _create_best_practices_template(self) -> PromptTemplate:
        """Create best practices template."""
        system_prompt = """You are an expert software architect with access to official documentation and best practices guides.

TASK: Provide authoritative best practices recommendations based on official documentation in the knowledge base.

GUIDELINES:
1. Prioritize official documentation and authoritative sources
2. Provide clear, actionable recommendations
3. Explain the reasoning behind each recommendation
4. Include practical examples
5. Mention common alternatives and when to use them
6. Highlight potential pitfalls to avoid
7. Always cite sources, especially official documentation

RESPONSE FORMAT:
- Clear recommendation with reasoning
- Practical example or code snippet
- Alternative approaches (if applicable)
- Common pitfalls or anti-patterns to avoid
- Source citations (prioritize official docs)"""

        user_template = """### Best Practices Documentation

{context}

### Question about Best Practices
{query}

Please provide best practices guidance based on the official documentation above."""

        response_format = """## Recommended Approach
[Clear recommendation with explanation of why this is best practice]

## Reasoning
[Explain the benefits and rationale behind this recommendation]

## Example
```
[Practical code example or implementation]
```

## Alternative Approaches
- **Alternative 1**: [Description]
  - Use when: [Scenario]
- **Alternative 2**: [Description]
  - Use when: [Scenario]

## Common Pitfalls to Avoid
- [Anti-pattern 1]: [Why to avoid]
- [Anti-pattern 2]: [Why to avoid]

## Additional Context
[Any relevant performance, security, or maintainability considerations]

## Official Documentation
[Source: official_docs.md] - Priority: Official Documentation
[Source: examples.md] - Priority: Community Examples"""

        few_shot_examples = []

        return PromptTemplate(
            system_prompt=system_prompt,
            user_template=user_template,
            response_format=response_format,
            few_shot_examples=few_shot_examples,
            metadata={"type": "best_practices", "priority": 7, "requires_authoritative": True}
        )

    def detect_prompt_type(self, query: str) -> PromptType:
        """Detect appropriate prompt type for query.

        Args:
            query: User query

        Returns:
            Detected prompt type
        """
        query_lower = query.lower()

        # Best practices (check first as it's specific)
        if any(kw in query_lower for kw in ['best practice', 'recommended', 'should i', 'idiomatic', 'convention', 'standard', 'anti-pattern', 'avoid', 'is it good']):
            return PromptType.BEST_PRACTICES

        # Troubleshooting
        if any(kw in query_lower for kw in ['error', 'exception', 'failed', 'not working', 'issue', 'problem', 'bug']):
            return PromptType.TROUBLESHOOTING

        # How-to
        if any(kw in query_lower for kw in ['how to', 'how do i', 'how can i', 'steps to', 'guide to']):
            return PromptType.HOW_TO

        # Comparison
        if any(kw in query_lower for kw in ['vs', 'versus', 'compare', 'difference between', 'which is better', 'or']):
            return PromptType.COMPARISON

        # Code explanation
        if any(kw in query_lower for kw in ['explain code', 'what does this code', 'how does this function', 'code review']):
            return PromptType.CODE_EXPLANATION

        # Technical docs (specific technical terms)
        if any(kw in query_lower for kw in ['api', 'configuration', 'parameters', 'settings', 'documentation']):
            return PromptType.TECHNICAL_DOCS

        # Default to general Q&A
        return PromptType.GENERAL_QA

    def get_template(self, prompt_type: Optional[PromptType] = None, query: Optional[str] = None) -> PromptTemplate:
        """Get prompt template by type or auto-detect from query.

        Args:
            prompt_type: Specific prompt type (optional)
            query: User query for auto-detection (optional)

        Returns:
            Prompt template
        """
        if prompt_type is None:
            if query is None:
                prompt_type = PromptType.GENERAL_QA
            else:
                prompt_type = self.detect_prompt_type(query)

        template = self.templates.get(prompt_type, self.templates[PromptType.GENERAL_QA])

        logger.debug("Selected prompt template",
                     prompt_type=prompt_type.value,
                     auto_detected=prompt_type is None)

        return template

    def format_prompt(self,
                      query: str,
                      context: str,
                      prompt_type: Optional[PromptType] = None) -> Dict[str, str]:
        """Format a complete prompt with system and user messages.

        Args:
            query: User query
            context: Retrieved context
            prompt_type: Specific prompt type (auto-detected if None)

        Returns:
            Dictionary with 'system' and 'user' messages
        """
        template = self.get_template(prompt_type, query)

        user_message = template.user_template.format(
            query=query,
            context=context
        )

        return {
            "system": template.system_prompt,
            "user": user_message,
            "type": template.metadata["type"]
        }


# Global instance
_prompt_manager: Optional[PromptTemplateManager] = None


def get_prompt_manager() -> PromptTemplateManager:
    """Get or create the global prompt manager.

    Returns:
        Prompt template manager instance
    """
    global _prompt_manager
    if _prompt_manager is None:
        _prompt_manager = PromptTemplateManager()
    return _prompt_manager


def format_prompt_for_query(query: str, context: str, prompt_type: Optional[PromptType] = None) -> Dict[str, str]:
    """Convenience function to format a prompt.

    Args:
        query: User query
        context: Retrieved context
        prompt_type: Optional prompt type

    Returns:
        Formatted prompt dictionary
    """
    manager = get_prompt_manager()
    return manager.format_prompt(query, context, prompt_type)
