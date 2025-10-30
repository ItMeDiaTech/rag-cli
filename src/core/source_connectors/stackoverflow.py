"""Stack Overflow API connector for Q&A content retrieval.

Fetches questions, answers, and solutions from Stack Overflow.
"""

import requests
import time
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import html

logger = logging.getLogger(__name__)


@dataclass
class StackOverflowAnswer:
    """Represents an answer from Stack Overflow."""
    question_title: str
    question_body: str
    answer_body: str
    question_id: int
    answer_id: int
    question_url: str
    score: int
    is_accepted: bool
    tags: List[str]
    created_date: datetime
    last_activity: datetime
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_document(self) -> str:
        """Convert to a formatted document string.

        Returns:
            Formatted document with question and answer
        """
        doc = f"# {self.question_title}\n\n"
        doc += f"**Tags**: {', '.join(self.tags)}\n\n"
        doc += f"## Question\n\n{self.question_body}\n\n"
        doc += f"## Answer (Score: {self.score}"
        if self.is_accepted:
            doc += ", Accepted"
        doc += ")\n\n"
        doc += f"{self.answer_body}\n\n"
        doc += f"**Source**: {self.question_url}\n"
        return doc


class StackOverflowConnector:
    """Connector for Stack Overflow API to fetch Q&A content."""

    def __init__(self, api_key: Optional[str] = None, rate_limit: int = 300,
                 timeout: int = 10, min_score: int = 5):
        """Initialize Stack Overflow connector.

        Args:
            api_key: Optional API key for higher rate limits
            rate_limit: Maximum requests per day (300 for free tier)
            timeout: Request timeout in seconds
            min_score: Minimum answer score to consider
        """
        self.api_key = api_key
        self.timeout = timeout
        self.min_score = min_score
        self.base_url = "https://api.stackexchange.com/2.3"
        self.session = requests.Session()

        # Rate limiting (Stack Overflow uses daily limits)
        self.rate_limit = rate_limit
        self.requests_made = []
        self.daily_window = timedelta(days=1)

    def _wait_if_needed(self):
        """Wait if rate limit would be exceeded."""
        now = datetime.now()
        cutoff = now - self.daily_window

        # Remove old requests outside window
        self.requests_made = [req_time for req_time in self.requests_made if req_time > cutoff]

        if len(self.requests_made) >= self.rate_limit:
            # Need to wait until tomorrow
            oldest = self.requests_made[0]
            wait_until = oldest + self.daily_window
            wait_seconds = (wait_until - now).total_seconds()

            if wait_seconds > 0:
                logger.warning(f"Stack Overflow rate limit reached, need to wait {wait_seconds/3600:.1f} hours")
                # Don't actually wait a full day - just raise an error
                raise Exception("Stack Overflow daily rate limit exceeded")

        self.requests_made.append(now)

    def _make_request(self, endpoint: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Make API request to Stack Overflow.

        Args:
            endpoint: API endpoint
            params: Query parameters

        Returns:
            Response data or None if error
        """
        self._wait_if_needed()

        # Add required parameters
        params['site'] = 'stackoverflow'
        if self.api_key:
            params['key'] = self.api_key

        url = f"{self.base_url}/{endpoint}"

        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()

            # Check for API errors
            if 'error_id' in data:
                logger.error(f"Stack Overflow API error: {data.get('error_message', 'Unknown error')}")
                return None

            # Check quota
            if 'quota_remaining' in data:
                logger.debug(f"Stack Overflow quota remaining: {data['quota_remaining']}")

            return data

        except requests.exceptions.RequestException as e:
            logger.error(f"Error making Stack Overflow request: {e}")
            return None

    def search_questions(self, query: str, tags: Optional[List[str]] = None,
                        max_results: int = 10, sort: str = 'relevance') -> List[Dict[str, Any]]:
        """Search for questions matching query.

        Args:
            query: Search query
            tags: Optional list of tags to filter by
            max_results: Maximum number of results
            sort: Sort order (relevance, votes, activity, creation)

        Returns:
            List of questions
        """
        params = {
            'order': 'desc',
            'sort': sort,
            'q': query,
            'pagesize': min(max_results, 100),  # API max is 100
            'filter': 'withbody'  # Include question body
        }

        if tags:
            params['tagged'] = ';'.join(tags)

        data = self._make_request('search/advanced', params)
        if data:
            return data.get('items', [])
        return []

    def get_question_answers(self, question_id: int) -> List[Dict[str, Any]]:
        """Get all answers for a question.

        Args:
            question_id: Stack Overflow question ID

        Returns:
            List of answers
        """
        params = {
            'order': 'desc',
            'sort': 'votes',
            'filter': 'withbody'
        }

        data = self._make_request(f'questions/{question_id}/answers', params)
        if data:
            return data.get('items', [])
        return []

    def search_with_answers(self, query: str, tags: Optional[List[str]] = None,
                           max_results: int = 5, max_age_days: int = 730) -> List[StackOverflowAnswer]:
        """Search for questions and get their best answers.

        Args:
            query: Search query
            tags: Optional tags to filter by
            max_results: Maximum number of question-answer pairs to return
            max_age_days: Only include content from last N days

        Returns:
            List of StackOverflowAnswer objects
        """
        results = []

        # Search for questions
        questions = self.search_questions(query, tags=tags, max_results=max_results)

        cutoff_date = datetime.now() - timedelta(days=max_age_days)

        for question in questions:
            # Check if question is recent enough
            question_date = datetime.fromtimestamp(question.get('creation_date', 0))
            if question_date < cutoff_date:
                continue

            # Get answers for this question
            question_id = question['question_id']
            answers = self.get_question_answers(question_id)

            # Filter answers by score and find best one
            good_answers = [a for a in answers if a.get('score', 0) >= self.min_score]
            if not good_answers:
                continue

            # Sort by score and whether accepted
            good_answers.sort(key=lambda a: (a.get('is_accepted', False), a.get('score', 0)), reverse=True)
            best_answer = good_answers[0]

            # Create StackOverflowAnswer object
            so_answer = StackOverflowAnswer(
                question_title=self._decode_html(question.get('title', '')),
                question_body=self._decode_html(question.get('body', '')),
                answer_body=self._decode_html(best_answer.get('body', '')),
                question_id=question_id,
                answer_id=best_answer.get('answer_id', 0),
                question_url=question.get('link', ''),
                score=best_answer.get('score', 0),
                is_accepted=best_answer.get('is_accepted', False),
                tags=question.get('tags', []),
                created_date=datetime.fromtimestamp(question.get('creation_date', 0)),
                last_activity=datetime.fromtimestamp(question.get('last_activity_date', 0)),
                metadata={
                    'view_count': question.get('view_count', 0),
                    'answer_count': question.get('answer_count', 0),
                    'question_score': question.get('score', 0)
                }
            )

            results.append(so_answer)

        return results

    def search_by_error(self, error_message: str, language: Optional[str] = None,
                       max_results: int = 5) -> List[StackOverflowAnswer]:
        """Search for solutions to an error message.

        Args:
            error_message: Error message or exception text
            language: Optional programming language tag
            max_results: Maximum results to return

        Returns:
            List of StackOverflowAnswer objects with solutions
        """
        # Extract key parts of error message
        query = self._extract_error_keywords(error_message)

        # Build tags list
        tags = []
        if language:
            tags.append(language.lower())

        # Search with error-specific keywords
        return self.search_with_answers(query, tags=tags, max_results=max_results)

    def get_top_questions_by_tag(self, tag: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Get top questions for a specific tag.

        Args:
            tag: Tag to search (e.g., 'python', 'javascript')
            max_results: Maximum results to return

        Returns:
            List of top questions
        """
        params = {
            'order': 'desc',
            'sort': 'votes',
            'tagged': tag,
            'pagesize': max_results,
            'filter': 'withbody'
        }

        data = self._make_request('questions', params)
        if data:
            return data.get('items', [])
        return []

    def _decode_html(self, text: str) -> str:
        """Decode HTML entities in text.

        Args:
            text: Text with HTML entities

        Returns:
            Decoded text
        """
        return html.unescape(text)

    def _extract_error_keywords(self, error_message: str) -> str:
        """Extract searchable keywords from error message.

        Args:
            error_message: Full error message

        Returns:
            Cleaned query string
        """
        # Remove file paths
        import re
        query = re.sub(r'[A-Za-z]:\\[^\s]+', '', error_message)
        query = re.sub(r'/[^\s]+\.py', '', query)

        # Remove line numbers and memory addresses
        query = re.sub(r'line \d+', '', query)
        query = re.sub(r'0x[0-9a-fA-F]+', '', query)

        # Keep main error type and message
        lines = query.split('\n')
        if lines:
            # Usually last line has the main error
            query = lines[-1] if lines[-1].strip() else lines[-2] if len(lines) > 1 else query

        # Limit length for API
        words = query.split()[:20]  # Max 20 words
        return ' '.join(words)

    def get_statistics(self) -> Dict[str, Any]:
        """Get API usage statistics.

        Returns:
            Statistics about API usage
        """
        now = datetime.now()
        cutoff = now - self.daily_window
        recent_requests = [r for r in self.requests_made if r > cutoff]

        return {
            'requests_today': len(recent_requests),
            'quota_remaining': self.rate_limit - len(recent_requests),
            'rate_limit': self.rate_limit,
            'has_api_key': self.api_key is not None
        }
