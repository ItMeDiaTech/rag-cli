"""Source connectors for online documentation retrieval.

This package contains connectors for various documentation sources:
- GitHub API for repositories and documentation
- Stack Overflow API for Q&A content
- ReadTheDocs/DevDocs for aggregated documentation
- Official documentation sites
"""


__all__ = [
    'GitHubConnector',
    'StackOverflowConnector',
    'ReadTheDocsConnector',
    'OfficialDocsConnector',
]

# Import connectors (will be available after implementing each module)
try:
    from .github import GitHubConnector
except ImportError:
    pass

try:
    from .stackoverflow import StackOverflowConnector
except ImportError:
    pass

try:
    from .readthedocs import ReadTheDocsConnector
except ImportError:
    pass

try:
    from .official_docs import OfficialDocsConnector
except ImportError:
    pass
