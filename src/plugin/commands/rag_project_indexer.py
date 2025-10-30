#!/usr/bin/env python3
"""RAG Project Indexer - Automatically index project-relevant documentation.

This script analyzes a project to detect languages, frameworks, and dependencies,
then searches for and indexes relevant documentation.
"""

import sys
import json
import time
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src.core.config import get_config
from src.core.online_retriever import OnlineRetriever
from src.core.document_processor import get_document_processor
from src.core.vector_store import get_vector_store
from src.core.embeddings import get_embedding_generator
from src.core.output import warning, error
from src.core.web_scraper import DocumentationScraperFactory
from src.monitoring.logger import get_logger

logger = get_logger(__name__)

# Simple print-based output for better compatibility


def print_header(text: str):
    """Print header text."""
    print(f"\n{'=' * 60}")
    print(text)
    print('=' * 60)


@dataclass
class DetectedTechnology:
    """Detected technology in project."""
    name: str
    type: str  # 'language', 'framework', 'library', 'tool'
    version: Optional[str] = None
    confidence: float = 1.0
    source_file: Optional[str] = None


@dataclass
class DocumentationSource:
    """Documentation source to fetch."""
    name: str
    url: str
    priority: int  # 1 = highest
    doc_type: str  # 'official', 'tutorial', 'reference', 'examples'


@dataclass
class IndexingResult:
    """Result of indexing operation."""
    source: str
    documents_added: int
    success: bool
    error: Optional[str] = None
    duration_seconds: float = 0.0


class ProjectAnalyzer:
    """Analyzes project to detect technologies."""

    def __init__(self, project_path: Path):
        """Initialize project analyzer.

        Args:
            project_path: Path to project root
        """
        self.project_path = project_path
        self.detected_tech: List[DetectedTechnology] = []

    def analyze(self) -> List[DetectedTechnology]:
        """Analyze project and detect technologies.

        Returns:
            List of detected technologies
        """
        print("Analyzing project structure...")

        # Detect from package files
        self._analyze_python()
        self._analyze_javascript()
        self._analyze_typescript()
        self._analyze_rust()
        self._analyze_go()
        self._analyze_java()

        # Detect from file extensions
        self._analyze_file_extensions()

        # Detect frameworks from code patterns
        self._detect_frameworks()

        print(f"[OK] Detected {len(self.detected_tech)} technologies")

        return self.detected_tech

    def _analyze_python(self):
        """Analyze Python project files."""
        # requirements.txt
        req_file = self.project_path / "requirements.txt"
        if req_file.exists():
            self.detected_tech.append(DetectedTechnology(
                name="Python",
                type="language",
                confidence=1.0,
                source_file="requirements.txt"
            ))

            with open(req_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Parse package name
                        match = re.match(r'^([a-zA-Z0-9\-_]+)', line)
                        if match:
                            package = match.group(1).lower()

                            # Detect major frameworks
                            framework_map = {
                                'django': ('Django', 'framework'),
                                'flask': ('Flask', 'framework'),
                                'fastapi': ('FastAPI', 'framework'),
                                'anthropic': ('Anthropic SDK', 'library'),
                                'langchain': ('LangChain', 'framework'),
                                'faiss': ('FAISS', 'library'),
                                'numpy': ('NumPy', 'library'),
                                'pandas': ('Pandas', 'library'),
                                'pytorch': ('PyTorch', 'framework'),
                                'tensorflow': ('TensorFlow', 'framework'),
                            }

                            if package in framework_map:
                                name, tech_type = framework_map[package]
                                self.detected_tech.append(DetectedTechnology(
                                    name=name,
                                    type=tech_type,
                                    source_file="requirements.txt"
                                ))

        # pyproject.toml
        pyproject = self.project_path / "pyproject.toml"
        if pyproject.exists():
            self.detected_tech.append(DetectedTechnology(
                name="Python",
                type="language",
                confidence=1.0,
                source_file="pyproject.toml"
            ))

    def _analyze_javascript(self):
        """Analyze JavaScript project files."""
        package_json = self.project_path / "package.json"
        if package_json.exists():
            self.detected_tech.append(DetectedTechnology(
                name="JavaScript",
                type="language",
                confidence=1.0,
                source_file="package.json"
            ))

            try:
                with open(package_json, 'r') as f:
                    data = json.load(f)

                # Check dependencies
                deps = {**data.get('dependencies', {}), **data.get('devDependencies', {})}

                framework_map = {
                    'react': ('React', 'framework'),
                    'vue': ('Vue.js', 'framework'),
                    'angular': ('Angular', 'framework'),
                    'express': ('Express', 'framework'),
                    'next': ('Next.js', 'framework'),
                    'svelte': ('Svelte', 'framework'),
                }

                for package, version in deps.items():
                    package_lower = package.lower()
                    if package_lower in framework_map:
                        name, tech_type = framework_map[package_lower]
                        self.detected_tech.append(DetectedTechnology(
                            name=name,
                            type=tech_type,
                            version=version,
                            source_file="package.json"
                        ))

            except Exception as e:
                logger.warning(f"Failed to parse package.json: {e}")

    def _analyze_typescript(self):
        """Analyze TypeScript project files."""
        tsconfig = self.project_path / "tsconfig.json"
        if tsconfig.exists():
            self.detected_tech.append(DetectedTechnology(
                name="TypeScript",
                type="language",
                confidence=1.0,
                source_file="tsconfig.json"
            ))

    def _analyze_rust(self):
        """Analyze Rust project files."""
        cargo_toml = self.project_path / "Cargo.toml"
        if cargo_toml.exists():
            self.detected_tech.append(DetectedTechnology(
                name="Rust",
                type="language",
                confidence=1.0,
                source_file="Cargo.toml"
            ))

    def _analyze_go(self):
        """Analyze Go project files."""
        go_mod = self.project_path / "go.mod"
        if go_mod.exists():
            self.detected_tech.append(DetectedTechnology(
                name="Go",
                type="language",
                confidence=1.0,
                source_file="go.mod"
            ))

    def _analyze_java(self):
        """Analyze Java project files."""
        pom_xml = self.project_path / "pom.xml"
        build_gradle = self.project_path / "build.gradle"

        if pom_xml.exists():
            self.detected_tech.append(DetectedTechnology(
                name="Java",
                type="language",
                confidence=1.0,
                source_file="pom.xml"
            ))
        elif build_gradle.exists():
            self.detected_tech.append(DetectedTechnology(
                name="Java",
                type="language",
                confidence=1.0,
                source_file="build.gradle"
            ))

    def _analyze_file_extensions(self):
        """Analyze file extensions to detect languages."""
        extension_map = {
            '.py': 'Python',
            '.js': 'JavaScript',
            '.ts': 'TypeScript',
            '.rs': 'Rust',
            '.go': 'Go',
            '.java': 'Java',
            '.cpp': 'C++',
            '.c': 'C',
            '.rb': 'Ruby',
            '.php': 'PHP',
        }

        extension_counts = defaultdict(int)

        # Sample files from project
        for ext, lang in extension_map.items():
            count = len(list(self.project_path.rglob(f'*{ext}')))
            if count > 0:
                extension_counts[lang] = count

        # Add languages with significant file counts
        for lang, count in extension_counts.items():
            if count >= 3:  # At least 3 files
                # Check if not already detected
                if not any(t.name == lang for t in self.detected_tech):
                    self.detected_tech.append(DetectedTechnology(
                        name=lang,
                        type="language",
                        confidence=0.8,
                        source_file=f"file_analysis ({count} files)"
                    ))

    def _detect_frameworks(self):
        """Detect frameworks from code patterns and directory structure."""
        # Django detection
        if (self.project_path / "manage.py").exists():
            if not any(t.name == "Django" for t in self.detected_tech):
                self.detected_tech.append(DetectedTechnology(
                    name="Django",
                    type="framework",
                    confidence=1.0,
                    source_file="manage.py"
                ))

        # Flask detection
        app_files = list(self.project_path.rglob("app.py"))
        if app_files:
            # Check if Flask is imported
            for app_file in app_files[:3]:  # Check first 3
                try:
                    content = app_file.read_text()
                    if 'from flask import' in content or 'import flask' in content:
                        if not any(t.name == "Flask" for t in self.detected_tech):
                            self.detected_tech.append(DetectedTechnology(
                                name="Flask",
                                type="framework",
                                confidence=0.9,
                                source_file=str(app_file.relative_to(self.project_path))
                            ))
                        break
                except Exception as e:
                    logger.debug(f"Could not parse {app_file.name} for Flask detection: {e}")


class DocumentationFetcher:
    """Fetches documentation for detected technologies."""

    # Documentation source mappings
    DOC_SOURCES = {
        # Languages
        "Python": [
            ("https://docs.python.org/3/", 1, "official"),
            ("https://realpython.com/", 2, "tutorial"),
        ],
        "JavaScript": [
            ("https://developer.mozilla.org/en-US/docs/Web/JavaScript", 1, "official"),
            ("https://javascript.info/", 2, "tutorial"),
        ],
        "TypeScript": [
            ("https://www.typescriptlang.org/docs/", 1, "official"),
        ],
        "Rust": [
            ("https://doc.rust-lang.org/book/", 1, "official"),
            ("https://rust-lang.github.io/async-book/", 2, "official"),
        ],
        "Go": [
            ("https://go.dev/doc/", 1, "official"),
        ],
        "Java": [
            ("https://docs.oracle.com/javase/tutorial/", 1, "official"),
        ],

        # Frameworks
        "Django": [
            ("https://docs.djangoproject.com/", 1, "official"),
        ],
        "Flask": [
            ("https://flask.palletsprojects.com/", 1, "official"),
        ],
        "FastAPI": [
            ("https://fastapi.tiangolo.com/", 1, "official"),
        ],
        "React": [
            ("https://react.dev/", 1, "official"),
        ],
        "Vue.js": [
            ("https://vuejs.org/guide/", 1, "official"),
        ],
        "Angular": [
            ("https://angular.io/docs", 1, "official"),
        ],
        "Express": [
            ("https://expressjs.com/", 1, "official"),
        ],
        "Next.js": [
            ("https://nextjs.org/docs", 1, "official"),
        ],

        # Libraries
        "LangChain": [
            ("https://python.langchain.com/docs/", 1, "official"),
        ],
        "Anthropic SDK": [
            ("https://docs.anthropic.com/", 1, "official"),
        ],
        "FAISS": [
            ("https://github.com/facebookresearch/faiss/wiki", 1, "official"),
        ],
        "NumPy": [
            ("https://numpy.org/doc/stable/", 1, "official"),
        ],
        "Pandas": [
            ("https://pandas.pydata.org/docs/", 1, "official"),
        ],
        "PyTorch": [
            ("https://pytorch.org/docs/stable/index.html", 1, "official"),
        ],
        "TensorFlow": [
            ("https://www.tensorflow.org/api_docs", 1, "official"),
        ],
    }

    def get_sources(self, technologies: List[DetectedTechnology]) -> List[DocumentationSource]:
        """Get documentation sources for detected technologies.

        Args:
            technologies: List of detected technologies

        Returns:
            List of documentation sources to fetch
        """
        sources = []

        for tech in technologies:
            if tech.name in self.DOC_SOURCES:
                for url, priority, doc_type in self.DOC_SOURCES[tech.name]:
                    sources.append(DocumentationSource(
                        name=tech.name,
                        url=url,
                        priority=priority,
                        doc_type=doc_type
                    ))

        # Sort by priority
        sources.sort(key=lambda x: x.priority)

        return sources


class ProjectIndexer:
    """Orchestrates project documentation indexing."""

    def __init__(self):
        """Initialize project indexer."""
        self.config = get_config()
        self.document_processor = get_document_processor()
        self.vector_store = get_vector_store()
        self.embedding_generator = get_embedding_generator()
        self.online_retriever = OnlineRetriever(self.config)

    def index_project(self, project_path: Path) -> Dict[str, Any]:
        """Index documentation for a project.

        Args:
            project_path: Path to project root

        Returns:
            Summary of indexing results
        """
        start_time = time.time()

        print(f"\nStarting project indexing for: {project_path}\n")

        # Step 1: Analyze project
        print_header("Step 1: Analyzing Project")
        analyzer = ProjectAnalyzer(project_path)
        technologies = analyzer.analyze()

        if not technologies:
            warning("No technologies detected in project")
            return {
                "success": False,
                "error": "No technologies detected",
                "duration_seconds": time.time() - start_time
            }

        # Display detected technologies
        print("\nDetected Technologies:")
        for tech in technologies:
            confidence_str = f"({tech.confidence:.0%})" if tech.confidence < 1.0 else ""
            print(f"  - {tech.name} [{tech.type}] {confidence_str}")
            if tech.source_file:
                print(f"    from {tech.source_file}")

        # Step 2: Get documentation sources
        print()
        print_header("Step 2: Finding Documentation Sources")
        fetcher = DocumentationFetcher()
        sources = fetcher.get_sources(technologies)

        if not sources:
            warning("No documentation sources found for detected technologies")
            return {
                "success": False,
                "error": "No documentation sources available",
                "technologies": [asdict(t) for t in technologies],
                "duration_seconds": time.time() - start_time
            }

        print(f"\nFound {len(sources)} documentation source(s):")
        for source in sources:
            print(f"  - {source.name}: {source.url} [{source.doc_type}]")

        # Step 3: Fetch and index documentation
        print()
        print_header("Step 3: Fetching and Indexing Documentation")
        print("This may take several minutes...\n")

        results = []
        total_docs = 0

        for source in sources:
            try:
                result = self._index_source(source)
                results.append(result)
                if result.success:
                    total_docs += result.documents_added
                    print(f"  [OK] {source.name}: {result.documents_added} documents ({result.duration_seconds:.1f}s)")
                else:
                    print(f"  [FAILED] {source.name}: {result.error}")
            except Exception as e:
                logger.error(f"Failed to index {source.name}: {e}")
                results.append(IndexingResult(
                    source=source.name,
                    documents_added=0,
                    success=False,
                    error=str(e)
                ))

        # Summary
        elapsed = time.time() - start_time
        print()
        print_header("Indexing Complete!")
        print("\nSummary:")
        print(f"  - Technologies detected: {len(technologies)}")
        print(f"  - Documentation sources: {len(sources)}")
        print(f"  - Total documents indexed: {total_docs}")
        print(f"  - Total time: {elapsed:.1f}s")

        return {
            "success": True,
            "technologies": [asdict(t) for t in technologies],
            "sources": [asdict(s) for s in sources],
            "results": [asdict(r) for r in results],
            "total_documents": total_docs,
            "duration_seconds": elapsed
        }

    def _index_source(self, source: DocumentationSource) -> IndexingResult:
        """Index a single documentation source.

        Args:
            source: Documentation source

        Returns:
            Indexing result
        """
        start_time = time.time()

        try:
            logger.info(f"Indexing documentation from: {source.url}")

            # Step 1: Scrape documentation
            scraper = DocumentationScraperFactory.create_scraper(source.name, source.url)
            scraped_docs = scraper.scrape_documentation(
                source.url,
                source.name,
                source.doc_type
            )

            if not scraped_docs:
                return IndexingResult(
                    source=source.name,
                    documents_added=0,
                    success=False,
                    error="No documents scraped from source",
                    duration_seconds=time.time() - start_time
                )

            logger.info(f"Scraped {len(scraped_docs)} documents from {source.name}")

            # Step 2: Process and chunk documents
            all_chunks = []
            for doc in scraped_docs:
                # Create document text with title and content
                full_text = f"# {doc.title}\n\n{doc.content}"

                # Process document into chunks
                chunks = self.document_processor.process_text(
                    text=full_text,
                    metadata={
                        'source': doc.metadata['source'],
                        'url': doc.url,
                        'title': doc.title,
                        'doc_type': doc.doc_type,
                        'scraped_at': doc.metadata['scraped_at']
                    }
                )

                all_chunks.extend(chunks)

            if not all_chunks:
                return IndexingResult(
                    source=source.name,
                    documents_added=0,
                    success=False,
                    error="No chunks generated from scraped content",
                    duration_seconds=time.time() - start_time
                )

            logger.info(f"Generated {len(all_chunks)} chunks from {source.name}")

            # Step 3: Generate embeddings and add to vector store
            texts = [chunk.content for chunk in all_chunks]
            metadatas = [chunk.metadata for chunk in all_chunks]

            embeddings = self.embedding_generator.encode(texts, show_progress=False)

            # Add to vector store
            self.vector_store.add(embeddings, metadatas)

            # Save vector store
            self.vector_store.save()

            logger.info(f"Successfully indexed {len(all_chunks)} chunks from {source.name}")

            return IndexingResult(
                source=source.name,
                documents_added=len(all_chunks),
                success=True,
                duration_seconds=time.time() - start_time
            )

        except Exception as e:
            logger.error(f"Failed to index {source.name}: {e}", exc_info=True)
            return IndexingResult(
                source=source.name,
                documents_added=0,
                success=False,
                error=str(e),
                duration_seconds=time.time() - start_time
            )


def main():
    """Main entry point for project indexing."""
    try:
        # Get current working directory as project path
        project_path = Path.cwd()

        print("=" * 60)
        print("RAG Project Documentation Indexer")
        print("=" * 60)
        print()

        # Create indexer and run
        indexer = ProjectIndexer()
        result = indexer.index_project(project_path)

        # Save result
        result_file = project_path / "data" / "project_indexing_result.json"
        result_file.parent.mkdir(parents=True, exist_ok=True)

        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)

        print()
        print(f"Results saved to: {result_file}")
        print()
        print("Next Steps:")
        print("  1. The system has analyzed your project")
        print("  2. Relevant documentation sources have been identified")
        print("  3. Use online fallback queries for real-time documentation access")
        print("  4. For offline indexing, implement full doc crawler (see TODO)")

        return 0 if result.get("success") else 1

    except Exception as e:
        logger.error(f"Project indexing failed: {e}", exc_info=True)
        error(f"Indexing failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
