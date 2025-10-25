#!/usr/bin/env python3
"""Setup script for RAG-CLI."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_file = Path(__file__).parent / "README.md"
if readme_file.exists():
    long_description = readme_file.read_text(encoding="utf-8")
else:
    long_description = "RAG-CLI: A Retrieval-Augmented Generation CLI tool"

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file) as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
else:
    requirements = []

setup(
    name="rag-cli",
    version="0.1.0",
    author="RAG-CLI Team",
    author_email="",
    description="A Retrieval-Augmented Generation CLI tool with Claude integration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ItMeDiaTech/rag-cli",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "rag-index=scripts.index:main",
            "rag-retrieve=scripts.retrieve:main",
            "rag-monitor=src.monitoring.tcp_server:main",
            "rag-skill=src.plugin.skills.rag_retrieval.retrieve:main",
            "rag-sync=sync_plugin:main",
        ],
    },
    scripts=[
        "sync_plugin.py",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    keywords="rag retrieval augmented generation claude ai nlp",
    project_urls={
        "Bug Reports": "https://github.com/ItMeDiaTech/rag-cli/issues",
        "Source": "https://github.com/ItMeDiaTech/rag-cli",
    },
)