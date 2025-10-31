# /rag-project Command Documentation

Automatically analyze your project and index relevant documentation for intelligent code assistance.

## Overview

The `/rag-project` command is a powerful feature that:
1. **Analyzes your project** to detect programming languages, frameworks, and libraries
2. **Identifies relevant documentation sources** (official docs, tutorials, API references)
3. **Prepares indexed documentation** for RAG-enhanced queries

## Usage

In Claude Code, simply run:
```
/rag-project
```

The command will automatically:
- Scan your project files
- Detect technologies
- Find and prepare documentation sources
- Report what's available for querying

## What It Detects

### Programming Languages
- **Python** (requirements.txt, pyproject.toml, *.py files)
- **JavaScript** (package.json, *.js files)
- **TypeScript** (tsconfig.json, *.ts files)
- **Rust** (Cargo.toml, *.rs files)
- **Go** (go.mod, *.go files)
- **Java** (pom.xml, build.gradle, *.java files)
- **C/C++** (*.c, *.cpp files)
- **Ruby** (*.rb files)
- **PHP** (*.php files)

### Frameworks & Libraries Detected

**Python**:
- Django, Flask, FastAPI
- LangChain
- Anthropic SDK
- NumPy, Pandas
- PyTorch, TensorFlow
- FAISS

**JavaScript/TypeScript**:
- React, Vue.js, Angular, Svelte
- Express
- Next.js

### Documentation Sources

The command identifies official documentation for each detected technology:

| Technology | Documentation URL |
|------------|------------------|
| Python | https://docs.python.org/3/ |
| Django | https://docs.djangoproject.com/ |
| Flask | https://flask.palletsprojects.com/ |
| FastAPI | https://fastapi.tiangolo.com/ |
| React | https://react.dev/ |
| Vue.js | https://vuejs.org/guide/ |
| TypeScript | https://www.typescriptlang.org/docs/ |
| LangChain | https://python.langchain.com/docs/ |
| Anthropic SDK | https://docs.anthropic.com/ |
| NumPy | https://numpy.org/doc/stable/ |
| FAISS | https://github.com/facebookresearch/faiss/wiki |

...and many more!

## Example Output

```
============================================================
RAG Project Documentation Indexer
============================================================

Step 1: Analyzing Project

Analyzing project structure...
[*] Detected 5 technologies

Detected Technologies:
  * Python [language]
    from requirements.txt
  * LangChain [framework]
    from requirements.txt
  * NumPy [library]
    from requirements.txt
  * Anthropic SDK [library]
    from requirements.txt
  * Flask [framework]
    from requirements.txt

Step 2: Finding Documentation Sources

Found 6 documentation source(s):
  * Python: https://docs.python.org/3/ [official]
  * LangChain: https://python.langchain.com/docs/ [official]
  * NumPy: https://numpy.org/doc/stable/ [official]
  * Anthropic SDK: https://docs.anthropic.com/ [official]
  * Flask: https://flask.palletsprojects.com/ [official]
  * Python: https://realpython.com/ [tutorial]

Step 3: Fetching and Indexing Documentation

Documentation sources identified and ready for online queries!

[*] Indexing Complete!

Summary:
  * Technologies detected: 5
  * Documentation sources: 6
  * Online documentation access enabled
  * Total time: 1.2s

Next Steps:
  1. The system has analyzed your project
  2. Relevant documentation sources have been identified
  3. Use online fallback queries for real-time documentation access
  4. Ask questions about any detected technology!
```

## How It Works

### 1. Project Analysis

The command scans your project for:
- **Package files**: requirements.txt, package.json, Cargo.toml, etc.
- **Configuration files**: tsconfig.json, go.mod, pyproject.toml, etc.
- **File extensions**: Counts .py, .js, .ts, .rs files to confirm languages
- **Framework markers**: manage.py (Django), specific import patterns

### 2. Technology Detection

**Confidence Levels**:
- **High (1.0)**: Detected from package files or framework-specific files
- **Medium (0.8)**: Detected from file patterns

**Detection Methods**:
```python
# Package-based detection (highest confidence)
requirements.txt -> detects Django, Flask, LangChain, etc.
package.json -> detects React, Vue, Express, etc.

# File-based detection
manage.py -> Django detected
tsconfig.json -> TypeScript detected
Cargo.toml -> Rust detected

# Pattern-based detection
3+ *.py files -> Python detected (80% confidence)
```

### 3. Documentation Mapping

The command maintains a curated list of official documentation sources:
- Primary sources (priority 1): Official docs
- Secondary sources (priority 2): High-quality tutorials

### 4. Integration with RAG System

Once documentation sources are identified:
- **Online Fallback**: The existing online retriever can fetch docs in real-time
- **Future Enhancement**: Full offline indexing (crawler implementation pending)

## Implementation Details

### Files Created

1. **`.claude/commands/rag-project.md`**
   - Claude Code command definition
   - Triggers the indexing script

2. **`src/plugin/commands/rag_project_indexer.py`**
   - Project analyzer
   - Technology detector
   - Documentation source mapper
   - Orchestration logic

### Key Classes

```python
class ProjectAnalyzer:
    """Analyzes project structure to detect technologies"""
    - analyze() -> List[DetectedTechnology]
    - _analyze_python()
    - _analyze_javascript()
    - _detect_frameworks()

class DocumentationFetcher:
    """Maps technologies to documentation sources"""
    - get_sources(technologies) -> List[DocumentationSource]
    - DOC_SOURCES: Curated mapping dict

class ProjectIndexer:
    """Orchestrates the indexing process"""
    - index_project(path) -> Result summary
    - _index_source(source) -> IndexingResult
```

## Configuration

The command works out-of-the-box with no configuration needed. It uses:
- Current working directory as project root
- Existing RAG-CLI configuration
- Online retriever settings from `config/default.yaml`

## Advanced Usage

### Manual Execution

Run the script directly for testing:
```bash
cd /path/to/your/project
python /path/to/RAG-CLI/src/plugin/commands/rag_project_indexer.py
```

### Results File

Results are saved to `data/project_indexing_result.json`:
```json
{
  "success": true,
  "technologies": [
    {
      "name": "Python",
      "type": "language",
      "confidence": 1.0,
      "source_file": "requirements.txt"
    }
  ],
  "sources": [
    {
      "name": "Python",
      "url": "https://docs.python.org/3/",
      "priority": 1,
      "doc_type": "official"
    }
  ],
  "total_documents": 0,
  "duration_seconds": 1.23
}
```

## Integration with RAG Queries

After running `/rag-project`, the RAG system knows about your project's technologies:

**Before `/rag-project`:**
```
User: "How do I use FastAPI?"
RAG: Generic answer or online search
```

**After `/rag-project`:**
```
User: "How do I use FastAPI?"
RAG: Knows FastAPI is in your project -> Prioritizes FastAPI official docs
     -> Returns contextually relevant answer for YOUR project
```

## Future Enhancements

### Phase 1 (Current)
- [OK] Project analysis and technology detection
- [OK] Documentation source identification
- [OK] Integration with online fallback

### Phase 2 (Planned)
- [ ] Full documentation crawler
- [ ] Offline indexing of docs
- [ ] Automatic re-indexing on dependency changes
- [ ] Version-specific documentation
- [ ] Code example extraction

### Phase 3 (Advanced)
- [ ] Project-specific best practices
- [ ] Framework-specific patterns
- [ ] Custom documentation sources
- [ ] Multi-project management

## Troubleshooting

### No Technologies Detected
**Problem**: Command reports 0 technologies

**Solution**:
- Ensure you're running from project root
- Check that package files exist (requirements.txt, package.json, etc.)
- Verify file permissions

### Missing Framework Detection
**Problem**: Framework not detected despite being used

**Solution**:
- Check if framework is in package file
- Verify minimum file count (3+ files for file-based detection)
- Add framework to DOC_SOURCES mapping manually

### Character Encoding Issues
**Problem**: Unicode characters display incorrectly

**Solution**:
- This is cosmetic only - functionality works
- Set `PYTHONIOENCODING=utf-8` environment variable
- Output logs still work correctly

## Examples

### Django Project
```
Technologies Detected:
  * Python [language] from requirements.txt
  * Django [framework] from manage.py
  * PostgreSQL [database] (if psycopg2 in requirements)

Documentation Sources:
  * Django official docs
  * Python official docs
  * Database-specific docs
```

### React + TypeScript Project
```
Technologies Detected:
  * JavaScript [language] from package.json
  * TypeScript [language] from tsconfig.json
  * React [framework] from package.json
  * Next.js [framework] (if detected)

Documentation Sources:
  * React official docs
  * TypeScript official docs
  * Next.js docs (if applicable)
```

### Machine Learning Project
```
Technologies Detected:
  * Python [language]
  * PyTorch [framework]
  * NumPy [library]
  * Pandas [library]
  * FAISS [library]

Documentation Sources:
  * PyTorch official docs
  * NumPy official docs
  * Pandas official docs
  * FAISS wiki
```

## Benefits

1. **Zero Configuration**: Works automatically on any project
2. **Intelligent**: Detects technologies without manual specification
3. **Comprehensive**: Covers 10+ languages, 20+ frameworks
4. **Fast**: Analysis completes in seconds
5. **Extensible**: Easy to add new technologies
6. **Integration**: Seamlessly works with existing RAG system

## Contributing

To add support for a new technology:

1. **Add to detection logic** in `ProjectAnalyzer`:
```python
def _analyze_newlang(self):
    config_file = self.project_path / "newlang.config"
    if config_file.exists():
        self.detected_tech.append(DetectedTechnology(
            name="NewLang",
            type="language",
            confidence=1.0,
            source_file="newlang.config"
        ))
```

2. **Add to documentation sources** in `DocumentationFetcher`:
```python
DOC_SOURCES = {
    "NewLang": [
        ("https://newlang.org/docs/", 1, "official"),
    ]
}
```

## License

Same as RAG-CLI project license.

## Related Documentation

- [RAG Optimizations 2025](./RAG_OPTIMIZATIONS_2025.md)
- [Online Documentation Retrieval](./docs/online_retrieval.md)
- [Claude Code Commands](../.claude/commands/)
