# Building a RAG-Enhanced Claude Code CLI Plugin

**As of October 2025, Claude Code supports five plugin components that enable powerful RAG integration.** The newest addition—Agent Skills—offers progressive disclosure architecture specifically designed for context-heavy applications like RAG. A production-ready plugin combining these components can intercept every user query, enhance responses with retrieved context, and provide seamless toggle functionality through Claude Code's settings system.

This comprehensive guide provides technical implementation details, architectural decisions, and production-ready patterns for building a RAG-integrated plugin. The recommended approach uses Agent Skills for RAG logic, Hooks for query interception, slash commands for user control, and MCP servers for external tool integration. With this architecture, you can achieve sub-5-second response times, 70-85% retrieval accuracy, and enterprise-grade security.

## Claude Code plugin system architecture

Claude Code's plugin ecosystem consists of five extensible components that work together. **Slash Commands** provide user-invoked shortcuts defined as Markdown files with frontmatter metadata. **Subagents** are specialized AI agents for specific tasks that Claude can invoke automatically. **Hooks** are event handlers responding to lifecycle events like PreToolUse, PostToolUse, and UserPromptSubmit. **MCP Servers** connect Claude to external tools via the Model Context Protocol. **Agent Skills** (introduced October 2025) contain model-invoked capabilities that Claude autonomously uses based on context, featuring progressive disclosure where only metadata loads at startup (~100 tokens per skill), with full instructions and supporting files loading only when relevant.

The plugin directory structure follows this pattern:

```
rag-plugin/
├── .claude-plugin/
│   └── plugin.json          # Required manifest
├── commands/                 # Slash commands
│   ├── rag-enable.md
│   └── rag-disable.md
├── skills/                   # Agent Skills  
│   └── rag-retrieval/
│       ├── SKILL.md
│       └── scripts/
│           └── retrieve.py
├── hooks/
│   └── hooks.json           # Query interception
├── .mcp.json                # Vector DB connection
└── README.md
```

The plugin manifest (plugin.json) requires name, version, description, and author fields. Environment variables like ${CLAUDE_PLUGIN_ROOT} provide the absolute path to the plugin directory, essential for referencing scripts in hooks and MCP configurations. The layered configuration system loads defaults first, then user settings from ~/.claude/settings.json, then project settings from .claude/settings.json, and finally CLI flags override everything.

## Hooking into the query/response pipeline

The plugin system provides multiple integration points throughout Claude's execution pipeline. At session start, Skills metadata loads into the system prompt, SessionStart hooks execute initialization logic, and MCP servers start automatically. When users submit input, **UserPromptSubmit hooks** pre-process or validate queries—this is your primary interception point for RAG. During tool execution, PreToolUse hooks validate parameters before execution while PostToolUse hooks can validate results or run additional processing. For response generation, Notification hooks respond to permission requests, Stop hooks control when Claude finishes, and SubagentStop hooks control subagent completion.

**The recommended RAG integration pattern uses a three-layer approach.** First, UserPromptSubmit hooks intercept all queries and invoke your RAG retrieval logic. Second, an Agent Skill contains the RAG instructions and retrieval scripts that run outside Claude's context window. Third, an MCP server connects to your vector database, exposing search tools that Claude can use.

Here's a complete UserPromptSubmit hook configuration:

```json
{
  "description": "RAG context enhancement",
  "hooks": {
    "UserPromptSubmit": [
      {
        "matcher": "*",
        "hooks": [
          {
            "type": "command",
            "command": "${CLAUDE_PLUGIN_ROOT}/scripts/rag-enhance.sh",
            "timeout": 5
          }
        ]
      }
    ]
  }
}
```

The hook script receives query information via stdin as JSON and can modify the system by returning JSON via stdout. Exit code 0 means continue normally, while other codes can block operations. The script should retrieve relevant context from your vector database, then inject it into Claude's context by modifying the query or adding it to CLAUDE.md files.

## RAG architecture and technology stack

A production RAG system for CLI environments requires careful technology choices balancing performance, cost, and ease of deployment. The architecture follows this flow: User Query → Query Embedding → Vector Search → Top-K Retrieval → Context Assembly → LLM Prompting → Answer Generation → Validation → Response.

**For vector databases in CLI applications, FAISS and Chroma excel for local development** (free, fast, no network latency), while Qdrant Cloud or Pinecone work best for production deployments under 10M vectors. Performance benchmarks on 1M vectors show Zilliz Cloud averaging 50-60ms latency at $45-75/month, Pinecone at 70-80ms for $70/month, and Qdrant at 75-85ms for $30-50/month. FAISS running locally typically achieves under 10ms latency with zero cost but lacks persistence by default. **For CLI plugins, start with FAISS for development and prototype testing, then migrate to Qdrant Cloud for production** due to its excellent Python SDK, resource-based pricing, and self-hosting option.

For embedding models, the decision depends on whether you prioritize performance or want offline capability. OpenAI's text-embedding-3-small provides the best cost-performance ratio at $0.02 per 1M tokens with 1536 dimensions and excellent retrieval scores (64.6 on MTEB benchmark). For local/offline operation, **BAAI/bge-large-en-v1.5 offers comparable performance** (61.5 MTEB score) with 1024 dimensions and zero API costs. The lightweight sentence-transformers/all-MiniLM-L6-v2 (384 dimensions) encodes 100 documents in ~0.5 seconds locally, making it ideal for rapid CLI operations despite slightly lower accuracy (56.0 MTEB).

Document chunking strategy significantly impacts retrieval quality. **Use semantic chunking with 400-500 token chunks and 10-20% overlap** for most applications. LangChain's RecursiveCharacterTextSplitter handles this well:

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " "]
)
chunks = splitter.split_text(document_text)
```

Add contextual headers to chunks (e.g., document title, section name) to improve retrieval precision. For technical documentation, reduce chunk size to 150-200 tokens. For code, use 100-150 tokens per logical unit like functions or classes. Always preserve complete sentences at boundaries and store metadata including source, section, and timestamp with each chunk.

**Implement hybrid search combining vector and keyword methods for 20-40% higher accuracy.** Run both vector similarity search (using cosine similarity or L2 distance) and keyword search (BM25 or TF-IDF) in parallel, then merge results using reciprocal rank fusion with typical weights of 0.7 semantic and 0.3 lexical. Follow this with two-stage retrieval: fetch top 10-20 candidates quickly with vector search, then rerank to top 3-5 using a cross-encoder model like cross-encoder/ms-marco-MiniLM-L-6-v2. This adds 50-200ms latency but improves accuracy by 20-40%.

## Recommended implementation architecture

The optimal architecture for a RAG-enhanced Claude Code plugin uses Agent Skills as the core RAG engine. Agent Skills support progressive disclosure (loading only metadata initially), can execute Python scripts outside the context window, and integrate seamlessly with Claude's decision-making. Here's the complete SKILL.md structure:

```markdown
---
name: rag-retrieval
description: Retrieve relevant context from knowledge base. Use when user questions relate to documented information.
allowed-tools: ["Read", "Bash"]
---

# RAG Retrieval System

## Instructions

When the user asks a question that might be answered by documentation:

1. Extract key terms and concepts from the query
2. Use the retrieval script to search the knowledge base
3. Analyze retrieved chunks for relevance
4. Include top 3-5 most relevant chunks in your response
5. Cite sources when using retrieved information

## When to Activate RAG Retrieval

### Code and Implementation Scenarios

Activate retrieval when the user asks about:
- Specific functions, classes, or modules in their codebase
- Implementation patterns or examples from existing code
- How certain features were previously implemented
- Code architecture or design decisions
- API usage patterns from their own code
- Function signatures and parameter documentation
- Best practices demonstrated in their codebase

### Coding Documentation Scenarios

Retrieve context for:
- API documentation queries
- README files and setup instructions
- Configuration documentation
- Inline code comments and docstrings
- Architecture decision records (ADRs)
- Technical specifications
- Development guidelines and standards

### Debug and Troubleshooting Scenarios

Use retrieval for:
- Error messages that have appeared before
- Stack traces and their solutions
- Performance optimization patterns
- Bug fixes and their explanations
- Test failures and resolutions
- Logging output analysis
- System behavior investigations
- Implementation debugging steps

## Retrieval Process

Run: `python ${CLAUDE_PLUGIN_ROOT}/skills/rag-retrieval/scripts/retrieve.py --query "$QUERY"`

This returns JSON with relevant chunks and metadata.

## Supporting Files

- `scripts/retrieve.py`: Main retrieval engine
- `reference/config.yaml`: Vector DB configuration
```

The retrieval script implements the complete RAG pipeline:

```python
#!/usr/bin/env python3
import sys
import json
import argparse
from sentence_transformers import SentenceTransformer
import faiss
import pickle

def load_index():
    """Load FAISS index and metadata"""
    index = faiss.read_index("./data/vectors.index")
    with open("./data/metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    return index, metadata

def retrieve(query, top_k=5):
    """Retrieve relevant chunks for query"""
    # Load model and index
    model = SentenceTransformer('all-MiniLM-L6-v2')
    index, metadata = load_index()
    
    # Embed query
    query_vec = model.encode([query])
    
    # Search
    distances, indices = index.search(query_vec, top_k)
    
    # Format results
    results = []
    for idx, dist in zip(indices[0], distances[0]):
        results.append({
            "text": metadata[idx]["text"],
            "source": metadata[idx]["source"],
            "score": float(1 / (1 + dist))  # Convert distance to similarity
        })
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    
    results = retrieve(args.query, args.top_k)
    print(json.dumps(results, indent=2))
```

For cloud vector databases, use an MCP server instead. Create .mcp.json in your plugin:

```json
{
  "mcpServers": {
    "qdrant": {
      "command": "npx",
      "args": ["-y", "@your-org/qdrant-mcp-server"],
      "env": {
        "QDRANT_URL": "${QDRANT_URL}",
        "QDRANT_API_KEY": "${QDRANT_API_KEY}"
      }
    }
  }
}
```

The MCP server exposes vector search as tools like `mcp__qdrant__search` that Claude can invoke directly. This approach keeps credentials secure and allows the Skill to focus on retrieval strategy rather than database connections.

## Settings integration and toggle functionality

Claude Code's configuration system enables seamless plugin control through multiple mechanisms. **Users can enable/disable plugins via the /plugin command**:

```bash
/plugin enable rag-plugin      # Activate plugin
/plugin disable rag-plugin     # Deactivate without uninstalling
```

For programmatic control, plugins store state in settings.json:

```json
{
  "plugins": {
    "rag-plugin": {
      "enabled": true,
      "settings": {
        "vectorDb": "local",
        "topK": 5,
        "rerank": true,
        "chunkSize": 500
      }
    }
  }
}
```

**Create custom slash commands for plugin-specific controls.** Save this as commands/rag-config.md:

```markdown
---
description: Configure RAG settings
argument-hint: [setting] [value]
---

# Configure RAG Plugin

Modify RAG plugin settings. Available settings:

- `topK`: Number of chunks to retrieve (default: 5)
- `rerank`: Enable reranking (true/false)
- `chunkSize`: Chunk size in tokens (default: 500)

Usage: `/rag-config topK 10`

## Steps

1. Parse the setting name and value from $ARGUMENTS
2. Read current config from `.claude/settings.json`
3. Update the specified setting
4. Write back to settings file
5. Confirm change to user

Settings file location: `.claude/settings.json`
Update path: `plugins.rag-plugin.settings.[setting]`
```

For runtime activation control, implement hooks that check the enabled state before executing:

```bash
#!/bin/bash
# scripts/rag-enhance.sh

# Check if plugin is enabled
ENABLED=$(jq -r '.plugins["rag-plugin"].enabled // true' ~/.claude/settings.json)

if [ "$ENABLED" != "true" ]; then
    # Plugin disabled, pass through without modification
    exit 0
fi

# Plugin enabled, perform RAG retrieval
QUERY=$(echo "$1" | jq -r '.query')
python "${CLAUDE_PLUGIN_ROOT}/skills/rag-retrieval/scripts/retrieve.py" --query "$QUERY"
```

This pattern ensures hooks respect the plugin's enabled state without requiring Claude Code restart. Changes to settings.json typically take effect immediately for hooks and commands, though MCP server changes may require restart.

## Implementing custom slash commands

Slash commands provide the user interface for RAG control. Claude Code uses Markdown files with frontmatter for command definitions, making them extremely simple to create and modify. Commands support arguments, file references, tool restrictions, and model selection.

**Create a command to manually trigger RAG search:**

```markdown
---
description: Search knowledge base and answer question with retrieved context
argument-hint: [your question]
allowed-tools: ["Read", "Bash"]
model: claude-3-5-sonnet-20241022
---

# RAG Search

Search the knowledge base for relevant information and provide an answer.

## Instructions

1. Extract the question from $ARGUMENTS
2. Run the RAG retrieval script:
   `python ~/.claude/plugins/rag-plugin/scripts/retrieve.py --query "$ARGUMENTS"`
3. Parse the JSON results
4. Analyze the retrieved chunks for relevance
5. Synthesize an answer using the retrieved context
6. Cite sources using [Source: filename] format
7. If no relevant information found, clearly state this

## Response Format

Provide a clear, concise answer based on the retrieved context. Always cite sources.

Example:
"Based on the documentation, you can configure X by doing Y [Source: config.md]. 
This approach is recommended because Z [Source: best-practices.md]."
```

Save as `commands/search.md` and invoke with `/search How do I configure the API?`

**Commands support namespacing through directory structure.** Place commands in subdirectories to organize them:

```
commands/
├── rag/
│   ├── search.md       # Invoked as /rag:search
│   ├── index.md        # Invoked as /rag:index
│   └── clear.md        # Invoked as /rag:clear
└── config.md           # Invoked as /config
```

The frontmatter supports several powerful options. Use `allowed-tools` to restrict which tools the command can access (security best practice). Set `model` to specify which Claude model executes the command (use opus for complex RAG reasoning, haiku for simple retrieval). The `disable-model-invocation` field prevents Claude from automatically choosing this command, keeping it manual-only. The `argument-hint` provides users with guidance on expected arguments.

Commands can reference files directly using @filepath syntax, automatically adding them to context. Variables like $ARGUMENTS capture all passed arguments, while $1, $2, etc. provide positional access (though $ARGUMENTS is officially supported). Commands have access to all environment variables including ${CLAUDE_PROJECT_DIR} for project-relative paths.

## Knowledge base management and indexing

Effective knowledge base management requires a robust document ingestion and indexing pipeline. The workflow consists of four stages: collection, processing, embedding, and indexing.

**For document collection**, support multiple sources based on your use case. Local CLI tools typically index from file systems, while enterprise applications integrate with Confluence, SharePoint, or S3. Handle diverse formats including PDF (use pdftotext or MinerU), Markdown, HTML (Pandoc for conversion), DOCX, and plain text. Extract metadata during collection—document title, author, creation date, modification date, source path, and document type—as this metadata enables powerful filtering during retrieval.

The processing pipeline chunks documents appropriately for your content type. Implement recursive character splitting that respects semantic boundaries:

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_document(text, metadata):
    """Chunk document with context preservation"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = splitter.split_text(text)
    
    # Enrich chunks with metadata and context
    enriched = []
    for i, chunk in enumerate(chunks):
        enriched.append({
            "text": chunk,
            "metadata": {
                **metadata,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "char_count": len(chunk)
            }
        })
    
    return enriched
```

**Add contextual headers to each chunk** to improve retrieval precision significantly. For documentation, prepend the document title and section heading. For code, include the file path and function/class name. For conversations or emails, add participants and subject line. This context helps the embedding model understand what each chunk discusses and improves ranking accuracy by 10-20%.

For embedding generation, maintain consistency by using the same model for both indexing and querying. Batch embeddings to reduce API costs and latency—OpenAI's API supports up to 2048 inputs per request. Cache embeddings to avoid recomputing unchanged documents. Monitor embedding costs carefully as they can become significant at scale.

The indexing script ties everything together:

```python
#!/usr/bin/env python3
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import json

def index_documents(docs_dir, output_dir):
    """Index documents from directory"""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Collect and chunk documents
    all_chunks = []
    for doc_path in Path(docs_dir).rglob("*.md"):
        text = doc_path.read_text()
        metadata = {
            "source": str(doc_path),
            "filename": doc_path.name
        }
        chunks = chunk_document(text, metadata)
        all_chunks.extend(chunks)
    
    print(f"Processing {len(all_chunks)} chunks...")
    
    # Generate embeddings
    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    
    # Build FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    # Save index and metadata
    Path(output_dir).mkdir(exist_ok=True)
    faiss.write_index(index, f"{output_dir}/vectors.index")
    
    with open(f"{output_dir}/metadata.pkl", "wb") as f:
        pickle.dump(all_chunks, f)
    
    print(f"Indexed {len(all_chunks)} chunks to {output_dir}")

if __name__ == "__main__":
    index_documents("./docs", "./data")
```

Run this during plugin installation and provide a slash command to reindex:

```markdown
---
description: Rebuild the RAG knowledge base index
---

# Rebuild Index

Reindex all documents in the knowledge base.

Run: `python ${CLAUDE_PLUGIN_ROOT}/scripts/index.py`

This will:
1. Scan all documents in the docs directory
2. Chunk them appropriately
3. Generate embeddings
4. Build the FAISS index
5. Save to disk

Time required: ~1-5 minutes depending on corpus size.
```

**For production systems, implement incremental updates** rather than full reindexing. Track document modification times and only reprocess changed files. Use vector database upsert operations to update existing chunks. For rapidly changing content, consider streaming ingestion where documents index immediately upon creation or modification. Most production deployments run full reindexing weekly or monthly, with incremental updates daily or hourly.

## Performance optimization for CLI environments

CLI tools must feel responsive, requiring aggressive performance optimization. Target metrics for RAG operations include vector search under 100ms, embedding generation under 50ms for local models (200ms for API calls), LLM generation 1-3 seconds with streaming, and total end-to-end response under 5 seconds.

**Caching provides the highest-impact optimization.** Implement three cache layers: query cache for embedding results (saves 50-200ms on repeated queries), result cache for complete RAG responses to identical queries, and embedding cache for static documents. Use an LRU cache for query embeddings:

```python
from functools import lru_cache
from sentence_transformers import SentenceTransformer

class EmbeddingCache:
    def __init__(self, model_name, cache_size=1000):
        self.model = SentenceTransformer(model_name)
        self._embed = lru_cache(maxsize=cache_size)(self._embed_impl)
    
    def _embed_impl(self, text):
        """Cache-able embedding function"""
        return tuple(self.model.encode(text).tolist())
    
    def embed(self, text):
        """Get embedding with caching"""
        return list(self._embed(text))
```

For disk-based caching across sessions, use GPTCache or a simple SQLite database storing query hashes mapped to results.

**Optimize FAISS index selection based on corpus size.** For under 100K vectors, use IndexFlatL2 (exact search, ~10ms). For 100K-1M vectors, use IndexHNSWFlat (approximate search, ~20-50ms, 95%+ recall). For over 1M vectors, use IndexIVFPQ with product quantization (compressed storage, ~50-100ms). Configure FAISS for optimal CPU usage:

```python
import faiss

# For CPU optimization
faiss.omp_set_num_threads(4)

# Create optimized index for 100K-1M vectors
index = faiss.IndexHNSWFlat(384, 32)  # 384 dims, 32 connections
index.hnsw.efConstruction = 200
index.hnsw.efSearch = 100
```

Reduce dimensionality for CLI applications where storage and speed matter more than maximum accuracy. The 384-dimensional all-MiniLM-L6-v2 model provides 85-90% of the quality of 1024-dimensional models at one-third the storage and search time.

**Implement async processing for independent operations.** Run vector search and keyword search in parallel for hybrid retrieval. Process multiple queries concurrently. Batch API calls for embedding generation. This parallel execution can reduce latency by 30-50%:

```python
import asyncio
import aiohttp

async def hybrid_search(query, top_k=5):
    """Run vector and keyword search in parallel"""
    async def vector_search():
        embedding = await embed_async(query)
        return await search_vector_db(embedding, top_k)
    
    async def keyword_search():
        return await search_bm25(query, top_k)
    
    # Run both searches concurrently
    vector_results, keyword_results = await asyncio.gather(
        vector_search(),
        keyword_search()
    )
    
    # Merge results with reciprocal rank fusion
    return merge_results(vector_results, keyword_results)
```

For LLM generation, always stream responses to improve perceived performance. Users see output immediately rather than waiting for complete generation. Claude's API supports streaming via Server-Sent Events:

```python
import anthropic

client = anthropic.Anthropic()

def stream_response(prompt, context):
    """Stream LLM response"""
    full_prompt = f"Context:\n{context}\n\nQuestion: {prompt}\nAnswer:"
    
    with client.messages.stream(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[{"role": "user", "content": full_prompt}]
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
```

Memory optimization matters for CLI tools. Use memory-mapped files for large FAISS indexes to avoid loading everything into RAM. Implement document streaming when processing large corpora. Limit concurrent operations—embedding 2-4 documents at a time prevents memory spikes. Profile your application with memory_profiler to identify hotspots.

## Security and privacy considerations

Security requires multi-layer protection across data, storage, retrieval, and generation stages. **The most critical consideration is PII handling**—personally identifiable information, health records, financial data, and proprietary business information require special treatment.

For local deployment with sensitive data, use completely local models (Mistral 7B, Llama models via Ollama) and local vector databases (FAISS, Chroma). This ensures no data leaves the user's machine, providing complete privacy and compliance with HIPAA, GDPR, and similar regulations. Install with:

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull models
ollama pull mistral
ollama pull nomic-embed-text

# Use in plugin
python3 -c "
from langchain.llms import Ollama
from langchain.embeddings import OllamaEmbeddings

llm = Ollama(model='mistral')
embedder = OllamaEmbeddings(model='nomic-embed-text')
"
```

**For cloud deployments, implement data anonymization before indexing.** Tokenize PII using reversible encryption where you need to retrieve original values, or use one-way hashing for values that only need matching. Apply named entity recognition to detect and redact sensitive information:

```python
import re
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

def anonymize_text(text):
    """Remove PII before indexing"""
    # Detect PII
    results = analyzer.analyze(
        text=text,
        language='en',
        entities=['PERSON', 'EMAIL_ADDRESS', 'PHONE_NUMBER', 'SSN']
    )
    
    # Replace with placeholders
    anonymized = anonymizer.anonymize(
        text=text,
        analyzer_results=results
    )
    
    return anonymized.text
```

Encrypt data at rest and in transit. Use AES-256 for storage encryption and TLS 1.3 for network connections. For vector databases, most cloud providers lack built-in encryption, so implement application-layer encryption:

```python
from cryptography.fernet import Fernet

class EncryptedVectorStore:
    def __init__(self, key, vector_store):
        self.cipher = Fernet(key)
        self.store = vector_store
    
    def add_texts(self, texts, metadatas):
        """Encrypt before storing"""
        encrypted_texts = [
            self.cipher.encrypt(t.encode()).decode()
            for t in texts
        ]
        return self.store.add_texts(encrypted_texts, metadatas)
    
    def search(self, query, k=5):
        """Search encrypted, decrypt results"""
        encrypted_query = self.cipher.encrypt(query.encode()).decode()
        results = self.store.search(encrypted_query, k)
        
        return [
            self.cipher.decrypt(r.encode()).decode()
            for r in results
        ]
```

**Implement robust access controls using role-based access control (RBAC).** Store user roles in metadata and filter results based on permissions:

```python
def search_with_permissions(query, user_role, top_k=5):
    """Filter results by access level"""
    # Get more results initially
    candidates = vector_store.search(query, top_k * 3)
    
    # Filter by permission
    allowed = [
        doc for doc in candidates
        if user_role in doc.metadata.get('allowed_roles', [])
    ]
    
    # Return top k allowed results
    return allowed[:top_k]
```

Protect against prompt injection attacks by validating queries before processing. Check for patterns attempting to override instructions, exfiltrate data, or bypass security controls:

```python
def validate_query(query):
    """Check for injection attempts"""
    dangerous_patterns = [
        r'ignore (previous|above) instructions',
        r'system:',
        r'<admin>',
        r'reveal.*password',
        r'bypass.*security'
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, query, re.IGNORECASE):
            raise SecurityError(f"Potentially malicious query: {pattern}")
    
    return True
```

Monitor for anomalous behavior indicating attacks. Track query patterns, retrieval frequency, and result diversity. Alert on unusual activity like rapid repeated queries, queries with no results (potential reconnaissance), or queries from unusual locations.

## Testing strategies for RAG systems

Comprehensive testing requires evaluating both retrieval and generation quality. **Create a golden dataset of 10-50 test queries covering main use cases**, diverse question types (simple lookups, complex reasoning, multi-hop), edge cases (ambiguous questions, out-of-scope topics), and real-world examples from actual users.

Each test case should include the question, expected ground truth answer, relevant documents that should be retrieved, and success criteria (required precision/recall, acceptable response patterns). Store in JSON for automated testing:

```json
{
  "test_cases": [
    {
      "question": "How do I configure the API key?",
      "ground_truth": "Set ANTHROPIC_API_KEY environment variable",
      "expected_docs": ["config.md", "api-reference.md"],
      "min_precision": 0.8,
      "min_recall": 0.9
    }
  ]
}
```

**Use RAGAS framework for automated evaluation**—it provides four critical metrics computed via LLM judging:

```python
from ragas import evaluate
from ragas.metrics import (
    context_precision,    # Relevant docs in retrieved set?
    context_recall,       # All relevant docs retrieved?
    faithfulness,         # Answer based only on context?
    answer_relevancy      # Answer addresses the question?
)
from datasets import Dataset

# Run evaluation
results = evaluate(
    dataset=Dataset.from_dict({
        'question': [case['question'] for case in test_cases],
        'answer': [generated_answers],
        'contexts': [retrieved_contexts],
        'ground_truth': [case['ground_truth'] for case in test_cases]
    }),
    metrics=[
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy
    ]
)

print(results)
# {'context_precision': 0.87, 'context_recall': 0.82, ...}
```

Implement CI/CD integration to run tests automatically on every change:

```python
import pytest
from rag_plugin import RAGSystem

@pytest.fixture
def rag_system():
    return RAGSystem(config="test_config.yaml")

def test_retrieval_quality(rag_system):
    """Test retrieval metrics"""
    query = "How to configure logging?"
    docs = rag_system.retrieve(query, top_k=5)
    
    # Check retrieval quality
    assert len(docs) == 5
    assert all(doc.score > 0.7 for doc in docs)
    assert any("logging" in doc.text.lower() for doc in docs)

def test_generation_faithfulness(rag_system):
    """Test generation quality"""
    query = "How to configure logging?"
    response = rag_system.query(query)
    context = "\n".join([d.text for d in response.retrieved_docs])
    
    # Check faithfulness
    faithfulness = evaluate_faithfulness(
        response.answer,
        context
    )
    assert faithfulness > 0.8
```

For security testing, implement adversarial test cases attempting prompt injection, data exfiltration, and access control bypass. Use the OWASP Top 10 for LLM Applications as a testing checklist. Conduct regular red-team exercises where team members attempt to break the system.

**Monitor production quality continuously** by collecting user feedback (thumbs up/down), tracking query latency distributions, measuring retrieval precision on subset of queries, and analyzing failure modes. Set up alerts for quality degradation—if faithfulness drops below 0.7, if retrieval precision falls below 0.6, if average latency exceeds 10 seconds, or if error rate exceeds 5%.

## Packaging and distribution

Claude Code plugins distribute through marketplace repositories on GitHub or similar platforms. **Create a marketplace repository structure**:

```
my-marketplace/
├── .claude-plugin/
│   └── marketplace.json
├── rag-plugin/
│   ├── .claude-plugin/
│   │   └── plugin.json
│   ├── commands/
│   ├── skills/
│   ├── scripts/
│   └── README.md
└── other-plugin/
    └── ...
```

The marketplace.json lists available plugins:

```json
{
  "name": "my-marketplace",
  "description": "Collection of RAG and productivity plugins",
  "plugins": [
    {
      "source": "./rag-plugin"
    },
    {
      "source": "github:other-org/external-plugin"
    }
  ]
}
```

Users install your marketplace once, then access all plugins:

```bash
# Add marketplace
/plugin marketplace add https://github.com/your-org/my-marketplace.git

# Install specific plugin
/plugin install rag-plugin@my-marketplace

# Or just
/plugin install rag-plugin
```

**For team deployments, configure automatic installation** via repository settings. Add to .claude/settings.json in your team's repository:

```json
{
  "extraKnownMarketplaces": {
    "team-tools": {
      "source": {
        "source": "github",
        "repo": "your-org/claude-plugins"
      }
    }
  },
  "plugins": {
    "team-tools/rag-plugin": {
      "enabled": true
    }
  }
}
```

When team members trust the repository, plugins install automatically on first run. This ensures consistent tooling across the team and simplifies onboarding.

Follow semantic versioning for releases (MAJOR.MINOR.PATCH). Use MAJOR for breaking changes to plugin interface, MINOR for new features that are backward compatible, and PATCH for bug fixes. Maintain a CHANGELOG.md documenting changes. Tag releases in Git with version numbers.

**Implement security scanning before distribution.** Scan dependencies for vulnerabilities using npm audit or pip-audit. Review all scripts and hooks for security issues. Test in sandboxed environment before releasing. Include security documentation clearly stating what permissions the plugin requires and what data it accesses.

The plugin README should cover installation instructions, configuration options, usage examples with screenshots or command output, security considerations and data handling practices, troubleshooting common issues, and contribution guidelines. Clear documentation significantly increases adoption.

## Reference implementations and resources

Several open-source projects demonstrate RAG integration with CLI tools. **The rag-cli project (okwilkins/rag-cli)** provides a complete Rust/Python implementation with Ollama and Qdrant integration, supporting embed, vector-store, and rag commands. The gptme-rag tool (gptme/gptme-rag) offers local RAG with ChromaDB using pipx installation and watch mode for automatic reindexing. Canopy (pinecone-io/canopy) provides a Pinecone-powered RAG framework with chat interface.

For Claude Code specifically, explore the official repository (anthropics/claude-code) containing example plugins and documentation. The awesome-claude-code community repository (hesreallyhim/awesome-claude-code) curates useful commands, workflows, and plugins. Community marketplaces like jeremylongshore/claude-code-plugins-plus offer 227+ plugins demonstrating various integration patterns.

LangChain (langchain-ai/langchain) remains the most comprehensive RAG framework with 100+ data source connectors, extensive LLM support, and production-tested components. LlamaIndex (run-llama/llama_index) excels at indexing and retrieval with advanced query engines. Both provide excellent examples and documentation for building RAG systems.

For testing and evaluation, examine RAGAS (explodinggradients/ragas) for automated RAG evaluation, TruLens for step-by-step tracking and debugging, and DeepEval for LLM evaluation with pytest integration. These tools provide production-ready testing infrastructure.

## Production deployment best practices

Deploying RAG plugins to production requires attention to reliability, monitoring, and cost optimization. **Implement retry logic with exponential backoff** for all external API calls (embedding APIs, vector databases, LLMs). Start with 1-second delays and double on each retry up to maximum 3-5 attempts. This handles transient network errors gracefully.

Create graceful degradation paths for component failures. If the vector database is unavailable, fall back to keyword search. If the LLM API is down, return retrieved context without generation. If embedding generation fails, use cached results or simplified retrieval. Always provide a fallback response rather than complete failure:

```python
def query_with_fallback(question):
    """Multi-level fallback strategy"""
    try:
        # Primary: Full RAG pipeline
        return rag_system.query(question)
    except VectorDBError:
        # Fallback 1: Keyword search
        logger.warning("Vector DB unavailable, using keyword search")
        return keyword_search_fallback(question)
    except LLMError:
        # Fallback 2: Context only
        logger.warning("LLM unavailable, returning context")
        return context_only_response(question)
    except Exception as e:
        # Fallback 3: User-friendly error
        logger.error(f"All systems failed: {e}")
        return "The system is temporarily unavailable. Please try again."
```

**Use circuit breakers to prevent cascade failures.** When a service repeatedly fails, stop calling it temporarily to avoid wasting resources and time. After a cooldown period, try again:

```python
from datetime import datetime, timedelta

class CircuitBreaker:
    def __init__(self, failure_threshold=5, timeout=60):
        self.failures = 0
        self.threshold = failure_threshold
        self.timeout = timeout
        self.opened_at = None
        self.state = "CLOSED"
    
    def call(self, func, *args):
        if self.state == "OPEN":
            if datetime.now() - self.opened_at > timedelta(seconds=self.timeout):
                self.state = "HALF_OPEN"
            else:
                raise CircuitOpenError("Service temporarily unavailable")
        
        try:
            result = func(*args)
            self.on_success()
            return result
        except Exception as e:
            self.on_failure()
            raise
    
    def on_success(self):
        self.failures = 0
        self.state = "CLOSED"
    
    def on_failure(self):
        self.failures += 1
        if self.failures >= self.threshold:
            self.state = "OPEN"
            self.opened_at = datetime.now()
```

Monitor critical metrics continuously using Prometheus and Grafana or similar tools. Track end-to-end latency (p50, p95, p99), retrieval precision and recall, faithfulness and answer relevancy scores, error rates by type, cost per query (API tokens, database operations), and throughput (queries per second). Set up alerts when metrics exceed thresholds—latency p95 over 10 seconds, error rate above 5%, faithfulness below 0.7, or costs exceeding budget.

**Cost optimization becomes critical at scale.** Implement aggressive caching for repeated queries, batch API calls to reduce per-request overhead, use smaller/cheaper models where appropriate (haiku for simple tasks, sonnet for complex reasoning), monitor token usage and optimize prompt lengths, and implement rate limiting to prevent abuse. Calculate cost per query and optimize the most expensive components first.

For observability, use structured logging with context:

```python
import logging
import json

logger = logging.getLogger('rag-plugin')

def log_rag_query(query, retrieved_docs, response, latency):
    """Structured logging for analysis"""
    logger.info(json.dumps({
        'event': 'rag_query',
        'query': query,
        'num_docs_retrieved': len(retrieved_docs),
        'avg_doc_score': sum(d.score for d in retrieved_docs) / len(retrieved_docs),
        'response_length': len(response),
        'latency_ms': latency * 1000,
        'timestamp': datetime.now().isoformat()
    }))
```

This enables powerful analysis with tools like Elasticsearch or Splunk, allowing you to identify patterns in failures, track performance trends over time, understand usage patterns, and calculate costs accurately.

## Key architectural decisions summary

Building a production-ready RAG plugin requires careful decisions across multiple dimensions. **For the plugin architecture, use Agent Skills as the core RAG engine** (progressive disclosure, executes outside context, model-invoked), UserPromptSubmit hooks for query interception, slash commands for user control interface, and MCP servers for vector database connections. This architecture minimizes context window usage while providing seamless integration.

**For the technology stack, select based on your deployment model.** Local development benefits from FAISS vector database with all-MiniLM-L6-v2 embeddings and Ollama for LLMs. Production deployments should use Qdrant Cloud or Pinecone, OpenAI text-embedding-3-small, and Claude API with streaming. Hybrid deployments can use FAISS with OpenAI APIs, providing good performance at moderate cost.

**For retrieval strategy, implement hybrid search combining vector and keyword approaches**, use two-stage retrieval (retrieve 10, rerank to 3-5), employ cross-encoder reranking for 20-40% accuracy improvement, and chunk at 400-500 tokens with 10-20% overlap and contextual headers. This combination provides optimal precision and recall.

**For production operations, implement multi-level caching** (query, embedding, result), use retry logic with exponential backoff and circuit breakers, monitor continuously with alerts on key metrics, and implement graceful degradation with fallback strategies. These patterns ensure reliability and performance.

Security requires data anonymization before indexing for sensitive information, encryption at rest and in transit, RBAC for access control, prompt injection detection and prevention, and regular security audits and penetration testing. For maximum privacy, deploy fully locally with Ollama and FAISS.

The complete plugin combines these elements into a cohesive system that intercepts queries, enriches them with retrieved context, generates high-quality responses, and provides users with intuitive controls—all while maintaining security, performance, and reliability.