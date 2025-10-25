#!/usr/bin/env python3
"""MCP (Model Context Protocol) server for RAG-CLI.

This server provides RAG functionality through the MCP protocol,
allowing Claude Code to interact with the document knowledge base.
"""

import sys
import json
import asyncio
from typing import Dict, Any, List, Optional
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src.core.config import get_config
from src.core.vector_store import get_vector_store
from src.core.embeddings import get_embedding_model
from src.core.retrieval_pipeline import HybridRetriever
from src.core.claude_integration import ClaudeAssistant
from src.monitoring.logger import get_logger
from src.monitoring.tcp_server import metrics_collector

logger = get_logger(__name__)


class RAGMCPServer:
    """MCP server for RAG operations."""

    def __init__(self):
        """Initialize the MCP server."""
        self.config = get_config()
        self.vector_store = None
        self.embedding_model = None
        self.retriever = None
        self.assistant = None
        self.initialized = False

        logger.info("RAG MCP server initialized")

    async def initialize(self):
        """Initialize server components."""
        if self.initialized:
            return

        try:
            # Check if vector store exists
            vector_store_path = project_root / "data" / "vectors" / "faiss_index"
            if not vector_store_path.exists():
                logger.warning("No vector index found")
                return

            # Initialize components
            self.vector_store = get_vector_store()
            self.embedding_model = get_embedding_model()
            self.retriever = HybridRetriever(
                vector_store=self.vector_store,
                embedding_model=self.embedding_model,
                config=self.config
            )
            self.assistant = ClaudeAssistant(self.config)

            self.initialized = True
            logger.info("RAG MCP server components initialized")

        except Exception as e:
            logger.error(f"Failed to initialize MCP server: {e}")
            raise

    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle an MCP request.

        Args:
            request: MCP request object

        Returns:
            MCP response object
        """
        method = request.get("method", "")
        params = request.get("params", {})
        request_id = request.get("id")

        try:
            # Initialize if needed
            if not self.initialized:
                await self.initialize()

            # Route to appropriate handler
            if method == "search":
                result = await self.handle_search(params)
            elif method == "index":
                result = await self.handle_index(params)
            elif method == "status":
                result = await self.handle_status(params)
            elif method == "configure":
                result = await self.handle_configure(params)
            else:
                return self.error_response(
                    request_id,
                    f"Unknown method: {method}"
                )

            return self.success_response(request_id, result)

        except Exception as e:
            logger.error(f"Request handling failed: {e}")
            return self.error_response(request_id, str(e))

    async def handle_search(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a search request.

        Args:
            params: Search parameters

        Returns:
            Search results
        """
        if not self.initialized:
            raise RuntimeError("Server not initialized")

        query = params.get("query", "")
        top_k = params.get("top_k", 5)
        use_llm = params.get("use_llm", True)

        # Perform search
        documents = self.retriever.search(query, top_k=top_k)

        # Generate response if requested
        if use_llm and documents:
            response = self.assistant.generate_response(query, documents)
            answer = response.get("answer", "")
        else:
            answer = None

        # Record metrics
        metrics_collector.record_query()

        return {
            "query": query,
            "answer": answer,
            "documents": documents,
            "total": len(documents)
        }

    async def handle_index(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle an indexing request.

        Args:
            params: Indexing parameters

        Returns:
            Indexing result
        """
        from src.core.document_processor import DocumentProcessor

        path = params.get("path", "")
        if not path:
            raise ValueError("Path parameter required")

        # Process and index documents
        processor = DocumentProcessor(self.config)
        documents = processor.process_directory(Path(path))

        # Generate embeddings and add to store
        embeddings = self.embedding_model.encode_batch(
            [doc["content"] for doc in documents]
        )

        self.vector_store.add_documents(documents, embeddings)

        return {
            "indexed": len(documents),
            "path": path,
            "status": "success"
        }

    async def handle_status(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a status request.

        Args:
            params: Status parameters

        Returns:
            Server status
        """
        status = {
            "initialized": self.initialized,
            "components": {
                "vector_store": self.vector_store is not None,
                "embedding_model": self.embedding_model is not None,
                "retriever": self.retriever is not None,
                "assistant": self.assistant is not None
            }
        }

        if self.initialized and self.vector_store:
            status["statistics"] = {
                "total_documents": self.vector_store.index.ntotal,
                "embedding_dimensions": self.config.embeddings.model_dim
            }

        return status

    async def handle_configure(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a configuration request.

        Args:
            params: Configuration parameters

        Returns:
            Configuration result
        """
        setting = params.get("setting", "")
        value = params.get("value")

        if not setting:
            # Return current configuration
            return {
                "retrieval": {
                    "top_k": self.config.retrieval.top_k,
                    "hybrid_ratio": self.config.retrieval.hybrid_ratio
                },
                "claude": {
                    "model": self.config.claude.model,
                    "max_tokens": self.config.claude.max_tokens
                }
            }

        # Update configuration
        if setting == "retrieval.top_k":
            self.config.retrieval.top_k = int(value)
        elif setting == "retrieval.hybrid_ratio":
            self.config.retrieval.hybrid_ratio = float(value)
        elif setting == "claude.max_tokens":
            self.config.claude.max_tokens = int(value)
        else:
            raise ValueError(f"Unknown setting: {setting}")

        return {
            "setting": setting,
            "value": value,
            "status": "updated"
        }

    def success_response(self, request_id: Any, result: Any) -> Dict[str, Any]:
        """Create a success response.

        Args:
            request_id: Request ID
            result: Result data

        Returns:
            MCP response object
        """
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": result
        }

    def error_response(self, request_id: Any, message: str) -> Dict[str, Any]:
        """Create an error response.

        Args:
            request_id: Request ID
            message: Error message

        Returns:
            MCP error response
        """
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": -32603,
                "message": message
            }
        }

    async def run(self):
        """Run the MCP server."""
        logger.info("Starting RAG MCP server")

        # Initialize server
        await self.initialize()

        # Read requests from stdin
        while True:
            try:
                line = await asyncio.get_event_loop().run_in_executor(
                    None, sys.stdin.readline
                )

                if not line:
                    break

                # Parse request
                request = json.loads(line.strip())

                # Handle request
                response = await self.handle_request(request)

                # Send response
                print(json.dumps(response))
                sys.stdout.flush()

            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON: {e}")
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Server error: {e}")

        logger.info("RAG MCP server stopped")


async def main():
    """Main entry point."""
    server = RAGMCPServer()
    await server.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server interrupted")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Server failed: {e}")
        sys.exit(1)