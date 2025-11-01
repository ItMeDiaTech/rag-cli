#!/usr/bin/env python3
"""Unified MCP server for RAG-CLI.

Combines service management and RAG search functionality into a single
MCP server that can be accessed from any Claude Code project.
Provides both async class-based architecture and comprehensive tool support.
"""

import sys
import json
import os
import asyncio
from pathlib import Path
from typing import Dict, Any

from rag_cli_plugin.services.logger import get_logger
from rag_cli_plugin.services.service_manager import (
    ensure_services_running,
    get_services_status,
    open_dashboard_in_browser
)
from rag_cli_plugin.services.output_formatter import OutputFormatter

logger = get_logger(__name__)

# Resolve project root for path operations
project_root = Path(__file__).resolve().parents[3]

# Input validation constants
MAX_QUERY_LENGTH = 10000
MAX_TOP_K = 100

class UnifiedMCPServer:
    """Unified MCP server combining RAG operations and service management."""

    def __init__(self):
        """Initialize the MCP server."""
        from core.config import get_config

        self.config = get_config()
        self.vector_store = None
        self.embedding_model = None
        self.retriever = None
        self.assistant = None
        self.initialized = False

        logger.info("Unified MCP server initialized")

    async def initialize(self):
        """Initialize server components lazily."""
        if self.initialized:
            return

        try:
            # Ensure monitoring services are running
            logger.info("Ensuring monitoring services are running...")
            try:
                services_status = ensure_services_running()
                logger.info(f"Monitoring services status: {services_status}")
            except Exception as e:
                logger.warning(f"Failed to start monitoring services: {e}")

            # Check if vector store exists
            vector_store_path = project_root / "data" / "vectors" / "vectors.index"
            if not vector_store_path.exists():
                logger.warning("No vector index found")
                # Don't return - allow service management tools to work
                self.initialized = True
                return

            # Initialize RAG components
            from core.vector_store import get_vector_store
            from core.embeddings import get_embedding_generator
            from core.retrieval_pipeline import HybridRetriever
            from core.claude_integration import ClaudeAssistant

            self.vector_store = get_vector_store()
            self.embedding_model = get_embedding_generator()
            self.retriever = HybridRetriever(
                vector_store=self.vector_store,
                embedding_generator=self.embedding_model,
                config=self.config
            )
            self.assistant = ClaudeAssistant(self.config)

            self.initialized = True
            logger.info("RAG components initialized")

        except Exception as e:
            logger.error(f"Failed to initialize MCP server: {e}")
            # Set as initialized anyway to allow service management
            self.initialized = True

    async def handle_request(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle an MCP protocol request.

        Args:
            message: MCP protocol message

        Returns:
            MCP protocol response
        """
        request_id = message.get("id", 0)
        method = message.get("method")
        params = message.get("params", {})

        logger.debug(f"MCP request: {method}")

        try:
            # Route to appropriate handler
            if method == "initialize":
                return self.handle_mcp_initialize(request_id)
            elif method == "tools/list":
                return self.handle_mcp_list_tools(request_id)
            elif method == "tools/call":
                tool_name = params.get("name")
                arguments = params.get("arguments", {})
                return await self.handle_mcp_call_tool(request_id, tool_name, arguments)
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32601,
                        "message": f"Unknown method: {method}"
                    }
                }

        except Exception as e:
            logger.error(f"Request handling failed: {e}")
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32603,
                    "message": f"Internal error: {str(e)}"
                }
            }

    def handle_mcp_initialize(self, request_id: int) -> Dict[str, Any]:
        """Handle MCP initialize message."""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {
                        "listChanged": True
                    }
                },
                "serverInfo": {
                    "name": "RAG-CLI-Unified",
                    "version": "0.3.0"
                }
            }
        }

    def handle_mcp_list_tools(self, request_id: int) -> Dict[str, Any]:
        """Handle MCP list tools request."""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "tools": [
                    # Service Management Tools
                    {
                        "name": "start_services",
                        "description": "Start RAG-CLI monitoring services (TCP server and web dashboard)",
                        "inputSchema": {
                            "type": "object",
                            "properties": {},
                            "required": []
                        }
                    },
                    {
                        "name": "get_services_status_tool",
                        "description": "Get the current status of RAG-CLI services",
                        "inputSchema": {
                            "type": "object",
                            "properties": {},
                            "required": []
                        }
                    },
                    {
                        "name": "open_dashboard",
                        "description": "Open the RAG-CLI web dashboard in the default browser",
                        "inputSchema": {
                            "type": "object",
                            "properties": {},
                            "required": []
                        }
                    },
                    # RAG Search Tools
                    {
                        "name": "rag_search",
                        "description": "Search the document knowledge base and get AI-generated responses",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "The search query"
                                },
                                "top_k": {
                                    "type": "integer",
                                    "description": "Number of documents to retrieve (default: 5)",
                                    "default": 5
                                },
                                "use_llm": {
                                    "type": "boolean",
                                    "description": "Generate AI response using Claude (default: true)",
                                    "default": True
                                }
                            },
                            "required": ["query"]
                        }
                    },
                    {
                        "name": "rag_index",
                        "description": "Index documents from a directory into the RAG knowledge base",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "path": {
                                    "type": "string",
                                    "description": "Path to directory containing documents to index"
                                },
                                "recursive": {
                                    "type": "boolean",
                                    "description": "Recursively index subdirectories (default: true)",
                                    "default": True
                                }
                            },
                            "required": ["path"]
                        }
                    },
                    {
                        "name": "rag_status",
                        "description": "Get the current status of the RAG system",
                        "inputSchema": {
                            "type": "object",
                            "properties": {},
                            "required": []
                        }
                    },
                    {
                        "name": "rag_configure",
                        "description": "Update RAG system configuration settings",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "setting": {
                                    "type": "string",
                                    "description": "Configuration setting name"
                                },
                                "value": {
                                    "description": "New value for the setting"
                                }
                            },
                            "required": []
                        }
                    },
                    # Hook Management Tools
                    {
                        "name": "rag_configure_hooks",
                        "description": "Enable/disable individual RAG hooks",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "hook_name": {
                                    "type": "string",
                                    "description": "Hook name"
                                },
                                "enabled": {
                                    "type": "boolean",
                                    "description": "Enable or disable the hook"
                                }
                            },
                            "required": ["hook_name", "enabled"]
                        }
                    },
                    {
                        "name": "rag_set_citation_format",
                        "description": "Configure citation format for ResponsePost hook",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "format": {
                                    "type": "string",
                                    "description": "Citation format (inline, footnotes, collapsible)"
                                },
                                "max_citations": {
                                    "type": "integer",
                                    "description": "Maximum number of citations to include"
                                }
                            },
                            "required": ["format"]
                        }
                    },
                    {
                        "name": "rag_get_hook_status",
                        "description": "Check which RAG hooks are currently active",
                        "inputSchema": {
                            "type": "object",
                            "properties": {},
                            "required": []
                        }
                    },
                    {
                        "name": "rag_set_error_mode",
                        "description": "Configure error handling behavior",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "mode": {
                                    "type": "string",
                                    "description": "Error mode (inline_warning, silent_fallback, block_query)"
                                }
                            },
                            "required": ["mode"]
                        }
                    },
                    # Multi-Agent Framework Tools
                    {
                        "name": "maf_execute",
                        "description": "Execute a task using the multi-agent framework",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "task": {
                                    "type": "string",
                                    "description": "Task description for agents to execute"
                                },
                                "workflow": {
                                    "type": "string",
                                    "description": "Workflow type (code_generation, bug_fix, optimization)",
                                    "default": "code_generation"
                                },
                                "use_rag": {
                                    "type": "boolean",
                                    "description": "Enhance agent knowledge with RAG retrieval",
                                    "default": True
                                }
                            },
                            "required": ["task"]
                        }
                    },
                    {
                        "name": "maf_status",
                        "description": "Get status of multi-agent framework",
                        "inputSchema": {
                            "type": "object",
                            "properties": {},
                            "required": []
                        }
                    },
                    {
                        "name": "maf_classify",
                        "description": "Classify a query to determine which agents should handle it",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "Query to classify"
                                }
                            },
                            "required": ["query"]
                        }
                    }
                ]
            }
        }

    async def handle_mcp_call_tool(self, request_id: int, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP tool call request."""
        try:
            # Ensure initialization
            await self.initialize()

            # Service Management Tools
            if tool_name == "start_services":
                return await self.handle_start_services(request_id, arguments)
            elif tool_name == "get_services_status_tool":
                return await self.handle_get_services_status(request_id, arguments)
            elif tool_name == "open_dashboard":
                return await self.handle_open_dashboard(request_id, arguments)

            # RAG Search Tools
            elif tool_name == "rag_search":
                return await self.handle_rag_search(request_id, arguments)
            elif tool_name == "rag_index":
                return await self.handle_rag_index(request_id, arguments)
            elif tool_name == "rag_status":
                return await self.handle_rag_status(request_id, arguments)
            elif tool_name == "rag_configure":
                return await self.handle_rag_configure(request_id, arguments)

            # Hook Management Tools
            elif tool_name == "rag_configure_hooks":
                return await self.handle_rag_configure_hooks(request_id, arguments)
            elif tool_name == "rag_set_citation_format":
                return await self.handle_rag_set_citation_format(request_id, arguments)
            elif tool_name == "rag_get_hook_status":
                return await self.handle_rag_get_hook_status(request_id, arguments)
            elif tool_name == "rag_set_error_mode":
                return await self.handle_rag_set_error_mode(request_id, arguments)

            # Multi-Agent Framework Tools
            elif tool_name == "maf_execute":
                return await self.handle_maf_execute(request_id, arguments)
            elif tool_name == "maf_status":
                return await self.handle_maf_status(request_id, arguments)
            elif tool_name == "maf_classify":
                return await self.handle_maf_classify(request_id, arguments)

            else:
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32601,
                        "message": f"Unknown tool: {tool_name}"
                    }
                }

        except Exception as e:
            logger.error(f"Error calling tool {tool_name}: {e}", exc_info=True)
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32603,
                    "message": f"Internal error: {str(e)}"
                }
            }

    # Service Management Tool Handlers

    async def handle_start_services(self, request_id: int, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle start services request."""
        try:
            results = ensure_services_running()
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Services started:\n{json.dumps(results, indent=2)}"
                        }
                    ]
                }
            }
        except Exception as e:
            logger.error(f"Failed to start services: {e}")
            return self.error_response(request_id, f"Failed to start services: {str(e)}")

    async def handle_get_services_status(self, request_id: int, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle get services status request."""
        try:
            status = get_services_status()
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Service status:\n{json.dumps(status, indent=2)}"
                        }
                    ]
                }
            }
        except Exception as e:
            logger.error(f"Failed to get service status: {e}")
            return self.error_response(request_id, f"Failed to get service status: {str(e)}")

    async def handle_open_dashboard(self, request_id: int, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle open dashboard request."""
        try:
            open_dashboard_in_browser()
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": "Web dashboard opened in default browser at http://localhost:5000"
                        }
                    ]
                }
            }
        except Exception as e:
            logger.error(f"Failed to open dashboard: {e}")
            return self.error_response(request_id, f"Failed to open dashboard: {str(e)}")

    # RAG Tool Handlers

    async def handle_rag_search(self, request_id: int, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle RAG search request."""
        query = arguments.get("query", "").strip()

        # Validate query
        if not query:
            return self.error_response(request_id, "Missing required parameter: query")
        if len(query) > MAX_QUERY_LENGTH:
            return self.error_response(request_id, f"Query too long (max {MAX_QUERY_LENGTH} characters)")

        # Validate top_k
        top_k = arguments.get("top_k", 5)
        if not isinstance(top_k, int) or top_k < 1 or top_k > MAX_TOP_K:
            return self.error_response(request_id, f"Invalid top_k (must be 1-{MAX_TOP_K})")

        use_llm = arguments.get("use_llm", True)

        # Check if vector store exists
        vector_store_path = project_root / "data" / "vectors" / "vectors.index"
        if not vector_store_path.exists():
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": "No vector index found. Please index documents first using the rag_index tool."
                        }
                    ]
                }
            }

        try:
            import time
            formatter = OutputFormatter(verbose=False)

            # Perform search
            search_start = time.time()
            documents = self.retriever.search(query, top_k=top_k)
            search_time_ms = (time.time() - search_start) * 1000

            # Generate response if requested
            answer = None
            if use_llm and documents and self.assistant:
                synthesis_start = time.time()
                response = self.assistant.generate_response(query, documents)
                (time.time() - synthesis_start) * 1000
                answer = response.get("answer", "")

            # Format results with clean output
            result_text = formatter.format_header("RAG Search Results", 1)
            result_text += formatter.format_search_results(
                num_results=len(documents),
                search_time_ms=search_time_ms
            )

            if answer:
                result_text += formatter.format_header("Response", 2)
                result_text += f"{answer}\n\n"
                result_text += formatter.format_synthesis(
                    num_sources=len(documents)
                )

            # Add document previews
            result_text += formatter.format_header("Retrieved Documents", 2)
            for i, doc in enumerate(documents, 1):
                result_text += formatter.format_document_preview(
                    title=f"{i}. {doc.get('source', 'Unknown')} (score: {doc.get('score', 0):.3f})",
                    content=doc.get('content', ''),
                    max_length=200
                )

            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": result_text
                        }
                    ]
                }
            }

        except Exception as e:
            logger.error(f"RAG search failed: {e}", exc_info=True)
            return self.error_response(request_id, f"Search failed: {str(e)}")

    async def handle_rag_index(self, request_id: int, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle RAG index request."""
        from core.document_processor import DocumentProcessor

        path = arguments.get("path", "")
        recursive = arguments.get("recursive", True)

        if not path:
            return self.error_response(request_id, "Missing required parameter: path")

        path_obj = Path(path)
        if not path_obj.exists():
            return self.error_response(request_id, f"Path does not exist: {path}")

        try:
            # Re-initialize components if needed
            if not self.vector_store:
                from core.vector_store import get_vector_store
                from core.embeddings import get_embedding_generator
                self.vector_store = get_vector_store()
                self.embedding_model = get_embedding_generator()

            # Process documents
            processor = DocumentProcessor(self.config)
            if path_obj.is_dir():
                documents = processor.process_directory(path_obj, recursive=recursive)
            else:
                documents = processor.process_file(path_obj)
                documents = [documents] if documents else []

            if not documents:
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": f"No documents found to index at {path}"
                            }
                        ]
                    }
                }

            # Generate embeddings and add to store
            embeddings = self.embedding_model.encode_batch(
                [doc["content"] for doc in documents]
            )
            self.vector_store.add_documents(documents, embeddings)

            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Successfully indexed {len(documents)} documents from {path}"
                        }
                    ]
                }
            }

        except Exception as e:
            logger.error(f"RAG indexing failed: {e}", exc_info=True)
            return self.error_response(request_id, f"Indexing failed: {str(e)}")

    async def handle_rag_status(self, request_id: int, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle RAG status request."""
        try:
            # Check if vector store exists
            vector_store_path = project_root / "data" / "vectors" / "vectors.index"
            has_index = vector_store_path.exists()

            status = {
                "initialized": has_index,
                "vector_store_path": str(vector_store_path),
                "configuration": {
                    "embedding_model": self.config.embeddings.model_name,
                    "embedding_dimensions": self.config.embeddings.dimensions,
                    "retrieval_top_k": self.config.retrieval.final_results,
                    "hybrid_ratio": self.config.retrieval.vector_weight
                }
            }

            if has_index and self.vector_store:
                status["statistics"] = {
                    "total_documents": self.vector_store.index.ntotal,
                    "index_type": type(self.vector_store.index).__name__
                }

            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": f"RAG System Status:\n{json.dumps(status, indent=2)}"
                        }
                    ]
                }
            }

        except Exception as e:
            logger.error(f"RAG status check failed: {e}")
            return self.error_response(request_id, f"Status check failed: {str(e)}")

    async def handle_rag_configure(self, request_id: int, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle RAG configure request."""
        setting = arguments.get("setting", "")
        value = arguments.get("value")

        try:
            if not setting:
                # Return current configuration
                config_dict = {
                    "retrieval": {
                        "top_k": self.config.retrieval.final_results,
                        "hybrid_ratio": self.config.retrieval.vector_weight
                    },
                    "claude": {
                        "model": self.config.claude.model,
                        "max_tokens": self.config.claude.max_tokens
                    }
                }
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": f"Current Configuration:\n{json.dumps(config_dict, indent=2)}"
                            }
                        ]
                    }
                }

            # Update configuration
            if setting == "retrieval.top_k":
                self.config.retrieval.final_results = int(value)
            elif setting == "retrieval.hybrid_ratio":
                self.config.retrieval.vector_weight = float(value)
            elif setting == "claude.max_tokens":
                self.config.claude.max_tokens = int(value)
            else:
                return self.error_response(request_id, f"Unknown setting: {setting}")

            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Configuration updated: {setting} = {value}"
                        }
                    ]
                }
            }

        except Exception as e:
            logger.error(f"RAG configuration failed: {e}")
            return self.error_response(request_id, f"Configuration failed: {str(e)}")

    # Hook Management Tool Handlers

    async def handle_rag_configure_hooks(self, request_id: int, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle hook configuration request."""
        hook_name = arguments.get("hook_name", "")
        enabled = arguments.get("enabled", True)

        hook_config_file = project_root / "config" / "hook_config.json"

        try:
            # Load current config
            if hook_config_file.exists():
                with open(hook_config_file, 'r') as f:
                    config = json.load(f)
            else:
                config = {}

            # Update hook status
            config[hook_name] = {"enabled": enabled}

            # Save config
            hook_config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(hook_config_file, 'w') as f:
                json.dump(config, f, indent=2)

            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Hook '{hook_name}' {'enabled' if enabled else 'disabled'}"
                        }
                    ]
                }
            }

        except Exception as e:
            logger.error(f"Hook configuration failed: {e}")
            return self.error_response(request_id, f"Hook configuration failed: {str(e)}")

    async def handle_rag_set_citation_format(self, request_id: int, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle citation format configuration request."""
        format_type = arguments.get("format", "inline")
        max_citations = arguments.get("max_citations", 3)

        citation_config_file = project_root / "config" / "citation_config.json"

        try:
            config = {
                "format": format_type,
                "max_citations": max_citations
            }

            citation_config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(citation_config_file, 'w') as f:
                json.dump(config, f, indent=2)

            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Citation format set to '{format_type}' with max {max_citations} citations"
                        }
                    ]
                }
            }

        except Exception as e:
            logger.error(f"Citation format configuration failed: {e}")
            return self.error_response(request_id, f"Citation format configuration failed: {str(e)}")

    async def handle_rag_get_hook_status(self, request_id: int, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle hook status request."""
        hook_config_file = project_root / "config" / "hook_config.json"

        try:
            # Load hook configuration
            if hook_config_file.exists():
                with open(hook_config_file, 'r') as f:
                    config = json.load(f)
            else:
                # Default status
                config = {
                    "response_post": {"enabled": True},
                    "error_handler": {"enabled": True},
                    "plugin_state_change": {"enabled": True},
                    "document_indexing": {"enabled": False}
                }

            # Check which hooks are actually available
            hooks_dir = project_root / "src" / "plugin" / "hooks"
            available_hooks = []
            if hooks_dir.exists():
                for hook_file in hooks_dir.glob("*.py"):
                    if hook_file.stem not in ['__init__', 'user-prompt-submit', 'update-rag-hook']:
                        available_hooks.append(hook_file.stem)

            status = {
                "configuration": config,
                "available_hooks": available_hooks,
                "hooks_directory": str(hooks_dir)
            }

            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Hook Status:\n{json.dumps(status, indent=2)}"
                        }
                    ]
                }
            }

        except Exception as e:
            logger.error(f"Hook status check failed: {e}")
            return self.error_response(request_id, f"Hook status check failed: {str(e)}")

    async def handle_rag_set_error_mode(self, request_id: int, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle error mode configuration request."""
        mode = arguments.get("mode", "inline_warning")

        error_config_file = project_root / "config" / "error_config.json"

        try:
            from datetime import datetime

            config = {
                "mode": mode,
                "updated_at": datetime.now().isoformat()
            }

            error_config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(error_config_file, 'w') as f:
                json.dump(config, f, indent=2)

            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Error handling mode set to '{mode}'"
                        }
                    ]
                }
            }

        except Exception as e:
            logger.error(f"Error mode configuration failed: {e}")
            return self.error_response(request_id, f"Error mode configuration failed: {str(e)}")

    # Multi-Agent Framework Tool Handlers

    async def handle_maf_execute(self, request_id: int, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle multi-agent framework execution request.

        Uses embedded MAF at src/agents/maf/ via maf_connector.
        """
        task = arguments.get("task", "")
        workflow = arguments.get("workflow", "code_generation")
        use_rag = arguments.get("use_rag", True)

        if not task:
            return self.error_response(request_id, "Missing required parameter: task")

        try:
            # Use embedded MAF via maf_connector
            from integrations.maf_connector import get_maf_connector

            maf_connector = get_maf_connector()

            if not maf_connector.is_available():
                return self.error_response(
                    request_id,
                    "Multi-agent framework not available. Embedded MAF may not be properly installed."
                )

            # Map workflow to agent name
            workflow_agent_map = {
                "bug_fix": "debugger",
                "code_generation": "developer",
                "code_review": "reviewer",
                "testing": "tester",
                "design": "architect",
                "documentation": "documenter",
                "optimization": "optimizer"
            }

            agent_name = workflow_agent_map.get(workflow, "developer")

            # Execute via embedded MAF
            task_data = {
                "task": task,
                "workflow": workflow,
                "use_rag": use_rag
            }

            result = await maf_connector.execute_agent(agent_name, task_data, timeout=30.0)

            if result and result.status == 'completed':
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": f"Multi-Agent Execution Result ({agent_name}):\n\n{result.content}\n\nExecution time: {result.execution_time:.2f}s"
                            }
                        ]
                    }
                }
            else:
                error_msg = result.content if result else "Task execution failed"
                return self.error_response(request_id, f"MAF execution failed: {error_msg}")

        except Exception as e:
            logger.error(f"Multi-agent execution failed: {e}", exc_info=True)
            return self.error_response(request_id, f"Multi-agent execution failed: {str(e)}")

    async def handle_maf_status(self, request_id: int, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle multi-agent framework status request.

        Uses embedded MAF at src/agents/maf/ via maf_connector.
        """
        try:
            # Use embedded MAF health check
            from integrations.maf_connector import get_maf_connector

            maf_connector = get_maf_connector()
            health = await maf_connector.health_check()

            # Add RAG integration status
            status = {
                **health,
                "rag_integration": self.retriever is not None,
                "retriever_initialized": self.retriever is not None,
                "vector_store_loaded": self.vector_store is not None
            }

            # Add component count
            if status['maf_available']:
                status['agent_count'] = len(status['available_agents'])

            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Multi-Agent Framework Status:\n{json.dumps(status, indent=2)}"
                        }
                    ]
                }
            }

        except Exception as e:
            logger.error(f"Multi-agent status check failed: {e}")
            return self.error_response(request_id, f"Status check failed: {str(e)}")

    async def handle_maf_classify(self, request_id: int, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle query classification request.

        Uses embedded query classifier from src/core which integrates with embedded MAF.
        """
        query = arguments.get("query", "")

        if not query:
            return self.error_response(request_id, "Missing required parameter: query")

        try:
            # Import query classifier
            from core.query_classifier import get_query_classifier

            classifier = get_query_classifier()
            classification = classifier.classify(query)

            result = {
                "query": query,
                "intent": classification.get("intent"),
                "recommended_agents": classification.get("agents", []),
                "confidence": classification.get("confidence", 0.0),
                "use_rag": classification.get("use_rag", False)
            }

            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Query Classification:\n{json.dumps(result, indent=2)}"
                        }
                    ]
                }
            }

        except Exception as e:
            logger.error(f"Query classification failed: {e}", exc_info=True)
            return self.error_response(request_id, f"Classification failed: {str(e)}")

    # Helper methods

    def error_response(self, request_id: Any, message: str) -> Dict[str, Any]:
        """Create an error response."""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": -32603,
                "message": message
            }
        }

    async def run(self):
        """Run the MCP server, reading from stdin and writing to stdout."""
        logger.info("Starting Unified MCP server")

        # Auto-start monitoring services
        logger.info("Auto-starting monitoring services...")
        try:
            await self.initialize()
        except Exception as e:
            logger.warning(f"Failed to auto-start services: {e}")

        # Read and process MCP messages from stdin
        try:
            for line in sys.stdin:
                line = line.strip()
                if not line:
                    continue

                try:
                    message = json.loads(line)
                    response = await self.handle_request(message)

                    # Send response
                    json_response = json.dumps(response)
                    sys.stdout.write(json_response + '\n')
                    sys.stdout.flush()

                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON received: {e}")
                except Exception as e:
                    logger.error(f"Error processing MCP message: {e}", exc_info=True)

        except KeyboardInterrupt:
            logger.info("MCP server interrupted")
        except Exception as e:
            logger.error(f"MCP server error: {e}", exc_info=True)

        logger.info("Unified MCP server stopped")

async def main():
    """Main entry point."""
    if not sys.stdin.isatty():
        # Run as MCP server
        server = UnifiedMCPServer()
        await server.run()
    else:
        # CLI mode - show help
        print("RAG-CLI Unified MCP Server v0.3.0")
        print("=" * 50)
        print("\nThis server provides service management, RAG search, and multi-agent tools.")
        print("\nAvailable tool categories:")
        print("\n  Service Management:")
        print("    - start_services, get_services_status_tool, open_dashboard")
        print("\n  RAG Search:")
        print("    - rag_search, rag_index, rag_status, rag_configure")
        print("\n  Hook Management:")
        print("    - rag_configure_hooks, rag_set_citation_format")
        print("    - rag_get_hook_status, rag_set_error_mode")
        print("\n  Multi-Agent Framework:")
        print("    - maf_execute, maf_status, maf_classify")
        print("\nTo use this server, configure it in Claude Code's MCP settings.")

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server interrupted")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Server failed: {e}", exc_info=True)
        sys.exit(1)
