"""Core component tests for RAG-CLI."""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import faiss

# Add project root to path
import sys
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.core.embeddings import EmbeddingModel, get_embedding_model
from src.core.vector_store import VectorStore, get_vector_store
from src.core.document_processor import DocumentProcessor
from src.core.retrieval_pipeline import HybridRetriever
from src.core.claude_integration import ClaudeAssistant


class TestEmbeddings:
    """Test embedding system."""

    @patch('sentence_transformers.SentenceTransformer')
    def test_model_loading(self, mock_transformer):
        """Test embedding model loading."""
        mock_model = MagicMock()
        mock_transformer.return_value = mock_model

        model = EmbeddingModel(Mock())

        mock_transformer.assert_called_once()
        assert model.model == mock_model

    @patch('sentence_transformers.SentenceTransformer')
    def test_single_encoding(self, mock_transformer):
        """Test encoding a single text."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])
        mock_transformer.return_value = mock_model

        model = EmbeddingModel(Mock())
        embedding = model.encode("test text")

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (3,)
        mock_model.encode.assert_called_once()

    @patch('sentence_transformers.SentenceTransformer')
    def test_batch_encoding(self, mock_transformer):
        """Test batch encoding."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        mock_transformer.return_value = mock_model

        model = EmbeddingModel(Mock())
        embeddings = model.encode_batch(["text1", "text2", "text3"])

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (3, 2)
        mock_model.encode.assert_called_once()

    @patch('sentence_transformers.SentenceTransformer')
    def test_cache_functionality(self, mock_transformer):
        """Test LRU cache for repeated queries."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])
        mock_transformer.return_value = mock_model

        config = Mock()
        config.embeddings.cache_enabled = True
        config.embeddings.cache_size = 100

        model = EmbeddingModel(config)

        # Encode same text twice
        embedding1 = model.encode("test text")
        embedding2 = model.encode("test text")

        # Should only call encode once due to cache
        assert mock_model.encode.call_count == 1
        assert np.array_equal(embedding1, embedding2)


class TestVectorStore:
    """Test FAISS vector store."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_index_creation(self):
        """Test FAISS index creation."""
        config = Mock()
        config.vector_store.index_type = "flat"
        config.embeddings.model_dim = 384

        store = VectorStore(config)

        assert store.index is not None
        assert store.index.d == 384  # Dimension
        assert store.index.ntotal == 0  # No vectors yet

    def test_add_documents(self):
        """Test adding documents to store."""
        config = Mock()
        config.vector_store.index_type = "flat"
        config.embeddings.model_dim = 3

        store = VectorStore(config)

        documents = [
            {"id": "1", "content": "doc1", "metadata": {"source": "file1.txt"}},
            {"id": "2", "content": "doc2", "metadata": {"source": "file2.txt"}}
        ]

        embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

        store.add_documents(documents, embeddings)

        assert store.index.ntotal == 2
        assert len(store.documents) == 2
        assert store.documents[0]["content"] == "doc1"

    def test_search(self):
        """Test vector search."""
        config = Mock()
        config.vector_store.index_type = "flat"
        config.embeddings.model_dim = 3

        store = VectorStore(config)

        # Add some documents
        documents = [
            {"id": "1", "content": "similar doc", "metadata": {}},
            {"id": "2", "content": "different doc", "metadata": {}},
            {"id": "3", "content": "another similar", "metadata": {}}
        ]

        embeddings = np.array([
            [0.9, 0.1, 0.1],  # similar to query
            [0.1, 0.9, 0.1],  # different from query
            [0.8, 0.2, 0.1]   # somewhat similar
        ])

        store.add_documents(documents, embeddings)

        # Search with query similar to first doc
        query_embedding = np.array([0.85, 0.15, 0.1])
        results = store.search(query_embedding, top_k=2)

        assert len(results) == 2
        # First result should be most similar
        assert results[0]["content"] == "similar doc"

    def test_save_load(self, temp_dir):
        """Test saving and loading index."""
        config = Mock()
        config.vector_store.index_type = "flat"
        config.embeddings.model_dim = 3
        config.vector_store.save_path = temp_dir

        # Create and populate store
        store1 = VectorStore(config)
        documents = [{"id": "1", "content": "test", "metadata": {}}]
        embeddings = np.array([[0.1, 0.2, 0.3]])
        store1.add_documents(documents, embeddings)

        # Save
        save_path = Path(temp_dir) / "test_index"
        store1.save(save_path)

        # Load in new store
        store2 = VectorStore(config)
        store2.load(save_path)

        assert store2.index.ntotal == 1
        assert len(store2.documents) == 1
        assert store2.documents[0]["content"] == "test"


class TestDocumentProcessor:
    """Test document processing."""

    @pytest.fixture
    def sample_documents(self, tmp_path):
        """Create sample documents."""
        # Create text file
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("This is a test document with some content.")

        # Create markdown file
        md_file = tmp_path / "test.md"
        md_file.write_text("# Title\n\nThis is markdown content.\n\n## Section\n\nMore content here.")

        return tmp_path

    def test_text_loading(self, sample_documents):
        """Test loading text files."""
        config = Mock()
        config.document_processing.chunk_size = 100
        config.document_processing.chunk_overlap = 20

        processor = DocumentProcessor(config)

        txt_file = sample_documents / "test.txt"
        content = processor.load_document(txt_file)

        assert content is not None
        assert "test document" in content

    def test_chunking(self):
        """Test document chunking."""
        config = Mock()
        config.document_processing.chunk_size = 50
        config.document_processing.chunk_overlap = 10

        processor = DocumentProcessor(config)

        # Create long text
        text = " ".join(["word"] * 100)  # 100 words

        chunks = processor.chunk_text(text, source="test.txt")

        assert len(chunks) > 1  # Should be split into multiple chunks
        assert all("content" in chunk for chunk in chunks)
        assert all("metadata" in chunk for chunk in chunks)
        assert all(chunk["metadata"]["source"] == "test.txt" for chunk in chunks)

    @patch('src.core.document_processor.RecursiveCharacterTextSplitter')
    def test_recursive_splitter(self, mock_splitter_class):
        """Test recursive character text splitter."""
        mock_splitter = MagicMock()
        mock_splitter.split_text.return_value = ["chunk1", "chunk2"]
        mock_splitter_class.return_value = mock_splitter

        config = Mock()
        config.document_processing.chunk_size = 500
        config.document_processing.chunk_overlap = 50

        processor = DocumentProcessor(config)
        chunks = processor.chunk_text("long text", source="test.txt")

        mock_splitter_class.assert_called_once()
        assert len(chunks) == 2

    def test_process_directory(self, sample_documents):
        """Test processing entire directory."""
        config = Mock()
        config.document_processing.chunk_size = 100
        config.document_processing.chunk_overlap = 20
        config.document_processing.file_extensions = [".txt", ".md"]

        processor = DocumentProcessor(config)
        documents = processor.process_directory(sample_documents)

        assert len(documents) > 0
        # Should have processed both .txt and .md files
        sources = [doc["metadata"]["source"] for doc in documents]
        assert any("test.txt" in s for s in sources)
        assert any("test.md" in s for s in sources)


class TestRetrievalPipeline:
    """Test hybrid retrieval pipeline."""

    @patch('src.core.retrieval_pipeline.CrossEncoder')
    def test_hybrid_search(self, mock_cross_encoder):
        """Test hybrid search combining vector and keyword."""
        # Setup mocks
        mock_reranker = MagicMock()
        mock_reranker.predict.return_value = np.array([0.9, 0.7, 0.5])
        mock_cross_encoder.return_value = mock_reranker

        mock_vector_store = MagicMock()
        mock_vector_store.search.return_value = [
            {"content": "vector result 1", "score": 0.8},
            {"content": "vector result 2", "score": 0.6}
        ]

        mock_embedding_model = MagicMock()
        mock_embedding_model.encode.return_value = np.array([0.1, 0.2, 0.3])

        config = Mock()
        config.retrieval.hybrid_ratio = 0.7
        config.retrieval.rerank = True
        config.retrieval.reranker_model = "test-reranker"

        retriever = HybridRetriever(
            vector_store=mock_vector_store,
            embedding_model=mock_embedding_model,
            config=config
        )

        # Mock keyword search
        with patch.object(retriever, 'keyword_search', return_value=[
            {"content": "keyword result 1", "score": 0.7}
        ]):
            results = retriever.search("test query", top_k=3)

        assert len(results) > 0
        mock_vector_store.search.assert_called_once()
        mock_embedding_model.encode.assert_called_once()

    def test_vector_only_search(self):
        """Test vector-only search."""
        mock_vector_store = MagicMock()
        mock_vector_store.search.return_value = [
            {"content": "result 1", "score": 0.9},
            {"content": "result 2", "score": 0.7}
        ]

        mock_embedding_model = MagicMock()
        mock_embedding_model.encode.return_value = np.array([0.1, 0.2, 0.3])

        config = Mock()

        retriever = HybridRetriever(
            vector_store=mock_vector_store,
            embedding_model=mock_embedding_model,
            config=config
        )

        results = retriever.vector_search("test query", top_k=2)

        assert len(results) == 2
        assert results[0]["content"] == "result 1"
        mock_vector_store.search.assert_called_once()

    @patch('rank_bm25.BM25Okapi')
    def test_keyword_search(self, mock_bm25_class):
        """Test keyword-only search."""
        mock_bm25 = MagicMock()
        mock_bm25.get_scores.return_value = np.array([0.5, 0.8, 0.3])
        mock_bm25_class.return_value = mock_bm25

        mock_vector_store = MagicMock()
        mock_vector_store.documents = [
            {"content": "doc 1", "metadata": {}},
            {"content": "doc 2 with keyword", "metadata": {}},
            {"content": "doc 3", "metadata": {}}
        ]

        config = Mock()

        retriever = HybridRetriever(
            vector_store=mock_vector_store,
            embedding_model=MagicMock(),
            config=config
        )

        # Initialize BM25 index
        retriever._build_keyword_index()

        results = retriever.keyword_search("keyword", top_k=2)

        assert len(results) == 2
        # Highest scoring doc should be first
        assert results[0]["score"] == 0.8


class TestClaudeIntegration:
    """Test Claude API integration."""

    @patch('anthropic.Anthropic')
    def test_assistant_creation(self, mock_anthropic_class):
        """Test Claude assistant initialization."""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        config = Mock()
        config.claude.api_key = "test-key"
        config.claude.model = "claude-haiku"
        config.claude.max_tokens = 1000

        assistant = ClaudeAssistant(config)

        mock_anthropic_class.assert_called_once_with(api_key="test-key")
        assert assistant.client == mock_client

    @patch('anthropic.Anthropic')
    def test_response_generation(self, mock_anthropic_class):
        """Test generating response with context."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Generated answer")]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_class.return_value = mock_client

        config = Mock()
        config.claude.api_key = "test-key"
        config.claude.model = "claude-haiku"
        config.claude.max_tokens = 1000
        config.claude.temperature = 0.7
        config.claude.system_prompt = "You are a helpful assistant."

        assistant = ClaudeAssistant(config)

        documents = [
            {"content": "Context doc 1", "metadata": {"source": "file1.txt"}},
            {"content": "Context doc 2", "metadata": {"source": "file2.txt"}}
        ]

        response = assistant.generate_response("Test query", documents)

        assert "answer" in response
        assert response["answer"] == "Generated answer"
        mock_client.messages.create.assert_called_once()

    @patch('anthropic.Anthropic')
    def test_streaming_response(self, mock_anthropic_class):
        """Test streaming response generation."""
        # Create mock stream
        mock_stream = MagicMock()
        mock_stream.__iter__.return_value = [
            MagicMock(delta=MagicMock(text="Part 1")),
            MagicMock(delta=MagicMock(text=" Part 2"))
        ]

        mock_client = MagicMock()
        mock_client.messages.stream.return_value.__enter__.return_value = mock_stream
        mock_anthropic_class.return_value = mock_client

        config = Mock()
        config.claude.api_key = "test-key"
        config.claude.model = "claude-haiku"
        config.claude.max_tokens = 1000
        config.claude.streaming = True
        config.claude.system_prompt = "Test prompt"

        assistant = ClaudeAssistant(config)

        # Collect streamed response
        chunks = list(assistant.generate_stream("Test query", []))
        full_response = "".join(chunks)

        assert full_response == "Part 1 Part 2"

    @patch('anthropic.Anthropic')
    def test_error_handling(self, mock_anthropic_class):
        """Test error handling in Claude integration."""
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("API Error")
        mock_anthropic_class.return_value = mock_client

        config = Mock()
        config.claude.api_key = "test-key"
        config.claude.model = "claude-haiku"
        config.claude.max_tokens = 1000
        config.claude.system_prompt = "Test"

        assistant = ClaudeAssistant(config)

        with pytest.raises(Exception) as exc_info:
            assistant.generate_response("Test query", [])

        assert "API Error" in str(exc_info.value)


class TestIntegration:
    """Integration tests for full pipeline."""

    @patch('src.core.claude_integration.Anthropic')
    @patch('sentence_transformers.SentenceTransformer')
    def test_end_to_end_retrieval(self, mock_transformer, mock_anthropic):
        """Test complete retrieval pipeline."""
        # Setup embedding model mock
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_transformer.return_value = mock_model

        # Setup Claude mock
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Answer based on context")]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        # Create config
        config = Mock()
        config.embeddings.model_name = "test-model"
        config.embeddings.model_dim = 3
        config.embeddings.cache_enabled = False
        config.vector_store.index_type = "flat"
        config.retrieval.hybrid_ratio = 0.7
        config.retrieval.rerank = False
        config.claude.api_key = "test-key"
        config.claude.model = "claude-haiku"
        config.claude.max_tokens = 1000
        config.claude.system_prompt = "Test"

        # Create components
        embedding_model = EmbeddingModel(config)
        vector_store = VectorStore(config)
        retriever = HybridRetriever(vector_store, embedding_model, config)
        assistant = ClaudeAssistant(config)

        # Add some documents
        documents = [
            {"content": "Important information", "metadata": {"source": "doc1.txt"}}
        ]
        embeddings = np.array([[0.1, 0.2, 0.3]])
        vector_store.add_documents(documents, embeddings)

        # Perform retrieval
        retrieved_docs = retriever.vector_search("query about information", top_k=1)
        assert len(retrieved_docs) > 0

        # Generate response
        response = assistant.generate_response("query", retrieved_docs)
        assert "answer" in response
        assert response["answer"] == "Answer based on context"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])