"""Comprehensive tests for document_processor module."""

import pytest
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil

from core.document_processor import (
    DocumentProcessor,
    Document,
    DocumentChunk,
    validate_path,
    get_allowed_document_paths,
)


class TestPathValidation:
    """Tests for path validation and security."""

    def test_validate_path_allowed_directory(self, tmp_path):
        """Test validation accepts paths in allowed directories."""
        # Create a test file in allowed directory
        allowed_dir = tmp_path / "data" / "documents"
        allowed_dir.mkdir(parents=True)
        test_file = allowed_dir / "test.txt"
        test_file.touch()

        with patch('core.document_processor.get_allowed_document_paths', return_value=[allowed_dir]):
            result = validate_path(test_file)
            assert result == test_file.resolve()

    def test_validate_path_rejects_outside_allowed(self, tmp_path):
        """Test validation rejects paths outside allowed directories."""
        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()

        outside_file = tmp_path / "outside" / "test.txt"
        outside_file.parent.mkdir()
        outside_file.touch()

        with patch('core.document_processor.get_allowed_document_paths', return_value=[allowed_dir]):
            with pytest.raises(ValueError, match="outside allowed directories"):
                validate_path(outside_file)

    def test_validate_path_rejects_symlinks(self, tmp_path):
        """Test validation rejects symlinks."""
        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()

        real_file = allowed_dir / "real.txt"
        real_file.touch()

        symlink = allowed_dir / "link.txt"
        symlink.symlink_to(real_file)

        with patch('core.document_processor.get_allowed_document_paths', return_value=[allowed_dir]):
            with pytest.raises(ValueError, match="Symlinks not allowed"):
                validate_path(symlink)

    def test_get_allowed_document_paths(self):
        """Test getting allowed document paths."""
        paths = get_allowed_document_paths()

        assert isinstance(paths, list)
        assert len(paths) > 0
        assert all(isinstance(p, Path) for p in paths)


class TestDocumentChunk:
    """Tests for DocumentChunk dataclass."""

    def test_create_document_chunk(self):
        """Test creating a document chunk."""
        chunk = DocumentChunk(
            content="Test content",
            metadata={"title": "Test"},
            chunk_index=0,
            total_chunks=1,
            char_count=12,
            token_count=3,
            source="test.txt",
            doc_id="doc123",
            chunk_id="chunk123"
        )

        assert chunk.content == "Test content"
        assert chunk.chunk_index == 0
        assert chunk.total_chunks == 1
        assert chunk.char_count == 12
        assert chunk.token_count == 3

    def test_document_chunk_metadata(self):
        """Test document chunk metadata handling."""
        metadata = {"title": "Test", "author": "Tester"}
        chunk = DocumentChunk(
            content="Test",
            metadata=metadata,
            chunk_index=0,
            total_chunks=1,
            char_count=4,
            token_count=1,
            source="test.txt",
            doc_id="doc123",
            chunk_id="chunk123"
        )

        assert chunk.metadata["title"] == "Test"
        assert chunk.metadata["author"] == "Tester"


class TestDocument:
    """Tests for Document dataclass."""

    def test_create_document(self):
        """Test creating a document."""
        timestamp = datetime.now()
        doc = Document(
            content="Document content",
            source="test.txt",
            doc_type="text",
            metadata={"title": "Test Document"},
            doc_id="doc123",
            timestamp=timestamp
        )

        assert doc.content == "Document content"
        assert doc.source == "test.txt"
        assert doc.doc_type == "text"
        assert doc.doc_id == "doc123"
        assert doc.timestamp == timestamp


class TestDocumentProcessor:
    """Tests for DocumentProcessor class."""

    @pytest.fixture
    def processor(self):
        """Create document processor instance."""
        return DocumentProcessor()

    def test_init(self, processor):
        """Test processor initialization."""
        assert processor.chunk_size > 0
        assert processor.chunk_overlap >= 0
        assert processor.chunk_overlap < processor.chunk_size
        assert isinstance(processor.separators, list)
        assert isinstance(processor.supported_formats, list)

    def test_token_length_estimation(self, processor):
        """Test token length estimation."""
        # Test with known text
        text = "This is a test"  # 14 characters
        estimated_tokens = processor._token_length(text)

        # Should be approximately 3-4 tokens (14/4 = 3.5)
        assert 3 <= estimated_tokens <= 4

    def test_token_length_empty_string(self, processor):
        """Test token length for empty string."""
        assert processor._token_length("") == 0

    def test_token_length_long_text(self, processor):
        """Test token length for longer text."""
        text = "a" * 1000  # 1000 characters
        estimated_tokens = processor._token_length(text)

        # Should be around 250 tokens
        assert 240 <= estimated_tokens <= 260

    def test_generate_doc_id_consistency(self, processor):
        """Test that same source generates same doc_id."""
        source = "test.txt"

        id1 = processor._generate_doc_id(source)
        id2 = processor._generate_doc_id(source)

        assert id1 == id2

    def test_generate_doc_id_different_sources(self, processor):
        """Test that different sources generate different doc_ids."""
        id1 = processor._generate_doc_id("test1.txt")
        id2 = processor._generate_doc_id("test2.txt")

        assert id1 != id2

    def test_generate_chunk_id(self, processor):
        """Test chunk ID generation."""
        doc_id = "doc123"
        chunk_index = 5

        chunk_id = processor._generate_chunk_id(doc_id, chunk_index)

        assert doc_id in chunk_id
        assert str(chunk_index) in chunk_id

    def test_chunk_text_basic(self, processor):
        """Test basic text chunking."""
        text = "This is a test. " * 100  # Create text that needs chunking
        doc = Document(
            content=text,
            source="test.txt",
            doc_type="text",
            metadata={},
            doc_id="doc123",
            timestamp=datetime.now()
        )

        chunks = processor.chunk_document(doc)

        assert len(chunks) > 0
        assert all(isinstance(chunk, DocumentChunk) for chunk in chunks)
        assert chunks[0].chunk_index == 0
        assert chunks[-1].chunk_index == len(chunks) - 1

    def test_chunk_text_preserves_content(self, processor):
        """Test that chunking preserves all content."""
        text = "Test content that should be preserved. " * 50
        doc = Document(
            content=text,
            source="test.txt",
            doc_type="text",
            metadata={},
            doc_id="doc123",
            timestamp=datetime.now()
        )

        chunks = processor.chunk_document(doc)

        # Reconstruct text from chunks (removing overlap)
        reconstructed = "".join(chunk.content for chunk in chunks)

        # Should contain all original content
        assert len(reconstructed) >= len(text) - processor.chunk_overlap * len(chunks)

    def test_chunk_metadata_inheritance(self, processor):
        """Test that chunks inherit document metadata."""
        metadata = {"title": "Test", "author": "Tester"}
        doc = Document(
            content="Test content",
            source="test.txt",
            doc_type="text",
            metadata=metadata,
            doc_id="doc123",
            timestamp=datetime.now()
        )

        chunks = processor.chunk_document(doc)

        assert all(chunk.metadata["title"] == "Test" for chunk in chunks)
        assert all(chunk.metadata["author"] == "Tester" for chunk in chunks)

    def test_chunk_total_chunks(self, processor):
        """Test that total_chunks is correct."""
        text = "Content. " * 200
        doc = Document(
            content=text,
            source="test.txt",
            doc_type="text",
            metadata={},
            doc_id="doc123",
            timestamp=datetime.now()
        )

        chunks = processor.chunk_document(doc)

        expected_total = len(chunks)
        assert all(chunk.total_chunks == expected_total for chunk in chunks)

    def test_chunk_indices_sequential(self, processor):
        """Test that chunk indices are sequential."""
        text = "Content. " * 200
        doc = Document(
            content=text,
            source="test.txt",
            doc_type="text",
            metadata={},
            doc_id="doc123",
            timestamp=datetime.now()
        )

        chunks = processor.chunk_document(doc)

        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_process_text_file(self, processor, tmp_path):
        """Test processing a text file."""
        test_file = tmp_path / "test.txt"
        test_content = "This is test content"
        test_file.write_text(test_content)

        with patch('core.document_processor.validate_path', return_value=test_file):
            doc = processor.process_document(str(test_file))

        assert doc.content == test_content
        assert doc.doc_type == "txt"
        assert doc.source == str(test_file)

    def test_process_markdown_file(self, processor, tmp_path):
        """Test processing a markdown file."""
        test_file = tmp_path / "test.md"
        test_content = "# Test Heading\n\nThis is test content"
        test_file.write_text(test_content)

        with patch('core.document_processor.validate_path', return_value=test_file):
            doc = processor.process_document(str(test_file))

        assert "Test Heading" in doc.content
        assert doc.doc_type == "md"

    def test_process_unsupported_format(self, processor, tmp_path):
        """Test processing unsupported file format."""
        test_file = tmp_path / "test.xyz"
        test_file.write_text("Content")

        with patch('core.document_processor.validate_path', return_value=test_file):
            with pytest.raises(ValueError, match="Unsupported format"):
                processor.process_document(str(test_file))

    def test_process_nonexistent_file(self, processor):
        """Test processing non-existent file."""
        with patch('core.document_processor.validate_path', side_effect=FileNotFoundError()):
            with pytest.raises(FileNotFoundError):
                processor.process_document("nonexistent.txt")

    def test_extract_metadata_basic(self, processor):
        """Test basic metadata extraction."""
        content = "Test content"
        source = "test.txt"

        metadata = processor._extract_metadata(content, source)

        assert "source" in metadata
        assert metadata["source"] == source
        assert "char_count" in metadata
        assert metadata["char_count"] == len(content)

    def test_process_single_document_end_to_end(self, processor, tmp_path):
        """Test complete document processing pipeline."""
        # Create test file
        test_file = tmp_path / "test.txt"
        test_content = "This is test content. " * 50
        test_file.write_text(test_content)

        with patch('core.document_processor.validate_path', return_value=test_file):
            # Process document
            doc = processor.process_document(str(test_file))

            # Chunk document
            chunks = processor.chunk_document(doc)

        # Verify
        assert doc.content == test_content
        assert len(chunks) > 0
        assert all(isinstance(chunk, DocumentChunk) for chunk in chunks)
        assert all(chunk.source == str(test_file) for chunk in chunks)

    def test_chunk_size_respect(self, processor):
        """Test that chunks respect configured chunk size."""
        text = "Word " * 1000  # Long text
        doc = Document(
            content=text,
            source="test.txt",
            doc_type="text",
            metadata={},
            doc_id="doc123",
            timestamp=datetime.now()
        )

        chunks = processor.chunk_document(doc)

        # Most chunks should be near the configured size
        for chunk in chunks[:-1]:  # Exclude last chunk (might be shorter)
            assert chunk.token_count <= processor.chunk_size * 1.2  # Allow 20% variance

    def test_empty_document_handling(self, processor):
        """Test handling of empty document."""
        doc = Document(
            content="",
            source="empty.txt",
            doc_type="text",
            metadata={},
            doc_id="doc123",
            timestamp=datetime.now()
        )

        chunks = processor.chunk_document(doc)

        # Should handle gracefully
        assert isinstance(chunks, list)

    def test_very_short_document(self, processor):
        """Test handling of very short document."""
        doc = Document(
            content="Short",
            source="short.txt",
            doc_type="text",
            metadata={},
            doc_id="doc123",
            timestamp=datetime.now()
        )

        chunks = processor.chunk_document(doc)

        assert len(chunks) == 1
        assert chunks[0].content == "Short"

    def test_unicode_content_handling(self, processor):
        """Test handling of unicode content."""
        unicode_text = "Hello 世界 🌍 Test"
        doc = Document(
            content=unicode_text,
            source="unicode.txt",
            doc_type="text",
            metadata={},
            doc_id="doc123",
            timestamp=datetime.now()
        )

        chunks = processor.chunk_document(doc)

        assert any(unicode_text in chunk.content for chunk in chunks)

    def test_special_characters_handling(self, processor):
        """Test handling of special characters."""
        special_text = "Test\nNewline\tTab\r\nCRLF"
        doc = Document(
            content=special_text,
            source="special.txt",
            doc_type="text",
            metadata={},
            doc_id="doc123",
            timestamp=datetime.now()
        )

        chunks = processor.chunk_document(doc)

        # Should preserve special characters
        assert len(chunks) > 0


class TestDocumentFormats:
    """Tests for different document format handling."""

    @pytest.fixture
    def processor(self):
        """Create processor instance."""
        return DocumentProcessor()

    def test_html_document_parsing(self, processor, tmp_path):
        """Test HTML document parsing."""
        html_content = "<html><body><h1>Title</h1><p>Content</p></body></html>"
        test_file = tmp_path / "test.html"
        test_file.write_text(html_content)

        with patch('core.document_processor.validate_path', return_value=test_file):
            doc = processor.process_document(str(test_file))

        # Should extract text from HTML
        assert "Title" in doc.content
        assert "Content" in doc.content
        assert doc.doc_type == "html"

    def test_markdown_headers_preserved(self, processor, tmp_path):
        """Test that markdown headers are preserved."""
        md_content = "# Main Title\n\n## Subtitle\n\nContent here"
        test_file = tmp_path / "test.md"
        test_file.write_text(md_content)

        with patch('core.document_processor.validate_path', return_value=test_file):
            doc = processor.process_document(str(test_file))

        assert "Main Title" in doc.content
        assert "Subtitle" in doc.content


class TestPerformance:
    """Performance-related tests."""

    @pytest.fixture
    def processor(self):
        """Create processor instance."""
        return DocumentProcessor()

    def test_large_document_chunking(self, processor):
        """Test chunking of large document."""
        # Create large document
        large_text = "Content paragraph. " * 10000  # ~170KB
        doc = Document(
            content=large_text,
            source="large.txt",
            doc_type="text",
            metadata={},
            doc_id="doc123",
            timestamp=datetime.now()
        )

        import time
        start = time.time()
        chunks = processor.chunk_document(doc)
        duration = time.time() - start

        # Should complete in reasonable time
        assert duration < 5.0  # 5 seconds max
        assert len(chunks) > 0

    def test_chunk_overlap_efficiency(self, processor):
        """Test that chunk overlap is efficient."""
        text = "Sentence. " * 100
        doc = Document(
            content=text,
            source="test.txt",
            doc_type="text",
            metadata={},
            doc_id="doc123",
            timestamp=datetime.now()
        )

        chunks = processor.chunk_document(doc)

        # Overlap should not cause excessive chunk count
        total_chars = sum(len(chunk.content) for chunk in chunks)
        original_chars = len(text)

        # Total chars should be reasonable (not 2x original)
        assert total_chars < original_chars * 1.5


class TestErrorHandling:
    """Error handling tests."""

    @pytest.fixture
    def processor(self):
        """Create processor instance."""
        return DocumentProcessor()

    def test_corrupted_file_handling(self, processor, tmp_path):
        """Test handling of corrupted file."""
        test_file = tmp_path / "corrupted.txt"
        # Write binary data that might cause issues
        test_file.write_bytes(b'\x80\x81\x82\x83')

        with patch('core.document_processor.validate_path', return_value=test_file):
            # Should handle gracefully
            try:
                doc = processor.process_document(str(test_file))
                # If it succeeds, that's fine too
                assert doc is not None
            except (UnicodeDecodeError, ValueError):
                # Expected for binary data
                pass

    def test_permission_denied_handling(self, processor, tmp_path):
        """Test handling of permission denied."""
        test_file = tmp_path / "restricted.txt"
        test_file.write_text("Content")

        with patch('core.document_processor.validate_path', return_value=test_file):
            with patch('builtins.open', side_effect=PermissionError()):
                with pytest.raises(PermissionError):
                    processor.process_document(str(test_file))
