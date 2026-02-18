"""Tests for PDF ingestion module."""

import pytest
from pathlib import Path

from src.ingestion.pdf_loader import PDFLoader, Document
from src.ingestion.text_chunker import TextChunker, Chunk, SemanticChunker


class TestPDFLoader:
    """Tests for PDFLoader class."""
    
    def test_loader_initialization(self):
        """Test loader initializes correctly."""
        loader = PDFLoader(use_pdfplumber=True)
        assert loader.use_pdfplumber is True
        
        loader = PDFLoader(use_pdfplumber=False)
        assert loader.use_pdfplumber is False
    
    def test_load_nonexistent_file(self):
        """Test loading non-existent file raises error."""
        loader = PDFLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_pdf("nonexistent.pdf")
    
    def test_load_nonexistent_directory(self):
        """Test loading non-existent directory raises error."""
        loader = PDFLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_directory("nonexistent_dir")
    
    def test_format_table(self):
        """Test table formatting."""
        loader = PDFLoader()
        
        table = [
            ["Header1", "Header2"],
            ["Value1", "Value2"],
        ]
        
        result = loader._format_table(table)
        assert "Header1 | Header2" in result
        assert "Value1 | Value2" in result
    
    def test_format_empty_table(self):
        """Test empty table formatting."""
        loader = PDFLoader()
        assert loader._format_table([]) == ""
        assert loader._format_table(None) == ""


class TestTextChunker:
    """Tests for TextChunker class."""
    
    def test_chunker_initialization(self):
        """Test chunker initializes with correct defaults."""
        chunker = TextChunker()
        assert chunker.chunk_size == 1000
        assert chunker.chunk_overlap == 200
        assert chunker.separator == "\n\n"
    
    def test_chunker_custom_params(self):
        """Test chunker with custom parameters."""
        chunker = TextChunker(chunk_size=500, chunk_overlap=100, separator="\n")
        assert chunker.chunk_size == 500
        assert chunker.chunk_overlap == 100
        assert chunker.separator == "\n"
    
    def test_chunk_small_document(self):
        """Test chunking a small document."""
        chunker = TextChunker(chunk_size=1000)
        
        doc = Document(
            content="This is a small document.",
            metadata={"filename": "test.pdf", "page": 1},
            page_number=1,
            source="test.pdf"
        )
        
        chunks = chunker.chunk_document(doc)
        assert len(chunks) == 1
        assert chunks[0].content == "This is a small document."
    
    def test_chunk_large_document(self):
        """Test chunking a large document."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        
        # Create a document larger than chunk_size
        content = "This is a test sentence. " * 20
        doc = Document(
            content=content,
            metadata={"filename": "test.pdf", "page": 1},
            page_number=1,
            source="test.pdf"
        )
        
        chunks = chunker.chunk_document(doc)
        assert len(chunks) > 1
        
        # Verify all chunks have content
        for chunk in chunks:
            assert len(chunk.content) > 0
    
    def test_chunk_metadata_preserved(self):
        """Test that metadata is preserved in chunks."""
        chunker = TextChunker()
        
        doc = Document(
            content="Test content for chunking.",
            metadata={"filename": "policy.pdf", "page": 5, "custom": "value"},
            page_number=5,
            source="policy.pdf"
        )
        
        chunks = chunker.chunk_document(doc)
        
        assert chunks[0].metadata["filename"] == "policy.pdf"
        assert chunks[0].metadata["page"] == 5
        assert chunks[0].metadata["custom"] == "value"
    
    def test_chunk_id_generation(self):
        """Test chunk ID generation."""
        chunker = TextChunker()
        
        doc = Document(
            content="Test content.",
            metadata={"filename": "test.pdf"},
            page_number=1,
            source="test.pdf"
        )
        
        chunks = chunker.chunk_document(doc)
        assert "test.pdf_p1_c0" in chunks[0].chunk_id
    
    def test_find_break_point(self):
        """Test break point finding."""
        chunker = TextChunker()
        
        # Test sentence break
        text = "First sentence. Second sentence. Third sentence."
        break_point = chunker._find_break_point(text, 30)
        assert text[break_point-1] == "."
        
        # Test word break when no sentence
        text = "word1 word2 word3 word4 word5"
        break_point = chunker._find_break_point(text, 15)
        assert text[break_point] == " " or break_point == 15


class TestSemanticChunker:
    """Tests for SemanticChunker class."""
    
    def test_semantic_chunker_initialization(self):
        """Test semantic chunker initializes correctly."""
        chunker = SemanticChunker()
        assert chunker.section_regex is not None
    
    def test_split_by_sections(self):
        """Test section splitting."""
        chunker = SemanticChunker()
        
        text = """# Section 1
        Content for section 1.
        
        # Section 2
        Content for section 2.
        """
        
        sections = chunker._split_by_sections(text)
        assert len(sections) >= 2
    
    def test_healthcare_section_patterns(self):
        """Test healthcare-specific section patterns."""
        chunker = SemanticChunker()
        
        text = """POLICY: Patient Privacy
        
        This policy covers patient data.
        
        PROCEDURE: Data Handling
        
        Follow these steps for data handling.
        """
        
        sections = chunker._split_by_sections(text)
        assert len(sections) >= 2


class TestChunkDocuments:
    """Tests for batch document chunking."""
    
    def test_chunk_multiple_documents(self):
        """Test chunking multiple documents."""
        chunker = TextChunker()
        
        docs = [
            Document(
                content="Document 1 content.",
                metadata={"filename": "doc1.pdf"},
                page_number=1,
                source="doc1.pdf"
            ),
            Document(
                content="Document 2 content.",
                metadata={"filename": "doc2.pdf"},
                page_number=1,
                source="doc2.pdf"
            ),
        ]
        
        chunks = chunker.chunk_documents(docs)
        assert len(chunks) == 2
    
    def test_empty_document_list(self):
        """Test chunking empty document list."""
        chunker = TextChunker()
        chunks = chunker.chunk_documents([])
        assert len(chunks) == 0
