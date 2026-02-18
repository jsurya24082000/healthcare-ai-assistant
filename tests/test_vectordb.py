"""Tests for FAISS vector store module."""

import pytest
import tempfile
import shutil
from pathlib import Path

from src.ingestion.text_chunker import Chunk
from src.vectordb.faiss_store import FAISSVectorStore, EmbeddingModel


class TestEmbeddingModel:
    """Tests for EmbeddingModel class."""
    
    def test_local_model_initialization(self):
        """Test local embedding model initialization."""
        model = EmbeddingModel(model_name="all-MiniLM-L6-v2", use_openai=False)
        assert model.model_name == "all-MiniLM-L6-v2"
        assert model.use_openai is False
    
    def test_openai_model_initialization(self):
        """Test OpenAI embedding model initialization."""
        model = EmbeddingModel(model_name="text-embedding-3-small", use_openai=True)
        assert model.model_name == "text-embedding-3-small"
        assert model.use_openai is True
    
    def test_openai_dimension(self):
        """Test OpenAI embedding dimension."""
        model = EmbeddingModel(use_openai=True)
        assert model.dimension == 1536


class TestFAISSVectorStore:
    """Tests for FAISSVectorStore class."""
    
    @pytest.fixture
    def sample_chunks(self):
        """Create sample chunks for testing."""
        return [
            Chunk(
                content="Patient privacy is protected under HIPAA regulations.",
                metadata={"filename": "privacy.pdf", "page": 1},
                chunk_id="privacy.pdf_p1_c0"
            ),
            Chunk(
                content="Informed consent must be obtained before any procedure.",
                metadata={"filename": "consent.pdf", "page": 1},
                chunk_id="consent.pdf_p1_c0"
            ),
            Chunk(
                content="Medical records must be retained for seven years.",
                metadata={"filename": "records.pdf", "page": 1},
                chunk_id="records.pdf_p1_c0"
            ),
        ]
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path)
    
    def test_store_initialization_flat(self):
        """Test vector store initialization with flat index."""
        store = FAISSVectorStore(index_type="flat", dimension=384)
        assert store.index_type == "flat"
        assert store.dimension == 384
        assert store.index is not None
    
    def test_store_initialization_hnsw(self):
        """Test vector store initialization with HNSW index."""
        store = FAISSVectorStore(index_type="hnsw", dimension=384)
        assert store.index_type == "hnsw"
    
    def test_invalid_index_type(self):
        """Test invalid index type raises error."""
        with pytest.raises(ValueError):
            FAISSVectorStore(index_type="invalid", dimension=384)
    
    def test_get_stats_empty(self):
        """Test stats on empty store."""
        store = FAISSVectorStore(dimension=384)
        stats = store.get_stats()
        
        assert stats["num_chunks"] == 0
        assert stats["total_vectors"] == 0
    
    def test_search_empty_store(self):
        """Test search on empty store."""
        store = FAISSVectorStore(dimension=384)
        results = store.search("test query")
        assert len(results) == 0


class TestVectorStoreIntegration:
    """Integration tests requiring actual embedding model."""
    
    @pytest.fixture
    def embedding_model(self):
        """Create embedding model for tests."""
        # Use a small local model for testing
        return EmbeddingModel(model_name="all-MiniLM-L6-v2", use_openai=False)
    
    @pytest.fixture
    def sample_chunks(self):
        """Create sample chunks."""
        return [
            Chunk(
                content="Patient privacy is protected under HIPAA regulations.",
                metadata={"filename": "privacy.pdf", "page": 1},
                chunk_id="privacy.pdf_p1_c0"
            ),
            Chunk(
                content="Informed consent must be obtained before any procedure.",
                metadata={"filename": "consent.pdf", "page": 1},
                chunk_id="consent.pdf_p1_c0"
            ),
            Chunk(
                content="Medical records must be retained for seven years.",
                metadata={"filename": "records.pdf", "page": 1},
                chunk_id="records.pdf_p1_c0"
            ),
        ]
    
    @pytest.mark.slow
    def test_add_and_search(self, embedding_model, sample_chunks):
        """Test adding chunks and searching."""
        store = FAISSVectorStore(embedding_model=embedding_model)
        store.add_chunks(sample_chunks)
        
        assert len(store.chunks) == 3
        
        # Search for privacy-related content
        results = store.search("What are the privacy regulations?", top_k=2)
        
        assert len(results) > 0
        assert results[0][0].chunk_id == "privacy.pdf_p1_c0"
    
    @pytest.mark.slow
    def test_search_with_metadata_filter(self, embedding_model, sample_chunks):
        """Test search with metadata filtering."""
        store = FAISSVectorStore(embedding_model=embedding_model)
        store.add_chunks(sample_chunks)
        
        results = store.search_with_metadata_filter(
            "healthcare policy",
            top_k=5,
            metadata_filter={"filename": "consent.pdf"}
        )
        
        # Should only return chunks from consent.pdf
        for chunk, score in results:
            assert chunk.metadata["filename"] == "consent.pdf"
    
    @pytest.mark.slow
    def test_save_and_load(self, embedding_model, sample_chunks, tmp_path):
        """Test saving and loading vector store."""
        store = FAISSVectorStore(embedding_model=embedding_model)
        store.add_chunks(sample_chunks)
        
        # Save
        save_path = tmp_path / "vectorstore"
        store.save(str(save_path))
        
        # Verify files exist
        assert (save_path / "index.faiss").exists()
        assert (save_path / "chunks.pkl").exists()
        assert (save_path / "config.json").exists()
        
        # Load
        loaded_store = FAISSVectorStore.load(str(save_path), embedding_model)
        
        assert len(loaded_store.chunks) == 3
        assert loaded_store.index.ntotal == 3
    
    @pytest.mark.slow
    def test_search_score_threshold(self, embedding_model, sample_chunks):
        """Test search with score threshold."""
        store = FAISSVectorStore(embedding_model=embedding_model)
        store.add_chunks(sample_chunks)
        
        # High threshold should return fewer results
        results_high = store.search(
            "random unrelated query",
            top_k=5,
            score_threshold=0.9
        )
        
        results_low = store.search(
            "random unrelated query",
            top_k=5,
            score_threshold=0.1
        )
        
        assert len(results_high) <= len(results_low)
