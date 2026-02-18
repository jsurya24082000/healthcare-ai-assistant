"""FAISS vector store for document embeddings and retrieval."""

import os
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import faiss
from tqdm import tqdm

from src.ingestion.text_chunker import Chunk


class EmbeddingModel:
    """Wrapper for embedding models."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", use_openai: bool = False):
        """
        Initialize embedding model.
        
        Args:
            model_name: Name of the sentence-transformers model or OpenAI model.
            use_openai: Use OpenAI embeddings instead of sentence-transformers.
        """
        self.model_name = model_name
        self.use_openai = use_openai
        self._model = None
        self._openai_client = None
        
    def _load_model(self):
        """Lazy load the embedding model."""
        if self.use_openai:
            from openai import OpenAI
            self._openai_client = OpenAI()
        else:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed.
            
        Returns:
            Numpy array of embeddings.
        """
        if self._model is None and self._openai_client is None:
            self._load_model()
        
        if self.use_openai:
            return self._embed_openai(texts)
        else:
            return self._embed_local(texts)
    
    def _embed_local(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using local sentence-transformers model."""
        embeddings = self._model.encode(
            texts,
            show_progress_bar=len(texts) > 10,
            convert_to_numpy=True
        )
        return embeddings.astype(np.float32)
    
    def _embed_openai(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        """Generate embeddings using OpenAI API."""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self._openai_client.embeddings.create(
                model=self.model_name,
                input=batch
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
        
        return np.array(all_embeddings, dtype=np.float32)
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        if self.use_openai:
            # OpenAI text-embedding-3-small dimension
            return 1536
        else:
            if self._model is None:
                self._load_model()
            return self._model.get_sentence_embedding_dimension()


class FAISSVectorStore:
    """FAISS-based vector store for semantic search."""
    
    def __init__(
        self,
        embedding_model: Optional[EmbeddingModel] = None,
        index_type: str = "flat",
        dimension: Optional[int] = None
    ):
        """
        Initialize FAISS vector store.
        
        Args:
            embedding_model: Model for generating embeddings.
            index_type: Type of FAISS index ('flat', 'ivf', 'hnsw').
            dimension: Embedding dimension (auto-detected if not provided).
        """
        self.embedding_model = embedding_model or EmbeddingModel()
        self.index_type = index_type
        self.dimension = dimension or self.embedding_model.dimension
        
        self.index: Optional[faiss.Index] = None
        self.chunks: List[Chunk] = []
        self.chunk_id_to_idx: Dict[str, int] = {}
        
        self._initialize_index()
    
    def _initialize_index(self):
        """Initialize the FAISS index."""
        if self.index_type == "flat":
            # Exact search - best for small datasets
            self.index = faiss.IndexFlatIP(self.dimension)
        elif self.index_type == "ivf":
            # Approximate search - better for large datasets
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
        elif self.index_type == "hnsw":
            # HNSW - good balance of speed and accuracy
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
    
    def add_chunks(self, chunks: List[Chunk], batch_size: int = 100):
        """
        Add chunks to the vector store.
        
        Args:
            chunks: List of Chunk objects to add.
            batch_size: Batch size for embedding generation.
        """
        if not chunks:
            return
        
        print(f"Adding {len(chunks)} chunks to vector store...")
        
        # Generate embeddings in batches
        all_embeddings = []
        
        for i in tqdm(range(0, len(chunks), batch_size), desc="Generating embeddings"):
            batch = chunks[i:i + batch_size]
            texts = [chunk.content for chunk in batch]
            embeddings = self.embedding_model.embed(texts)
            all_embeddings.append(embeddings)
        
        embeddings_array = np.vstack(all_embeddings)
        
        # Normalize for cosine similarity (using inner product)
        faiss.normalize_L2(embeddings_array)
        
        # Train index if needed (for IVF)
        if self.index_type == "ivf" and not self.index.is_trained:
            print("Training IVF index...")
            self.index.train(embeddings_array)
        
        # Add to index
        start_idx = len(self.chunks)
        self.index.add(embeddings_array)
        
        # Store chunks and mapping
        for i, chunk in enumerate(chunks):
            idx = start_idx + i
            self.chunks.append(chunk)
            self.chunk_id_to_idx[chunk.chunk_id] = idx
        
        print(f"Vector store now contains {len(self.chunks)} chunks")
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: Optional[float] = None
    ) -> List[Tuple[Chunk, float]]:
        """
        Search for similar chunks.
        
        Args:
            query: Query text.
            top_k: Number of results to return.
            score_threshold: Minimum similarity score (0-1).
            
        Returns:
            List of (Chunk, score) tuples.
        """
        if not self.chunks:
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_model.embed([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, min(top_k, len(self.chunks)))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # FAISS returns -1 for missing results
                continue
            
            if score_threshold is not None and score < score_threshold:
                continue
            
            results.append((self.chunks[idx], float(score)))
        
        return results
    
    def search_with_metadata_filter(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Chunk, float]]:
        """
        Search with metadata filtering.
        
        Args:
            query: Query text.
            top_k: Number of results to return.
            metadata_filter: Dict of metadata key-value pairs to filter by.
            
        Returns:
            List of (Chunk, score) tuples matching the filter.
        """
        # Get more results than needed to account for filtering
        results = self.search(query, top_k=top_k * 3)
        
        if metadata_filter is None:
            return results[:top_k]
        
        filtered = []
        for chunk, score in results:
            match = all(
                chunk.metadata.get(key) == value
                for key, value in metadata_filter.items()
            )
            if match:
                filtered.append((chunk, score))
            
            if len(filtered) >= top_k:
                break
        
        return filtered
    
    def save(self, directory: str):
        """
        Save vector store to disk.
        
        Args:
            directory: Directory to save to.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(directory / "index.faiss"))
        
        # Save chunks and metadata
        with open(directory / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)
        
        with open(directory / "chunk_mapping.json", "w") as f:
            json.dump(self.chunk_id_to_idx, f)
        
        # Save config
        config = {
            "index_type": self.index_type,
            "dimension": self.dimension,
            "num_chunks": len(self.chunks)
        }
        with open(directory / "config.json", "w") as f:
            json.dump(config, f)
        
        print(f"Vector store saved to {directory}")
    
    @classmethod
    def load(
        cls,
        directory: str,
        embedding_model: Optional[EmbeddingModel] = None
    ) -> "FAISSVectorStore":
        """
        Load vector store from disk.
        
        Args:
            directory: Directory to load from.
            embedding_model: Embedding model to use for queries.
            
        Returns:
            Loaded FAISSVectorStore instance.
        """
        directory = Path(directory)
        
        # Load config
        with open(directory / "config.json", "r") as f:
            config = json.load(f)
        
        # Create instance
        store = cls(
            embedding_model=embedding_model,
            index_type=config["index_type"],
            dimension=config["dimension"]
        )
        
        # Load FAISS index
        store.index = faiss.read_index(str(directory / "index.faiss"))
        
        # Load chunks
        with open(directory / "chunks.pkl", "rb") as f:
            store.chunks = pickle.load(f)
        
        with open(directory / "chunk_mapping.json", "r") as f:
            store.chunk_id_to_idx = json.load(f)
        
        print(f"Loaded vector store with {len(store.chunks)} chunks")
        return store
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        return {
            "num_chunks": len(self.chunks),
            "index_type": self.index_type,
            "dimension": self.dimension,
            "index_trained": getattr(self.index, "is_trained", True),
            "total_vectors": self.index.ntotal if self.index else 0
        }
