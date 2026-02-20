"""
Hybrid Search: BM25 + Embeddings with Reranking.

Combines:
- BM25 for exact term matching (catches "HIPAA", "PHI")
- Embeddings for semantic matching
- Cross-encoder reranker for final ranking

This typically improves top-1 relevance significantly.
"""

import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass
from rank_bm25 import BM25Okapi
import re


@dataclass
class SearchResult:
    """Single search result with scores."""
    chunk_id: str
    content: str
    metadata: Dict[str, Any]
    bm25_score: float
    embedding_score: float
    combined_score: float
    rerank_score: Optional[float] = None


class HybridSearcher:
    """
    Hybrid search combining BM25 and embedding-based retrieval.
    
    Features:
    - BM25 for keyword/exact matching
    - FAISS for semantic similarity
    - Reciprocal Rank Fusion (RRF) for combining scores
    - Optional cross-encoder reranking
    """
    
    def __init__(
        self,
        index_dir: str = "data/scaled_vectorstore",
        use_reranker: bool = True,
        bm25_weight: float = 0.3,
        embedding_weight: float = 0.7
    ):
        self.index_dir = Path(index_dir)
        self.use_reranker = use_reranker
        self.bm25_weight = bm25_weight
        self.embedding_weight = embedding_weight
        
        self.chunks = None
        self.chunk_id_list = None
        self.bm25 = None
        self.faiss_index = None
        self.embedding_model = None
        self.reranker = None
        
    def load(self):
        """Load all components."""
        import faiss
        from sentence_transformers import SentenceTransformer
        
        print("Loading hybrid search components...")
        
        # Load chunks
        with open(self.index_dir / "chunks.json", "r") as f:
            self.chunks = json.load(f)
        
        self.chunk_id_list = [c["chunk_id"] for c in self.chunks]
        self.chunk_content = {c["chunk_id"]: c["content"] for c in self.chunks}
        
        print(f"  Loaded {len(self.chunks)} chunks")
        
        # Build BM25 index
        print("  Building BM25 index...")
        tokenized_corpus = [self._tokenize(c["content"]) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        # Load FAISS index
        print("  Loading FAISS index...")
        self.faiss_index = faiss.read_index(str(self.index_dir / "index.faiss"))
        
        # Load embedding model
        print("  Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load reranker if enabled
        if self.use_reranker:
            print("  Loading cross-encoder reranker...")
            try:
                from sentence_transformers import CrossEncoder
                self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            except Exception as e:
                print(f"  Reranker not available: {e}")
                self.use_reranker = False
        
        print("✅ Hybrid search ready")
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25."""
        # Simple tokenization: lowercase, split on non-alphanumeric
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def search_bm25(self, query: str, top_k: int = 50) -> List[Tuple[int, float]]:
        """Search using BM25."""
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        return [(int(idx), float(scores[idx])) for idx in top_indices]
    
    def search_embedding(self, query: str, top_k: int = 50) -> List[Tuple[int, float]]:
        """Search using embeddings."""
        import faiss
        
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.faiss_index.search(query_embedding, top_k)
        
        return [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0]) if idx >= 0]
    
    def reciprocal_rank_fusion(
        self,
        bm25_results: List[Tuple[int, float]],
        embedding_results: List[Tuple[int, float]],
        k: int = 60
    ) -> List[Tuple[int, float]]:
        """
        Combine rankings using Reciprocal Rank Fusion (RRF).
        
        RRF score = sum(1 / (k + rank_i)) for each ranking
        """
        scores = {}
        
        # BM25 contribution
        for rank, (idx, _) in enumerate(bm25_results):
            if idx not in scores:
                scores[idx] = 0
            scores[idx] += self.bm25_weight * (1 / (k + rank + 1))
        
        # Embedding contribution
        for rank, (idx, _) in enumerate(embedding_results):
            if idx not in scores:
                scores[idx] = 0
            scores[idx] += self.embedding_weight * (1 / (k + rank + 1))
        
        # Sort by combined score
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_results
    
    def rerank(
        self,
        query: str,
        candidates: List[Tuple[int, float]],
        top_k: int = 10
    ) -> List[Tuple[int, float]]:
        """Rerank candidates using cross-encoder."""
        if not self.reranker or not candidates:
            return candidates[:top_k]
        
        # Prepare pairs for reranking
        pairs = [
            (query, self.chunks[idx]["content"][:512])
            for idx, _ in candidates[:min(50, len(candidates))]
        ]
        
        # Get reranker scores
        rerank_scores = self.reranker.predict(pairs)
        
        # Combine with original indices
        reranked = [
            (candidates[i][0], float(score))
            for i, score in enumerate(rerank_scores)
        ]
        
        # Sort by rerank score
        reranked.sort(key=lambda x: x[1], reverse=True)
        
        return reranked[:top_k]
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        use_reranker: Optional[bool] = None
    ) -> List[SearchResult]:
        """
        Perform hybrid search.
        
        Args:
            query: Search query
            top_k: Number of results to return
            use_reranker: Override default reranker setting
            
        Returns:
            List of SearchResult objects
        """
        # Get results from both methods
        bm25_results = self.search_bm25(query, top_k=50)
        embedding_results = self.search_embedding(query, top_k=50)
        
        # Combine using RRF
        combined = self.reciprocal_rank_fusion(bm25_results, embedding_results)
        
        # Optional reranking
        should_rerank = use_reranker if use_reranker is not None else self.use_reranker
        if should_rerank and self.reranker:
            final_results = self.rerank(query, combined, top_k)
        else:
            final_results = combined[:top_k]
        
        # Build result objects
        bm25_scores = dict(bm25_results)
        embedding_scores = dict(embedding_results)
        
        results = []
        for idx, combined_score in final_results:
            chunk = self.chunks[idx]
            results.append(SearchResult(
                chunk_id=chunk["chunk_id"],
                content=chunk["content"],
                metadata={
                    "doc_id": chunk.get("doc_id"),
                    "topic": chunk.get("topic"),
                    "department": chunk.get("department")
                },
                bm25_score=bm25_scores.get(idx, 0.0),
                embedding_score=embedding_scores.get(idx, 0.0),
                combined_score=combined_score,
                rerank_score=combined_score if should_rerank else None
            ))
        
        return results
    
    def search_simple(self, query: str, top_k: int = 10) -> List[str]:
        """Simple search returning just chunk IDs."""
        results = self.search(query, top_k)
        return [r.chunk_id for r in results]


def benchmark_hybrid_vs_embedding(
    index_dir: str = "data/scaled_vectorstore",
    labeled_queries_file: str = "evaluation/labeled_queries.json"
) -> Dict[str, Any]:
    """Compare hybrid search vs embedding-only search."""
    
    # Load labeled queries
    with open(labeled_queries_file, "r") as f:
        data = json.load(f)
    
    queries = [q for q in data["queries"] if q["relevant_chunks"]][:50]
    
    # Initialize searcher
    searcher = HybridSearcher(index_dir, use_reranker=True)
    searcher.load()
    
    # Evaluate embedding-only
    print("\nEvaluating embedding-only search...")
    embedding_recalls = []
    for q in queries:
        results = searcher.search_embedding(q["query"], top_k=10)
        retrieved = [searcher.chunk_id_list[idx] for idx, _ in results]
        relevant = set(q["relevant_chunks"])
        recall = len(set(retrieved) & relevant) / len(relevant) if relevant else 0
        embedding_recalls.append(recall)
    
    # Evaluate hybrid
    print("Evaluating hybrid search...")
    hybrid_recalls = []
    for q in queries:
        results = searcher.search(q["query"], top_k=10, use_reranker=False)
        retrieved = [r.chunk_id for r in results]
        relevant = set(q["relevant_chunks"])
        recall = len(set(retrieved) & relevant) / len(relevant) if relevant else 0
        hybrid_recalls.append(recall)
    
    # Evaluate hybrid + reranker
    print("Evaluating hybrid + reranker...")
    reranked_recalls = []
    for q in queries:
        results = searcher.search(q["query"], top_k=10, use_reranker=True)
        retrieved = [r.chunk_id for r in results]
        relevant = set(q["relevant_chunks"])
        recall = len(set(retrieved) & relevant) / len(relevant) if relevant else 0
        reranked_recalls.append(recall)
    
    comparison = {
        "num_queries": len(queries),
        "embedding_only": {
            "recall@10": round(sum(embedding_recalls) / len(embedding_recalls), 4)
        },
        "hybrid_bm25_embedding": {
            "recall@10": round(sum(hybrid_recalls) / len(hybrid_recalls), 4)
        },
        "hybrid_with_reranker": {
            "recall@10": round(sum(reranked_recalls) / len(reranked_recalls), 4)
        }
    }
    
    print("\n" + "=" * 60)
    print("📊 HYBRID SEARCH COMPARISON")
    print("=" * 60)
    print(f"Queries: {comparison['num_queries']}")
    print(f"\nRecall@10:")
    print(f"  Embedding only:      {comparison['embedding_only']['recall@10']:.4f}")
    print(f"  Hybrid (BM25+Emb):   {comparison['hybrid_bm25_embedding']['recall@10']:.4f}")
    print(f"  Hybrid + Reranker:   {comparison['hybrid_with_reranker']['recall@10']:.4f}")
    print("=" * 60)
    
    return comparison


if __name__ == "__main__":
    benchmark_hybrid_vs_embedding()
