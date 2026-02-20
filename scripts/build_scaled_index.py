"""
Build Scaled FAISS Index from Synthetic Corpus.

Reports:
- Index build time
- Index size (MB/GB)
- p50/p95 retrieval latency
- Memory usage
"""

import os
import sys
import json
import time
import psutil
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent.parent))


def get_memory_usage_mb() -> float:
    """Get current memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def build_index(
    corpus_dir: str = "data/synthetic_corpus",
    output_dir: str = "data/scaled_vectorstore",
    batch_size: int = 100
) -> Dict[str, Any]:
    """Build FAISS index from synthetic corpus."""
    
    from sentence_transformers import SentenceTransformer
    import faiss
    
    corpus_path = Path(corpus_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load chunks
    print("📂 Loading corpus...")
    with open(corpus_path / "chunks.json", "r") as f:
        chunks = json.load(f)
    
    print(f"   Loaded {len(chunks)} chunks")
    
    # Initialize embedding model
    print("\n🧠 Loading embedding model...")
    memory_before_model = get_memory_usage_mb()
    model_start = time.time()
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    model_load_time = time.time() - model_start
    memory_after_model = get_memory_usage_mb()
    
    print(f"   Model loaded in {model_load_time:.2f}s")
    print(f"   Memory: {memory_after_model:.1f}MB (+{memory_after_model - memory_before_model:.1f}MB)")
    
    # Generate embeddings in batches
    print(f"\n🔢 Generating embeddings (batch_size={batch_size})...")
    embedding_start = time.time()
    
    all_embeddings = []
    texts = [c["content"] for c in chunks]
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        embeddings = model.encode(batch, show_progress_bar=False)
        all_embeddings.append(embeddings)
        
        if (i + batch_size) % 1000 == 0 or i + batch_size >= len(texts):
            print(f"   Processed {min(i + batch_size, len(texts))}/{len(texts)} chunks")
    
    embeddings_array = np.vstack(all_embeddings).astype(np.float32)
    embedding_time = time.time() - embedding_start
    memory_after_embeddings = get_memory_usage_mb()
    
    print(f"   Embeddings generated in {embedding_time:.2f}s")
    print(f"   Shape: {embeddings_array.shape}")
    print(f"   Memory: {memory_after_embeddings:.1f}MB")
    
    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings_array)
    
    # Build FAISS index
    print("\n📊 Building FAISS index...")
    index_start = time.time()
    
    dimension = embeddings_array.shape[1]
    
    # Use IVF index for large scale (faster search)
    if len(chunks) > 10000:
        nlist = min(int(np.sqrt(len(chunks))), 1000)  # Number of clusters
        quantizer = faiss.IndexFlatIP(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
        
        print(f"   Training IVF index with {nlist} clusters...")
        index.train(embeddings_array)
        index.nprobe = 10  # Search 10 clusters
    else:
        index = faiss.IndexFlatIP(dimension)
    
    index.add(embeddings_array)
    index_build_time = time.time() - index_start
    
    print(f"   Index built in {index_build_time:.2f}s")
    print(f"   Index contains {index.ntotal} vectors")
    
    # Save index
    print("\n💾 Saving index...")
    faiss.write_index(index, str(output_path / "index.faiss"))
    
    # Save chunk mapping
    chunk_mapping = {c["chunk_id"]: i for i, c in enumerate(chunks)}
    with open(output_path / "chunk_mapping.json", "w") as f:
        json.dump(chunk_mapping, f)
    
    # Save chunks with metadata
    with open(output_path / "chunks.json", "w") as f:
        json.dump(chunks, f)
    
    # Calculate index size
    index_size_bytes = os.path.getsize(output_path / "index.faiss")
    index_size_mb = index_size_bytes / 1024 / 1024
    
    # Save config
    config = {
        "index_type": "IVF" if len(chunks) > 10000 else "Flat",
        "dimension": dimension,
        "num_chunks": len(chunks),
        "nlist": nlist if len(chunks) > 10000 else None,
        "nprobe": 10 if len(chunks) > 10000 else None
    }
    with open(output_path / "config.json", "w") as f:
        json.dump(config, f)
    
    # Benchmark retrieval latency
    print("\n⏱️ Benchmarking retrieval latency...")
    latencies = benchmark_retrieval(model, index, chunks, num_queries=100)
    
    # Final statistics
    stats = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "corpus": {
            "num_chunks": len(chunks),
            "source_dir": str(corpus_path)
        },
        "model": {
            "name": "all-MiniLM-L6-v2",
            "dimension": dimension,
            "load_time_seconds": round(model_load_time, 2)
        },
        "indexing": {
            "embedding_time_seconds": round(embedding_time, 2),
            "index_build_time_seconds": round(index_build_time, 2),
            "total_time_seconds": round(model_load_time + embedding_time + index_build_time, 2),
            "throughput_chunks_per_second": round(len(chunks) / (embedding_time + index_build_time), 1)
        },
        "index": {
            "type": config["index_type"],
            "size_mb": round(index_size_mb, 2),
            "size_gb": round(index_size_mb / 1024, 3),
            "vectors": index.ntotal
        },
        "memory": {
            "model_mb": round(memory_after_model - memory_before_model, 1),
            "embeddings_mb": round(memory_after_embeddings - memory_after_model, 1),
            "total_mb": round(get_memory_usage_mb(), 1)
        },
        "retrieval_latency": latencies,
        "output_dir": str(output_path)
    }
    
    # Save stats
    with open(output_path / "index_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    print("\n" + "=" * 60)
    print("📈 INDEX BUILD COMPLETE")
    print("=" * 60)
    print(f"Chunks indexed: {stats['corpus']['num_chunks']:,}")
    print(f"Index size: {stats['index']['size_mb']:.2f} MB")
    print(f"Build time: {stats['indexing']['total_time_seconds']:.1f}s")
    print(f"Throughput: {stats['indexing']['throughput_chunks_per_second']:.1f} chunks/s")
    print(f"\nRetrieval Latency:")
    print(f"  p50: {stats['retrieval_latency']['p50_ms']:.2f}ms")
    print(f"  p95: {stats['retrieval_latency']['p95_ms']:.2f}ms")
    print(f"  p99: {stats['retrieval_latency']['p99_ms']:.2f}ms")
    print(f"\nMemory Usage: {stats['memory']['total_mb']:.1f} MB")
    print("=" * 60)
    
    return stats


def benchmark_retrieval(model, index, chunks, num_queries: int = 100) -> Dict[str, float]:
    """Benchmark retrieval latency."""
    import random
    
    # Generate random queries from chunk content
    sample_chunks = random.sample(chunks, min(num_queries, len(chunks)))
    queries = [c["content"][:100] for c in sample_chunks]
    
    latencies = []
    
    for query in queries:
        start = time.time()
        
        # Encode query
        query_embedding = model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = index.search(query_embedding, k=10)
        
        latency_ms = (time.time() - start) * 1000
        latencies.append(latency_ms)
    
    latencies.sort()
    
    return {
        "num_queries": len(latencies),
        "mean_ms": round(sum(latencies) / len(latencies), 2),
        "min_ms": round(min(latencies), 2),
        "max_ms": round(max(latencies), 2),
        "p50_ms": round(latencies[int(len(latencies) * 0.50)], 2),
        "p95_ms": round(latencies[int(len(latencies) * 0.95)], 2),
        "p99_ms": round(latencies[int(len(latencies) * 0.99)], 2)
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build scaled FAISS index")
    parser.add_argument("--corpus", type=str, default="data/synthetic_corpus", help="Corpus directory")
    parser.add_argument("--output", type=str, default="data/scaled_vectorstore", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=100, help="Embedding batch size")
    
    args = parser.parse_args()
    
    stats = build_index(
        corpus_dir=args.corpus,
        output_dir=args.output,
        batch_size=args.batch_size
    )
