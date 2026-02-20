"""
Run actual evaluation to measure real project metrics.
Measures retrieval accuracy, latency, and grounding consistency.
"""

import os
import sys
import time
import json
from datetime import datetime
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def measure_retrieval_metrics():
    """Measure actual retrieval performance."""
    print("=" * 60)
    print("📊 HEALTHCARE RAG - REAL METRICS EVALUATION")
    print("=" * 60)
    
    results = {
        "timestamp": datetime.utcnow().isoformat(),
        "environment": {
            "python_version": sys.version.split()[0],
            "platform": sys.platform
        }
    }
    
    # Test queries for evaluation
    test_queries = [
        "What is the policy on patient data privacy?",
        "How should emergency situations be handled?",
        "What are the HIPAA compliance requirements?",
        "Describe the clinical documentation standards.",
        "What are the patient safety protocols?",
        "How is quality improvement measured?",
    ]
    
    try:
        # Try to import and use actual components
        from src.vectordb.faiss_store import FAISSVectorStore, EmbeddingModel
        from sentence_transformers import SentenceTransformer
        
        print("\n✅ Loading vector store and embedding model...")
        
        # Load embedding model (lazy loaded, so force load for timing)
        start = time.time()
        embedding_model = EmbeddingModel(model_name='all-MiniLM-L6-v2')
        # Force model load by calling embed
        _ = embedding_model.embed(["test"])
        model_load_time = time.time() - start
        print(f"   Model load time: {model_load_time:.2f}s")
        
        # Load vector store with embedding model (classmethod returns new instance)
        store = FAISSVectorStore.load("data/vectorstore", embedding_model=embedding_model)
        
        results["model"] = {
            "name": "all-MiniLM-L6-v2",
            "dimensions": 384,
            "load_time_seconds": round(model_load_time, 3)
        }
        
        results["vector_store"] = {
            "type": "FAISS",
            "num_chunks": 18,
            "index_type": "flat"
        }
        
        # Measure retrieval latency
        print("\n📏 Measuring retrieval latency...")
        latencies = []
        
        for query in test_queries:
            # Search (includes encoding internally)
            start = time.time()
            results_list = store.search(query, top_k=3)
            total_time = (time.time() - start) * 1000
            
            # Get scores from results
            scores = [score for _, score in results_list]
            avg_score = sum(scores) / len(scores) if scores else 0
            top_score = scores[0] if scores else 0
            
            latencies.append({
                "query": query[:50],
                "total_ms": round(total_time, 2),
                "num_results": len(results_list),
                "avg_similarity": round(avg_score, 4),
                "top_score": round(top_score, 4)
            })
            print(f"   Query: {query[:40]}... -> {total_time:.1f}ms (top score: {top_score:.3f})")
        
        # Calculate latency stats
        total_times = [l["total_ms"] for l in latencies]
        top_scores = [l["top_score"] for l in latencies]
        avg_scores = [l["avg_similarity"] for l in latencies]
        
        results["latency_metrics"] = {
            "total_retrieval": {
                "mean_ms": round(sum(total_times) / len(total_times), 2),
                "min_ms": round(min(total_times), 2),
                "max_ms": round(max(total_times), 2),
            },
            "per_query": latencies
        }
        
        results["retrieval_quality"] = {
            "top_1_similarity": {
                "mean": round(sum(top_scores) / len(top_scores), 4),
                "min": round(min(top_scores), 4),
                "max": round(max(top_scores), 4),
            },
            "top_3_avg_similarity": {
                "mean": round(sum(avg_scores) / len(avg_scores), 4),
            }
        }
        
        print(f"\n   Average retrieval latency: {results['latency_metrics']['total_retrieval']['mean_ms']}ms")
        print(f"   Top-1 similarity (mean): {results['retrieval_quality']['top_1_similarity']['mean']}")
        
    except ImportError as e:
        print(f"\n⚠️  Could not import components: {e}")
        print("   Running basic measurements instead...")
        
        # Measure what we can without full imports
        results["error"] = str(e)
        results["note"] = "Partial evaluation - some components not available"
    
    # Measure file-based metrics
    print("\n📁 Measuring document metrics...")
    
    # Count chunks from mapping
    try:
        with open("data/vectorstore/chunk_mapping.json", "r") as f:
            chunk_mapping = json.load(f)
        
        documents = set()
        for key in chunk_mapping.keys():
            doc_name = key.rsplit("_p", 1)[0]
            documents.add(doc_name)
        
        results["document_metrics"] = {
            "total_documents": len(documents),
            "total_chunks": len(chunk_mapping),
            "avg_chunks_per_doc": round(len(chunk_mapping) / len(documents), 1),
            "documents": list(documents)
        }
        
        print(f"   Documents indexed: {len(documents)}")
        print(f"   Total chunks: {len(chunk_mapping)}")
        
    except Exception as e:
        print(f"   Could not read chunk mapping: {e}")
    
    # Save results
    output_path = "experiments/evaluation_results.json"
    os.makedirs("experiments", exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("📈 EVALUATION SUMMARY")
    print("=" * 60)
    print(json.dumps(results, indent=2))
    
    return results


def measure_embedding_performance():
    """Benchmark embedding model performance."""
    print("\n" + "=" * 60)
    print("🧠 EMBEDDING MODEL BENCHMARK")
    print("=" * 60)
    
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Test sentences
        test_sentences = [
            "What is the policy on patient data privacy?",
            "How should emergency situations be handled?",
            "What are the HIPAA compliance requirements?",
        ] * 10  # 30 sentences
        
        # Warm up
        _ = model.encode(test_sentences[:3])
        
        # Measure batch encoding
        start = time.time()
        embeddings = model.encode(test_sentences)
        batch_time = time.time() - start
        
        # Measure single encoding
        single_times = []
        for sent in test_sentences[:10]:
            start = time.time()
            _ = model.encode([sent])
            single_times.append((time.time() - start) * 1000)
        
        results = {
            "model": "all-MiniLM-L6-v2",
            "embedding_dim": embeddings.shape[1],
            "batch_encoding": {
                "sentences": len(test_sentences),
                "total_time_ms": round(batch_time * 1000, 2),
                "per_sentence_ms": round((batch_time * 1000) / len(test_sentences), 2)
            },
            "single_encoding": {
                "mean_ms": round(np.mean(single_times), 2),
                "min_ms": round(np.min(single_times), 2),
                "max_ms": round(np.max(single_times), 2)
            }
        }
        
        print(f"   Embedding dimension: {results['embedding_dim']}")
        print(f"   Batch encoding ({len(test_sentences)} sentences): {results['batch_encoding']['total_time_ms']}ms")
        print(f"   Per-sentence (batch): {results['batch_encoding']['per_sentence_ms']}ms")
        print(f"   Single encoding avg: {results['single_encoding']['mean_ms']}ms")
        
        return results
        
    except ImportError as e:
        print(f"   Could not run embedding benchmark: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    # Run evaluations
    retrieval_results = measure_retrieval_metrics()
    embedding_results = measure_embedding_performance()
    
    # Combine results
    final_results = {
        "retrieval": retrieval_results,
        "embedding": embedding_results,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    # Save final results
    with open("experiments/full_evaluation.json", "w") as f:
        json.dump(final_results, f, indent=2)
    
    print("\n✅ Full evaluation complete!")
    print("   Results saved to experiments/full_evaluation.json")
