"""
Information Retrieval Metrics for RAG Evaluation.

Computes standard IR metrics:
- Recall@k (k=1,3,5,10)
- Precision@k
- MRR (Mean Reciprocal Rank)
- nDCG@k (Normalized Discounted Cumulative Gain)
"""

import json
import math
import numpy as np
from typing import List, Dict, Any, Set
from pathlib import Path
from dataclasses import dataclass


@dataclass
class RetrievalResult:
    """Single retrieval result."""
    chunk_id: str
    score: float
    rank: int


@dataclass
class QueryEvaluation:
    """Evaluation results for a single query."""
    query_id: str
    query: str
    retrieved: List[str]
    relevant: Set[str]
    relevance_scores: Dict[str, int]
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    recall_at_10: float
    precision_at_1: float
    precision_at_3: float
    precision_at_5: float
    precision_at_10: float
    mrr: float
    ndcg_at_10: float


def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """Calculate Recall@k."""
    if not relevant:
        return 0.0
    retrieved_k = set(retrieved[:k])
    return len(retrieved_k & relevant) / len(relevant)


def precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """Calculate Precision@k."""
    if k == 0:
        return 0.0
    retrieved_k = set(retrieved[:k])
    return len(retrieved_k & relevant) / k


def mean_reciprocal_rank(retrieved: List[str], relevant: Set[str]) -> float:
    """Calculate Mean Reciprocal Rank (MRR)."""
    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant:
            return 1.0 / (i + 1)
    return 0.0


def dcg_at_k(retrieved: List[str], relevance_scores: Dict[str, int], k: int) -> float:
    """Calculate Discounted Cumulative Gain at k."""
    dcg = 0.0
    for i, doc_id in enumerate(retrieved[:k]):
        rel = relevance_scores.get(doc_id, 0)
        dcg += (2 ** rel - 1) / math.log2(i + 2)  # i+2 because log2(1) = 0
    return dcg


def ndcg_at_k(retrieved: List[str], relevance_scores: Dict[str, int], k: int) -> float:
    """Calculate Normalized Discounted Cumulative Gain at k."""
    dcg = dcg_at_k(retrieved, relevance_scores, k)
    
    # Ideal DCG: sort by relevance scores
    ideal_order = sorted(relevance_scores.keys(), key=lambda x: relevance_scores[x], reverse=True)
    idcg = dcg_at_k(ideal_order, relevance_scores, k)
    
    if idcg == 0:
        return 0.0
    return dcg / idcg


def evaluate_query(
    query_id: str,
    query: str,
    retrieved: List[str],
    relevant: List[str],
    relevance_scores: List[int]
) -> QueryEvaluation:
    """Evaluate a single query."""
    relevant_set = set(relevant)
    rel_scores_dict = dict(zip(relevant, relevance_scores))
    
    return QueryEvaluation(
        query_id=query_id,
        query=query,
        retrieved=retrieved,
        relevant=relevant_set,
        relevance_scores=rel_scores_dict,
        recall_at_1=recall_at_k(retrieved, relevant_set, 1),
        recall_at_3=recall_at_k(retrieved, relevant_set, 3),
        recall_at_5=recall_at_k(retrieved, relevant_set, 5),
        recall_at_10=recall_at_k(retrieved, relevant_set, 10),
        precision_at_1=precision_at_k(retrieved, relevant_set, 1),
        precision_at_3=precision_at_k(retrieved, relevant_set, 3),
        precision_at_5=precision_at_k(retrieved, relevant_set, 5),
        precision_at_10=precision_at_k(retrieved, relevant_set, 10),
        mrr=mean_reciprocal_rank(retrieved, relevant_set),
        ndcg_at_10=ndcg_at_k(retrieved, rel_scores_dict, 10)
    )


def aggregate_metrics(evaluations: List[QueryEvaluation]) -> Dict[str, float]:
    """Aggregate metrics across all queries."""
    n = len(evaluations)
    if n == 0:
        return {}
    
    return {
        "num_queries": n,
        "recall@1": round(sum(e.recall_at_1 for e in evaluations) / n, 4),
        "recall@3": round(sum(e.recall_at_3 for e in evaluations) / n, 4),
        "recall@5": round(sum(e.recall_at_5 for e in evaluations) / n, 4),
        "recall@10": round(sum(e.recall_at_10 for e in evaluations) / n, 4),
        "precision@1": round(sum(e.precision_at_1 for e in evaluations) / n, 4),
        "precision@3": round(sum(e.precision_at_3 for e in evaluations) / n, 4),
        "precision@5": round(sum(e.precision_at_5 for e in evaluations) / n, 4),
        "precision@10": round(sum(e.precision_at_10 for e in evaluations) / n, 4),
        "mrr": round(sum(e.mrr for e in evaluations) / n, 4),
        "ndcg@10": round(sum(e.ndcg_at_10 for e in evaluations) / n, 4)
    }


class RetrievalEvaluator:
    """Evaluator for retrieval systems using labeled queries."""
    
    def __init__(self, labeled_queries_file: str):
        """Load labeled queries."""
        with open(labeled_queries_file, "r") as f:
            data = json.load(f)
        
        self.queries = data["queries"]
        self.metadata = data["metadata"]
        print(f"Loaded {len(self.queries)} labeled queries")
    
    def evaluate(self, retrieval_fn, top_k: int = 10) -> Dict[str, Any]:
        """
        Evaluate retrieval function against labeled queries.
        
        Args:
            retrieval_fn: Function that takes query string and returns list of chunk_ids
            top_k: Number of results to retrieve
            
        Returns:
            Evaluation results with per-query and aggregate metrics
        """
        evaluations = []
        
        for q in self.queries:
            # Skip queries without relevance judgments
            if not q["relevant_chunks"]:
                continue
            
            # Get retrieved results
            retrieved = retrieval_fn(q["query"], top_k)
            
            # Evaluate
            eval_result = evaluate_query(
                query_id=q["query_id"],
                query=q["query"],
                retrieved=retrieved,
                relevant=q["relevant_chunks"],
                relevance_scores=q["relevance_scores"]
            )
            evaluations.append(eval_result)
        
        # Aggregate
        aggregate = aggregate_metrics(evaluations)
        
        # Per-query results
        per_query = [
            {
                "query_id": e.query_id,
                "query": e.query[:50] + "...",
                "recall@1": e.recall_at_1,
                "recall@5": e.recall_at_5,
                "mrr": e.mrr,
                "ndcg@10": e.ndcg_at_10
            }
            for e in evaluations
        ]
        
        return {
            "aggregate": aggregate,
            "per_query": per_query,
            "num_evaluated": len(evaluations)
        }


def run_evaluation(
    index_dir: str = "data/scaled_vectorstore",
    labeled_queries_file: str = "evaluation/labeled_queries.json",
    output_file: str = "evaluation/ir_metrics_results.json"
) -> Dict[str, Any]:
    """Run full IR evaluation."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from sentence_transformers import SentenceTransformer
    import faiss
    
    # Load index
    print("Loading index...")
    index = faiss.read_index(str(Path(index_dir) / "index.faiss"))
    
    with open(Path(index_dir) / "chunks.json", "r") as f:
        chunks = json.load(f)
    
    chunk_id_list = [c["chunk_id"] for c in chunks]
    
    # Load model
    print("Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Create retrieval function
    def retrieve(query: str, top_k: int = 10) -> List[str]:
        query_embedding = model.encode([query])
        faiss.normalize_L2(query_embedding)
        scores, indices = index.search(query_embedding, top_k)
        return [chunk_id_list[i] for i in indices[0] if i >= 0]
    
    # Run evaluation
    print("Running evaluation...")
    evaluator = RetrievalEvaluator(labeled_queries_file)
    results = evaluator.evaluate(retrieve, top_k=10)
    
    # Save results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 IR EVALUATION RESULTS")
    print("=" * 60)
    agg = results["aggregate"]
    print(f"Queries evaluated: {agg['num_queries']}")
    print(f"\nRecall:")
    print(f"  Recall@1:  {agg['recall@1']:.4f}")
    print(f"  Recall@3:  {agg['recall@3']:.4f}")
    print(f"  Recall@5:  {agg['recall@5']:.4f}")
    print(f"  Recall@10: {agg['recall@10']:.4f}")
    print(f"\nPrecision:")
    print(f"  Precision@1:  {agg['precision@1']:.4f}")
    print(f"  Precision@3:  {agg['precision@3']:.4f}")
    print(f"  Precision@5:  {agg['precision@5']:.4f}")
    print(f"  Precision@10: {agg['precision@10']:.4f}")
    print(f"\nRanking:")
    print(f"  MRR:      {agg['mrr']:.4f}")
    print(f"  nDCG@10:  {agg['ndcg@10']:.4f}")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    run_evaluation()
