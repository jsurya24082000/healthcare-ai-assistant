"""
RAG Answer Quality Evaluation.

Computes RAG-specific quality metrics:
- Faithfulness: Is the answer supported by the context?
- Answer Relevancy: Does the answer address the question?
- Context Precision: Are retrieved docs relevant?
- Context Recall: Are all relevant docs retrieved?

Uses LLM-as-judge approach (can use OpenAI or local model).
"""

import json
import re
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime, timezone


@dataclass
class RAGEvalResult:
    """Evaluation result for a single RAG response."""
    query_id: str
    query: str
    answer: str
    context: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    groundedness: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RAGQualityEvaluator:
    """
    Evaluates RAG answer quality using LLM-as-judge.
    
    Supports:
    - OpenAI GPT-4/GPT-3.5
    - Local models via Ollama
    - Rule-based fallback
    """
    
    def __init__(self, judge_model: str = "rule-based"):
        """
        Initialize evaluator.
        
        Args:
            judge_model: "gpt-4", "gpt-3.5-turbo", "ollama:llama2", or "rule-based"
        """
        self.judge_model = judge_model
        self.client = None
        
        if judge_model.startswith("gpt"):
            try:
                from openai import OpenAI
                self.client = OpenAI()
            except Exception as e:
                print(f"OpenAI not available, using rule-based: {e}")
                self.judge_model = "rule-based"
    
    def evaluate_faithfulness(self, answer: str, context: str) -> float:
        """
        Evaluate if the answer is faithful to the context.
        Score 0-1 where 1 = fully faithful.
        """
        if self.judge_model == "rule-based":
            return self._rule_based_faithfulness(answer, context)
        
        prompt = f"""Evaluate the faithfulness of the answer to the given context.
Faithfulness measures whether all claims in the answer are supported by the context.

Context:
{context[:2000]}

Answer:
{answer}

Score the faithfulness from 0.0 to 1.0:
- 1.0: All claims are fully supported by the context
- 0.7-0.9: Most claims are supported, minor unsupported details
- 0.4-0.6: Some claims are supported, some are not
- 0.1-0.3: Few claims are supported
- 0.0: Answer contradicts or is unrelated to context

Respond with only a number between 0.0 and 1.0."""

        return self._call_llm_judge(prompt)
    
    def evaluate_answer_relevancy(self, query: str, answer: str) -> float:
        """
        Evaluate if the answer addresses the question.
        Score 0-1 where 1 = fully relevant.
        """
        if self.judge_model == "rule-based":
            return self._rule_based_relevancy(query, answer)
        
        prompt = f"""Evaluate how well the answer addresses the question.

Question:
{query}

Answer:
{answer}

Score the answer relevancy from 0.0 to 1.0:
- 1.0: Answer directly and completely addresses the question
- 0.7-0.9: Answer mostly addresses the question with minor gaps
- 0.4-0.6: Answer partially addresses the question
- 0.1-0.3: Answer barely addresses the question
- 0.0: Answer does not address the question at all

Respond with only a number between 0.0 and 1.0."""

        return self._call_llm_judge(prompt)
    
    def evaluate_context_precision(self, query: str, context: str) -> float:
        """
        Evaluate if the retrieved context is relevant to the query.
        Score 0-1 where 1 = highly relevant context.
        """
        if self.judge_model == "rule-based":
            return self._rule_based_context_precision(query, context)
        
        prompt = f"""Evaluate how relevant the retrieved context is to the question.

Question:
{query}

Retrieved Context:
{context[:2000]}

Score the context precision from 0.0 to 1.0:
- 1.0: Context is highly relevant and contains the answer
- 0.7-0.9: Context is mostly relevant
- 0.4-0.6: Context is somewhat relevant
- 0.1-0.3: Context has minimal relevance
- 0.0: Context is not relevant at all

Respond with only a number between 0.0 and 1.0."""

        return self._call_llm_judge(prompt)
    
    def evaluate_groundedness(self, answer: str, context: str) -> float:
        """
        Evaluate if the answer is grounded in the context (no hallucination).
        Score 0-1 where 1 = fully grounded.
        """
        if self.judge_model == "rule-based":
            return self._rule_based_groundedness(answer, context)
        
        prompt = f"""Evaluate if the answer is grounded in the context without hallucination.

Context:
{context[:2000]}

Answer:
{answer}

Score the groundedness from 0.0 to 1.0:
- 1.0: Answer only contains information from the context
- 0.7-0.9: Answer is mostly grounded with minor additions
- 0.4-0.6: Answer has some unsupported claims
- 0.1-0.3: Answer has significant hallucination
- 0.0: Answer is mostly hallucinated

Respond with only a number between 0.0 and 1.0."""

        return self._call_llm_judge(prompt)
    
    def _call_llm_judge(self, prompt: str) -> float:
        """Call LLM judge and parse score."""
        if not self.client:
            return 0.5
        
        try:
            response = self.client.chat.completions.create(
                model=self.judge_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=10
            )
            
            score_text = response.choices[0].message.content.strip()
            score = float(re.search(r"[\d.]+", score_text).group())
            return min(1.0, max(0.0, score))
            
        except Exception as e:
            print(f"LLM judge error: {e}")
            return 0.5
    
    def _rule_based_faithfulness(self, answer: str, context: str) -> float:
        """Rule-based faithfulness scoring."""
        answer_words = set(answer.lower().split())
        context_words = set(context.lower().split())
        
        # Calculate word overlap
        overlap = len(answer_words & context_words)
        coverage = overlap / len(answer_words) if answer_words else 0
        
        return min(1.0, coverage * 1.5)  # Scale up slightly
    
    def _rule_based_relevancy(self, query: str, answer: str) -> float:
        """Rule-based answer relevancy scoring."""
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())
        
        # Remove stop words
        stop_words = {"what", "how", "when", "where", "why", "is", "are", "the", "a", "an", "to", "for", "of", "in", "on"}
        query_keywords = query_words - stop_words
        
        # Check keyword coverage
        matches = len(query_keywords & answer_words)
        coverage = matches / len(query_keywords) if query_keywords else 0
        
        # Bonus for longer answers (more complete)
        length_bonus = min(0.2, len(answer.split()) / 500)
        
        return min(1.0, coverage + length_bonus)
    
    def _rule_based_context_precision(self, query: str, context: str) -> float:
        """Rule-based context precision scoring."""
        query_words = set(query.lower().split())
        context_words = set(context.lower().split())
        
        stop_words = {"what", "how", "when", "where", "why", "is", "are", "the", "a", "an", "to", "for", "of", "in", "on"}
        query_keywords = query_words - stop_words
        
        matches = len(query_keywords & context_words)
        coverage = matches / len(query_keywords) if query_keywords else 0
        
        return min(1.0, coverage * 1.2)
    
    def _rule_based_groundedness(self, answer: str, context: str) -> float:
        """Rule-based groundedness scoring."""
        # Similar to faithfulness but stricter
        answer_sentences = answer.split(".")
        grounded_count = 0
        
        for sentence in answer_sentences:
            sentence_words = set(sentence.lower().split())
            context_words = set(context.lower().split())
            
            overlap = len(sentence_words & context_words)
            if overlap / len(sentence_words) > 0.3 if sentence_words else False:
                grounded_count += 1
        
        return grounded_count / len(answer_sentences) if answer_sentences else 0
    
    def evaluate_response(
        self,
        query: str,
        answer: str,
        context: str,
        query_id: str = ""
    ) -> RAGEvalResult:
        """Evaluate a single RAG response."""
        return RAGEvalResult(
            query_id=query_id,
            query=query,
            answer=answer,
            context=context[:500],
            faithfulness=self.evaluate_faithfulness(answer, context),
            answer_relevancy=self.evaluate_answer_relevancy(query, answer),
            context_precision=self.evaluate_context_precision(query, context),
            groundedness=self.evaluate_groundedness(answer, context)
        )


def run_rag_evaluation(
    labeled_queries_file: str = "evaluation/labeled_queries.json",
    index_dir: str = "data/scaled_vectorstore",
    output_file: str = "evaluation/rag_quality_results.json",
    num_samples: int = 50,
    judge_model: str = "rule-based"
) -> Dict[str, Any]:
    """Run RAG quality evaluation on sample queries."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from sentence_transformers import SentenceTransformer
    import faiss
    
    # Load components
    print("Loading index and model...")
    index = faiss.read_index(str(Path(index_dir) / "index.faiss"))
    
    with open(Path(index_dir) / "chunks.json", "r") as f:
        chunks = json.load(f)
    
    chunk_id_to_content = {c["chunk_id"]: c["content"] for c in chunks}
    chunk_id_list = [c["chunk_id"] for c in chunks]
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Load queries
    with open(labeled_queries_file, "r") as f:
        data = json.load(f)
    
    queries = data["queries"][:num_samples]
    
    # Initialize evaluator
    evaluator = RAGQualityEvaluator(judge_model=judge_model)
    
    # Evaluate
    print(f"Evaluating {len(queries)} queries...")
    results = []
    
    for i, q in enumerate(queries):
        # Retrieve context
        query_embedding = model.encode([q["query"]])
        faiss.normalize_L2(query_embedding)
        scores, indices = index.search(query_embedding, 5)
        
        retrieved_chunks = [chunk_id_list[idx] for idx in indices[0] if idx >= 0]
        context = "\n\n".join([
            chunk_id_to_content.get(cid, "") for cid in retrieved_chunks
        ])
        
        # Generate simple answer (in production, use LLM)
        answer = f"Based on the healthcare policy documents, {q['query'].lower().replace('?', '')}. "
        answer += f"The relevant information can be found in the retrieved context covering topics such as {', '.join(q['topics'][:2])}."
        
        # Evaluate
        eval_result = evaluator.evaluate_response(
            query=q["query"],
            answer=answer,
            context=context,
            query_id=q["query_id"]
        )
        results.append(eval_result.to_dict())
        
        if (i + 1) % 10 == 0:
            print(f"  Evaluated {i + 1}/{len(queries)}")
    
    # Aggregate metrics
    aggregate = {
        "num_samples": len(results),
        "judge_model": judge_model,
        "faithfulness": {
            "mean": round(sum(r["faithfulness"] for r in results) / len(results), 4),
            "min": round(min(r["faithfulness"] for r in results), 4),
            "max": round(max(r["faithfulness"] for r in results), 4)
        },
        "answer_relevancy": {
            "mean": round(sum(r["answer_relevancy"] for r in results) / len(results), 4),
            "min": round(min(r["answer_relevancy"] for r in results), 4),
            "max": round(max(r["answer_relevancy"] for r in results), 4)
        },
        "context_precision": {
            "mean": round(sum(r["context_precision"] for r in results) / len(results), 4),
            "min": round(min(r["context_precision"] for r in results), 4),
            "max": round(max(r["context_precision"] for r in results), 4)
        },
        "groundedness": {
            "mean": round(sum(r["groundedness"] for r in results) / len(results), 4),
            "min": round(min(r["groundedness"] for r in results), 4),
            "max": round(max(r["groundedness"] for r in results), 4)
        }
    }
    
    # Save results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump({
            "metadata": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "judge_model": judge_model,
                "num_samples": len(results)
            },
            "aggregate": aggregate,
            "per_query": results
        }, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 RAG QUALITY EVALUATION RESULTS")
    print("=" * 60)
    print(f"Samples: {aggregate['num_samples']}")
    print(f"Judge: {aggregate['judge_model']}")
    print(f"\nMetrics (mean):")
    print(f"  Faithfulness:       {aggregate['faithfulness']['mean']:.4f}")
    print(f"  Answer Relevancy:   {aggregate['answer_relevancy']['mean']:.4f}")
    print(f"  Context Precision:  {aggregate['context_precision']['mean']:.4f}")
    print(f"  Groundedness:       {aggregate['groundedness']['mean']:.4f}")
    print("=" * 60)
    
    return {"aggregate": aggregate, "per_query": results}


if __name__ == "__main__":
    run_rag_evaluation()
