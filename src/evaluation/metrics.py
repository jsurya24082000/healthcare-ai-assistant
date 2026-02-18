"""Evaluation metrics for RAG systems."""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import numpy as np
from openai import OpenAI


@dataclass
class MetricResult:
    """Result of a metric evaluation."""
    name: str
    score: float
    details: Dict[str, Any]
    explanation: str = ""


class RAGMetrics:
    """Metrics for evaluating RAG system responses."""
    
    def __init__(self, model: str = "gpt-4-turbo-preview"):
        """
        Initialize metrics evaluator.
        
        Args:
            model: OpenAI model for LLM-based evaluations.
        """
        self.model = model
        self.client = OpenAI()
    
    def evaluate_all(
        self,
        question: str,
        answer: str,
        context: str,
        ground_truth: Optional[str] = None
    ) -> Dict[str, MetricResult]:
        """
        Run all evaluation metrics.
        
        Args:
            question: Original question.
            answer: Generated answer.
            context: Retrieved context used for generation.
            ground_truth: Optional ground truth answer.
            
        Returns:
            Dictionary of metric names to MetricResult objects.
        """
        results = {}
        
        # Faithfulness - is the answer supported by context?
        results["faithfulness"] = self.evaluate_faithfulness(answer, context)
        
        # Answer relevancy - does the answer address the question?
        results["answer_relevancy"] = self.evaluate_answer_relevancy(question, answer)
        
        # Citation coverage - are claims properly cited?
        results["citation_coverage"] = self.evaluate_citation_coverage(answer, context)
        
        # Hallucination detection
        results["hallucination"] = self.detect_hallucination(answer, context)
        
        # If ground truth available, compute accuracy
        if ground_truth:
            results["answer_correctness"] = self.evaluate_correctness(
                answer, ground_truth
            )
        
        return results
    
    def evaluate_faithfulness(self, answer: str, context: str) -> MetricResult:
        """
        Evaluate if the answer is faithful to the context.
        
        Faithfulness measures whether all claims in the answer
        are supported by the provided context.
        """
        prompt = f"""Evaluate the faithfulness of the following answer to the given context.

CONTEXT:
{context}

ANSWER:
{answer}

For each claim in the answer, determine if it is:
1. SUPPORTED - directly stated or clearly implied by the context
2. PARTIALLY_SUPPORTED - somewhat related but not fully supported
3. UNSUPPORTED - not found in or contradicts the context

Provide your evaluation as:
1. List each claim and its support status
2. Calculate the faithfulness score (supported claims / total claims)
3. Overall assessment

Format your response as:
CLAIMS:
- [claim]: [SUPPORTED/PARTIALLY_SUPPORTED/UNSUPPORTED]
...

SCORE: [0.0-1.0]
ASSESSMENT: [brief explanation]"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        result_text = response.choices[0].message.content
        
        # Parse score from response
        score = self._extract_score(result_text)
        
        return MetricResult(
            name="faithfulness",
            score=score,
            details={"raw_evaluation": result_text},
            explanation="Measures if answer claims are supported by context"
        )
    
    def evaluate_answer_relevancy(self, question: str, answer: str) -> MetricResult:
        """
        Evaluate if the answer is relevant to the question.
        """
        prompt = f"""Evaluate how well the answer addresses the question.

QUESTION: {question}

ANSWER: {answer}

Rate the answer relevancy on these criteria:
1. Does it directly address the question asked?
2. Is the information provided useful for answering the question?
3. Is the response appropriately scoped (not too broad/narrow)?

Provide:
RELEVANCY_SCORE: [0.0-1.0]
REASONING: [explanation]"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        result_text = response.choices[0].message.content
        score = self._extract_score(result_text, pattern=r"RELEVANCY_SCORE:\s*([\d.]+)")
        
        return MetricResult(
            name="answer_relevancy",
            score=score,
            details={"raw_evaluation": result_text},
            explanation="Measures if answer addresses the question"
        )
    
    def evaluate_citation_coverage(self, answer: str, context: str) -> MetricResult:
        """
        Evaluate citation coverage in the answer.
        """
        # Find citations in answer
        citation_patterns = [
            r'\[Source:.*?\]',
            r'\[Document \d+.*?\]',
            r'\(Source:.*?\)',
            r'\[p\.\s*\d+\]',
            r'\[Page \d+\]'
        ]
        
        citations_found = []
        for pattern in citation_patterns:
            citations_found.extend(re.findall(pattern, answer, re.IGNORECASE))
        
        # Count sentences that should have citations
        sentences = re.split(r'[.!?]+', answer)
        factual_sentences = [s for s in sentences if len(s.strip()) > 20]
        
        # Calculate coverage
        if not factual_sentences:
            coverage = 1.0
        else:
            # Rough heuristic: each citation covers ~2 sentences
            expected_citations = max(1, len(factual_sentences) // 2)
            coverage = min(1.0, len(citations_found) / expected_citations)
        
        return MetricResult(
            name="citation_coverage",
            score=coverage,
            details={
                "citations_found": len(citations_found),
                "citation_examples": citations_found[:5],
                "factual_sentences": len(factual_sentences)
            },
            explanation="Measures if claims are properly cited"
        )
    
    def detect_hallucination(self, answer: str, context: str) -> MetricResult:
        """
        Detect potential hallucinations in the answer.
        
        Returns a score where:
        - 1.0 = No hallucination detected
        - 0.0 = High hallucination detected
        """
        prompt = f"""Analyze the following answer for potential hallucinations - information that is fabricated or not supported by the context.

CONTEXT:
{context}

ANSWER:
{answer}

Identify:
1. Any specific facts, numbers, dates, or names in the answer not found in context
2. Any claims that contradict the context
3. Any overly specific details that seem fabricated
4. Any generalizations not supported by the context

Rate the hallucination level:
- 0.0 = Severe hallucination (major fabricated content)
- 0.3 = Moderate hallucination (some unsupported claims)
- 0.7 = Minor hallucination (slight embellishments)
- 1.0 = No hallucination (fully grounded)

Provide:
HALLUCINATIONS_FOUND:
- [list any hallucinated content]

HALLUCINATION_SCORE: [0.0-1.0] (1.0 = no hallucination)
RISK_LEVEL: [LOW/MEDIUM/HIGH/CRITICAL]
EXPLANATION: [details]"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        result_text = response.choices[0].message.content
        score = self._extract_score(result_text, pattern=r"HALLUCINATION_SCORE:\s*([\d.]+)")
        
        # Extract risk level
        risk_match = re.search(r"RISK_LEVEL:\s*(\w+)", result_text)
        risk_level = risk_match.group(1) if risk_match else "UNKNOWN"
        
        return MetricResult(
            name="hallucination",
            score=score,
            details={
                "raw_evaluation": result_text,
                "risk_level": risk_level
            },
            explanation="Detects fabricated or unsupported information (1.0 = no hallucination)"
        )
    
    def evaluate_correctness(
        self, 
        answer: str, 
        ground_truth: str
    ) -> MetricResult:
        """
        Evaluate answer correctness against ground truth.
        """
        prompt = f"""Compare the generated answer to the ground truth answer.

GENERATED ANSWER:
{answer}

GROUND TRUTH:
{ground_truth}

Evaluate:
1. Factual accuracy - does the answer contain correct information?
2. Completeness - does it cover the key points from ground truth?
3. No contradictions - does it avoid stating incorrect information?

Provide:
CORRECTNESS_SCORE: [0.0-1.0]
MISSING_INFORMATION: [what's missing from ground truth]
INCORRECT_INFORMATION: [any errors]
ASSESSMENT: [overall evaluation]"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        result_text = response.choices[0].message.content
        score = self._extract_score(result_text, pattern=r"CORRECTNESS_SCORE:\s*([\d.]+)")
        
        return MetricResult(
            name="answer_correctness",
            score=score,
            details={"raw_evaluation": result_text},
            explanation="Measures accuracy against ground truth"
        )
    
    def _extract_score(
        self, 
        text: str, 
        pattern: str = r"SCORE:\s*([\d.]+)"
    ) -> float:
        """Extract numeric score from evaluation text."""
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                score = float(match.group(1))
                return min(1.0, max(0.0, score))
            except ValueError:
                pass
        return 0.5  # Default if parsing fails
    
    def compute_aggregate_score(
        self, 
        results: Dict[str, MetricResult],
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Compute weighted aggregate score.
        
        Args:
            results: Dictionary of metric results.
            weights: Optional weights for each metric.
            
        Returns:
            Weighted average score.
        """
        default_weights = {
            "faithfulness": 0.3,
            "answer_relevancy": 0.2,
            "citation_coverage": 0.15,
            "hallucination": 0.25,
            "answer_correctness": 0.1
        }
        
        weights = weights or default_weights
        
        total_weight = 0
        weighted_sum = 0
        
        for name, result in results.items():
            weight = weights.get(name, 0.1)
            weighted_sum += result.score * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
