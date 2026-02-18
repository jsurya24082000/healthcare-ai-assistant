"""Prompt evaluation experiments framework."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict

from src.llm.qa_chain import HealthcareQAChain, QAResponse
from src.llm.prompts import PromptTemplates
from .metrics import RAGMetrics, MetricResult


@dataclass
class TestCase:
    """A single test case for evaluation."""
    question: str
    ground_truth: Optional[str] = None
    expected_sources: Optional[List[str]] = None
    category: str = "general"
    difficulty: str = "medium"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentResult:
    """Result of a single experiment run."""
    test_case: TestCase
    response: QAResponse
    metrics: Dict[str, MetricResult]
    aggregate_score: float
    template_name: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "test_case": asdict(self.test_case),
            "response": self.response.to_dict(),
            "metrics": {
                name: {
                    "score": m.score,
                    "explanation": m.explanation,
                    "details": m.details
                }
                for name, m in self.metrics.items()
            },
            "aggregate_score": self.aggregate_score,
            "template_name": self.template_name,
            "timestamp": self.timestamp
        }


@dataclass
class PromptExperiment:
    """Configuration for a prompt experiment."""
    name: str
    description: str
    templates_to_test: List[str]
    test_cases: List[TestCase]
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromptExperiment":
        """Create from dictionary."""
        test_cases = [
            TestCase(**tc) if isinstance(tc, dict) else tc
            for tc in data.get("test_cases", [])
        ]
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            templates_to_test=data.get("templates_to_test", ["grounded_qa"]),
            test_cases=test_cases
        )


class ExperimentRunner:
    """Run and analyze prompt experiments."""
    
    def __init__(
        self,
        qa_chain: HealthcareQAChain,
        output_dir: str = "experiments"
    ):
        """
        Initialize experiment runner.
        
        Args:
            qa_chain: Q&A chain to test.
            output_dir: Directory for experiment outputs.
        """
        self.qa_chain = qa_chain
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = RAGMetrics()
    
    def run_experiment(
        self,
        experiment: PromptExperiment,
        save_results: bool = True
    ) -> Dict[str, List[ExperimentResult]]:
        """
        Run a complete experiment.
        
        Args:
            experiment: Experiment configuration.
            save_results: Whether to save results to disk.
            
        Returns:
            Dictionary mapping template names to results.
        """
        print(f"\n{'='*60}")
        print(f"Running Experiment: {experiment.name}")
        print(f"Description: {experiment.description}")
        print(f"Templates: {experiment.templates_to_test}")
        print(f"Test Cases: {len(experiment.test_cases)}")
        print(f"{'='*60}\n")
        
        all_results = {}
        
        for template_name in experiment.templates_to_test:
            print(f"\n--- Testing Template: {template_name} ---")
            template_results = []
            
            for i, test_case in enumerate(experiment.test_cases, 1):
                print(f"  Test {i}/{len(experiment.test_cases)}: {test_case.question[:50]}...")
                
                # Run query
                response = self.qa_chain.query(
                    test_case.question,
                    template_name=template_name
                )
                
                # Build context string from sources
                context = "\n\n".join([
                    s.get("content_preview", "") 
                    for s in response.sources
                ])
                
                # Evaluate
                metrics = self.metrics.evaluate_all(
                    question=test_case.question,
                    answer=response.answer,
                    context=context,
                    ground_truth=test_case.ground_truth
                )
                
                aggregate = self.metrics.compute_aggregate_score(metrics)
                
                result = ExperimentResult(
                    test_case=test_case,
                    response=response,
                    metrics=metrics,
                    aggregate_score=aggregate,
                    template_name=template_name
                )
                
                template_results.append(result)
                print(f"    Score: {aggregate:.3f}")
            
            all_results[template_name] = template_results
        
        # Generate summary
        summary = self._generate_summary(experiment, all_results)
        
        if save_results:
            self._save_results(experiment, all_results, summary)
        
        return all_results
    
    def _generate_summary(
        self,
        experiment: PromptExperiment,
        results: Dict[str, List[ExperimentResult]]
    ) -> Dict[str, Any]:
        """Generate experiment summary statistics."""
        summary = {
            "experiment_name": experiment.name,
            "timestamp": datetime.now().isoformat(),
            "templates_tested": len(results),
            "test_cases": len(experiment.test_cases),
            "template_scores": {}
        }
        
        print(f"\n{'='*60}")
        print("EXPERIMENT SUMMARY")
        print(f"{'='*60}")
        
        for template_name, template_results in results.items():
            scores = [r.aggregate_score for r in template_results]
            
            template_summary = {
                "mean_score": sum(scores) / len(scores),
                "min_score": min(scores),
                "max_score": max(scores),
                "metric_averages": {}
            }
            
            # Average by metric
            metric_names = template_results[0].metrics.keys()
            for metric_name in metric_names:
                metric_scores = [
                    r.metrics[metric_name].score 
                    for r in template_results
                ]
                template_summary["metric_averages"][metric_name] = (
                    sum(metric_scores) / len(metric_scores)
                )
            
            summary["template_scores"][template_name] = template_summary
            
            print(f"\n{template_name}:")
            print(f"  Mean Score: {template_summary['mean_score']:.3f}")
            print(f"  Range: [{template_summary['min_score']:.3f}, {template_summary['max_score']:.3f}]")
            print("  Metrics:")
            for metric, avg in template_summary["metric_averages"].items():
                print(f"    - {metric}: {avg:.3f}")
        
        # Find best template
        best_template = max(
            summary["template_scores"].items(),
            key=lambda x: x[1]["mean_score"]
        )
        summary["best_template"] = best_template[0]
        summary["best_score"] = best_template[1]["mean_score"]
        
        print(f"\n{'='*60}")
        print(f"BEST TEMPLATE: {summary['best_template']} (score: {summary['best_score']:.3f})")
        print(f"{'='*60}\n")
        
        return summary
    
    def _save_results(
        self,
        experiment: PromptExperiment,
        results: Dict[str, List[ExperimentResult]],
        summary: Dict[str, Any]
    ):
        """Save experiment results to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = self.output_dir / f"{experiment.name}_{timestamp}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save summary
        with open(exp_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        # Save detailed results per template
        for template_name, template_results in results.items():
            results_data = [r.to_dict() for r in template_results]
            with open(exp_dir / f"results_{template_name}.json", "w") as f:
                json.dump(results_data, f, indent=2)
        
        print(f"Results saved to: {exp_dir}")
    
    def compare_templates(
        self,
        test_cases: List[TestCase],
        templates: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Quick comparison of templates on test cases.
        
        Args:
            test_cases: Test cases to evaluate.
            templates: Templates to compare (all if None).
            
        Returns:
            Comparison summary.
        """
        templates = templates or [t["name"] for t in PromptTemplates.list_templates()]
        
        experiment = PromptExperiment(
            name="template_comparison",
            description="Quick template comparison",
            templates_to_test=templates,
            test_cases=test_cases
        )
        
        results = self.run_experiment(experiment, save_results=False)
        
        return self._generate_summary(experiment, results)


# Pre-defined test cases for healthcare domain
HEALTHCARE_TEST_CASES = [
    TestCase(
        question="What is the policy on patient data privacy?",
        category="privacy",
        difficulty="easy"
    ),
    TestCase(
        question="How should healthcare providers handle informed consent?",
        category="consent",
        difficulty="medium"
    ),
    TestCase(
        question="What are the requirements for medical record retention?",
        category="records",
        difficulty="medium"
    ),
    TestCase(
        question="Explain the procedure for reporting adverse events.",
        category="safety",
        difficulty="hard"
    ),
    TestCase(
        question="What are the guidelines for telemedicine consultations?",
        category="telemedicine",
        difficulty="medium"
    ),
    TestCase(
        question="How does HIPAA affect data sharing between providers?",
        category="compliance",
        difficulty="hard"
    ),
    TestCase(
        question="What is the protocol for handling patient complaints?",
        category="operations",
        difficulty="easy"
    ),
    TestCase(
        question="Describe the requirements for clinical trial documentation.",
        category="research",
        difficulty="hard"
    ),
]


def create_hallucination_test_experiment() -> PromptExperiment:
    """Create an experiment focused on hallucination detection."""
    test_cases = [
        TestCase(
            question="What specific medications are mentioned in the policy?",
            category="hallucination_test",
            difficulty="hard",
            metadata={"focus": "specific_details"}
        ),
        TestCase(
            question="What are the exact penalty amounts for HIPAA violations?",
            category="hallucination_test", 
            difficulty="hard",
            metadata={"focus": "numbers"}
        ),
        TestCase(
            question="Who is the designated privacy officer mentioned in the documents?",
            category="hallucination_test",
            difficulty="hard",
            metadata={"focus": "names"}
        ),
        TestCase(
            question="What happened in the 2019 policy update?",
            category="hallucination_test",
            difficulty="hard",
            metadata={"focus": "dates_events"}
        ),
    ]
    
    return PromptExperiment(
        name="hallucination_test",
        description="Test prompts for resistance to hallucination on specific details",
        templates_to_test=["grounded_qa", "strict_citation", "safety_first"],
        test_cases=test_cases
    )
