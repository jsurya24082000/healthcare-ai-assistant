"""Hallucination detection for healthcare AI responses."""

import re
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from openai import OpenAI


class HallucinationSeverity(Enum):
    """Severity levels for hallucination."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class HallucinationReport:
    """Detailed hallucination analysis report."""
    severity: HallucinationSeverity
    confidence_score: float  # 0-1, how confident we are in the assessment
    grounding_score: float  # 0-1, how well grounded the response is
    flagged_claims: List[Dict[str, Any]]
    recommendations: List[str]
    requires_human_review: bool
    raw_analysis: str


class HallucinationDetector:
    """Detect and analyze hallucinations in AI responses."""
    
    # Patterns that often indicate hallucination risk
    RISK_PATTERNS = [
        (r'\b(exactly|precisely|specifically)\s+\d+', "specific_numbers"),
        (r'\b(Dr\.|Professor|Director)\s+[A-Z][a-z]+\s+[A-Z][a-z]+', "specific_names"),
        (r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}', "specific_dates"),
        (r'\$[\d,]+(?:\.\d{2})?', "specific_amounts"),
        (r'\b\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)', "specific_times"),
        (r'(?:Section|Article|Paragraph)\s+\d+(?:\.\d+)*', "specific_sections"),
        (r'\b(?:always|never|must|shall|guaranteed|certainly)\b', "absolute_language"),
    ]
    
    def __init__(self, model: str = "gpt-4-turbo-preview"):
        """
        Initialize hallucination detector.
        
        Args:
            model: OpenAI model for analysis.
        """
        self.model = model
        self.client = OpenAI()
    
    def detect(
        self,
        response: str,
        context: str,
        question: Optional[str] = None
    ) -> HallucinationReport:
        """
        Detect hallucinations in a response.
        
        Args:
            response: The AI-generated response to check.
            context: The source context used for generation.
            question: Optional original question.
            
        Returns:
            HallucinationReport with detailed analysis.
        """
        # Step 1: Pattern-based risk detection
        pattern_risks = self._detect_pattern_risks(response)
        
        # Step 2: LLM-based claim verification
        llm_analysis = self._llm_verify_claims(response, context, question)
        
        # Step 3: Combine analyses
        report = self._create_report(pattern_risks, llm_analysis)
        
        return report
    
    def _detect_pattern_risks(self, response: str) -> List[Dict[str, Any]]:
        """Detect risky patterns that often indicate hallucination."""
        risks = []
        
        for pattern, risk_type in self.RISK_PATTERNS:
            matches = re.finditer(pattern, response, re.IGNORECASE)
            for match in matches:
                risks.append({
                    "type": risk_type,
                    "text": match.group(),
                    "position": match.start(),
                    "risk_level": "medium"
                })
        
        return risks
    
    def _llm_verify_claims(
        self,
        response: str,
        context: str,
        question: Optional[str]
    ) -> Dict[str, Any]:
        """Use LLM to verify claims against context."""
        prompt = f"""You are a fact-checker for healthcare AI systems. Your job is to identify potential hallucinations - information in the response that is NOT supported by the provided context.

CONTEXT (Source Documents):
{context}

{"QUESTION: " + question if question else ""}

RESPONSE TO VERIFY:
{response}

Analyze the response and identify:

1. UNSUPPORTED CLAIMS: List any factual claims that are NOT found in the context
2. CONTRADICTIONS: List any claims that contradict the context
3. FABRICATED DETAILS: List any specific details (names, dates, numbers, procedures) that appear fabricated
4. APPROPRIATE HEDGING: Note if the response appropriately hedges uncertain information
5. MISSING CITATIONS: Claims that should have citations but don't

For each issue found, provide:
- The problematic text
- Why it's problematic
- Severity: LOW (minor embellishment), MEDIUM (unsupported claim), HIGH (fabricated fact), CRITICAL (dangerous misinformation)

Format your response as:

OVERALL_GROUNDING_SCORE: [0.0-1.0] (1.0 = fully grounded)
OVERALL_SEVERITY: [NONE/LOW/MEDIUM/HIGH/CRITICAL]

ISSUES:
1. [Issue description]
   - Text: "[problematic text]"
   - Type: [UNSUPPORTED/CONTRADICTION/FABRICATED/MISSING_CITATION]
   - Severity: [LOW/MEDIUM/HIGH/CRITICAL]
   - Explanation: [why this is problematic]

POSITIVE_OBSERVATIONS:
- [What the response does well regarding grounding]

RECOMMENDATIONS:
- [How to improve the response]

REQUIRES_HUMAN_REVIEW: [YES/NO]
REVIEW_REASON: [If yes, why]"""

        response_obj = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        analysis_text = response_obj.choices[0].message.content
        
        # Parse the analysis
        return self._parse_llm_analysis(analysis_text)
    
    def _parse_llm_analysis(self, analysis: str) -> Dict[str, Any]:
        """Parse LLM analysis into structured format."""
        result = {
            "grounding_score": 0.5,
            "severity": "MEDIUM",
            "issues": [],
            "recommendations": [],
            "requires_human_review": False,
            "raw_analysis": analysis
        }
        
        # Extract grounding score
        score_match = re.search(r"OVERALL_GROUNDING_SCORE:\s*([\d.]+)", analysis)
        if score_match:
            result["grounding_score"] = float(score_match.group(1))
        
        # Extract severity
        severity_match = re.search(r"OVERALL_SEVERITY:\s*(\w+)", analysis)
        if severity_match:
            result["severity"] = severity_match.group(1).upper()
        
        # Extract human review requirement
        review_match = re.search(r"REQUIRES_HUMAN_REVIEW:\s*(YES|NO)", analysis, re.IGNORECASE)
        if review_match:
            result["requires_human_review"] = review_match.group(1).upper() == "YES"
        
        # Extract issues (simplified parsing)
        issues_section = re.search(r"ISSUES:(.*?)(?:POSITIVE_OBSERVATIONS|RECOMMENDATIONS|$)", 
                                   analysis, re.DOTALL)
        if issues_section:
            issue_text = issues_section.group(1)
            # Find individual issues
            issue_matches = re.findall(
                r'Text:\s*"([^"]+)".*?Severity:\s*(\w+)',
                issue_text,
                re.DOTALL
            )
            for text, severity in issue_matches:
                result["issues"].append({
                    "text": text,
                    "severity": severity.upper()
                })
        
        # Extract recommendations
        rec_section = re.search(r"RECOMMENDATIONS:(.*?)(?:REQUIRES_HUMAN_REVIEW|$)", 
                                analysis, re.DOTALL)
        if rec_section:
            recs = re.findall(r"-\s*(.+?)(?:\n|$)", rec_section.group(1))
            result["recommendations"] = [r.strip() for r in recs if r.strip()]
        
        return result
    
    def _create_report(
        self,
        pattern_risks: List[Dict[str, Any]],
        llm_analysis: Dict[str, Any]
    ) -> HallucinationReport:
        """Combine analyses into final report."""
        # Determine severity
        severity_map = {
            "NONE": HallucinationSeverity.NONE,
            "LOW": HallucinationSeverity.LOW,
            "MEDIUM": HallucinationSeverity.MEDIUM,
            "HIGH": HallucinationSeverity.HIGH,
            "CRITICAL": HallucinationSeverity.CRITICAL
        }
        
        severity = severity_map.get(
            llm_analysis.get("severity", "MEDIUM"),
            HallucinationSeverity.MEDIUM
        )
        
        # Combine flagged claims
        flagged_claims = llm_analysis.get("issues", [])
        for risk in pattern_risks:
            flagged_claims.append({
                "text": risk["text"],
                "type": f"pattern_risk_{risk['type']}",
                "severity": risk["risk_level"].upper()
            })
        
        # Calculate confidence based on analysis quality
        confidence = 0.8  # Base confidence
        if len(flagged_claims) > 5:
            confidence = 0.9  # More issues = more confident in detection
        
        return HallucinationReport(
            severity=severity,
            confidence_score=confidence,
            grounding_score=llm_analysis.get("grounding_score", 0.5),
            flagged_claims=flagged_claims,
            recommendations=llm_analysis.get("recommendations", []),
            requires_human_review=llm_analysis.get("requires_human_review", False),
            raw_analysis=llm_analysis.get("raw_analysis", "")
        )
    
    def quick_check(self, response: str, context: str) -> Tuple[bool, float]:
        """
        Quick hallucination check without full analysis.
        
        Args:
            response: Response to check.
            context: Source context.
            
        Returns:
            Tuple of (is_likely_hallucinated, confidence_score).
        """
        # Pattern-based quick check
        pattern_risks = self._detect_pattern_risks(response)
        high_risk_patterns = [r for r in pattern_risks if r["risk_level"] in ["high", "critical"]]
        
        # Check if response contains content not in context
        response_words = set(response.lower().split())
        context_words = set(context.lower().split())
        
        # Calculate word overlap
        overlap = len(response_words & context_words) / len(response_words) if response_words else 1.0
        
        # Heuristic scoring
        is_hallucinated = len(high_risk_patterns) > 2 or overlap < 0.3
        confidence = 0.6 + (0.2 * len(high_risk_patterns) / 5)  # Rough confidence
        
        return is_hallucinated, min(confidence, 1.0)
