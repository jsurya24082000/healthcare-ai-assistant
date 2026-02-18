"""Grounding verification for AI responses."""

import re
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

from openai import OpenAI


@dataclass
class GroundingResult:
    """Result of grounding verification."""
    is_grounded: bool
    grounding_score: float  # 0-1
    supported_claims: List[str]
    unsupported_claims: List[str]
    source_coverage: float  # How much of the response is covered by sources
    citation_accuracy: float  # How accurate the citations are
    details: Dict[str, Any]


class GroundingChecker:
    """Verify that AI responses are grounded in source documents."""
    
    def __init__(self, model: str = "gpt-4-turbo-preview"):
        """
        Initialize grounding checker.
        
        Args:
            model: OpenAI model for verification.
        """
        self.model = model
        self.client = OpenAI()
    
    def check_grounding(
        self,
        response: str,
        sources: List[Dict[str, Any]],
        strict_mode: bool = False
    ) -> GroundingResult:
        """
        Check if response is grounded in sources.
        
        Args:
            response: AI-generated response.
            sources: List of source documents with content.
            strict_mode: If True, require explicit citations for all claims.
            
        Returns:
            GroundingResult with detailed analysis.
        """
        # Extract claims from response
        claims = self._extract_claims(response)
        
        # Build source context
        source_context = self._format_sources(sources)
        
        # Verify each claim
        verification = self._verify_claims(claims, source_context, strict_mode)
        
        # Calculate scores
        supported = verification["supported_claims"]
        unsupported = verification["unsupported_claims"]
        
        total_claims = len(supported) + len(unsupported)
        grounding_score = len(supported) / total_claims if total_claims > 0 else 1.0
        
        # Check citation accuracy
        citation_accuracy = self._check_citation_accuracy(response, sources)
        
        # Calculate source coverage
        source_coverage = self._calculate_source_coverage(response, source_context)
        
        return GroundingResult(
            is_grounded=grounding_score >= 0.8 and len(unsupported) == 0,
            grounding_score=grounding_score,
            supported_claims=supported,
            unsupported_claims=unsupported,
            source_coverage=source_coverage,
            citation_accuracy=citation_accuracy,
            details=verification
        )
    
    def _extract_claims(self, response: str) -> List[str]:
        """Extract factual claims from response."""
        # Split into sentences
        sentences = re.split(r'[.!?]+', response)
        
        claims = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:
                continue
            
            # Skip meta-sentences
            skip_patterns = [
                r'^(I |Based on|According to|The document|This)',
                r'^(However|Therefore|Thus|In conclusion)',
                r'(should consult|recommend|disclaimer)',
            ]
            
            is_meta = any(re.search(p, sentence, re.IGNORECASE) for p in skip_patterns)
            
            if not is_meta and len(sentence) > 20:
                claims.append(sentence)
        
        return claims
    
    def _format_sources(self, sources: List[Dict[str, Any]]) -> str:
        """Format sources into context string."""
        formatted = []
        for i, source in enumerate(sources, 1):
            content = source.get("content", source.get("content_preview", ""))
            filename = source.get("filename", f"Source {i}")
            page = source.get("page", "?")
            formatted.append(f"[{filename}, Page {page}]\n{content}")
        
        return "\n\n---\n\n".join(formatted)
    
    def _verify_claims(
        self,
        claims: List[str],
        source_context: str,
        strict_mode: bool
    ) -> Dict[str, Any]:
        """Verify claims against source context."""
        if not claims:
            return {"supported_claims": [], "unsupported_claims": [], "details": []}
        
        claims_text = "\n".join([f"{i+1}. {c}" for i, c in enumerate(claims)])
        
        strictness = "STRICT" if strict_mode else "STANDARD"
        
        prompt = f"""Verify each claim against the source documents.

SOURCE DOCUMENTS:
{source_context}

CLAIMS TO VERIFY:
{claims_text}

MODE: {strictness}
{"In STRICT mode, a claim is only SUPPORTED if it can be directly traced to a specific passage." if strict_mode else "In STANDARD mode, reasonable inferences from the sources are acceptable."}

For each claim, determine:
- SUPPORTED: The claim is directly stated or clearly implied by the sources
- UNSUPPORTED: The claim cannot be verified from the sources
- PARTIALLY_SUPPORTED: Some aspects are supported, others are not

Format your response as:
CLAIM 1: [SUPPORTED/UNSUPPORTED/PARTIALLY_SUPPORTED]
- Evidence: [quote or reference from source if supported]
- Issue: [what's unsupported if applicable]

CLAIM 2: ...

SUMMARY:
- Supported claims: [list numbers]
- Unsupported claims: [list numbers]
- Partially supported: [list numbers]"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        analysis = response.choices[0].message.content
        
        # Parse results
        supported = []
        unsupported = []
        details = []
        
        for i, claim in enumerate(claims, 1):
            pattern = rf"CLAIM {i}:\s*(\w+)"
            match = re.search(pattern, analysis, re.IGNORECASE)
            
            if match:
                status = match.group(1).upper()
                if status == "SUPPORTED":
                    supported.append(claim)
                elif status == "UNSUPPORTED":
                    unsupported.append(claim)
                else:  # PARTIALLY_SUPPORTED
                    unsupported.append(claim)  # Treat as unsupported for safety
                
                details.append({
                    "claim": claim,
                    "status": status,
                    "index": i
                })
        
        return {
            "supported_claims": supported,
            "unsupported_claims": unsupported,
            "details": details,
            "raw_analysis": analysis
        }
    
    def _check_citation_accuracy(
        self,
        response: str,
        sources: List[Dict[str, Any]]
    ) -> float:
        """Check if citations in response accurately reference sources."""
        # Find citations in response
        citation_patterns = [
            r'\[Source:\s*([^\]]+)\]',
            r'\[Document\s*(\d+)[^\]]*\]',
            r'\(([^)]+,\s*(?:p\.|Page)\s*\d+)\)',
        ]
        
        citations = []
        for pattern in citation_patterns:
            citations.extend(re.findall(pattern, response, re.IGNORECASE))
        
        if not citations:
            return 0.0  # No citations found
        
        # Check if cited sources exist
        source_names = [s.get("filename", "").lower() for s in sources]
        
        accurate_citations = 0
        for citation in citations:
            citation_lower = citation.lower()
            # Check if any source name is mentioned in citation
            if any(name in citation_lower or citation_lower in name 
                   for name in source_names if name):
                accurate_citations += 1
        
        return accurate_citations / len(citations) if citations else 0.0
    
    def _calculate_source_coverage(
        self,
        response: str,
        source_context: str
    ) -> float:
        """Calculate how much of the response is covered by sources."""
        # Simple word overlap calculation
        response_words = set(re.findall(r'\b\w{4,}\b', response.lower()))
        source_words = set(re.findall(r'\b\w{4,}\b', source_context.lower()))
        
        if not response_words:
            return 1.0
        
        overlap = len(response_words & source_words)
        coverage = overlap / len(response_words)
        
        return min(coverage * 1.5, 1.0)  # Scale up slightly, cap at 1.0
    
    def verify_citation(
        self,
        claim: str,
        cited_source: str,
        source_content: str
    ) -> Tuple[bool, str]:
        """
        Verify a specific citation.
        
        Args:
            claim: The claim being made.
            cited_source: The source being cited.
            source_content: The actual content of the source.
            
        Returns:
            Tuple of (is_valid, explanation).
        """
        prompt = f"""Verify if this citation is accurate.

CLAIM: {claim}

CITED SOURCE: {cited_source}

SOURCE CONTENT:
{source_content}

Is the claim supported by this source? Respond with:
VALID: [YES/NO]
EXPLANATION: [Brief explanation]"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=200
        )
        
        result = response.choices[0].message.content
        
        is_valid = "VALID: YES" in result.upper()
        explanation = re.search(r"EXPLANATION:\s*(.+)", result, re.IGNORECASE)
        explanation_text = explanation.group(1) if explanation else "Unable to verify"
        
        return is_valid, explanation_text
