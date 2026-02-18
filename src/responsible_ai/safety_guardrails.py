"""Safety guardrails for healthcare AI responses."""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from openai import OpenAI


class SafetyCategory(Enum):
    """Categories of safety concerns."""
    MEDICAL_ADVICE = "medical_advice"
    EMERGENCY = "emergency"
    MEDICATION = "medication"
    DIAGNOSIS = "diagnosis"
    TREATMENT = "treatment"
    LEGAL = "legal"
    PRIVACY = "privacy"
    MENTAL_HEALTH = "mental_health"


@dataclass
class SafetyCheck:
    """Result of a safety check."""
    is_safe: bool
    risk_level: str  # low, medium, high, critical
    categories_flagged: List[SafetyCategory]
    warnings: List[str]
    required_disclaimers: List[str]
    modified_response: Optional[str] = None


class SafetyGuardrails:
    """Safety guardrails for healthcare AI system."""
    
    # Patterns that indicate potentially unsafe content
    UNSAFE_PATTERNS = {
        SafetyCategory.MEDICAL_ADVICE: [
            r'\b(you should take|take \d+ mg|dosage of)\b',
            r'\b(prescribe|prescription|medication for you)\b',
            r'\b(diagnose you with|you have|you are suffering from)\b',
        ],
        SafetyCategory.EMERGENCY: [
            r'\b(heart attack|stroke|seizure|overdose|suicide|self-harm)\b',
            r'\b(call 911|emergency room|ER immediately)\b',
            r'\b(life-threatening|fatal|death)\b',
        ],
        SafetyCategory.MEDICATION: [
            r'\b(increase|decrease|stop taking|start taking)\s+\w+\s*(medication|drug|medicine)\b',
            r'\b(drug interaction|side effect|adverse reaction)\b',
            r'\b(opioid|narcotic|controlled substance)\b',
        ],
        SafetyCategory.DIAGNOSIS: [
            r'\b(you have|you are diagnosed|diagnosis is|you suffer from)\b',
            r'\b(symptoms indicate|signs of|consistent with)\b',
        ],
        SafetyCategory.MENTAL_HEALTH: [
            r'\b(depression|anxiety|bipolar|schizophrenia|PTSD)\b',
            r'\b(suicidal|self-harm|eating disorder)\b',
            r'\b(mental health crisis|psychiatric)\b',
        ],
    }
    
    # Required disclaimers by category
    DISCLAIMERS = {
        SafetyCategory.MEDICAL_ADVICE: (
            "This information is from policy documents and should not be considered "
            "medical advice. Please consult a qualified healthcare provider for "
            "personalized medical guidance."
        ),
        SafetyCategory.EMERGENCY: (
            "If you are experiencing a medical emergency, please call 911 or your "
            "local emergency services immediately."
        ),
        SafetyCategory.MEDICATION: (
            "Medication information provided is for reference only. Always consult "
            "your healthcare provider or pharmacist before making any changes to "
            "your medication regimen."
        ),
        SafetyCategory.DIAGNOSIS: (
            "This system cannot provide medical diagnoses. Only qualified healthcare "
            "professionals can diagnose medical conditions after proper examination."
        ),
        SafetyCategory.MENTAL_HEALTH: (
            "If you or someone you know is struggling with mental health issues, "
            "please reach out to a mental health professional or call the National "
            "Suicide Prevention Lifeline at 988."
        ),
        SafetyCategory.LEGAL: (
            "This information is not legal advice. Consult with a qualified legal "
            "professional for guidance on legal matters."
        ),
        SafetyCategory.PRIVACY: (
            "For specific questions about your personal health information and privacy "
            "rights, contact your healthcare provider's privacy officer."
        ),
    }
    
    def __init__(self, model: str = "gpt-4-turbo-preview", strict_mode: bool = True):
        """
        Initialize safety guardrails.
        
        Args:
            model: OpenAI model for safety analysis.
            strict_mode: If True, apply stricter safety checks.
        """
        self.model = model
        self.strict_mode = strict_mode
        self.client = OpenAI()
    
    def check_response(
        self,
        response: str,
        question: Optional[str] = None
    ) -> SafetyCheck:
        """
        Check response for safety concerns.
        
        Args:
            response: AI-generated response to check.
            question: Optional original question for context.
            
        Returns:
            SafetyCheck with analysis and recommendations.
        """
        # Pattern-based detection
        flagged_categories = self._pattern_check(response)
        
        # LLM-based safety analysis
        llm_analysis = self._llm_safety_check(response, question)
        
        # Combine results
        all_categories = list(set(flagged_categories + llm_analysis.get("categories", [])))
        
        # Determine risk level
        risk_level = self._calculate_risk_level(all_categories, llm_analysis)
        
        # Generate warnings and disclaimers
        warnings = llm_analysis.get("warnings", [])
        disclaimers = [
            self.DISCLAIMERS[cat] 
            for cat in all_categories 
            if cat in self.DISCLAIMERS
        ]
        
        # Determine if safe
        is_safe = risk_level in ["low", "medium"] and not llm_analysis.get("block", False)
        
        # Generate modified response if needed
        modified_response = None
        if not is_safe or disclaimers:
            modified_response = self._add_safety_modifications(
                response, disclaimers, warnings
            )
        
        return SafetyCheck(
            is_safe=is_safe,
            risk_level=risk_level,
            categories_flagged=all_categories,
            warnings=warnings,
            required_disclaimers=disclaimers,
            modified_response=modified_response
        )
    
    def _pattern_check(self, response: str) -> List[SafetyCategory]:
        """Check response against unsafe patterns."""
        flagged = []
        
        for category, patterns in self.UNSAFE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, response, re.IGNORECASE):
                    if category not in flagged:
                        flagged.append(category)
                    break
        
        return flagged
    
    def _llm_safety_check(
        self,
        response: str,
        question: Optional[str]
    ) -> Dict[str, Any]:
        """Use LLM for nuanced safety analysis."""
        prompt = f"""Analyze this healthcare AI response for safety concerns.

{"QUESTION: " + question if question else ""}

RESPONSE:
{response}

Check for:
1. MEDICAL ADVICE: Does it provide specific medical recommendations?
2. DIAGNOSIS: Does it attempt to diagnose conditions?
3. MEDICATION: Does it recommend specific medications or dosages?
4. EMERGENCY: Does it involve emergency situations?
5. MENTAL HEALTH: Does it address sensitive mental health topics?
6. HARMFUL: Could following this advice cause harm?

For each concern found:
- Identify the problematic text
- Explain why it's concerning
- Suggest how to make it safer

Format response as:
RISK_LEVEL: [LOW/MEDIUM/HIGH/CRITICAL]
SHOULD_BLOCK: [YES/NO]

CONCERNS:
1. Category: [category]
   Text: "[problematic text]"
   Issue: [explanation]
   Fix: [suggestion]

WARNINGS:
- [warning messages to show user]

OVERALL_ASSESSMENT: [brief summary]"""

        response_obj = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        analysis = response_obj.choices[0].message.content
        
        return self._parse_safety_analysis(analysis)
    
    def _parse_safety_analysis(self, analysis: str) -> Dict[str, Any]:
        """Parse LLM safety analysis."""
        result = {
            "risk_level": "medium",
            "block": False,
            "categories": [],
            "warnings": [],
            "raw_analysis": analysis
        }
        
        # Extract risk level
        risk_match = re.search(r"RISK_LEVEL:\s*(\w+)", analysis, re.IGNORECASE)
        if risk_match:
            result["risk_level"] = risk_match.group(1).lower()
        
        # Check if should block
        block_match = re.search(r"SHOULD_BLOCK:\s*(YES|NO)", analysis, re.IGNORECASE)
        if block_match:
            result["block"] = block_match.group(1).upper() == "YES"
        
        # Extract categories
        category_map = {
            "medical advice": SafetyCategory.MEDICAL_ADVICE,
            "diagnosis": SafetyCategory.DIAGNOSIS,
            "medication": SafetyCategory.MEDICATION,
            "emergency": SafetyCategory.EMERGENCY,
            "mental health": SafetyCategory.MENTAL_HEALTH,
            "legal": SafetyCategory.LEGAL,
            "privacy": SafetyCategory.PRIVACY,
        }
        
        for cat_name, cat_enum in category_map.items():
            if cat_name.lower() in analysis.lower():
                if cat_enum not in result["categories"]:
                    result["categories"].append(cat_enum)
        
        # Extract warnings
        warnings_section = re.search(r"WARNINGS:(.*?)(?:OVERALL_ASSESSMENT|$)", 
                                     analysis, re.DOTALL)
        if warnings_section:
            warnings = re.findall(r"-\s*(.+?)(?:\n|$)", warnings_section.group(1))
            result["warnings"] = [w.strip() for w in warnings if w.strip()]
        
        return result
    
    def _calculate_risk_level(
        self,
        categories: List[SafetyCategory],
        llm_analysis: Dict[str, Any]
    ) -> str:
        """Calculate overall risk level."""
        # High-risk categories
        high_risk = {
            SafetyCategory.EMERGENCY,
            SafetyCategory.MEDICATION,
            SafetyCategory.DIAGNOSIS
        }
        
        if any(cat in high_risk for cat in categories):
            if self.strict_mode:
                return "high"
            return "medium"
        
        if llm_analysis.get("risk_level") == "critical":
            return "critical"
        
        if len(categories) >= 3:
            return "high"
        elif len(categories) >= 1:
            return "medium"
        
        return "low"
    
    def _add_safety_modifications(
        self,
        response: str,
        disclaimers: List[str],
        warnings: List[str]
    ) -> str:
        """Add safety modifications to response."""
        modified = response
        
        # Add disclaimers at the end
        if disclaimers:
            disclaimer_text = "\n\n---\n**Important Disclaimers:**\n"
            for disclaimer in disclaimers:
                disclaimer_text += f"- {disclaimer}\n"
            modified += disclaimer_text
        
        return modified
    
    def check_question_safety(self, question: str) -> Tuple[bool, List[str]]:
        """
        Check if a question is safe to answer.
        
        Args:
            question: User's question.
            
        Returns:
            Tuple of (is_safe_to_answer, warnings).
        """
        # Check for emergency indicators
        emergency_patterns = [
            r'\b(overdose|suicide|kill myself|want to die)\b',
            r'\b(heart attack|stroke|can\'t breathe|chest pain)\b',
            r'\b(bleeding heavily|unconscious|seizure)\b',
        ]
        
        for pattern in emergency_patterns:
            if re.search(pattern, question, re.IGNORECASE):
                return False, [
                    "This appears to be an emergency situation. "
                    "Please call 911 or your local emergency services immediately."
                ]
        
        # Check for requests for specific medical advice
        advice_patterns = [
            r'\b(what medication should I take)\b',
            r'\b(should I stop taking)\b',
            r'\b(diagnose me|what do I have)\b',
        ]
        
        warnings = []
        for pattern in advice_patterns:
            if re.search(pattern, question, re.IGNORECASE):
                warnings.append(
                    "This system provides policy information only and cannot "
                    "provide personalized medical advice."
                )
                break
        
        return True, warnings
    
    def get_safe_response_template(self, category: SafetyCategory) -> str:
        """Get a safe response template for a category."""
        templates = {
            SafetyCategory.MEDICAL_ADVICE: (
                "I can provide information from healthcare policy documents, "
                "but I cannot give personalized medical advice. For specific "
                "medical guidance, please consult with your healthcare provider."
            ),
            SafetyCategory.EMERGENCY: (
                "If you're experiencing a medical emergency, please call 911 "
                "or go to your nearest emergency room immediately. I can help "
                "with policy questions once you're safe."
            ),
            SafetyCategory.MEDICATION: (
                "I can share policy information about medications, but any "
                "decisions about your medications should be made with your "
                "healthcare provider or pharmacist."
            ),
            SafetyCategory.DIAGNOSIS: (
                "I'm not able to diagnose medical conditions. Only qualified "
                "healthcare professionals can provide diagnoses after proper "
                "examination. I can help you understand healthcare policies."
            ),
        }
        
        return templates.get(category, self.DISCLAIMERS.get(category, ""))
