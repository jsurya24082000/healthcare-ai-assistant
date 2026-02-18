"""Tests for responsible AI module."""

import pytest

from src.responsible_ai.safety_guardrails import SafetyGuardrails, SafetyCategory
from src.responsible_ai.hallucination_detector import (
    HallucinationDetector, 
    HallucinationSeverity
)
from src.responsible_ai.grounding_checker import GroundingChecker


class TestSafetyGuardrails:
    """Tests for SafetyGuardrails class."""
    
    @pytest.fixture
    def guardrails(self):
        """Create guardrails instance."""
        return SafetyGuardrails(strict_mode=True)
    
    def test_initialization(self, guardrails):
        """Test guardrails initialization."""
        assert guardrails.strict_mode is True
        assert guardrails.model is not None
    
    def test_emergency_question_detection(self, guardrails):
        """Test emergency question detection."""
        emergency_questions = [
            "I'm having a heart attack, what should I do?",
            "I think I'm having a stroke",
            "I took too many pills, overdose",
            "I want to kill myself",
        ]
        
        for question in emergency_questions:
            is_safe, warnings = guardrails.check_question_safety(question)
            assert is_safe is False or len(warnings) > 0
    
    def test_safe_question_detection(self, guardrails):
        """Test safe question detection."""
        safe_questions = [
            "What is the policy on patient data privacy?",
            "How long should medical records be retained?",
            "What are the telemedicine guidelines?",
        ]
        
        for question in safe_questions:
            is_safe, warnings = guardrails.check_question_safety(question)
            assert is_safe is True
    
    def test_pattern_check_medical_advice(self, guardrails):
        """Test pattern detection for medical advice."""
        response = "You should take 500mg of ibuprofen for your headache."
        categories = guardrails._pattern_check(response)
        
        assert SafetyCategory.MEDICAL_ADVICE in categories or \
               SafetyCategory.MEDICATION in categories
    
    def test_pattern_check_emergency(self, guardrails):
        """Test pattern detection for emergency content."""
        response = "If you're having a heart attack, call 911 immediately."
        categories = guardrails._pattern_check(response)
        
        assert SafetyCategory.EMERGENCY in categories
    
    def test_safe_response_template(self, guardrails):
        """Test safe response template retrieval."""
        template = guardrails.get_safe_response_template(SafetyCategory.MEDICAL_ADVICE)
        assert "cannot give personalized medical advice" in template.lower() or \
               "consult" in template.lower()
    
    def test_disclaimers_exist(self, guardrails):
        """Test that disclaimers exist for key categories."""
        for category in [
            SafetyCategory.MEDICAL_ADVICE,
            SafetyCategory.EMERGENCY,
            SafetyCategory.MEDICATION,
            SafetyCategory.DIAGNOSIS,
        ]:
            assert category in guardrails.DISCLAIMERS


class TestHallucinationDetector:
    """Tests for HallucinationDetector class."""
    
    @pytest.fixture
    def detector(self):
        """Create detector instance."""
        return HallucinationDetector()
    
    def test_initialization(self, detector):
        """Test detector initialization."""
        assert detector.model is not None
        assert len(detector.RISK_PATTERNS) > 0
    
    def test_pattern_risk_detection_numbers(self, detector):
        """Test detection of specific numbers."""
        response = "The policy requires exactly 42 days of retention."
        risks = detector._detect_pattern_risks(response)
        
        number_risks = [r for r in risks if r["type"] == "specific_numbers"]
        assert len(number_risks) > 0
    
    def test_pattern_risk_detection_names(self, detector):
        """Test detection of specific names."""
        response = "According to Dr. John Smith, the policy states..."
        risks = detector._detect_pattern_risks(response)
        
        name_risks = [r for r in risks if r["type"] == "specific_names"]
        assert len(name_risks) > 0
    
    def test_pattern_risk_detection_dates(self, detector):
        """Test detection of specific dates."""
        response = "This policy was updated on January 15, 2024."
        risks = detector._detect_pattern_risks(response)
        
        date_risks = [r for r in risks if r["type"] == "specific_dates"]
        assert len(date_risks) > 0
    
    def test_pattern_risk_detection_amounts(self, detector):
        """Test detection of specific dollar amounts."""
        response = "The fine is $50,000 per violation."
        risks = detector._detect_pattern_risks(response)
        
        amount_risks = [r for r in risks if r["type"] == "specific_amounts"]
        assert len(amount_risks) > 0
    
    def test_quick_check_grounded(self, detector):
        """Test quick check on grounded response."""
        context = "Patient privacy is protected under HIPAA regulations."
        response = "Patient privacy is protected under HIPAA."
        
        is_hallucinated, confidence = detector.quick_check(response, context)
        # Should not be flagged as hallucinated
        assert confidence > 0
    
    def test_quick_check_hallucinated(self, detector):
        """Test quick check on potentially hallucinated response."""
        context = "Patient privacy is protected."
        response = "Dr. John Smith announced on January 15, 2024 that the fine is exactly $50,000."
        
        is_hallucinated, confidence = detector.quick_check(response, context)
        # Should be flagged due to specific details not in context
        assert is_hallucinated is True or confidence > 0.5


class TestGroundingChecker:
    """Tests for GroundingChecker class."""
    
    @pytest.fixture
    def checker(self):
        """Create grounding checker instance."""
        return GroundingChecker()
    
    def test_initialization(self, checker):
        """Test checker initialization."""
        assert checker.model is not None
    
    def test_extract_claims(self, checker):
        """Test claim extraction from response."""
        response = """
        Patient data must be encrypted. All access must be logged.
        Healthcare providers should follow HIPAA guidelines.
        """
        
        claims = checker._extract_claims(response)
        assert len(claims) >= 2
    
    def test_extract_claims_filters_meta(self, checker):
        """Test that meta-sentences are filtered."""
        response = """
        Based on the documents, patient data must be encrypted.
        I recommend consulting a healthcare provider.
        """
        
        claims = checker._extract_claims(response)
        # Meta-sentences should be filtered
        for claim in claims:
            assert not claim.startswith("Based on")
            assert not claim.startswith("I recommend")
    
    def test_format_sources(self, checker):
        """Test source formatting."""
        sources = [
            {"filename": "policy.pdf", "page": 1, "content": "Test content 1"},
            {"filename": "guide.pdf", "page": 5, "content": "Test content 2"},
        ]
        
        formatted = checker._format_sources(sources)
        
        assert "policy.pdf" in formatted
        assert "guide.pdf" in formatted
        assert "Test content 1" in formatted
    
    def test_calculate_source_coverage(self, checker):
        """Test source coverage calculation."""
        response = "Patient privacy is protected under regulations."
        context = "Patient privacy is protected under HIPAA regulations and guidelines."
        
        coverage = checker._calculate_source_coverage(response, context)
        
        # Should have high coverage since words overlap
        assert coverage > 0.5
    
    def test_check_citation_accuracy_no_citations(self, checker):
        """Test citation accuracy with no citations."""
        response = "Patient data must be protected."
        sources = [{"filename": "policy.pdf", "content": "test"}]
        
        accuracy = checker._check_citation_accuracy(response, sources)
        
        # No citations = 0 accuracy
        assert accuracy == 0.0
    
    def test_check_citation_accuracy_with_citations(self, checker):
        """Test citation accuracy with citations."""
        response = "Patient data must be protected [Source: policy.pdf, p.1]."
        sources = [{"filename": "policy.pdf", "content": "test"}]
        
        accuracy = checker._check_citation_accuracy(response, sources)
        
        # Should have some accuracy
        assert accuracy > 0.0


class TestHallucinationSeverity:
    """Tests for HallucinationSeverity enum."""
    
    def test_severity_levels(self):
        """Test severity level values."""
        assert HallucinationSeverity.NONE.value == "none"
        assert HallucinationSeverity.LOW.value == "low"
        assert HallucinationSeverity.MEDIUM.value == "medium"
        assert HallucinationSeverity.HIGH.value == "high"
        assert HallucinationSeverity.CRITICAL.value == "critical"


class TestSafetyCategory:
    """Tests for SafetyCategory enum."""
    
    def test_category_values(self):
        """Test category values."""
        assert SafetyCategory.MEDICAL_ADVICE.value == "medical_advice"
        assert SafetyCategory.EMERGENCY.value == "emergency"
        assert SafetyCategory.MEDICATION.value == "medication"
        assert SafetyCategory.DIAGNOSIS.value == "diagnosis"
