"""
Generate Synthetic Healthcare Corpus for Large-Scale RAG Evaluation.

Targets:
- 100-1,000 synthetic documents
- 10,000-100,000 chunks
- Realistic healthcare policy content

Reports:
- Index build time
- Index size (MB/GB)
- p50/p95 retrieval latency
- Memory usage
"""

import os
import sys
import json
import random
import hashlib
import time
from datetime import datetime, timezone
from typing import List, Dict, Any
from pathlib import Path

# Healthcare topics and templates for synthetic generation
HEALTHCARE_TOPICS = [
    "patient_privacy", "hipaa_compliance", "clinical_documentation",
    "emergency_procedures", "medication_administration", "infection_control",
    "patient_safety", "quality_improvement", "staff_training",
    "consent_procedures", "discharge_planning", "pain_management",
    "fall_prevention", "hand_hygiene", "isolation_protocols",
    "blood_transfusion", "surgical_safety", "radiology_procedures",
    "laboratory_protocols", "pharmacy_guidelines", "nutrition_services",
    "rehabilitation_therapy", "mental_health", "pediatric_care",
    "geriatric_care", "oncology_protocols", "cardiology_procedures",
    "respiratory_care", "wound_care", "diabetes_management"
]

POLICY_SECTIONS = [
    "Purpose", "Scope", "Definitions", "Policy Statement", "Procedures",
    "Responsibilities", "Documentation Requirements", "Training Requirements",
    "Compliance Monitoring", "References", "Revision History"
]

POLICY_TEMPLATES = {
    "patient_privacy": """
## {section}

### Patient Privacy and Confidentiality

All healthcare providers must maintain strict confidentiality of patient information 
in accordance with HIPAA regulations and organizational policies.

**Key Requirements:**
- Protected Health Information (PHI) must be secured at all times
- Access to patient records is limited to authorized personnel only
- Minimum necessary standard applies to all PHI disclosures
- Patient consent is required for most uses and disclosures
- Breach notification procedures must be followed within 72 hours

**Documentation Standards:**
- All access to patient records must be logged
- Audit trails must be maintained for a minimum of 6 years
- Electronic records must use encryption and access controls
- Paper records must be stored in locked cabinets

**Staff Responsibilities:**
- Complete annual HIPAA training
- Report suspected breaches immediately
- Use secure communication channels for PHI
- Verify patient identity before disclosing information
""",
    
    "medication_administration": """
## {section}

### Medication Administration Guidelines

Safe medication administration is critical to patient safety and requires 
adherence to the "Five Rights" of medication administration.

**The Five Rights:**
1. Right Patient - Verify using two identifiers
2. Right Medication - Check against order and label
3. Right Dose - Calculate and verify dosage
4. Right Route - Confirm administration method
5. Right Time - Administer within scheduled window

**High-Alert Medications:**
- Anticoagulants (heparin, warfarin)
- Insulin
- Opioids and narcotics
- Chemotherapy agents
- Concentrated electrolytes

**Documentation Requirements:**
- Document administration within 30 minutes
- Record patient response and any adverse reactions
- Note any refused or held medications with reason
- Maintain accurate medication reconciliation

**Error Reporting:**
- Report all medication errors immediately
- Complete incident report within 24 hours
- Participate in root cause analysis
- Implement corrective actions
""",

    "emergency_procedures": """
## {section}

### Emergency Response Procedures

All staff must be prepared to respond to medical emergencies following 
established protocols and chain of command.

**Code Blue (Cardiac Arrest):**
- Activate emergency response system immediately
- Begin CPR within 30 seconds
- Apply AED as soon as available
- Document resuscitation efforts in real-time

**Code Red (Fire):**
- RACE: Rescue, Alarm, Contain, Extinguish/Evacuate
- Close doors to contain smoke and fire
- Do not use elevators during evacuation
- Account for all patients and staff

**Rapid Response Team:**
- Criteria for activation: acute change in condition
- Response time: within 5 minutes
- Team composition: physician, nurse, respiratory therapist
- Documentation: complete within 1 hour

**Mass Casualty Incident:**
- Implement incident command structure
- Triage patients using START protocol
- Activate surge capacity plan
- Coordinate with external agencies
""",

    "infection_control": """
## {section}

### Infection Prevention and Control

Preventing healthcare-associated infections (HAIs) is a priority requiring 
adherence to evidence-based practices and standard precautions.

**Standard Precautions:**
- Hand hygiene before and after patient contact
- Use of personal protective equipment (PPE)
- Safe injection practices
- Respiratory hygiene and cough etiquette
- Environmental cleaning and disinfection

**Transmission-Based Precautions:**
- Contact precautions: gown and gloves
- Droplet precautions: surgical mask
- Airborne precautions: N95 respirator, negative pressure room

**Hand Hygiene Compliance:**
- Target compliance rate: >95%
- Use alcohol-based hand rub or soap and water
- Duration: minimum 20 seconds
- Monitor and report compliance rates monthly

**Device-Related Infections:**
- Central line-associated bloodstream infections (CLABSI)
- Catheter-associated urinary tract infections (CAUTI)
- Ventilator-associated pneumonia (VAP)
- Surgical site infections (SSI)
""",

    "clinical_documentation": """
## {section}

### Clinical Documentation Standards

Accurate and complete clinical documentation is essential for patient care, 
legal protection, and reimbursement.

**Documentation Principles:**
- Timely: Document as close to the event as possible
- Accurate: Record facts objectively
- Complete: Include all relevant information
- Legible: Use clear, readable writing or typing
- Authenticated: Sign and date all entries

**Required Elements:**
- Patient identification on every page
- Date and time of entry
- Chief complaint and history
- Physical examination findings
- Assessment and plan
- Provider signature and credentials

**Prohibited Practices:**
- Altering records without proper correction procedures
- Pre-dating or post-dating entries
- Using unapproved abbreviations
- Leaving blank spaces in records
- Documenting for another provider

**Electronic Health Records:**
- Use structured data entry when available
- Avoid copy-paste without review
- Complete required fields before signing
- Review auto-populated information for accuracy
"""
}

def generate_policy_document(topic: str, doc_id: int) -> Dict[str, Any]:
    """Generate a synthetic healthcare policy document."""
    
    # Get template or use generic
    template = POLICY_TEMPLATES.get(topic, POLICY_TEMPLATES["patient_privacy"])
    
    # Generate document content
    sections = []
    for section in random.sample(POLICY_SECTIONS, k=random.randint(5, 8)):
        content = template.format(section=section)
        # Add some variation
        content = content.replace("healthcare", random.choice(["healthcare", "medical", "clinical"]))
        sections.append({
            "section": section,
            "content": content
        })
    
    # Create document metadata
    doc = {
        "doc_id": f"DOC-{doc_id:05d}",
        "title": f"{topic.replace('_', ' ').title()} Policy - Version {random.randint(1, 5)}.{random.randint(0, 9)}",
        "topic": topic,
        "department": random.choice(["Nursing", "Medical Staff", "Administration", "Quality", "Compliance"]),
        "effective_date": f"202{random.randint(0, 4)}-{random.randint(1, 12):02d}-01",
        "review_date": f"202{random.randint(5, 6)}-{random.randint(1, 12):02d}-01",
        "sections": sections,
        "full_text": "\n\n".join([s["content"] for s in sections]),
        "word_count": sum(len(s["content"].split()) for s in sections)
    }
    
    return doc


def chunk_document(doc: Dict[str, Any], chunk_size: int = 512, overlap: int = 100) -> List[Dict[str, Any]]:
    """Chunk document with overlap and metadata preservation."""
    
    chunks = []
    text = doc["full_text"]
    words = text.split()
    
    start = 0
    chunk_idx = 0
    
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_text = " ".join(words[start:end])
        
        # Create chunk with metadata
        chunk = {
            "chunk_id": f"{doc['doc_id']}_chunk_{chunk_idx:03d}",
            "doc_id": doc["doc_id"],
            "doc_title": doc["title"],
            "topic": doc["topic"],
            "department": doc["department"],
            "chunk_index": chunk_idx,
            "content": chunk_text,
            "word_count": len(chunk_text.split()),
            "start_word": start,
            "end_word": end
        }
        
        chunks.append(chunk)
        chunk_idx += 1
        
        # Move start with overlap
        start = end - overlap if end < len(words) else len(words)
    
    return chunks


def generate_corpus(
    num_documents: int = 500,
    chunk_size: int = 512,
    overlap: int = 100,
    output_dir: str = "data/synthetic_corpus"
) -> Dict[str, Any]:
    """Generate full synthetic corpus."""
    
    print(f"Generating {num_documents} synthetic healthcare documents...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_documents = []
    all_chunks = []
    
    start_time = time.time()
    
    for i in range(num_documents):
        topic = random.choice(HEALTHCARE_TOPICS)
        doc = generate_policy_document(topic, i + 1)
        all_documents.append(doc)
        
        chunks = chunk_document(doc, chunk_size, overlap)
        all_chunks.extend(chunks)
        
        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{num_documents} documents, {len(all_chunks)} chunks")
    
    generation_time = time.time() - start_time
    
    # Save documents
    with open(output_path / "documents.json", "w") as f:
        json.dump(all_documents, f, indent=2)
    
    # Save chunks
    with open(output_path / "chunks.json", "w") as f:
        json.dump(all_chunks, f, indent=2)
    
    # Calculate statistics
    stats = {
        "num_documents": len(all_documents),
        "num_chunks": len(all_chunks),
        "avg_chunks_per_doc": len(all_chunks) / len(all_documents),
        "total_words": sum(c["word_count"] for c in all_chunks),
        "generation_time_seconds": round(generation_time, 2),
        "chunk_size": chunk_size,
        "overlap": overlap,
        "topics_covered": list(set(d["topic"] for d in all_documents)),
        "output_dir": str(output_path)
    }
    
    # Save stats
    with open(output_path / "corpus_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n✅ Corpus generated:")
    print(f"   Documents: {stats['num_documents']}")
    print(f"   Chunks: {stats['num_chunks']}")
    print(f"   Total words: {stats['total_words']:,}")
    print(f"   Generation time: {stats['generation_time_seconds']}s")
    print(f"   Output: {output_path}")
    
    return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic healthcare corpus")
    parser.add_argument("--num-docs", type=int, default=500, help="Number of documents")
    parser.add_argument("--chunk-size", type=int, default=512, help="Chunk size in words")
    parser.add_argument("--overlap", type=int, default=100, help="Overlap between chunks")
    parser.add_argument("--output", type=str, default="data/synthetic_corpus", help="Output directory")
    
    args = parser.parse_args()
    
    stats = generate_corpus(
        num_documents=args.num_docs,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        output_dir=args.output
    )
    
    print("\n📊 Corpus Statistics:")
    print(json.dumps(stats, indent=2))
