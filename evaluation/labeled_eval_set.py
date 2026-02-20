"""
Labeled Evaluation Set for IR Metrics.

Creates 100 queries with labeled relevant chunks for computing:
- Recall@k (k=1,3,5,10)
- MRR (Mean Reciprocal Rank)
- nDCG@k
- Precision@k
"""

import json
import random
from typing import List, Dict, Any
from pathlib import Path


# Query templates with expected topics/keywords
EVAL_QUERIES = [
    # Patient Privacy (10 queries)
    {"query": "What are the HIPAA requirements for patient data protection?", "topics": ["patient_privacy", "hipaa_compliance"], "keywords": ["HIPAA", "PHI", "protected health information"]},
    {"query": "How should staff handle patient confidentiality?", "topics": ["patient_privacy"], "keywords": ["confidentiality", "privacy", "patient"]},
    {"query": "What is the minimum necessary standard for PHI disclosure?", "topics": ["patient_privacy", "hipaa_compliance"], "keywords": ["minimum necessary", "PHI", "disclosure"]},
    {"query": "When is patient consent required for information sharing?", "topics": ["patient_privacy", "consent_procedures"], "keywords": ["consent", "sharing", "patient"]},
    {"query": "What are the breach notification requirements?", "topics": ["patient_privacy", "hipaa_compliance"], "keywords": ["breach", "notification", "72 hours"]},
    {"query": "How long must audit trails be maintained?", "topics": ["patient_privacy", "clinical_documentation"], "keywords": ["audit", "6 years", "records"]},
    {"query": "What encryption standards apply to electronic records?", "topics": ["patient_privacy"], "keywords": ["encryption", "electronic", "records"]},
    {"query": "How should paper records be stored securely?", "topics": ["patient_privacy"], "keywords": ["paper", "locked", "cabinets"]},
    {"query": "What training is required for HIPAA compliance?", "topics": ["patient_privacy", "staff_training"], "keywords": ["training", "annual", "HIPAA"]},
    {"query": "How should suspected breaches be reported?", "topics": ["patient_privacy"], "keywords": ["breach", "report", "immediately"]},
    
    # Medication Administration (10 queries)
    {"query": "What are the five rights of medication administration?", "topics": ["medication_administration"], "keywords": ["five rights", "patient", "medication", "dose", "route", "time"]},
    {"query": "How should patient identity be verified before giving medication?", "topics": ["medication_administration"], "keywords": ["verify", "two identifiers", "patient"]},
    {"query": "What are considered high-alert medications?", "topics": ["medication_administration"], "keywords": ["high-alert", "anticoagulants", "insulin", "opioids"]},
    {"query": "How quickly should medication administration be documented?", "topics": ["medication_administration", "clinical_documentation"], "keywords": ["document", "30 minutes", "administration"]},
    {"query": "What should be done if a patient refuses medication?", "topics": ["medication_administration"], "keywords": ["refused", "held", "reason"]},
    {"query": "How should medication errors be reported?", "topics": ["medication_administration", "patient_safety"], "keywords": ["error", "report", "incident"]},
    {"query": "What is medication reconciliation?", "topics": ["medication_administration"], "keywords": ["reconciliation", "accurate", "medication"]},
    {"query": "How should chemotherapy agents be handled?", "topics": ["medication_administration"], "keywords": ["chemotherapy", "high-alert"]},
    {"query": "What is the procedure for administering controlled substances?", "topics": ["medication_administration"], "keywords": ["controlled", "narcotics", "opioids"]},
    {"query": "How should adverse drug reactions be documented?", "topics": ["medication_administration", "clinical_documentation"], "keywords": ["adverse", "reaction", "document"]},
    
    # Emergency Procedures (10 queries)
    {"query": "What is the Code Blue response protocol?", "topics": ["emergency_procedures"], "keywords": ["Code Blue", "cardiac arrest", "CPR"]},
    {"query": "How quickly should CPR be initiated?", "topics": ["emergency_procedures"], "keywords": ["CPR", "30 seconds", "cardiac"]},
    {"query": "What does RACE stand for in fire emergencies?", "topics": ["emergency_procedures"], "keywords": ["RACE", "Rescue", "Alarm", "Contain", "fire"]},
    {"query": "When should the Rapid Response Team be activated?", "topics": ["emergency_procedures"], "keywords": ["Rapid Response", "acute change", "condition"]},
    {"query": "What is the response time for the Rapid Response Team?", "topics": ["emergency_procedures"], "keywords": ["5 minutes", "response time"]},
    {"query": "How should mass casualty incidents be handled?", "topics": ["emergency_procedures"], "keywords": ["mass casualty", "triage", "START protocol"]},
    {"query": "What is the incident command structure?", "topics": ["emergency_procedures"], "keywords": ["incident command", "structure"]},
    {"query": "Should elevators be used during fire evacuation?", "topics": ["emergency_procedures"], "keywords": ["elevator", "fire", "evacuation"]},
    {"query": "How should resuscitation efforts be documented?", "topics": ["emergency_procedures", "clinical_documentation"], "keywords": ["resuscitation", "document", "real-time"]},
    {"query": "What is the Code Red procedure?", "topics": ["emergency_procedures"], "keywords": ["Code Red", "fire"]},
    
    # Infection Control (10 queries)
    {"query": "What are standard precautions for infection control?", "topics": ["infection_control"], "keywords": ["standard precautions", "hand hygiene", "PPE"]},
    {"query": "What is the target hand hygiene compliance rate?", "topics": ["infection_control"], "keywords": ["95%", "hand hygiene", "compliance"]},
    {"query": "How long should hand washing last?", "topics": ["infection_control"], "keywords": ["20 seconds", "hand", "washing"]},
    {"query": "What PPE is required for contact precautions?", "topics": ["infection_control"], "keywords": ["contact precautions", "gown", "gloves"]},
    {"query": "When should an N95 respirator be used?", "topics": ["infection_control"], "keywords": ["N95", "airborne", "respirator"]},
    {"query": "What is CLABSI?", "topics": ["infection_control"], "keywords": ["CLABSI", "central line", "bloodstream"]},
    {"query": "How should negative pressure rooms be used?", "topics": ["infection_control"], "keywords": ["negative pressure", "airborne"]},
    {"query": "What are transmission-based precautions?", "topics": ["infection_control"], "keywords": ["transmission", "contact", "droplet", "airborne"]},
    {"query": "How often should hand hygiene compliance be reported?", "topics": ["infection_control"], "keywords": ["monthly", "compliance", "report"]},
    {"query": "What is VAP prevention?", "topics": ["infection_control"], "keywords": ["VAP", "ventilator", "pneumonia"]},
    
    # Clinical Documentation (10 queries)
    {"query": "What are the principles of clinical documentation?", "topics": ["clinical_documentation"], "keywords": ["timely", "accurate", "complete", "legible"]},
    {"query": "What elements are required in clinical documentation?", "topics": ["clinical_documentation"], "keywords": ["patient identification", "date", "time", "signature"]},
    {"query": "What practices are prohibited in medical records?", "topics": ["clinical_documentation"], "keywords": ["altering", "pre-dating", "blank spaces"]},
    {"query": "How should corrections be made to medical records?", "topics": ["clinical_documentation"], "keywords": ["correction", "proper", "procedures"]},
    {"query": "What are the risks of copy-paste in EHR?", "topics": ["clinical_documentation"], "keywords": ["copy-paste", "review", "EHR"]},
    {"query": "Who should sign clinical documentation?", "topics": ["clinical_documentation"], "keywords": ["signature", "credentials", "provider"]},
    {"query": "What abbreviations are approved for use?", "topics": ["clinical_documentation"], "keywords": ["abbreviations", "unapproved"]},
    {"query": "How should auto-populated information be handled?", "topics": ["clinical_documentation"], "keywords": ["auto-populated", "review", "accuracy"]},
    {"query": "What is the SOAP note format?", "topics": ["clinical_documentation"], "keywords": ["SOAP", "assessment", "plan"]},
    {"query": "How should verbal orders be documented?", "topics": ["clinical_documentation"], "keywords": ["verbal", "orders", "document"]},
    
    # Additional topics (50 more queries)
    {"query": "What are patient safety protocols?", "topics": ["patient_safety"], "keywords": ["safety", "protocols", "patient"]},
    {"query": "How is quality improvement measured?", "topics": ["quality_improvement"], "keywords": ["quality", "improvement", "measured"]},
    {"query": "What training is required for new staff?", "topics": ["staff_training"], "keywords": ["training", "new staff", "required"]},
    {"query": "What is the consent process for procedures?", "topics": ["consent_procedures"], "keywords": ["consent", "procedures", "process"]},
    {"query": "How should discharge planning be conducted?", "topics": ["discharge_planning"], "keywords": ["discharge", "planning"]},
    {"query": "What are pain management guidelines?", "topics": ["pain_management"], "keywords": ["pain", "management", "guidelines"]},
    {"query": "How can falls be prevented?", "topics": ["fall_prevention"], "keywords": ["fall", "prevention"]},
    {"query": "What are hand hygiene best practices?", "topics": ["hand_hygiene", "infection_control"], "keywords": ["hand hygiene", "best practices"]},
    {"query": "When should isolation protocols be used?", "topics": ["isolation_protocols", "infection_control"], "keywords": ["isolation", "protocols"]},
    {"query": "What are blood transfusion safety procedures?", "topics": ["blood_transfusion"], "keywords": ["blood", "transfusion", "safety"]},
    {"query": "What is the surgical safety checklist?", "topics": ["surgical_safety"], "keywords": ["surgical", "safety", "checklist"]},
    {"query": "How should radiology procedures be performed safely?", "topics": ["radiology_procedures"], "keywords": ["radiology", "procedures", "safely"]},
    {"query": "What are laboratory specimen handling protocols?", "topics": ["laboratory_protocols"], "keywords": ["laboratory", "specimen", "handling"]},
    {"query": "What are pharmacy dispensing guidelines?", "topics": ["pharmacy_guidelines"], "keywords": ["pharmacy", "dispensing", "guidelines"]},
    {"query": "How should nutrition services be provided?", "topics": ["nutrition_services"], "keywords": ["nutrition", "services"]},
    {"query": "What rehabilitation therapy options are available?", "topics": ["rehabilitation_therapy"], "keywords": ["rehabilitation", "therapy"]},
    {"query": "How should mental health patients be assessed?", "topics": ["mental_health"], "keywords": ["mental health", "assessed"]},
    {"query": "What are pediatric care considerations?", "topics": ["pediatric_care"], "keywords": ["pediatric", "care", "children"]},
    {"query": "What special considerations apply to geriatric patients?", "topics": ["geriatric_care"], "keywords": ["geriatric", "elderly"]},
    {"query": "What are oncology treatment protocols?", "topics": ["oncology_protocols"], "keywords": ["oncology", "treatment", "cancer"]},
    {"query": "How should cardiac emergencies be managed?", "topics": ["cardiology_procedures", "emergency_procedures"], "keywords": ["cardiac", "emergency"]},
    {"query": "What respiratory care procedures are standard?", "topics": ["respiratory_care"], "keywords": ["respiratory", "care"]},
    {"query": "How should wounds be assessed and treated?", "topics": ["wound_care"], "keywords": ["wound", "care", "treatment"]},
    {"query": "What is the diabetes management protocol?", "topics": ["diabetes_management"], "keywords": ["diabetes", "management", "glucose"]},
    {"query": "How should insulin be administered safely?", "topics": ["medication_administration", "diabetes_management"], "keywords": ["insulin", "safely", "high-alert"]},
]


def create_labeled_eval_set(
    chunks_file: str = "data/synthetic_corpus/chunks.json",
    output_file: str = "evaluation/labeled_queries.json"
) -> Dict[str, Any]:
    """Create labeled evaluation set by matching queries to relevant chunks."""
    
    # Load chunks
    with open(chunks_file, "r") as f:
        chunks = json.load(f)
    
    print(f"Loaded {len(chunks)} chunks")
    
    # Create chunk index by topic
    topic_index = {}
    for i, chunk in enumerate(chunks):
        topic = chunk.get("topic", "unknown")
        if topic not in topic_index:
            topic_index[topic] = []
        topic_index[topic].append(i)
    
    # Label queries with relevant chunks
    labeled_queries = []
    
    for q in EVAL_QUERIES:
        query_data = {
            "query_id": f"Q{len(labeled_queries) + 1:03d}",
            "query": q["query"],
            "topics": q["topics"],
            "keywords": q["keywords"],
            "relevant_chunks": [],
            "relevance_scores": []
        }
        
        # Find relevant chunks by topic
        for topic in q["topics"]:
            if topic in topic_index:
                # Get chunks from this topic
                topic_chunks = topic_index[topic]
                
                # Score by keyword matches
                for chunk_idx in topic_chunks:
                    chunk = chunks[chunk_idx]
                    content_lower = chunk["content"].lower()
                    
                    # Count keyword matches
                    matches = sum(1 for kw in q["keywords"] if kw.lower() in content_lower)
                    
                    if matches > 0:
                        # Relevance score: 3 = highly relevant, 2 = relevant, 1 = marginally relevant
                        relevance = min(3, matches)
                        
                        query_data["relevant_chunks"].append(chunk["chunk_id"])
                        query_data["relevance_scores"].append(relevance)
        
        # Limit to top 20 relevant chunks per query
        if len(query_data["relevant_chunks"]) > 20:
            # Sort by relevance and keep top 20
            pairs = list(zip(query_data["relevant_chunks"], query_data["relevance_scores"]))
            pairs.sort(key=lambda x: x[1], reverse=True)
            pairs = pairs[:20]
            query_data["relevant_chunks"] = [p[0] for p in pairs]
            query_data["relevance_scores"] = [p[1] for p in pairs]
        
        labeled_queries.append(query_data)
    
    # Calculate statistics
    stats = {
        "num_queries": len(labeled_queries),
        "queries_with_relevance": sum(1 for q in labeled_queries if q["relevant_chunks"]),
        "avg_relevant_per_query": sum(len(q["relevant_chunks"]) for q in labeled_queries) / len(labeled_queries),
        "total_relevance_judgments": sum(len(q["relevant_chunks"]) for q in labeled_queries)
    }
    
    # Save labeled queries
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump({
            "metadata": stats,
            "queries": labeled_queries
        }, f, indent=2)
    
    print(f"\n✅ Created labeled evaluation set:")
    print(f"   Queries: {stats['num_queries']}")
    print(f"   Queries with relevance: {stats['queries_with_relevance']}")
    print(f"   Avg relevant chunks per query: {stats['avg_relevant_per_query']:.1f}")
    print(f"   Total relevance judgments: {stats['total_relevance_judgments']}")
    print(f"   Output: {output_path}")
    
    return stats


if __name__ == "__main__":
    create_labeled_eval_set()
