"""Streamlit demo application for Healthcare AI Assistant."""

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Page config
st.set_page_config(
    page_title="Healthcare Policy Assistant",
    page_icon="🏥",
    layout="wide"
)


@st.cache_resource
def load_qa_system():
    """Load and cache the Q&A system."""
    from src.vectordb import FAISSVectorStore
    from src.vectordb.faiss_store import EmbeddingModel
    
    vectorstore_dir = Path("data/vectorstore")
    
    if not vectorstore_dir.exists():
        return None, None, None, None
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        # Return vector store only for search demo
        embedding_model = EmbeddingModel(model_name="all-MiniLM-L6-v2")
        vector_store = FAISSVectorStore.load(str(vectorstore_dir), embedding_model)
        return None, None, None, vector_store
    
    from src.llm import HealthcareQAChain
    from src.responsible_ai import SafetyGuardrails, HallucinationDetector
    
    # Use local embeddings by default for demo
    embedding_model = EmbeddingModel(model_name="all-MiniLM-L6-v2")
    
    vector_store = FAISSVectorStore.load(str(vectorstore_dir), embedding_model)
    
    qa_chain = HealthcareQAChain(
        vector_store=vector_store,
        model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
        top_k=5
    )
    
    guardrails = SafetyGuardrails()
    hallucination_detector = HallucinationDetector()
    
    return qa_chain, guardrails, hallucination_detector, vector_store


def main():
    # Header
    st.title("🏥 Healthcare Policy Assistant")
    st.markdown("""
    A retrieval-augmented generation (RAG) system for healthcare policy document Q&A 
    with built-in hallucination detection and safety guardrails.
    """)
    
    # Load system
    qa_chain, guardrails, hallucination_detector, vector_store = load_qa_system()
    
    if vector_store is None:
        st.error("""
        ⚠️ Vector store not found. Please run document ingestion first:
        ```bash
        python main.py ingest --pdf-dir data/pdfs
        ```
        """)
    elif qa_chain is None:
        st.warning("""
        ⚠️ OpenAI API key not set. Set `OPENAI_API_KEY` in your `.env` file for full Q&A functionality.
        
        **Search Demo Mode**: You can still test document retrieval below.
        """)
        
        # Show search-only demo
        st.subheader("🔍 Document Search Demo")
        search_query = st.text_input("Search documents:", placeholder="e.g., patient privacy")
        
        if search_query:
            results = vector_store.search(search_query, top_k=3)
            st.markdown("### Results")
            for i, (chunk, score) in enumerate(results, 1):
                with st.expander(f"Result {i} - Score: {score:.3f}"):
                    st.markdown(f"**Source:** {chunk.metadata.get('filename', 'Unknown')}, Page {chunk.metadata.get('page', '?')}")
                    st.markdown(chunk.content)
        return
        
        # Show upload option
        st.subheader("📄 Or Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload PDF documents",
            type=["pdf"],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            st.info(f"Uploaded {len(uploaded_files)} files. Run ingestion to process them.")
        
        return
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        template = st.selectbox(
            "Prompt Template",
            ["grounded_qa", "strict_citation", "conversational", "analytical", "safety_first"],
            index=0,
            help="Select the prompt strategy for responses"
        )
        
        st.markdown("---")
        
        enable_safety = st.checkbox("🛡️ Safety Check", value=True)
        enable_hallucination = st.checkbox("🔍 Hallucination Detection", value=True)
        show_sources = st.checkbox("📚 Show Sources", value=True)
        
        st.markdown("---")
        
        # Vector store stats
        if vector_store:
            stats = vector_store.get_stats()
            st.subheader("📊 Index Stats")
            st.metric("Documents Indexed", stats["num_chunks"])
            st.metric("Index Type", stats["index_type"])
        
        st.markdown("---")
        st.markdown("""
        **⚠️ Disclaimer**
        
        This is a prototype system for policy information only. 
        Do not use for medical advice or clinical decisions.
        """)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Ask a Question")
        
        # Question input
        question = st.text_area(
            "Enter your question about healthcare policies:",
            height=100,
            placeholder="e.g., What is the policy on patient data privacy?"
        )
        
        submit = st.button("🔍 Get Answer", type="primary")
        
        if submit and question:
            # Safety check on question
            if enable_safety:
                is_safe, warnings = guardrails.check_question_safety(question)
                for warning in warnings:
                    st.warning(warning)
                
                if not is_safe:
                    st.error("This question cannot be safely answered by this system.")
                    return
            
            with st.spinner("Searching documents and generating answer..."):
                # Get response
                response = qa_chain.query(question, template_name=template)
            
            # Display answer
            st.markdown("### 📝 Answer")
            st.markdown(response.answer)
            
            # Safety check on response
            if enable_safety:
                safety_result = guardrails.check_response(response.answer, question)
                
                if safety_result.risk_level in ["high", "critical"]:
                    st.error(f"⚠️ High risk response detected: {safety_result.risk_level}")
                elif safety_result.warnings:
                    for warning in safety_result.warnings:
                        st.warning(warning)
                
                if safety_result.required_disclaimers:
                    with st.expander("📋 Important Disclaimers"):
                        for disclaimer in safety_result.required_disclaimers:
                            st.info(disclaimer)
            
            # Hallucination check
            if enable_hallucination:
                context = "\n".join([s.get("content_preview", "") for s in response.sources])
                is_hallucinated, confidence = hallucination_detector.quick_check(
                    response.answer, context
                )
                
                if is_hallucinated:
                    st.warning(f"⚠️ Potential hallucination detected (confidence: {confidence:.0%})")
                else:
                    st.success(f"✅ Response appears well-grounded (confidence: {confidence:.0%})")
            
            # Show sources
            if show_sources and response.sources:
                st.markdown("### 📚 Sources")
                for i, source in enumerate(response.sources, 1):
                    with st.expander(
                        f"Source {i}: {source['filename']} (Page {source.get('page', '?')}) - "
                        f"Relevance: {source.get('relevance_score', 'N/A'):.2f}"
                    ):
                        st.markdown(source.get("content_preview", "No preview available"))
            
            # Store in session for follow-up
            if "conversation" not in st.session_state:
                st.session_state.conversation = []
            
            st.session_state.conversation.append({
                "question": question,
                "answer": response.answer,
                "sources": response.sources
            })
    
    with col2:
        st.subheader("📜 Conversation History")
        
        if "conversation" in st.session_state and st.session_state.conversation:
            for i, turn in enumerate(reversed(st.session_state.conversation[-5:])):
                with st.expander(f"Q: {turn['question'][:50]}...", expanded=(i == 0)):
                    st.markdown(f"**Q:** {turn['question']}")
                    st.markdown(f"**A:** {turn['answer'][:300]}...")
            
            if st.button("Clear History"):
                st.session_state.conversation = []
                st.rerun()
        else:
            st.info("No conversation history yet. Ask a question to get started!")
        
        # Example questions
        st.subheader("💡 Example Questions")
        examples = [
            "What is the policy on patient data privacy?",
            "How should informed consent be handled?",
            "What are the telemedicine guidelines?",
            "Explain the adverse event reporting procedure.",
        ]
        
        for example in examples:
            if st.button(example, key=f"example_{example[:20]}"):
                st.session_state.example_question = example
                st.rerun()
    
    # Handle example question selection
    if "example_question" in st.session_state:
        question = st.session_state.example_question
        del st.session_state.example_question


if __name__ == "__main__":
    main()
