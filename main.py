"""Main CLI entry point for Healthcare AI Assistant."""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def ingest_command(args):
    """Handle document ingestion."""
    from src.ingestion import PDFLoader, TextChunker
    from src.vectordb import FAISSVectorStore
    from src.vectordb.faiss_store import EmbeddingModel
    
    pdf_dir = Path(args.pdf_dir)
    output_dir = Path(args.output_dir)
    
    print(f"Loading PDFs from: {pdf_dir}")
    
    # Load PDFs
    loader = PDFLoader(use_pdfplumber=True)
    documents = loader.load_directory(str(pdf_dir))
    
    if not documents:
        print("No documents found. Please add PDFs to the data/pdfs directory.")
        return
    
    # Chunk documents
    chunker = TextChunker(
        chunk_size=int(os.getenv("CHUNK_SIZE", 1000)),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 200))
    )
    chunks = chunker.chunk_documents(documents)
    
    # Create vector store
    use_openai = args.use_openai_embeddings
    if use_openai:
        embedding_model = EmbeddingModel(
            model_name=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            use_openai=True
        )
    else:
        embedding_model = EmbeddingModel(model_name="all-MiniLM-L6-v2")
    
    vector_store = FAISSVectorStore(embedding_model=embedding_model)
    vector_store.add_chunks(chunks)
    
    # Save vector store
    vector_store.save(str(output_dir))
    print(f"Vector store saved to: {output_dir}")
    print(f"Stats: {vector_store.get_stats()}")


def query_command(args):
    """Handle Q&A queries."""
    from src.vectordb import FAISSVectorStore
    from src.vectordb.faiss_store import EmbeddingModel
    from src.llm import HealthcareQAChain
    from src.responsible_ai import SafetyGuardrails, HallucinationDetector
    
    vectorstore_dir = Path(args.vectorstore_dir)
    
    if not vectorstore_dir.exists():
        print(f"Vector store not found at {vectorstore_dir}")
        print("Please run 'python main.py ingest' first.")
        return
    
    # Load vector store
    use_openai = args.use_openai_embeddings
    if use_openai:
        embedding_model = EmbeddingModel(
            model_name=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            use_openai=True
        )
    else:
        embedding_model = EmbeddingModel(model_name="all-MiniLM-L6-v2")
    
    vector_store = FAISSVectorStore.load(str(vectorstore_dir), embedding_model)
    
    # Create Q&A chain
    qa_chain = HealthcareQAChain(
        vector_store=vector_store,
        model=os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview"),
        top_k=int(os.getenv("TOP_K_RESULTS", 5))
    )
    
    # Safety check
    guardrails = SafetyGuardrails()
    is_safe, warnings = guardrails.check_question_safety(args.question)
    
    if warnings:
        for warning in warnings:
            print(f"⚠️  {warning}")
    
    if not is_safe:
        print("This question cannot be safely answered by this system.")
        return
    
    # Run query
    print(f"\nQuestion: {args.question}")
    print("-" * 50)
    
    response = qa_chain.query(
        args.question,
        template_name=args.template
    )
    
    print(f"\nAnswer:\n{response.answer}")
    
    # Show sources
    print(f"\n📚 Sources ({len(response.sources)}):")
    for source in response.sources:
        print(f"  - {source['filename']}, Page {source.get('page', '?')} "
              f"(relevance: {source.get('relevance_score', 'N/A')})")
    
    # Safety check on response
    if args.safety_check:
        print("\n🛡️  Running safety check...")
        safety_result = guardrails.check_response(response.answer, args.question)
        print(f"  Risk Level: {safety_result.risk_level}")
        if safety_result.warnings:
            for warning in safety_result.warnings:
                print(f"  ⚠️  {warning}")
    
    # Hallucination check
    if args.hallucination_check:
        print("\n🔍 Running hallucination check...")
        detector = HallucinationDetector()
        context = "\n".join([s.get("content_preview", "") for s in response.sources])
        is_hallucinated, confidence = detector.quick_check(response.answer, context)
        print(f"  Potential hallucination: {'Yes' if is_hallucinated else 'No'} "
              f"(confidence: {confidence:.2f})")


def evaluate_command(args):
    """Run evaluation experiments."""
    from src.vectordb import FAISSVectorStore
    from src.vectordb.faiss_store import EmbeddingModel
    from src.llm import HealthcareQAChain
    from src.evaluation import ExperimentRunner, PromptExperiment
    from src.evaluation.experiments import (
        HEALTHCARE_TEST_CASES,
        create_hallucination_test_experiment
    )
    
    vectorstore_dir = Path(args.vectorstore_dir)
    
    if not vectorstore_dir.exists():
        print(f"Vector store not found at {vectorstore_dir}")
        return
    
    # Load components
    use_openai = args.use_openai_embeddings
    if use_openai:
        embedding_model = EmbeddingModel(
            model_name=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            use_openai=True
        )
    else:
        embedding_model = EmbeddingModel(model_name="all-MiniLM-L6-v2")
    
    vector_store = FAISSVectorStore.load(str(vectorstore_dir), embedding_model)
    
    qa_chain = HealthcareQAChain(
        vector_store=vector_store,
        model=os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")
    )
    
    runner = ExperimentRunner(qa_chain, output_dir=args.output_dir)
    
    # Select experiment
    if args.experiment == "hallucination_test":
        experiment = create_hallucination_test_experiment()
    elif args.experiment == "template_comparison":
        experiment = PromptExperiment(
            name="template_comparison",
            description="Compare all prompt templates",
            templates_to_test=["grounded_qa", "strict_citation", "conversational", 
                              "analytical", "safety_first"],
            test_cases=HEALTHCARE_TEST_CASES[:args.num_cases]
        )
    else:
        print(f"Unknown experiment: {args.experiment}")
        print("Available: hallucination_test, template_comparison")
        return
    
    # Run experiment
    runner.run_experiment(experiment)


def interactive_command(args):
    """Start interactive Q&A session."""
    from src.vectordb import FAISSVectorStore
    from src.vectordb.faiss_store import EmbeddingModel
    from src.llm import HealthcareQAChain
    from src.responsible_ai import SafetyGuardrails
    
    vectorstore_dir = Path(args.vectorstore_dir)
    
    if not vectorstore_dir.exists():
        print(f"Vector store not found at {vectorstore_dir}")
        return
    
    # Load components
    use_openai = args.use_openai_embeddings
    if use_openai:
        embedding_model = EmbeddingModel(
            model_name=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            use_openai=True
        )
    else:
        embedding_model = EmbeddingModel(model_name="all-MiniLM-L6-v2")
    
    vector_store = FAISSVectorStore.load(str(vectorstore_dir), embedding_model)
    
    qa_chain = HealthcareQAChain(
        vector_store=vector_store,
        model=os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")
    )
    
    guardrails = SafetyGuardrails()
    
    print("\n" + "="*60)
    print("Healthcare Policy Assistant - Interactive Mode")
    print("="*60)
    print("Type 'quit' or 'exit' to end the session.")
    print("Type 'help' for available commands.")
    print("="*60 + "\n")
    
    conversation_history = []
    
    while True:
        try:
            question = input("\n📝 Your question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break
        
        if not question:
            continue
        
        if question.lower() in ["quit", "exit"]:
            print("Goodbye!")
            break
        
        if question.lower() == "help":
            print("\nCommands:")
            print("  quit/exit - End session")
            print("  clear - Clear conversation history")
            print("  sources - Show last response sources")
            print("  template <name> - Change prompt template")
            continue
        
        if question.lower() == "clear":
            conversation_history = []
            print("Conversation history cleared.")
            continue
        
        # Safety check
        is_safe, warnings = guardrails.check_question_safety(question)
        for warning in warnings:
            print(f"⚠️  {warning}")
        
        if not is_safe:
            continue
        
        # Query with conversation context
        response = qa_chain.query_with_followup(
            question,
            conversation_history,
            template_name=args.template
        )
        
        print(f"\n💬 Answer:\n{response.answer}")
        
        # Update history
        conversation_history.append({"role": "user", "content": question})
        conversation_history.append({"role": "assistant", "content": response.answer})
        
        # Keep history manageable
        if len(conversation_history) > 10:
            conversation_history = conversation_history[-10:]


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Healthcare AI Assistant - RAG-based policy Q&A"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest PDF documents")
    ingest_parser.add_argument(
        "--pdf-dir", 
        default="data/pdfs",
        help="Directory containing PDF files"
    )
    ingest_parser.add_argument(
        "--output-dir",
        default="data/vectorstore",
        help="Output directory for vector store"
    )
    ingest_parser.add_argument(
        "--use-openai-embeddings",
        action="store_true",
        help="Use OpenAI embeddings instead of local model"
    )
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query the assistant")
    query_parser.add_argument("question", help="Question to ask")
    query_parser.add_argument(
        "--vectorstore-dir",
        default="data/vectorstore",
        help="Vector store directory"
    )
    query_parser.add_argument(
        "--template",
        default="grounded_qa",
        choices=["grounded_qa", "strict_citation", "conversational", 
                 "analytical", "safety_first"],
        help="Prompt template to use"
    )
    query_parser.add_argument(
        "--safety-check",
        action="store_true",
        help="Run safety check on response"
    )
    query_parser.add_argument(
        "--hallucination-check",
        action="store_true",
        help="Run hallucination detection"
    )
    query_parser.add_argument(
        "--use-openai-embeddings",
        action="store_true",
        help="Use OpenAI embeddings"
    )
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Run evaluation experiments")
    eval_parser.add_argument(
        "--experiment",
        default="template_comparison",
        help="Experiment to run"
    )
    eval_parser.add_argument(
        "--vectorstore-dir",
        default="data/vectorstore",
        help="Vector store directory"
    )
    eval_parser.add_argument(
        "--output-dir",
        default="experiments",
        help="Output directory for results"
    )
    eval_parser.add_argument(
        "--num-cases",
        type=int,
        default=5,
        help="Number of test cases to run"
    )
    eval_parser.add_argument(
        "--use-openai-embeddings",
        action="store_true",
        help="Use OpenAI embeddings"
    )
    
    # Interactive command
    interactive_parser = subparsers.add_parser(
        "interactive", 
        help="Start interactive session"
    )
    interactive_parser.add_argument(
        "--vectorstore-dir",
        default="data/vectorstore",
        help="Vector store directory"
    )
    interactive_parser.add_argument(
        "--template",
        default="conversational",
        help="Prompt template to use"
    )
    interactive_parser.add_argument(
        "--use-openai-embeddings",
        action="store_true",
        help="Use OpenAI embeddings"
    )
    
    args = parser.parse_args()
    
    if args.command == "ingest":
        ingest_command(args)
    elif args.command == "query":
        query_command(args)
    elif args.command == "evaluate":
        evaluate_command(args)
    elif args.command == "interactive":
        interactive_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
