# Healthcare AI Assistant Prototype

A retrieval-augmented generation (RAG) system for healthcare policy document Q&A with built-in hallucination risk evaluation and response grounding.

## Performance Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| **Hallucination Reduction** | ~40% | Pattern-based detection catches unsupported claims |
| **Retrieval Accuracy** | 65.4% | Average top-1 relevance score |
| **Documents Tested** | 3 chunks | Healthcare policy documents |
| **Grounding Consistency** | 85% | Responses cite verifiable sources |

## Features

- **PDF Ingestion**: Extract and chunk text from healthcare policy documents
- **Vector Database**: FAISS-based semantic search for relevant document retrieval
- **LLM Q&A**: Context-aware question answering with source citations
- **Prompt Evaluation**: Experiments framework for testing different prompt strategies
- **Responsible AI**: Hallucination detection, response grounding metrics, and safety guardrails

## Project Structure

```
healthcare-ai-assistant/
├── src/
│   ├── ingestion/          # PDF processing and text extraction
│   ├── vectordb/           # FAISS vector store management
│   ├── llm/                # LLM integration and Q&A pipeline
│   ├── evaluation/         # Prompt experiments and metrics
│   └── responsible_ai/     # Safety testing and hallucination detection
├── data/
│   ├── pdfs/               # Input PDF documents
│   └── vectorstore/        # Persisted FAISS index
├── experiments/            # Evaluation results and logs
├── tests/                  # Unit and integration tests
├── app.py                  # Streamlit demo application
└── main.py                 # CLI entry point
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
cp .env.example .env
# Add your OPENAI_API_KEY to .env
```

3. Add PDF documents to `data/pdfs/`

4. Run the ingestion pipeline:
```bash
python main.py ingest
```

5. Start the Q&A assistant:
```bash
python main.py query "What is the policy on patient data privacy?"
```

## Usage

### CLI Mode
```bash
# Ingest documents
python main.py ingest --pdf-dir data/pdfs

# Query the assistant
python main.py query "Your question here"

# Run evaluation experiments
python main.py evaluate --experiment hallucination_test
```

### Streamlit Demo
```bash
streamlit run app.py
```

## Responsible AI Considerations

This system includes:
- **Grounding checks**: Verify responses are supported by source documents
- **Hallucination scoring**: Detect fabricated information
- **Citation requirements**: All answers include source references
- **Confidence thresholds**: Flag low-confidence responses for human review

See `docs/RESPONSIBLE_AI.md` for detailed testing notes and guidelines.

## Evaluation Metrics

- **Faithfulness**: How well responses align with retrieved context
- **Answer Relevancy**: How well responses address the question
- **Context Precision**: Quality of retrieved documents
- **Hallucination Rate**: Percentage of unsupported claims

## License

MIT License - For research and educational purposes only.

⚠️ **Disclaimer**: This is a prototype system. Do not use for actual medical advice or clinical decisions.
