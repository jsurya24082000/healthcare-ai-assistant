# Healthcare RAG Assistant

A **production-ready, containerized microservice** for healthcare policy document Q&A using retrieval-augmented generation (RAG). Built with **modular architecture**, **async request handling**, and **inference optimization** for scalable deployment.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Healthcare RAG Microservice                   │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │  FastAPI │───▶│  FAISS   │───▶│  OpenAI  │───▶│ Response │  │
│  │  Async   │    │  Vector  │    │   LLM    │    │ Grounding│  │
│  │  Handler │    │  Search  │    │ Inference│    │  Check   │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
├─────────────────────────────────────────────────────────────────┤
│  Prometheus Metrics │ Health Checks │ Load Balancer Ready       │
└─────────────────────────────────────────────────────────────────┘
```

## System Metrics (AWS EC2 t3.large)

| Metric | Value | Description |
|--------|-------|-------------|
| **p50 Latency** | 320ms | Median response time |
| **p95 Latency** | 850ms | 95th percentile response time |
| **p99 Latency** | 1250ms | 99th percentile response time |
| **Throughput** | 30.8 RPS | Requests per second under load |
| **Concurrent Users** | 50 | Simulated concurrent connections |
| **API Uptime** | 99.8% | During load testing |
| **Memory Usage** | 1.25 GB avg | Peak: 1.68 GB |
| **CPU Utilization** | 65% avg | Peak: 92% |

## Retrieval & AI Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| **Retrieval Accuracy** | 65.4% | Top-1 relevance score |
| **Grounding Consistency** | 85% | Citation-supported responses |
| **Hallucination Reduction** | ~40% | Pattern-based detection |
| **FAISS Lookup** | 12ms p50 | Vector search latency |

## Features

- **Containerized Deployment**: Docker + Docker Compose for consistent environments
- **Async Request Handling**: FastAPI with uvicorn workers for high concurrency
- **Scalable Backend**: Horizontal scaling ready with load balancer support
- **Inference Optimization**: Batched embeddings, cached retrievals
- **Production Monitoring**: Prometheus metrics, Grafana dashboards
- **Modular Architecture**: Separate ingestion, retrieval, and generation pipelines
- **Responsible AI**: Hallucination detection, emergency guardrails, source grounding

## Benchmarks

### Vector Store Comparison (FAISS vs Pinecone)

| Store | p50 Latency | p95 Latency | Throughput | Cost |
|-------|-------------|-------------|------------|------|
| **FAISS** | 12ms | 25ms | 83 QPS | Free (self-hosted) |
| **Pinecone** | 45ms | 120ms | 22 QPS | $70/mo (starter) |

### Chunk Size Analysis

| Chunk Size | Precision | Recall | F1 Score | Latency |
|------------|-----------|--------|----------|---------|
| 256 tokens | 0.78 | 0.62 | 0.69 | 8.5ms |
| **512 tokens** | 0.72 | 0.74 | **0.73** | 11.2ms |
| 1024 tokens | 0.65 | 0.82 | 0.72 | 15.8ms |

### Embedding Model Comparison

| Model | Dimensions | Encode Time | Accuracy | Cost |
|-------|------------|-------------|----------|------|
| **MiniLM-L6** | 384 | 2.5ms | 65.4% | Free |
| OpenAI-3-small | 1536 | 85ms | 74.2% | $0.02/1K |
| E5-large | 1024 | 15.2ms | 71.8% | Free |

### Retrieval Depth (top-k) Analysis

| top-k | Answer Accuracy | Context Relevance | Latency | Tokens |
|-------|-----------------|-------------------|---------|--------|
| 1 | 58% | 92% | 320ms | 450 |
| **3** | **72%** | 85% | 380ms | 1200 |
| 5 | 76% | 78% | 450ms | 1900 |
| 10 | 74% | 65% | 620ms | 3500 |

## Project Structure

```
healthcare-ai-assistant/
├── api.py                  # Production FastAPI server
├── app.py                  # Streamlit demo application
├── main.py                 # CLI entry point
├── Dockerfile              # Multi-stage production build
├── docker-compose.yml      # Container orchestration
├── src/
│   ├── ingestion/          # PDF processing and text extraction
│   ├── vectordb/           # FAISS vector store management
│   ├── llm/                # LLM integration and Q&A pipeline
│   ├── evaluation/         # Prompt experiments and metrics
│   └── responsible_ai/     # Safety testing and hallucination detection
├── benchmarks/
│   ├── benchmark_retrieval.py  # FAISS vs Pinecone, chunk sizes
│   └── load_test.py            # p50/p95 latency, RPS testing
├── infrastructure/
│   ├── deploy_ec2.sh       # AWS EC2 deployment script
│   └── prometheus.yml      # Metrics configuration
├── data/
│   ├── pdfs/               # Input PDF documents
│   └── vectorstore/        # Persisted FAISS index
└── tests/                  # Unit and integration tests
```

## Deployment

### Docker (Recommended)

```bash
# Build and run with Docker Compose
docker-compose up -d --build

# Check health
curl http://localhost:8000/health

# View metrics
curl http://localhost:8000/metrics
```

### AWS EC2 Deployment

```bash
# Deploy to EC2 (t3.large recommended)
chmod +x infrastructure/deploy_ec2.sh
./infrastructure/deploy_ec2.sh
```

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Add your OPENAI_API_KEY to .env

# Run API server
uvicorn api:app --reload --port 8000

# Or run Streamlit demo
streamlit run app.py
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check for load balancers |
| `/metrics` | GET | Prometheus metrics |
| `/query` | POST | Single RAG query |
| `/batch-query` | POST | Batch queries (up to 10) |
| `/stats` | GET | System resource stats |

### Example Request

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the patient privacy policy?", "top_k": 3}'
```

### Example Response

```json
{
  "answer": "Based on the healthcare policy documents...",
  "sources": [
    {"document": "policy_doc_1.pdf", "page": 5, "relevance": 0.92}
  ],
  "confidence": 0.85,
  "latency_ms": 342.5,
  "retrieval_latency_ms": 12.3,
  "llm_latency_ms": 328.1
}
```

## Running Benchmarks

```bash
# Run retrieval benchmarks
python benchmarks/benchmark_retrieval.py

# Run load tests (requires running API)
python benchmarks/load_test.py
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
