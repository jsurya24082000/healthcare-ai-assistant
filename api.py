"""
Healthcare RAG Assistant - Production FastAPI Server
Microservice architecture with async request handling and inference optimization
"""

import os
import time
import asyncio
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

# Metrics for monitoring
REQUEST_COUNT = Counter('rag_requests_total', 'Total RAG requests', ['endpoint', 'status'])
REQUEST_LATENCY = Histogram('rag_request_latency_seconds', 'Request latency', 
                            ['endpoint'], buckets=[0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 10.0])
ACTIVE_REQUESTS = Gauge('rag_active_requests', 'Currently processing requests')
RETRIEVAL_LATENCY = Histogram('rag_retrieval_latency_seconds', 'Vector retrieval latency',
                               buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5])
LLM_LATENCY = Histogram('rag_llm_latency_seconds', 'LLM inference latency',
                         buckets=[0.5, 1.0, 2.0, 3.0, 5.0, 10.0])
MEMORY_USAGE = Gauge('rag_memory_usage_bytes', 'Current memory usage')


class QueryRequest(BaseModel):
    """Request model for RAG queries"""
    question: str = Field(..., min_length=3, max_length=1000)
    top_k: int = Field(default=3, ge=1, le=10)
    include_sources: bool = Field(default=True)


class QueryResponse(BaseModel):
    """Response model with metrics"""
    answer: str
    sources: Optional[List[Dict[str, Any]]] = None
    confidence: float
    latency_ms: float
    retrieval_latency_ms: float
    llm_latency_ms: float
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    uptime_seconds: float
    vector_store_loaded: bool
    model_loaded: bool


# Global state
startup_time = None
vector_store = None
llm_chain = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup/shutdown"""
    global startup_time, vector_store, llm_chain
    
    startup_time = time.time()
    
    # Initialize vector store and LLM on startup
    print("🚀 Initializing Healthcare RAG microservice...")
    
    # Lazy load to simulate production initialization
    # In production, load actual FAISS index and LLM here
    await asyncio.sleep(0.1)  # Simulate async initialization
    vector_store = True  # Placeholder
    llm_chain = True  # Placeholder
    
    print("✅ RAG service ready - vector store and LLM loaded")
    
    yield
    
    # Cleanup on shutdown
    print("🛑 Shutting down RAG service...")


app = FastAPI(
    title="Healthcare RAG Assistant API",
    description="Production-ready RAG microservice for healthcare policy Q&A",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for load balancer and monitoring"""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        uptime_seconds=time.time() - startup_time if startup_time else 0,
        vector_store_loaded=vector_store is not None,
        model_loaded=llm_chain is not None
    )


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """
    Main RAG query endpoint with async request handling
    
    Performance targets:
    - p50 latency: < 500ms
    - p95 latency: < 2000ms
    - Throughput: 50+ RPS
    """
    ACTIVE_REQUESTS.inc()
    start_time = time.time()
    
    try:
        # Simulate retrieval phase
        retrieval_start = time.time()
        await asyncio.sleep(0.05)  # Simulated FAISS lookup (~50ms)
        retrieval_time = (time.time() - retrieval_start) * 1000
        RETRIEVAL_LATENCY.observe(retrieval_time / 1000)
        
        # Simulate LLM inference phase
        llm_start = time.time()
        await asyncio.sleep(0.3)  # Simulated LLM call (~300ms)
        llm_time = (time.time() - llm_start) * 1000
        LLM_LATENCY.observe(llm_time / 1000)
        
        # Calculate total latency
        total_latency = (time.time() - start_time) * 1000
        REQUEST_LATENCY.labels(endpoint='/query').observe(total_latency / 1000)
        REQUEST_COUNT.labels(endpoint='/query', status='success').inc()
        
        response = QueryResponse(
            answer=f"Based on the healthcare policy documents, here is the answer to: {request.question}",
            sources=[
                {"document": "policy_doc_1.pdf", "page": 5, "relevance": 0.92},
                {"document": "policy_doc_2.pdf", "page": 12, "relevance": 0.87},
            ] if request.include_sources else None,
            confidence=0.85,
            latency_ms=round(total_latency, 2),
            retrieval_latency_ms=round(retrieval_time, 2),
            llm_latency_ms=round(llm_time, 2),
            timestamp=datetime.utcnow().isoformat()
        )
        
        return response
        
    except Exception as e:
        REQUEST_COUNT.labels(endpoint='/query', status='error').inc()
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        ACTIVE_REQUESTS.dec()


@app.post("/batch-query")
async def batch_query(questions: List[str], background_tasks: BackgroundTasks):
    """
    Batch query endpoint for high-throughput scenarios
    Processes multiple queries concurrently
    """
    if len(questions) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 questions per batch")
    
    start_time = time.time()
    
    # Process queries concurrently
    tasks = [
        query_rag(QueryRequest(question=q, include_sources=False))
        for q in questions
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    total_latency = (time.time() - start_time) * 1000
    
    return {
        "results": [r.dict() if not isinstance(r, Exception) else {"error": str(r)} for r in results],
        "batch_latency_ms": round(total_latency, 2),
        "queries_processed": len(questions)
    }


@app.get("/stats")
async def get_stats():
    """
    System statistics endpoint for monitoring dashboard
    """
    import psutil
    
    process = psutil.Process()
    memory_info = process.memory_info()
    
    MEMORY_USAGE.set(memory_info.rss)
    
    return {
        "memory_usage_mb": round(memory_info.rss / 1024 / 1024, 2),
        "cpu_percent": process.cpu_percent(),
        "uptime_seconds": round(time.time() - startup_time, 2) if startup_time else 0,
        "active_requests": ACTIVE_REQUESTS._value.get(),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
