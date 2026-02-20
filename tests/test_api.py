"""
API Integration Tests for Healthcare RAG Assistant
Tests REST endpoints, caching, rate limiting, and concurrent requests
"""

import pytest
import asyncio
from httpx import AsyncClient, ASGITransport
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api import app, cache, rate_limiter


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client():
    """Create async test client"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_health_endpoint(client):
    """Test health check returns correct structure"""
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "uptime_seconds" in data
    assert "instance_id" in data


@pytest.mark.asyncio
async def test_query_endpoint(client):
    """Test RAG query endpoint"""
    response = await client.post(
        "/query",
        json={"question": "What is the patient privacy policy?", "top_k": 3}
    )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "latency_ms" in data
    assert "confidence" in data


@pytest.mark.asyncio
async def test_query_caching(client):
    """Test that repeated queries are cached"""
    question = "Test caching question"
    
    # First request - should be cache miss
    response1 = await client.post("/query", json={"question": question})
    assert response1.status_code == 200
    data1 = response1.json()
    assert data1["cached"] == False
    
    # Second request - should be cache hit
    response2 = await client.post("/query", json={"question": question})
    assert response2.status_code == 200
    data2 = response2.json()
    assert data2["cached"] == True
    assert data2["latency_ms"] < data1["latency_ms"]


@pytest.mark.asyncio
async def test_batch_query(client):
    """Test batch query endpoint"""
    questions = [
        "What is HIPAA?",
        "What are patient rights?",
        "How to file a complaint?"
    ]
    
    response = await client.post("/batch-query", json=questions)
    assert response.status_code == 200
    data = response.json()
    assert data["queries_processed"] == 3
    assert len(data["results"]) == 3


@pytest.mark.asyncio
async def test_batch_query_limit(client):
    """Test batch query rejects more than 10 questions"""
    questions = [f"Question {i}" for i in range(15)]
    
    response = await client.post("/batch-query", json=questions)
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_stats_endpoint(client):
    """Test stats endpoint returns system metrics"""
    response = await client.get("/stats")
    assert response.status_code == 200
    data = response.json()
    assert "memory_usage_mb" in data
    assert "instance_id" in data
    assert "cache" in data


@pytest.mark.asyncio
async def test_cache_stats(client):
    """Test cache stats endpoint"""
    response = await client.get("/cache/stats")
    assert response.status_code == 200
    data = response.json()
    assert "stats" in data
    assert "ttl_seconds" in data


@pytest.mark.asyncio
async def test_scaling_info(client):
    """Test scaling info endpoint"""
    response = await client.get("/scaling/info")
    assert response.status_code == 200
    data = response.json()
    assert "instance_id" in data
    assert "capacity" in data
    assert "endpoints" in data


@pytest.mark.asyncio
async def test_concurrent_requests(client):
    """Test handling of concurrent requests"""
    async def make_request():
        return await client.post(
            "/query",
            json={"question": "Concurrent test", "top_k": 1}
        )
    
    # Make 10 concurrent requests
    tasks = [make_request() for _ in range(10)]
    responses = await asyncio.gather(*tasks)
    
    # All should succeed
    assert all(r.status_code == 200 for r in responses)


@pytest.mark.asyncio
async def test_query_validation(client):
    """Test query validation"""
    # Too short question
    response = await client.post("/query", json={"question": "Hi"})
    assert response.status_code == 422
    
    # Missing question
    response = await client.post("/query", json={})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_metrics_endpoint(client):
    """Test Prometheus metrics endpoint"""
    response = await client.get("/metrics")
    assert response.status_code == 200
    assert b"rag_requests_total" in response.content


class TestRateLimiter:
    """Unit tests for rate limiter"""
    
    def test_allows_under_limit(self):
        from api import RateLimiter
        limiter = RateLimiter(requests_per_minute=10)
        
        for _ in range(10):
            assert limiter.is_allowed("test_client") == True
    
    def test_blocks_over_limit(self):
        from api import RateLimiter
        limiter = RateLimiter(requests_per_minute=5)
        
        for _ in range(5):
            limiter.is_allowed("test_client")
        
        assert limiter.is_allowed("test_client") == False
    
    def test_different_clients(self):
        from api import RateLimiter
        limiter = RateLimiter(requests_per_minute=2)
        
        assert limiter.is_allowed("client_a") == True
        assert limiter.is_allowed("client_a") == True
        assert limiter.is_allowed("client_a") == False
        
        # Different client should still be allowed
        assert limiter.is_allowed("client_b") == True


class TestCacheLayer:
    """Unit tests for cache layer"""
    
    @pytest.mark.asyncio
    async def test_cache_set_get(self):
        from api import CacheLayer
        cache = CacheLayer(default_ttl=60)
        
        await cache.set("key1", {"data": "value"})
        result = await cache.get("key1")
        
        assert result == {"data": "value"}
    
    @pytest.mark.asyncio
    async def test_cache_miss(self):
        from api import CacheLayer
        cache = CacheLayer(default_ttl=60)
        
        result = await cache.get("nonexistent")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_cache_delete(self):
        from api import CacheLayer
        cache = CacheLayer(default_ttl=60)
        
        await cache.set("key1", "value")
        await cache.delete("key1")
        
        result = await cache.get("key1")
        assert result is None
