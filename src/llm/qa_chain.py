"""Healthcare Q&A chain with RAG."""

import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from openai import OpenAI

from src.vectordb.faiss_store import FAISSVectorStore
from src.ingestion.text_chunker import Chunk
from .prompts import PromptTemplates, PromptTemplate


@dataclass
class QAResponse:
    """Response from the Q&A system."""
    answer: str
    sources: List[Dict[str, Any]]
    query: str
    prompt_template: str
    model: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "answer": self.answer,
            "sources": self.sources,
            "query": self.query,
            "prompt_template": self.prompt_template,
            "model": self.model,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }


class HealthcareQAChain:
    """RAG-based Q&A chain for healthcare documents."""
    
    def __init__(
        self,
        vector_store: FAISSVectorStore,
        model: str = "gpt-4-turbo-preview",
        temperature: float = 0.1,
        max_tokens: int = 2000,
        top_k: int = 5,
        score_threshold: float = 0.3
    ):
        """
        Initialize Q&A chain.
        
        Args:
            vector_store: FAISS vector store with indexed documents.
            model: OpenAI model to use.
            temperature: LLM temperature (lower = more deterministic).
            max_tokens: Maximum tokens in response.
            top_k: Number of chunks to retrieve.
            score_threshold: Minimum similarity score for retrieval.
        """
        self.vector_store = vector_store
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_k = top_k
        self.score_threshold = score_threshold
        
        self.client = OpenAI()
        self.default_template = PromptTemplates.GROUNDED_QA
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Chunk, float]]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: User query.
            top_k: Number of results (uses default if None).
            metadata_filter: Optional metadata filter.
            
        Returns:
            List of (Chunk, score) tuples.
        """
        k = top_k or self.top_k
        
        if metadata_filter:
            results = self.vector_store.search_with_metadata_filter(
                query, top_k=k, metadata_filter=metadata_filter
            )
        else:
            results = self.vector_store.search(
                query, top_k=k, score_threshold=self.score_threshold
            )
        
        return results
    
    def generate(
        self,
        query: str,
        context_chunks: List[Tuple[Chunk, float]],
        template: Optional[PromptTemplate] = None
    ) -> str:
        """
        Generate answer using LLM.
        
        Args:
            query: User query.
            context_chunks: Retrieved chunks with scores.
            template: Prompt template to use.
            
        Returns:
            Generated answer string.
        """
        template = template or self.default_template
        
        # Format context
        context = PromptTemplates.format_context(context_chunks)
        
        # Build messages
        messages = [
            {"role": "system", "content": template.system_prompt},
            {
                "role": "user",
                "content": template.user_prompt_template.format(
                    context=context,
                    question=query
                )
            }
        ]
        
        # Call LLM
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        return response.choices[0].message.content
    
    def query(
        self,
        question: str,
        template_name: Optional[str] = None,
        top_k: Optional[int] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        include_scores: bool = True
    ) -> QAResponse:
        """
        Full RAG pipeline: retrieve and generate.
        
        Args:
            question: User question.
            template_name: Name of prompt template to use.
            top_k: Number of chunks to retrieve.
            metadata_filter: Optional metadata filter.
            include_scores: Include relevance scores in response.
            
        Returns:
            QAResponse object with answer and metadata.
        """
        # Get template
        if template_name:
            template = PromptTemplates.get_template(template_name)
        else:
            template = self.default_template
        
        # Retrieve relevant chunks
        retrieved_chunks = self.retrieve(
            question, 
            top_k=top_k,
            metadata_filter=metadata_filter
        )
        
        if not retrieved_chunks:
            return QAResponse(
                answer="I couldn't find any relevant information in the documents to answer your question.",
                sources=[],
                query=question,
                prompt_template=template.name,
                model=self.model,
                metadata={"retrieval_count": 0}
            )
        
        # Generate answer
        answer = self.generate(question, retrieved_chunks, template)
        
        # Format sources
        sources = []
        for chunk, score in retrieved_chunks:
            source_info = {
                "filename": chunk.metadata.get("filename", "Unknown"),
                "page": chunk.metadata.get("page"),
                "chunk_id": chunk.chunk_id,
                "content_preview": chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content
            }
            if include_scores:
                source_info["relevance_score"] = round(score, 3)
            sources.append(source_info)
        
        return QAResponse(
            answer=answer,
            sources=sources,
            query=question,
            prompt_template=template.name,
            model=self.model,
            metadata={
                "retrieval_count": len(retrieved_chunks),
                "avg_relevance": sum(s for _, s in retrieved_chunks) / len(retrieved_chunks),
                "temperature": self.temperature
            }
        )
    
    def query_with_followup(
        self,
        question: str,
        conversation_history: List[Dict[str, str]],
        template_name: Optional[str] = None
    ) -> QAResponse:
        """
        Query with conversation history for follow-up questions.
        
        Args:
            question: Current question.
            conversation_history: List of {"role": "user/assistant", "content": "..."}.
            template_name: Prompt template name.
            
        Returns:
            QAResponse object.
        """
        # Reformulate question with context
        if conversation_history:
            reformulation_prompt = f"""Given this conversation history:
{self._format_history(conversation_history)}

The user now asks: "{question}"

Reformulate this as a standalone question that captures the full context."""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": reformulation_prompt}],
                temperature=0,
                max_tokens=200
            )
            
            standalone_question = response.choices[0].message.content
        else:
            standalone_question = question
        
        # Run standard query with reformulated question
        result = self.query(standalone_question, template_name=template_name)
        result.metadata["original_question"] = question
        result.metadata["reformulated_question"] = standalone_question
        
        return result
    
    def _format_history(self, history: List[Dict[str, str]]) -> str:
        """Format conversation history."""
        formatted = []
        for msg in history[-5:]:  # Last 5 messages
            role = "User" if msg["role"] == "user" else "Assistant"
            formatted.append(f"{role}: {msg['content']}")
        return "\n".join(formatted)
    
    def batch_query(
        self,
        questions: List[str],
        template_name: Optional[str] = None
    ) -> List[QAResponse]:
        """
        Process multiple questions.
        
        Args:
            questions: List of questions.
            template_name: Prompt template name.
            
        Returns:
            List of QAResponse objects.
        """
        results = []
        for question in questions:
            result = self.query(question, template_name=template_name)
            results.append(result)
        return results
