"""Text chunking strategies for document processing."""

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .pdf_loader import Document


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    content: str
    metadata: Dict[str, Any]
    chunk_id: str
    

class TextChunker:
    """Split documents into chunks for embedding."""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separator: str = "\n\n"
    ):
        """
        Initialize text chunker.
        
        Args:
            chunk_size: Target size of each chunk in characters.
            chunk_overlap: Overlap between consecutive chunks.
            separator: Primary separator for splitting text.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator
    
    def chunk_document(self, document: Document) -> List[Chunk]:
        """
        Split a document into chunks.
        
        Args:
            document: Document to chunk.
            
        Returns:
            List of Chunk objects.
        """
        text = document.content
        chunks = []
        
        # First, split by primary separator
        sections = text.split(self.separator)
        
        current_chunk = ""
        chunk_index = 0
        
        for section in sections:
            section = section.strip()
            if not section:
                continue
            
            # If section fits in current chunk, add it
            if len(current_chunk) + len(section) + len(self.separator) <= self.chunk_size:
                if current_chunk:
                    current_chunk += self.separator + section
                else:
                    current_chunk = section
            else:
                # Save current chunk if it has content
                if current_chunk:
                    chunk = self._create_chunk(
                        current_chunk, document, chunk_index
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                    
                    # Start new chunk with overlap
                    overlap_text = self._get_overlap_text(current_chunk)
                    current_chunk = overlap_text + self.separator + section if overlap_text else section
                else:
                    current_chunk = section
                
                # Handle sections larger than chunk_size
                while len(current_chunk) > self.chunk_size:
                    # Find a good break point
                    break_point = self._find_break_point(
                        current_chunk, self.chunk_size
                    )
                    
                    chunk_text = current_chunk[:break_point].strip()
                    if chunk_text:
                        chunk = self._create_chunk(
                            chunk_text, document, chunk_index
                        )
                        chunks.append(chunk)
                        chunk_index += 1
                    
                    # Continue with remaining text plus overlap
                    remaining = current_chunk[break_point:].strip()
                    overlap_text = self._get_overlap_text(chunk_text)
                    current_chunk = overlap_text + " " + remaining if overlap_text else remaining
        
        # Don't forget the last chunk
        if current_chunk.strip():
            chunk = self._create_chunk(
                current_chunk.strip(), document, chunk_index
            )
            chunks.append(chunk)
        
        return chunks
    
    def _create_chunk(
        self, 
        text: str, 
        document: Document, 
        chunk_index: int
    ) -> Chunk:
        """Create a Chunk object with metadata."""
        chunk_id = f"{document.metadata.get('filename', 'unknown')}_p{document.page_number}_c{chunk_index}"
        
        metadata = {
            **document.metadata,
            "chunk_index": chunk_index,
            "chunk_size": len(text),
            "chunk_id": chunk_id
        }
        
        return Chunk(
            content=text,
            metadata=metadata,
            chunk_id=chunk_id
        )
    
    def _get_overlap_text(self, text: str) -> str:
        """Get the overlap portion from the end of text."""
        if len(text) <= self.chunk_overlap:
            return text
        
        overlap_start = len(text) - self.chunk_overlap
        
        # Try to start at a sentence boundary
        sentence_end = text.rfind(". ", 0, overlap_start)
        if sentence_end > overlap_start - 100:
            overlap_start = sentence_end + 2
        
        return text[overlap_start:]
    
    def _find_break_point(self, text: str, max_length: int) -> int:
        """Find a good break point near max_length."""
        if len(text) <= max_length:
            return len(text)
        
        # Try to break at paragraph
        para_break = text.rfind("\n\n", 0, max_length)
        if para_break > max_length * 0.5:
            return para_break
        
        # Try to break at sentence
        sentence_break = text.rfind(". ", 0, max_length)
        if sentence_break > max_length * 0.5:
            return sentence_break + 1
        
        # Try to break at word
        word_break = text.rfind(" ", 0, max_length)
        if word_break > max_length * 0.5:
            return word_break
        
        # Hard break at max_length
        return max_length
    
    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        """
        Chunk multiple documents.
        
        Args:
            documents: List of documents to chunk.
            
        Returns:
            List of all chunks from all documents.
        """
        all_chunks = []
        
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
        
        print(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        return all_chunks


class SemanticChunker(TextChunker):
    """
    Advanced chunker that respects semantic boundaries.
    Useful for healthcare documents with specific sections.
    """
    
    # Common healthcare document section headers
    SECTION_PATTERNS = [
        r"^#{1,3}\s+",  # Markdown headers
        r"^[A-Z][A-Z\s]{3,}:?\s*$",  # ALL CAPS HEADERS
        r"^\d+\.\s+[A-Z]",  # Numbered sections
        r"^(?:Section|Article|Chapter)\s+\d+",  # Formal sections
        r"^(?:POLICY|PROCEDURE|GUIDELINE|REGULATION):",  # Healthcare specific
    ]
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.section_regex = re.compile(
            "|".join(self.SECTION_PATTERNS), 
            re.MULTILINE | re.IGNORECASE
        )
    
    def chunk_document(self, document: Document) -> List[Chunk]:
        """Split document respecting semantic sections."""
        text = document.content
        
        # Find section boundaries
        sections = self._split_by_sections(text)
        
        chunks = []
        chunk_index = 0
        
        for section in sections:
            if not section.strip():
                continue
            
            # If section is small enough, keep it whole
            if len(section) <= self.chunk_size:
                chunk = self._create_chunk(section.strip(), document, chunk_index)
                chunks.append(chunk)
                chunk_index += 1
            else:
                # Split large sections using parent method
                temp_doc = Document(
                    content=section,
                    metadata=document.metadata,
                    page_number=document.page_number,
                    source=document.source
                )
                section_chunks = super().chunk_document(temp_doc)
                
                for sc in section_chunks:
                    sc.chunk_id = f"{document.metadata.get('filename', 'unknown')}_p{document.page_number}_c{chunk_index}"
                    sc.metadata["chunk_index"] = chunk_index
                    chunks.append(sc)
                    chunk_index += 1
        
        return chunks
    
    def _split_by_sections(self, text: str) -> List[str]:
        """Split text by section headers."""
        matches = list(self.section_regex.finditer(text))
        
        if not matches:
            return [text]
        
        sections = []
        prev_end = 0
        
        for match in matches:
            # Add text before this section header
            if match.start() > prev_end:
                sections.append(text[prev_end:match.start()])
            prev_end = match.start()
        
        # Add remaining text
        if prev_end < len(text):
            sections.append(text[prev_end:])
        
        return sections
