"""PDF document loading and text extraction."""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import pdfplumber
from pypdf import PdfReader
from tqdm import tqdm


@dataclass
class Document:
    """Represents a loaded document with metadata."""
    content: str
    metadata: Dict[str, Any]
    page_number: Optional[int] = None
    source: Optional[str] = None


class PDFLoader:
    """Load and extract text from PDF documents."""
    
    def __init__(self, use_pdfplumber: bool = True):
        """
        Initialize PDF loader.
        
        Args:
            use_pdfplumber: Use pdfplumber for extraction (better for tables).
                           Falls back to pypdf if False.
        """
        self.use_pdfplumber = use_pdfplumber
    
    def load_pdf(self, pdf_path: str) -> List[Document]:
        """
        Load a single PDF file and extract text by page.
        
        Args:
            pdf_path: Path to the PDF file.
            
        Returns:
            List of Document objects, one per page.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        documents = []
        
        if self.use_pdfplumber:
            documents = self._load_with_pdfplumber(pdf_path)
        else:
            documents = self._load_with_pypdf(pdf_path)
        
        return documents
    
    def _load_with_pdfplumber(self, pdf_path: Path) -> List[Document]:
        """Extract text using pdfplumber (better for complex layouts)."""
        documents = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                
                # Extract tables if present
                tables = page.extract_tables()
                table_text = ""
                for table in tables:
                    if table:
                        table_text += self._format_table(table) + "\n"
                
                full_text = text
                if table_text:
                    full_text += "\n\n[TABLES]\n" + table_text
                
                if full_text.strip():
                    doc = Document(
                        content=full_text.strip(),
                        metadata={
                            "source": str(pdf_path),
                            "filename": pdf_path.name,
                            "page": page_num,
                            "total_pages": len(pdf.pages),
                            "extraction_method": "pdfplumber"
                        },
                        page_number=page_num,
                        source=str(pdf_path)
                    )
                    documents.append(doc)
        
        return documents
    
    def _load_with_pypdf(self, pdf_path: Path) -> List[Document]:
        """Extract text using pypdf (faster, simpler)."""
        documents = []
        
        reader = PdfReader(pdf_path)
        total_pages = len(reader.pages)
        
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            
            if text.strip():
                doc = Document(
                    content=text.strip(),
                    metadata={
                        "source": str(pdf_path),
                        "filename": pdf_path.name,
                        "page": page_num,
                        "total_pages": total_pages,
                        "extraction_method": "pypdf"
                    },
                    page_number=page_num,
                    source=str(pdf_path)
                )
                documents.append(doc)
        
        return documents
    
    def _format_table(self, table: List[List[str]]) -> str:
        """Format extracted table as text."""
        if not table:
            return ""
        
        formatted_rows = []
        for row in table:
            if row:
                cleaned_row = [str(cell) if cell else "" for cell in row]
                formatted_rows.append(" | ".join(cleaned_row))
        
        return "\n".join(formatted_rows)
    
    def load_directory(self, directory: str, recursive: bool = True) -> List[Document]:
        """
        Load all PDFs from a directory.
        
        Args:
            directory: Path to directory containing PDFs.
            recursive: Search subdirectories if True.
            
        Returns:
            List of all Document objects from all PDFs.
        """
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        pattern = "**/*.pdf" if recursive else "*.pdf"
        pdf_files = list(directory.glob(pattern))
        
        if not pdf_files:
            print(f"No PDF files found in {directory}")
            return []
        
        all_documents = []
        
        for pdf_path in tqdm(pdf_files, desc="Loading PDFs"):
            try:
                docs = self.load_pdf(str(pdf_path))
                all_documents.extend(docs)
            except Exception as e:
                print(f"Error loading {pdf_path}: {e}")
                continue
        
        print(f"Loaded {len(all_documents)} pages from {len(pdf_files)} PDFs")
        return all_documents
