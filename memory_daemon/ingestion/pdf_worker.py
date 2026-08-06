# ingestion/pdf_worker.py
import pdfplumber
from pathlib import Path
from typing import List
from memory.models import MemoryRecord

class PDFWorker:
    def __init__(self, memory_system):
        self.memory = memory_system

    def extract_text(self, filepath: str, page_limit: int = 100) -> List[str]:
        """Extract text from each page of a PDF."""
        chunks = []
        with pdfplumber.open(filepath) as pdf:
            for i, page in enumerate(pdf.pages):
                if i >= page_limit:
                    break
                text = page.extract_text()
                if text and len(text) > 50:
                    chunks.append({
                        "text": text,
                        "page": i + 1,
                        "source_file": filepath
                    })
        return chunks

    def ingest_pdf(self, filepath: str, max_pages: int = 100):
        """Ingest a PDF into memory."""
        chunks = self.extract_text(filepath, page_limit=max_pages)
        for chunk in chunks:
            record = MemoryRecord(
                text=chunk["text"],
                memory_type="semantic",
                metadata={
                    "source_file": chunk["source_file"],
                    "page": chunk["page"],
                    "type": "pdf"
                }
            )
            self.memory.store(record.text)
        return len(chunks)
