# ingestion/pdf_worker.py
import pdfplumber
from pathlib import Path
from typing import List, Optional

from core.logger import debug
from memory.models import MemoryRecord


class PDFWorker:
    def __init__(self, memory_system):
        self.memory = memory_system

    def extract_text(
        self,
        filepath: str,
        page_limit: int = 100,
        min_text_length: int = 50
    ) -> List[dict]:
        """
        Extract text from each page of a PDF.

        Args:
            filepath: Path to PDF file
            page_limit: Maximum number of pages to process
            min_text_length: Minimum text length to consider (avoid empty/header-only pages)

        Returns:
            List of dicts with text, page number, source file
        """
        if not Path(filepath).exists():
            debug(f"[PDFWorker] File not found: {filepath}")
            return []

        chunks = []

        try:
            with pdfplumber.open(filepath) as pdf:
                total_pages = min(len(pdf.pages), page_limit)
                debug(f"[PDFWorker] Processing {total_pages} pages from {filepath}")

                for i, page in enumerate(pdf.pages):
                    if i >= page_limit:
                        break

                    try:
                        text = page.extract_text()
                        if text and len(text.strip()) >= min_text_length:
                            chunks.append({
                                "text": text.strip(),
                                "page": i + 1,
                                "source_file": filepath,
                                "total_pages": len(pdf.pages)
                            })
                    except Exception as e:
                        debug(f"[PDFWorker] Error on page {i+1}: {e}")
                        continue

        except Exception as e:
            debug(f"[PDFWorker] Failed to open PDF {filepath}: {e}")
            return []

        debug(f"[PDFWorker] Extracted {len(chunks)} text chunks from {filepath}")
        return chunks

    def ingest_pdf(
        self,
        filepath: str,
        max_pages: int = 100,
        memory_type: str = "semantic",
        min_text_length: int = 50,
        batch_size: int = 50
    ) -> int:
        """
        Ingest a PDF into memory.

        Args:
            filepath: Path to PDF file
            max_pages: Maximum number of pages to ingest
            memory_type: Memory type to assign to chunks
            min_text_length: Minimum text length per chunk
            batch_size: Number of chunks to batch per store operation

        Returns:
            Number of chunks ingested
        """
        chunks = self.extract_text(filepath, page_limit=max_pages, min_text_length=min_text_length)

        if not chunks:
            return 0

        # Batch process for efficiency
        total_ingested = 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            texts = [chunk["text"] for chunk in batch]

            # Use batch ingestion
            ids = self.memory.store_many(
                texts,
                memory_type=memory_type,
                metadata=[
                    {
                        "source_file": chunk["source_file"],
                        "page": chunk["page"],
                        "type": "pdf",
                        "total_pages": chunk["total_pages"]
                    }
                    for chunk in batch
                ]
            )

            total_ingested += len(ids)
            debug(f"[PDFWorker] Ingested batch {i//batch_size + 1}: {len(ids)} chunks")

        debug(f"[PDFWorker] Complete: {total_ingested} chunks ingested from {filepath}")
        return total_ingested

    def ingest_pdf_chunked(
        self,
        filepath: str,
        max_pages: int = 100,
        chunk_size: int = 1000  # characters per chunk
    ) -> int:
        """
        Ingest a PDF with page-level chunking into smaller pieces.
        Useful for long pages that exceed token limits.
        """
        chunks = self.extract_text(filepath, page_limit=max_pages)

        if not chunks:
            return 0

        total_ingested = 0
        for chunk in chunks:
            text = chunk["text"]

            # Split long pages into smaller chunks
            if len(text) > chunk_size:
                # Split by sentences or paragraphs
                import re
                sentences = re.split(r'(?<=[.!?])\s+', text)
                sub_chunks = []
                current = []

                for sentence in sentences:
                    current.append(sentence)
                    if len(" ".join(current)) >= chunk_size:
                        sub_chunks.append(" ".join(current))
                        current = []

                if current:
                    sub_chunks.append(" ".join(current))

                for sub_text in sub_chunks:
                    if len(sub_text.strip()) >= 50:
                        self.memory.store(
                            sub_text,
                            memory_type="semantic",
                            metadata={
                                "source_file": chunk["source_file"],
                                "page": chunk["page"],
                                "type": "pdf_chunk"
                            }
                        )
                        total_ingested += 1
            else:
                self.memory.store(
                    text,
                    memory_type="semantic",
                    metadata={
                        "source_file": chunk["source_file"],
                        "page": chunk["page"],
                        "type": "pdf"
                    }
                )
                total_ingested += 1

        debug(f"[PDFWorker] Complete: {total_ingested} chunks ingested from {filepath}")
        return total_ingested
