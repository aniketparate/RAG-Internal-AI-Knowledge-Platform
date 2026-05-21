"""Content extraction service for supported document formats."""

from __future__ import annotations

import asyncio
from io import BytesIO
from pathlib import Path

import aiofiles
import fitz
import pytesseract
from docx import Document as DocxDocument
from PIL import Image
from pypdf import PdfReader


CODE_FILE_TYPES = {
    "py",
    "python",
    "js",
    "javascript",
    "ts",
    "typescript",
    "go",
    "java",
    "rs",
    "cpp",
    "c",
    "cs",
}


class ContentExtractor:
    """Extract normalized text content from a file path."""

    async def extract(self, file_path: str, file_type: str) -> str:
        """Extract text content by file type."""

        normalized = file_type.lower()
        if normalized == "pdf":
            return await self._extract_pdf(file_path)
        if normalized in {"md", "markdown", "txt", "text"}:
            return await self._read_text(file_path)
        if normalized in {"docx"}:
            return await self._extract_docx(file_path)
        if normalized in CODE_FILE_TYPES:
            content = await self._read_text(file_path)
            return f"# File: {Path(file_path).name}\n\n{content}"
        raise ValueError(f"Unsupported file type: {file_type}")

    async def _read_text(self, file_path: str) -> str:
        """Read a text file asynchronously."""

        async with aiofiles.open(file_path, mode="r", encoding="utf-8", errors="ignore") as f:
            return await f.read()

    async def _read_bytes(self, file_path: str) -> bytes:
        """Read binary content asynchronously."""

        async with aiofiles.open(file_path, mode="rb") as f:
            return await f.read()

    async def _extract_pdf(self, file_path: str) -> str:
        """Extract PDF text content page-by-page."""

        file_bytes = await self._read_bytes(file_path)
        reader = PdfReader(BytesIO(file_bytes))
        page_texts: list[str] = []
        extracted_length = 0
        for idx, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                # Layout mode can recover text on PDFs where plain extraction is sparse.
                text = (page.extract_text(extraction_mode="layout") or "").strip()
            page_texts.append(text)
            extracted_length += len(text)

        if extracted_length > 100:
            return self._join_page_text(page_texts)

        ocr_page_texts = await asyncio.to_thread(self._extract_pdf_with_ocr, file_bytes)
        ocr_length = sum(len(text.strip()) for text in ocr_page_texts)
        if ocr_length == 0:
            raise ValueError("PDF text extraction returned no extractable text")
        return self._join_page_text(ocr_page_texts)

    def _extract_pdf_with_ocr(self, file_bytes: bytes) -> list[str]:
        """Extract text from PDF pages using rasterization + OCR."""

        page_texts: list[str] = []
        with fitz.open(stream=file_bytes, filetype="pdf") as pdf_doc:
            for page in pdf_doc:
                pixmap = page.get_pixmap(dpi=200)
                mode = "RGBA" if pixmap.alpha else "RGB"
                image = Image.frombytes(mode, [pixmap.width, pixmap.height], pixmap.samples)
                text = pytesseract.image_to_string(image).strip()
                page_texts.append(text)
        return page_texts

    def _join_page_text(self, page_texts: list[str]) -> str:
        """Join page text blocks with explicit page separators."""

        sections = [f"--- Page {idx} ---\n\n{text}" for idx, text in enumerate(page_texts, start=1)]
        return "\n\n".join(sections).strip()

    async def _extract_docx(self, file_path: str) -> str:
        """Extract DOCX paragraph text content."""

        file_bytes = await self._read_bytes(file_path)
        doc = DocxDocument(BytesIO(file_bytes))
        paragraphs = [paragraph.text for paragraph in doc.paragraphs]
        return "\n".join(paragraphs).strip()
