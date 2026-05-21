"""Chunking strategies for text and source code files."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tree_sitter import Parser

try:
    import tree_sitter_python
except ImportError:  # pragma: no cover
    tree_sitter_python = None


TEXT_TYPES = {"pdf", "md", "markdown", "txt", "text"}
CODE_TYPES = {"py", "python", "js", "javascript", "ts", "typescript", "go", "java", "rs", "cpp", "c", "cs"}


@dataclass
class Chunk:
    """A retrieval chunk and its metadata."""

    content: str
    chunk_index: int
    metadata: dict[str, Any]


class ChunkingService:
    """Create chunks using text or code-aware strategies."""

    def __init__(self) -> None:
        self._encoding = tiktoken.get_encoding("cl100k_base")
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=64,
            length_function=self.token_count,
        )
        # Keep PDF splitting explicit and deterministic at token boundaries.
        self._pdf_chunk_size = 512
        self._pdf_chunk_overlap = 64

    def token_count(self, text: str) -> int:
        """Count tokens using cl100k_base."""

        return len(self._encoding.encode(text))

    def chunk_content(self, content: str, file_type: str, base_metadata: dict[str, Any] | None = None) -> list[Chunk]:
        """Chunk content using file-type specific strategy."""

        metadata = base_metadata.copy() if base_metadata else {}
        normalized = file_type.lower()

        if normalized == "pdf":
            return self._chunk_pdf(content, metadata)
        if normalized in TEXT_TYPES:
            return self._chunk_text(content, metadata)
        if normalized in CODE_TYPES:
            return self._chunk_code(content, normalized, metadata)
        return self._chunk_text(content, metadata)

    def _chunk_pdf(self, content: str, base_metadata: dict[str, Any]) -> list[Chunk]:
        """Chunk PDF content to 512 tokens with 64-token overlap."""

        if not content.strip():
            return []
        return self._chunk_by_token_windows(content, base_metadata)

    def _chunk_text(self, content: str, base_metadata: dict[str, Any]) -> list[Chunk]:
        """Chunk non-code text content."""

        chunks = self._text_splitter.split_text(content)
        return self._build_chunks_from_strings(content, chunks, base_metadata)

    def _build_chunks_from_strings(
        self,
        source_content: str,
        chunk_strings: list[str],
        base_metadata: dict[str, Any],
    ) -> list[Chunk]:
        """Build chunk objects from pre-split strings."""

        output: list[Chunk] = []
        cursor = 0
        for idx, chunk in enumerate(chunk_strings):
            start = max(source_content.find(chunk, cursor), 0)
            end = start + len(chunk)
            cursor = end
            output.append(
                Chunk(
                    content=chunk,
                    chunk_index=idx,
                    metadata={
                        **base_metadata,
                        "function_name": None,
                        "class_name": None,
                        "char_start": start,
                        "char_end": end,
                    },
                )
            )
        return output

    def _chunk_by_token_windows(self, content: str, base_metadata: dict[str, Any]) -> list[Chunk]:
        """Split text by exact token windows with configured overlap."""

        tokens = self._encoding.encode(content)
        if not tokens:
            return []

        chunks: list[Chunk] = []
        step = self._pdf_chunk_size - self._pdf_chunk_overlap
        token_index = 0
        chunk_index = 0
        search_cursor = 0
        while token_index < len(tokens):
            window_tokens = tokens[token_index : token_index + self._pdf_chunk_size]
            window_text = self._encoding.decode(window_tokens).strip()
            if window_text:
                found = content.find(window_text, search_cursor)
                start = found if found >= 0 else 0
                end = start + len(window_text)
                search_cursor = end
                chunks.append(
                    Chunk(
                        content=window_text,
                        chunk_index=chunk_index,
                        metadata={
                            **base_metadata,
                            "function_name": None,
                            "class_name": None,
                            "char_start": start,
                            "char_end": end,
                        },
                    )
                )
                chunk_index += 1
            token_index += step
        return chunks

    def _chunk_code(self, content: str, file_type: str, base_metadata: dict[str, Any]) -> list[Chunk]:
        """Chunk code by top-level symbols, with tree-sitter fallback."""

        if file_type not in {"py", "python"}:
            return self._chunk_text(content, base_metadata)

        parser = self._python_parser()
        if parser is None:
            return self._chunk_text(content, base_metadata)

        try:
            tree = parser.parse(content.encode("utf-8", errors="ignore"))
            root = tree.root_node
        except Exception:
            return self._chunk_text(content, base_metadata)

        collected: list[Chunk] = []
        chunk_index = 0
        for node in root.children:
            if node.type not in {"function_definition", "class_definition"}:
                continue
            text = content[node.start_byte : node.end_byte]
            name_node = node.child_by_field_name("name")
            symbol_name = content[name_node.start_byte : name_node.end_byte] if name_node else None
            symbol_metadata = {
                **base_metadata,
                "function_name": symbol_name if node.type == "function_definition" else None,
                "class_name": symbol_name if node.type == "class_definition" else None,
                "char_start": node.start_byte,
                "char_end": node.end_byte,
            }
            if self.token_count(text) <= 512:
                collected.append(Chunk(content=text, chunk_index=chunk_index, metadata=symbol_metadata))
                chunk_index += 1
                continue

            split_parts = self._text_splitter.split_text(text)
            offset_cursor = node.start_byte
            for split in split_parts:
                split_start = max(text.find(split, offset_cursor - node.start_byte), 0) + node.start_byte
                split_end = split_start + len(split)
                offset_cursor = split_end
                collected.append(
                    Chunk(
                        content=split,
                        chunk_index=chunk_index,
                        metadata={
                            **symbol_metadata,
                            "char_start": split_start,
                            "char_end": split_end,
                        },
                    )
                )
                chunk_index += 1

        if not collected:
            return self._chunk_text(content, base_metadata)
        return collected

    def _python_parser(self) -> Parser | None:
        """Create a Python tree-sitter parser if available."""

        if tree_sitter_python is None:
            return None

        parser = Parser()
        try:
            parser.set_language(tree_sitter_python.language())
            return parser
        except Exception:
            return None
