import json
import re
from dataclasses import dataclass
from pathlib import Path

from loguru import logger


@dataclass
class LegalChunk:
    """A single chunk of legal text with metadata for retrieval."""

    chunk_id: str
    act_id: str
    act_title: str
    section_id: str
    section_number: str
    section_title: str
    text: str
    metadata: dict

    @property
    def display_source(self) -> str:
        """Human-readable source citation."""
        return f"{self.act_title}, ধারা {self.section_number}"


class LegalChunker:
    """Splits legal text by section (ধারা) boundaries instead of arbitrary token counts.

    Each section becomes one chunk. Sections longer than max_chunk_size are split
    at sub-section (উপ-ধারা) boundaries. Very short consecutive sections are never
    merged — each section stays a separate chunk to preserve citation accuracy.
    """

    # Bangla sub-section markers: (ক), (খ), (১), (২), (অ), (আ) etc.
    SUBSECTION_PATTERN = re.compile(
        r"\n(?=\([কখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহড়ঢ়য়০১২৩৪৫৬৭৮৯a-zA-Z]+\))"
    )

    def __init__(self, max_chunk_size: int = 2000):
        self.max_chunk_size = max_chunk_size

    def chunk_section(self, section: dict, act_id: str, act_title: str) -> list[LegalChunk]:
        """Split a single section into one or more chunks."""
        text = section.get("section_text", "").strip()
        if not text:
            return []

        section_id = section.get("section_id", "")
        section_number = section.get("section_number", "")
        section_title = section.get("section_title", "")

        base_metadata = {
            "act_id": act_id,
            "act_title": act_title,
            "section_id": section_id,
            "section_number": section_number,
            "section_title": section_title,
            "source_url": section.get("section_url", ""),
        }

        # If section fits in one chunk, return as-is
        if len(text) <= self.max_chunk_size:
            return [
                LegalChunk(
                    chunk_id=f"{act_id}_s{section_id}",
                    act_id=act_id,
                    act_title=act_title,
                    section_id=section_id,
                    section_number=section_number,
                    section_title=section_title,
                    text=text,
                    metadata=base_metadata,
                )
            ]

        # Split at sub-section boundaries
        return self._split_large_section(text, base_metadata)

    def _split_large_section(self, text: str, metadata: dict) -> list[LegalChunk]:
        """Split a large section at sub-section boundaries."""
        parts = self.SUBSECTION_PATTERN.split(text)
        chunks: list[LegalChunk] = []
        current_text = ""
        part_index = 0

        for part in parts:
            part = part.strip()
            if not part:
                continue

            # If adding this part exceeds limit, flush current chunk
            if current_text and len(current_text) + len(part) + 1 > self.max_chunk_size:
                chunks.append(self._make_chunk(current_text, metadata, part_index))
                part_index += 1
                current_text = part
            else:
                current_text = f"{current_text}\n{part}".strip() if current_text else part

        # Flush remaining text
        if current_text:
            chunks.append(self._make_chunk(current_text, metadata, part_index))

        # Edge case: if splitting produced nothing useful, force single chunk
        if not chunks:
            chunks.append(self._make_chunk(text[: self.max_chunk_size], metadata, 0))

        return chunks

    def _make_chunk(self, text: str, metadata: dict, part_index: int) -> LegalChunk:
        """Create a LegalChunk from text and metadata."""
        act_id = metadata["act_id"]
        section_id = metadata["section_id"]
        chunk_id = (
            f"{act_id}_s{section_id}_p{part_index}" if part_index > 0 else f"{act_id}_s{section_id}"
        )

        chunk_metadata = {**metadata, "part_index": part_index}

        return LegalChunk(
            chunk_id=chunk_id,
            act_id=act_id,
            act_title=metadata["act_title"],
            section_id=section_id,
            section_number=metadata["section_number"],
            section_title=metadata["section_title"],
            text=text,
            metadata=chunk_metadata,
        )

    def chunk_act(self, act_data: dict) -> list[LegalChunk]:
        """Chunk all sections of an act."""
        act_id = act_data.get("act_id", "")
        act_title = act_data.get("act_title", "")
        sections = act_data.get("sections", [])
        chunks: list[LegalChunk] = []

        for section in sections:
            section_chunks = self.chunk_section(section, act_id, act_title)
            chunks.extend(section_chunks)

        logger.debug(f"Act {act_id}: {len(sections)} sections -> {len(chunks)} chunks")
        return chunks

    def chunk_from_file(self, filepath: Path) -> list[LegalChunk]:
        """Load a scraped act JSON and chunk it."""
        data = json.loads(filepath.read_text(encoding="utf-8"))
        return self.chunk_act(data)

    def chunk_all_files(self, data_dir: Path) -> list[LegalChunk]:
        """Chunk all act JSON files in a directory."""
        all_chunks: list[LegalChunk] = []
        json_files = sorted(data_dir.glob("act_*.json"))

        if not json_files:
            logger.warning(f"No act files found in {data_dir}")
            return all_chunks

        for filepath in json_files:
            try:
                chunks = self.chunk_from_file(filepath)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Failed to chunk {filepath.name}: {e}")
                continue

        logger.info(f"Total: {len(json_files)} acts -> {len(all_chunks)} chunks")
        return all_chunks
