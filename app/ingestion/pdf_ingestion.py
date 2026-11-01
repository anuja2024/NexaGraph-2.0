
import json
import re
from pathlib import Path

import pymupdf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PDF_DIRECTORY = PROJECT_ROOT / "data" / "staging" / "siemens_pdf"
REGISTRY_PATH = PDF_DIRECTORY / "documents.json"


def load_document_registry() -> list[dict]:
    """Load the registered documents."""
    with REGISTRY_PATH.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def clean_text(text: str) -> str:
    """Normalize whitespace without removing paragraph boundaries."""
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_into_chunks(
    text: str,
    chunk_size: int = 1200,
    overlap: int = 200,
) -> list[str]:
    """Split text near sentence or word boundaries."""
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Require chunk_size > overlap >= 0.")

    text = clean_text(text)
    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            boundary = max(
                text.rfind(". ", start, end),
                text.rfind("? ", start, end),
                text.rfind("! ", start, end),
                text.rfind(" ", start, end),
            )
            if boundary > start:
                end = boundary + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(start + 1, end - overlap)

    return chunks


def ingest_pdf(
    document: dict,
    chunk_size: int = 1200,
    overlap: int = 200,
) -> list[dict]:
    """Extract page-aware chunks from one registered PDF."""
    if document.get("status") != "active":
        raise ValueError("Document is not active.")

    pdf_path = PDF_DIRECTORY / document["filename"]

    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path.name}")

    records = []

    with pymupdf.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf):
            page_text = clean_text(page.get_text("text"))

            if not page_text:
                continue

            chunks = split_into_chunks(
                page_text,
                chunk_size=chunk_size,
                overlap=overlap,
            )

            for chunk_index, chunk in enumerate(chunks):
                records.append(
                    {
                        "document_id": document["document_id"],
                        "project_id": document["project_id"],
                        "source_file": document["filename"],
                        "page_number": page_index + 1,
                        "chunk_index": chunk_index,
                        "text": chunk,
                    }
                )

    return records
