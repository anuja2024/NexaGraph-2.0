import re
from collections.abc import Mapping, Sequence
from typing import Any


_CITATION_PATTERN = re.compile(r"\[(S[1-9][0-9]*(?:\s*,\s*S[1-9][0-9]*)*)\]")


class CitationValidationError(ValueError):
    """Raised when an answer contains invalid or unverifiable citations."""


def build_source_registry(
    passages: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Assign deterministic citation IDs to backend-retrieved passages."""
    registry: dict[str, dict[str, Any]] = {}

    for index, passage in enumerate(passages, start=1):
        if not isinstance(passage, Mapping):
            raise CitationValidationError(f"Passage {index} must be a mapping.")

        document_id = passage.get("document_id")
        source_file = passage.get("source_file")
        page_number = passage.get("page_number")
        passage_text = passage.get("text")

        if not isinstance(document_id, str) or not document_id.strip():
            raise CitationValidationError(
                f"Passage {index} has an invalid document_id."
            )

        if not isinstance(source_file, str) or not source_file.strip():
            raise CitationValidationError(
                f"Passage {index} has an invalid source_file."
            )

        if (
            isinstance(page_number, bool)
            or not isinstance(page_number, int)
            or page_number < 1
        ):
            raise CitationValidationError(
                f"Passage {index} has an invalid page_number."
            )

        if not isinstance(passage_text, str) or not passage_text.strip():
            raise CitationValidationError(
                f"Passage {index} has empty or invalid text."
            )

        registry[f"S{index}"] = {
            "document_id": document_id,
            "source_file": source_file,
            "page_number": page_number,
            "text": passage_text,
        }

    return registry


def extract_citation_ids(answer: str) -> list[str]:
    """Extract citation IDs, including comma-separated IDs in one bracket."""
    if not isinstance(answer, str):
        raise CitationValidationError("Answer must be a string.")

    found: list[str] = []
    for match in _CITATION_PATTERN.finditer(answer):
        found.extend(
            item.strip()
            for item in match.group(1).split(",")
        )

    return list(dict.fromkeys(found))


def _validate_source(source_id: str, source: Any) -> None:
    if not isinstance(source, Mapping):
        raise CitationValidationError(
            f"Source [{source_id}] has invalid metadata."
        )

    try:
        build_source_registry([
            {
                "document_id": source.get("document_id"),
                "source_file": source.get("source_file"),
                "page_number": source.get("page_number"),
                "text": source.get("text"),
            }
        ])
    except CitationValidationError as exc:
        raise CitationValidationError(
            f"Source [{source_id}] has invalid metadata: {exc}"
        ) from exc


def validate_citations(
    answer: str,
    source_registry: Mapping[str, Mapping[str, Any]],
    *,
    require_citation: bool = True,
) -> list[str]:
    """Ensure all citation IDs map to retrieved passages with valid metadata."""
    if not isinstance(answer, str) or not answer.strip():
        raise CitationValidationError("Answer must not be empty.")

    if not isinstance(source_registry, Mapping):
        raise CitationValidationError("Source registry must be a mapping.")

    citation_ids = extract_citation_ids(answer)

    if require_citation and not citation_ids:
        raise CitationValidationError(
            "Answer contains no valid citation IDs."
        )

    for source_id in citation_ids:
        if source_id not in source_registry:
            raise CitationValidationError(
                f"Unknown citation ID: [{source_id}]."
            )
        _validate_source(source_id, source_registry[source_id])

    return citation_ids


def render_citation_metadata(
    citation_ids: Sequence[str],
    source_registry: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return citation metadata from trusted backend-retrieved sources."""
    output: list[dict[str, Any]] = []
    seen: set[str] = set()

    for source_id in citation_ids:
        if source_id in seen:
            continue
        seen.add(source_id)

        if source_id not in source_registry:
            raise CitationValidationError(
                f"Unknown citation ID: [{source_id}]."
            )

        source = source_registry[source_id]
        _validate_source(source_id, source)

        output.append({
            "citation_id": source_id,
            "document_id": source["document_id"],
            "source_file": source["source_file"],
            "page_number": source["page_number"],
        })

    return output
