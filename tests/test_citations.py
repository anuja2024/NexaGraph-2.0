import unittest

from app.citations import (
    CitationValidationError,
    build_source_registry,
    extract_citation_ids,
    render_citation_metadata,
    validate_citations,
)


class CitationValidationTests(unittest.TestCase):
    def setUp(self):
        self.passages = [
            {
                "document_id": "siemens-annual-report-2025",
                "source_file": "siemens_annual_report_2025.pdf",
                "page_number": 22,
                "text": "Revenue increased to EUR 78.9 billion.",
            },
            {
                "document_id": "siemens-annual-report-2025",
                "source_file": "siemens_annual_report_2025.pdf",
                "page_number": 162,
                "text": "Industrial Business profit increased.",
            },
        ]
        self.registry = build_source_registry(self.passages)

    def test_builds_backend_assigned_source_ids(self):
        self.assertEqual(
            list(self.registry.keys()),
            ["S1", "S2"],
        )
        self.assertEqual(
            self.registry["S1"]["page_number"],
            22,
        )

    def test_accepts_valid_citations(self):
        result = validate_citations(
            "Revenue increased [S1]. Industrial Business profit increased [S2].",
            self.registry,
        )
        self.assertEqual(result, ["S1", "S2"])

    def test_extracts_unique_citations_in_order(self):
        self.assertEqual(
            extract_citation_ids("First [S2], then [S1], then [S2]."),
            ["S2", "S1"],
        )

    def test_rejects_fabricated_source_id(self):
        with self.assertRaisesRegex(
            CitationValidationError,
            "Unknown citation ID",
        ):
            validate_citations("Revenue increased [S9].", self.registry)

    def test_rejects_answer_without_citations(self):
        with self.assertRaisesRegex(
            CitationValidationError,
            "no valid citation IDs",
        ):
            validate_citations("Revenue increased.", self.registry)

    def test_can_explicitly_allow_uncited_text(self):
        result = validate_citations(
            "General introductory text.",
            self.registry,
            require_citation=False,
        )
        self.assertEqual(result, [])

    def test_rejects_empty_answer(self):
        with self.assertRaises(CitationValidationError):
            validate_citations("  ", self.registry)

    def test_rejects_invalid_page_number(self):
        invalid_passage = [
            {
                "document_id": "doc-1",
                "source_file": "report.pdf",
                "page_number": 0,
                "text": "Evidence text.",
            }
        ]
        with self.assertRaisesRegex(
            CitationValidationError,
            "invalid page_number",
        ):
            build_source_registry(invalid_passage)

    def test_rejects_boolean_page_number(self):
        invalid_passage = [
            {
                "document_id": "doc-1",
                "source_file": "report.pdf",
                "page_number": True,
                "text": "Evidence text.",
            }
        ]
        with self.assertRaisesRegex(
            CitationValidationError,
            "invalid page_number",
        ):
            build_source_registry(invalid_passage)

    def test_rejects_missing_source_text(self):
        invalid_passage = [
            {
                "document_id": "doc-1",
                "source_file": "report.pdf",
                "page_number": 1,
                "text": " ",
            }
        ]
        with self.assertRaisesRegex(
            CitationValidationError,
            "empty or invalid text",
        ):
            build_source_registry(invalid_passage)

    def test_rejects_invalid_registry_metadata(self):
        invalid_registry = {
            "S1": {
                "document_id": "doc-1",
                "source_file": "report.pdf",
                "page_number": -1,
                "text": "Evidence text.",
            }
        }
        with self.assertRaisesRegex(
            CitationValidationError,
            "invalid metadata",
        ):
            validate_citations("Evidence [S1].", invalid_registry)

    def test_returns_metadata_from_backend_registry(self):
        citations = validate_citations(
            "Revenue increased [S1].",
            self.registry,
        )
        metadata = render_citation_metadata(citations, self.registry)

        self.assertEqual(
            metadata,
            [
                {
                    "citation_id": "S1",
                    "document_id": "siemens-annual-report-2025",
                    "source_file": "siemens_annual_report_2025.pdf",
                    "page_number": 22,
                }
            ],
        )

    def test_rejects_unknown_id_when_rendering_metadata(self):
        with self.assertRaisesRegex(
            CitationValidationError,
            "Unknown citation ID",
        ):
            render_citation_metadata(["S99"], self.registry)

    def test_rejects_missing_document_id(self):
        invalid_passage = [
            {
                "document_id": "",
                "source_file": "report.pdf",
                "page_number": 1,
                "text": "Evidence text.",
            }
        ]
        with self.assertRaisesRegex(
            CitationValidationError,
            "invalid document_id",
        ):
            build_source_registry(invalid_passage)


if __name__ == "__main__":
    unittest.main(verbosity=2)
