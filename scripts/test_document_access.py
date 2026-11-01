from app.access_control import (
    DOCUMENTS,
    can_access_document,
    can_access_project,
    documents_for_project,
)
import app.main as main


def run():
    checks = []

    def check(label, condition):
        checks.append((label, bool(condition)))

    project_to_doc = {
        "corporate": "siemens-annual-report-2025",
        "sustainability": "siemens-sustainability-statement-2025",
        "industrial_automation": "siemens-simatic-cybersecurity-2026",
    }
    memberships = {
        "emp1": "corporate",
        "emp2": "corporate",
        "emp3": "sustainability",
        "emp4": "sustainability",
        "emp5": "industrial_automation",
        "emp6": "industrial_automation",
    }

    # Document and project permission checks.
    for project_id, document_id in project_to_doc.items():
        check(
            f"manager accesses {document_id}",
            can_access_document("manager", document_id, project_id),
        )
        for user_id, assigned_project in memberships.items():
            check(
                f"{user_id} {'can' if assigned_project == project_id else 'cannot'} access {document_id}",
                can_access_document(
                    user_id, document_id, project_id
                ) == (assigned_project == project_id),
            )
        for user_id in ("emp7", "emp8", "emp9", "not-a-user"):
            check(
                f"{user_id} denied {document_id}",
                not can_access_document(user_id, document_id, project_id),
            )
        check(
            f"unknown document denied for {project_id}",
            not can_access_document("manager", "unknown-doc", project_id),
        )

    check(
        "project/document mismatch denied",
        not can_access_document(
            "manager",
            project_to_doc["corporate"],
            "sustainability",
        ),
    )
    check(
        "manager project access to all",
        all(can_access_project("manager", p) for p in project_to_doc),
    )
    check(
        "three registered documents",
        len(DOCUMENTS) == 3
        and sum(len(documents_for_project(p)) for p in project_to_doc) == 3,
    )

    sample_chunks = [
        {
            "project_id": project_id,
            "document_id": document_id,
            "source_file": DOCUMENTS[document_id].source_file,
            "page_number": 1,
            "text": (
                f"unique marker {project_id} document content "
                "security revenue sustainability"
            ),
            "chunk_index": 0,
        }
        for project_id, document_id in project_to_doc.items()
    ]

    original_load_chunks = main.load_chunks
    original_graph_retrieval = main.retrieve_graph_enhanced
    import app.hybrid_retrieval as hybrid_module
    original_hybrid_retrieval = hybrid_module.retrieve_hybrid
    captured = []

    def fake_graph_retrieval(
        *, project_id, query, authorized_chunks, limit
    ):
        captured.append((project_id, list(authorized_chunks)))
        return {
            "retrieval_method": "test_mock",
            "results": list(authorized_chunks)[:limit],
        }

    def fake_hybrid_retrieval(
        *, project_id, query, authorized_chunks, limit, candidate_limit
    ):
        captured.append((project_id, list(authorized_chunks)))
        return {
            "retrieval_method": "test_mock_fallback",
            "results": list(authorized_chunks)[:limit],
        }

    main.load_chunks = lambda: sample_chunks
    main.retrieve_graph_enhanced = fake_graph_retrieval
    hybrid_module.retrieve_hybrid = fake_hybrid_retrieval

    try:
        for user_id, assigned_project in memberships.items():
            result = main.retrieve_project_passages(
                assigned_project,
                "security revenue sustainability",
                10,
                user_id=user_id,
            )
            results = result["results"]
            expected_doc = project_to_doc[assigned_project]

            check(
                f"retrieval for {user_id} returns only assigned project's document",
                bool(results)
                and all(r["document_id"] == expected_doc for r in results),
            )

            other_project = next(
                p for p in project_to_doc if p != assigned_project
            )
            denied_result = main.retrieve_project_passages(
                other_project,
                "security revenue sustainability",
                10,
                user_id=user_id,
            )
            check(
                f"retrieval for {user_id} cannot return another project's document",
                len(denied_result["results"]) == 0,
            )

        manager_result = main.retrieve_project_passages(
            "corporate",
            "security revenue sustainability",
            10,
            user_id="manager",
        )
        manager_results = manager_result["results"]
        check(
            "manager retrieval includes corporate document",
            bool(manager_results)
            and all(
                r["document_id"] == project_to_doc["corporate"]
                for r in manager_results
            ),
        )
    finally:
        main.load_chunks = original_load_chunks
        main.retrieve_graph_enhanced = original_graph_retrieval
        hybrid_module.retrieve_hybrid = original_hybrid_retrieval

    failed = [label for label, passed in checks if not passed]
    print(
        f"Document authorization regression checks: "
        f"{len(checks) - len(failed)}/{len(checks)} passed"
    )
    for label, passed in checks:
        print(f"{'PASS' if passed else 'FAIL'}: {label}")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    run()