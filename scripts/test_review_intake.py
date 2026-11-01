from app.database import get_connection
from app.workflows.review_intake import (
    ReviewIntakeError,
    create_review_request,
)

created_ids = []


def check(condition, message):
    if not condition:
        raise AssertionError(f"FAILED: {message}")
    print(f"PASS: {message}")


try:
    result = create_review_request(
        project_id="corporate",
        requester_user_id="emp2",
        question="Temporary intake test: what is the reported revenue?",
        answer="Temporary test answer. [S1]",
        citations=[
            {
                "citation_id": "S1",
                "document_id": "test-document",
                "source_file": "test.pdf",
                "page_number": 1,
            }
        ],
        retrieval_method="BM25",
        review_reason="high_impact",
        request_id="review-intake-checkpoint-64",
    )

    review_id = result["review_id"]
    created_ids.append(review_id)

    check(result["status"] == "needs_review", "review starts in needs_review")
    check(
        result["assigned_reviewer_user_id"] == "emp1",
        "corporate reviewer assignment applied",
    )
    check(
        result["escalation_user_id"] == "manager",
        "manager escalation owner applied",
    )
    check(
        result["review_reason"] == "high_impact",
        "review reason preserved",
    )

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT question, original_answer, current_answer,
                       citations, retrieval_method, status, review_reason,
                       request_id
                FROM answer_reviews
                WHERE review_id = %s
                """,
                (review_id,),
            )
            stored = cursor.fetchone()

            cursor.execute(
                """
                SELECT actor_user_id, action, previous_status,
                       new_status, comment
                FROM answer_review_events
                WHERE review_id = %s
                ORDER BY event_id
                """,
                (review_id,),
            )
            events = cursor.fetchall()

    check(stored is not None, "review persisted in PostgreSQL")
    check(
        stored[0] == "Temporary intake test: what is the reported revenue?"
        and stored[1] == "Temporary test answer. [S1]"
        and stored[2] == stored[1],
        "question and answer stored without alteration",
    )
    check(
        stored[3] == [
            {
                "citation_id": "S1",
                "document_id": "test-document",
                "source_file": "test.pdf",
                "page_number": 1,
            }
        ],
        "citation metadata persisted",
    )
    check(
        stored[4] == "BM25"
        and stored[5] == "needs_review"
        and stored[6] == "high_impact"
        and stored[7] == "review-intake-checkpoint-64",
        "retrieval method, status, reason, and request ID persisted",
    )
    check(
        len(events) == 1
        and events[0][0] == "emp2"
        and events[0][1] == "assigned"
        and events[0][2] is None
        and events[0][3] == "needs_review",
        "initial assignment event recorded",
    )

    missing_assignment_denied = False
    try:
        create_review_request(
            project_id="unconfigured_project",
            requester_user_id="emp2",
            question="Temporary invalid-project test?",
            answer="Must not be saved.",
            citations=[],
            retrieval_method="BM25",
            review_reason="high_impact",
        )
    except ReviewIntakeError:
        missing_assignment_denied = True

    check(
        missing_assignment_denied,
        "unconfigured project cannot create a review",
    )

    print("\nREVIEW INTAKE CHECKS PASSED: 9")

finally:
    if created_ids:
        with get_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "DELETE FROM answer_review_events WHERE review_id = ANY(%s)",
                    (created_ids,),
                )
                cursor.execute(
                    "DELETE FROM answer_correction_feedback WHERE review_id = ANY(%s)",
                    (created_ids,),
                )
                cursor.execute(
                    "DELETE FROM answer_reviews WHERE review_id = ANY(%s)",
                    (created_ids,),
                )
        print("Temporary intake test records cleaned up.")
