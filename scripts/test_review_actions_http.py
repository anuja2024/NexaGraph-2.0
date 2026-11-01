import asyncio
import httpx

from app.database import get_connection
from app.review_app import app


created_ids = []


def create_test_review():
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO answer_reviews (
                    project_id, requester_user_id,
                    assigned_reviewer_user_id, escalation_user_id,
                    question, original_answer, current_answer,
                    citations, retrieval_method, status, review_reason
                )
                VALUES (
                    'corporate', 'emp2', 'emp1', 'manager',
                    'Temporary HTTP review test?',
                    'Original test answer.',
                    'Original test answer.',
                    '[]'::jsonb, 'BM25', 'needs_review', 'user_flagged'
                )
                RETURNING review_id
                """
            )
            review_id = cursor.fetchone()[0]

    created_ids.append(review_id)
    return review_id


async def main():
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://nexagraph.test",
    ) as client:

        # Correction submission must not change the stored answer.
        review_id = create_test_review()
        response = await client.post(
            f"/reviews/{review_id}/correction",
            headers={"X-Dev-User": "emp1"},
            json={
                "corrected_answer": "Proposed corrected answer.",
                "reviewer_comment": "Temporary HTTP test.",
            },
        )
        assert response.status_code == 200, response.text
        correction = response.json()
        assert correction["evaluation_status"] == "pending"
        assert correction["answer_modified"] is False
        print("PASS: correction submitted as pending evaluation")

        with get_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT original_answer, current_answer
                    FROM answer_reviews
                    WHERE review_id = %s
                    """,
                    (review_id,),
                )
                original, current = cursor.fetchone()

                cursor.execute(
                    """
                    SELECT evaluation_status
                    FROM answer_correction_feedback
                    WHERE feedback_id = %s
                    """,
                    (correction["feedback_id"],),
                )
                feedback_status = cursor.fetchone()[0]

        assert original == current == "Original test answer."
        assert feedback_status == "pending"
        print("PASS: correction did not overwrite the original answer")

        # Non-assigned project member must not approve.
        response = await client.post(
            f"/reviews/{review_id}/action",
            headers={"X-Dev-User": "emp2"},
            json={"action": "approved", "comment": "Unauthorized test."},
        )
        assert response.status_code == 403, response.text
        print("PASS: unauthorized reviewer action denied with HTTP 403")

        # Assigned reviewer can request revision, then approve.
        response = await client.post(
            f"/reviews/{review_id}/action",
            headers={"X-Dev-User": "emp1"},
            json={
                "action": "revision_requested",
                "comment": "Temporary revision test.",
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["status"] == "revision_requested"
        print("PASS: reviewer can request revision over HTTP")

        response = await client.post(
            f"/reviews/{review_id}/action",
            headers={"X-Dev-User": "emp1"},
            json={"action": "approved", "comment": "Temporary approval test."},
        )
        assert response.status_code == 200, response.text
        assert response.json()["status"] == "approved"
        print("PASS: reviewer can approve over HTTP")

        # Manager must not bypass escalation.
        review_id_2 = create_test_review()
        response = await client.post(
            f"/reviews/{review_id_2}/action",
            headers={"X-Dev-User": "manager"},
            json={"action": "approved", "comment": "Should be denied."},
        )
        assert response.status_code == 403, response.text
        print("PASS: manager cannot approve before escalation")

        # Reviewer escalates, then manager resolves it.
        response = await client.post(
            f"/reviews/{review_id_2}/action",
            headers={"X-Dev-User": "emp1"},
            json={"action": "escalated", "comment": "Temporary escalation test."},
        )
        assert response.status_code == 200, response.text
        assert response.json()["status"] == "escalated"
        print("PASS: reviewer can escalate over HTTP")

        response = await client.post(
            f"/reviews/{review_id_2}/action",
            headers={"X-Dev-User": "manager"},
            json={"action": "approved", "comment": "Temporary manager resolution."},
        )
        assert response.status_code == 200, response.text
        assert response.json()["status"] == "approved"
        print("PASS: manager can resolve escalated review over HTTP")

        # Invalid action values must fail request validation.
        response = await client.post(
            f"/reviews/{review_id_2}/action",
            headers={"X-Dev-User": "manager"},
            json={"action": "delete_everything", "comment": "Invalid action."},
        )
        assert response.status_code == 422, response.text
        print("PASS: invalid action rejected with HTTP 422")

        print("\nREVIEW ACTION HTTP CHECKS PASSED: 8")


try:
    asyncio.run(main())
finally:
    if created_ids:
        with get_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    DELETE FROM answer_correction_feedback
                    WHERE review_id = ANY(%s)
                    """,
                    (created_ids,),
                )
                cursor.execute(
                    """
                    DELETE FROM answer_review_events
                    WHERE review_id = ANY(%s)
                    """,
                    (created_ids,),
                )
                cursor.execute(
                    """
                    DELETE FROM answer_reviews
                    WHERE review_id = ANY(%s)
                    """,
                    (created_ids,),
                )
        print("Temporary test review records cleaned up.")
