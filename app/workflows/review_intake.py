from typing import Any

from psycopg.types.json import Jsonb

from app.database import get_connection


ALLOWED_REVIEW_REASONS = {
    "high_impact",
    "insufficient_evidence",
    "citation_validation",
    "conflicting_evidence",
    "user_flagged",
    "policy_required",
}


class ReviewIntakeError(Exception):
    """The review request could not be safely recorded."""


def create_review_request(
    *,
    project_id: str,
    requester_user_id: str,
    question: str,
    answer: str,
    citations: list[dict[str, Any]],
    retrieval_method: str,
    review_reason: str,
    request_id: str | None = None,
) -> dict[str, Any]:
    """Persist an answer awaiting review and its assignment event atomically."""
    if review_reason not in ALLOWED_REVIEW_REASONS:
        raise ValueError("Unsupported review reason")

    if not question.strip():
        raise ValueError("Question must not be empty")

    if not answer.strip():
        raise ValueError("Answer must not be empty")

    if not isinstance(citations, list):
        raise ValueError("Citations must be a list")

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT reviewer_user_id, escalation_user_id
                FROM project_review_assignments
                WHERE project_id = %s
                """,
                (project_id,),
            )
            assignment = cursor.fetchone()

            if assignment is None:
                raise ReviewIntakeError(
                    f"No reviewer assignment configured for project {project_id}"
                )

            reviewer_user_id, escalation_user_id = assignment

            cursor.execute(
                """
                INSERT INTO answer_reviews (
                    project_id,
                    requester_user_id,
                    assigned_reviewer_user_id,
                    escalation_user_id,
                    question,
                    original_answer,
                    current_answer,
                    citations,
                    retrieval_method,
                    status,
                    review_reason,
                    request_id
                )
                VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    'needs_review', %s, %s
                )
                RETURNING review_id, status, created_at
                """,
                (
                    project_id,
                    requester_user_id,
                    reviewer_user_id,
                    escalation_user_id,
                    question.strip(),
                    answer.strip(),
                    answer.strip(),
                    Jsonb(citations),
                    retrieval_method,
                    review_reason,
                    request_id,
                ),
            )

            review_id, status, created_at = cursor.fetchone()

            cursor.execute(
                """
                INSERT INTO answer_review_events (
                    review_id,
                    actor_user_id,
                    action,
                    previous_status,
                    new_status,
                    comment
                )
                VALUES (%s, %s, 'assigned', NULL, %s, %s)
                """,
                (
                    review_id,
                    requester_user_id,
                    status,
                    f"Assigned to {reviewer_user_id}; reason: {review_reason}",
                ),
            )

    return {
        "review_id": review_id,
        "project_id": project_id,
        "assigned_reviewer_user_id": reviewer_user_id,
        "escalation_user_id": escalation_user_id,
        "status": status,
        "review_reason": review_reason,
        "created_at": created_at.isoformat() if created_at else None,
    }
