import json
from typing import Any

from app.access_control import USERS, can_access_project
from app.database import get_connection


ACTIVE_REVIEW_STATUSES = (
    "needs_review",
    "revision_requested",
    "escalated",
)

REVIEWER_ACTIONS = {
    "approved",
    "rejected",
    "revision_requested",
    "escalated",
}

MANAGER_ACTIONS = {
    "approved",
    "rejected",
    "revision_requested",
}


class ReviewPermissionError(Exception):
    """The user is not authorized to access or act on this review."""


class ReviewNotFoundError(Exception):
    """The requested review does not exist or is not visible to the user."""


class ReviewTransitionError(Exception):
    """The requested review status transition is not allowed."""


def get_project_review_assignment(project_id: str) -> dict[str, str] | None:
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT project_id, reviewer_user_id, escalation_user_id
                FROM project_review_assignments
                WHERE project_id = %s
                """,
                (project_id,),
            )
            row = cursor.fetchone()

    if row is None:
        return None

    return {
        "project_id": row[0],
        "reviewer_user_id": row[1],
        "escalation_user_id": row[2],
    }


def can_access_review(
    *,
    user_id: str,
    project_id: str,
    assigned_reviewer_user_id: str,
    escalation_user_id: str,
) -> bool:
    user = USERS.get(user_id)
    if user is None or not can_access_project(user_id, project_id):
        return False

    if user_id == assigned_reviewer_user_id:
        return True

    return (
        user_id == escalation_user_id
        and user.role == "manager"
    )


def _decode_review(row: tuple) -> dict[str, Any]:
    citations = row[8]
    if isinstance(citations, str):
        citations = json.loads(citations)

    return {
        "review_id": row[0],
        "project_id": row[1],
        "requester_user_id": row[2],
        "assigned_reviewer_user_id": row[3],
        "escalation_user_id": row[4],
        "question": row[5],
        "original_answer": row[6],
        "current_answer": row[7],
        "citations": citations,
        "retrieval_method": row[9],
        "status": row[10],
        "review_reason": row[11],
        "request_id": row[12],
        "created_at": row[13].isoformat() if row[13] else None,
        "updated_at": row[14].isoformat() if row[14] else None,
        "reviewed_at": row[15].isoformat() if row[15] else None,
    }


REVIEW_SELECT = """
    SELECT review_id, project_id, requester_user_id,
           assigned_reviewer_user_id, escalation_user_id,
           question, original_answer, current_answer,
           citations, retrieval_method, status, review_reason,
           request_id, created_at, updated_at, reviewed_at
    FROM answer_reviews
"""


def list_review_queue(
    *,
    user_id: str,
    project_id: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    user = USERS.get(user_id)
    if user is None:
        raise ReviewPermissionError("Unknown user")

    if not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100")

    with get_connection() as connection:
        with connection.cursor() as cursor:
            if project_id is not None:
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
                    return []

                reviewer_id, escalation_id = assignment

                if not can_access_project(user_id, project_id):
                    raise ReviewPermissionError("Project access denied")

                if not can_access_review(
                    user_id=user_id,
                    project_id=project_id,
                    assigned_reviewer_user_id=reviewer_id,
                    escalation_user_id=escalation_id,
                ):
                    raise ReviewPermissionError("Review access denied")

                cursor.execute(
                    REVIEW_SELECT + """
                    WHERE project_id = %s
                      AND status = ANY(%s)
                      AND (
                          assigned_reviewer_user_id = %s
                          OR (%s = escalation_user_id AND %s = 'manager')
                      )
                    ORDER BY created_at ASC
                    LIMIT %s
                    """,
                    (
                        project_id, list(ACTIVE_REVIEW_STATUSES),
                        user_id, user_id, user.role, limit,
                    ),
                )
            elif user.role == "manager":
                cursor.execute(
                    REVIEW_SELECT + """
                    WHERE status = ANY(%s)
                    ORDER BY created_at ASC
                    LIMIT %s
                    """,
                    (list(ACTIVE_REVIEW_STATUSES), limit),
                )
            else:
                cursor.execute(
                    REVIEW_SELECT + """
                    WHERE assigned_reviewer_user_id = %s
                      AND status = ANY(%s)
                    ORDER BY created_at ASC
                    LIMIT %s
                    """,
                    (user_id, list(ACTIVE_REVIEW_STATUSES), limit),
                )

            rows = cursor.fetchall()

    return [_decode_review(row) for row in rows]


def get_review_for_user(
    *,
    review_id: int,
    user_id: str,
) -> dict[str, Any]:
    if user_id not in USERS:
        raise ReviewPermissionError("Unknown user")

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                REVIEW_SELECT + " WHERE review_id = %s",
                (review_id,),
            )
            row = cursor.fetchone()

    if row is None:
        raise ReviewNotFoundError("Review not found")

    if not can_access_review(
        user_id=user_id,
        project_id=row[1],
        assigned_reviewer_user_id=row[3],
        escalation_user_id=row[4],
    ):
        raise ReviewPermissionError("Review access denied")

    return _decode_review(row)


def _get_locked_review(cursor, review_id: int) -> tuple:
    cursor.execute(
        REVIEW_SELECT + " WHERE review_id = %s FOR UPDATE",
        (review_id,),
    )
    row = cursor.fetchone()

    if row is None:
        raise ReviewNotFoundError("Review not found")

    return row


def transition_review(
    *,
    review_id: int,
    actor_user_id: str,
    action: str,
    comment: str = "",
) -> dict[str, Any]:
    """Change review status and append its event atomically."""
    if action not in REVIEWER_ACTIONS:
        raise ReviewTransitionError("Unsupported review action")

    actor = USERS.get(actor_user_id)
    if actor is None:
        raise ReviewPermissionError("Unknown user")

    with get_connection() as connection:
        with connection.cursor() as cursor:
            row = _get_locked_review(cursor, review_id)

            (
                _, project_id, _, assigned_reviewer_id, escalation_id,
                _, _, _, _, _, current_status, _, _, _, _, _,
            ) = row

            if not can_access_review(
                user_id=actor_user_id,
                project_id=project_id,
                assigned_reviewer_user_id=assigned_reviewer_id,
                escalation_user_id=escalation_id,
            ):
                raise ReviewPermissionError("Review access denied")

            if actor_user_id == assigned_reviewer_id:
                if current_status not in ("needs_review", "revision_requested"):
                    raise ReviewTransitionError(
                        "Assigned reviewer cannot act on this review status"
                    )
                allowed_actions = REVIEWER_ACTIONS
            elif (
                actor.role == "manager"
                and actor_user_id == escalation_id
                and current_status == "escalated"
            ):
                allowed_actions = MANAGER_ACTIONS
            else:
                raise ReviewPermissionError(
                    "Only the assigned reviewer or authorized escalation manager may act"
                )

            if action not in allowed_actions:
                raise ReviewPermissionError("Action is not permitted")

            reviewed_at_sql = (
                "CURRENT_TIMESTAMP"
                if action in ("approved", "rejected")
                else "NULL"
            )

            cursor.execute(
                f"""
                UPDATE answer_reviews
                SET status = %s,
                    updated_at = CURRENT_TIMESTAMP,
                    reviewed_at = {reviewed_at_sql}
                WHERE review_id = %s
                RETURNING updated_at, reviewed_at
                """,
                (action, review_id),
            )
            updated_at, reviewed_at = cursor.fetchone()

            cursor.execute(
                """
                INSERT INTO answer_review_events (
                    review_id, actor_user_id, action,
                    previous_status, new_status, comment
                )
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    review_id, actor_user_id, action,
                    current_status, action, comment.strip(),
                ),
            )

    return {
        "review_id": review_id,
        "previous_status": current_status,
        "status": action,
        "updated_at": updated_at.isoformat() if updated_at else None,
        "reviewed_at": reviewed_at.isoformat() if reviewed_at else None,
    }


def submit_correction_feedback(
    *,
    review_id: int,
    actor_user_id: str,
    corrected_answer: str,
    reviewer_comment: str = "",
) -> dict[str, Any]:
    """Store a correction for evaluation without changing the answer."""
    corrected_answer = corrected_answer.strip()
    if not corrected_answer:
        raise ValueError("corrected_answer must not be empty")

    actor = USERS.get(actor_user_id)
    if actor is None:
        raise ReviewPermissionError("Unknown user")

    with get_connection() as connection:
        with connection.cursor() as cursor:
            row = _get_locked_review(cursor, review_id)

            (
                _, project_id, _, assigned_reviewer_id, escalation_id,
                _, original_answer, _, _, _, current_status, _, _, _, _, _,
            ) = row

            if not can_access_review(
                user_id=actor_user_id,
                project_id=project_id,
                assigned_reviewer_user_id=assigned_reviewer_id,
                escalation_user_id=escalation_id,
            ):
                raise ReviewPermissionError("Review access denied")

            is_assigned_reviewer = actor_user_id == assigned_reviewer_id
            is_escalation_manager = (
                actor.role == "manager"
                and actor_user_id == escalation_id
                and current_status == "escalated"
            )

            if not (
                is_assigned_reviewer
                and current_status in ("needs_review", "revision_requested")
            ) and not is_escalation_manager:
                raise ReviewPermissionError(
                    "Correction feedback is not permitted for this user or status"
                )

            cursor.execute(
                """
                INSERT INTO answer_correction_feedback (
                    review_id, submitted_by_user_id, original_answer,
                    corrected_answer, reviewer_comment, evaluation_status
                )
                VALUES (%s, %s, %s, %s, %s, 'pending')
                RETURNING feedback_id, evaluation_status, created_at
                """,
                (
                    review_id, actor_user_id, original_answer,
                    corrected_answer, reviewer_comment.strip(),
                ),
            )
            feedback_id, evaluation_status, created_at = cursor.fetchone()

            cursor.execute(
                """
                INSERT INTO answer_review_events (
                    review_id, actor_user_id, action,
                    previous_status, new_status, comment
                )
                VALUES (%s, %s, 'correction_submitted', %s, %s, %s)
                """,
                (
                    review_id, actor_user_id, current_status,
                    current_status, reviewer_comment.strip(),
                ),
            )

    return {
        "feedback_id": feedback_id,
        "review_id": review_id,
        "evaluation_status": evaluation_status,
        "created_at": created_at.isoformat() if created_at else None,
        "answer_modified": False,
    }
