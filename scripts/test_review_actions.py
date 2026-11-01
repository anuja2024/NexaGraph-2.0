from app.database import get_connection
from app.workflows.review_service import (
    ReviewPermissionError,
    ReviewTransitionError,
    submit_correction_feedback,
    transition_review,
)

created_ids = []

def check(condition, message):
    if not condition:
        raise AssertionError(f"FAILED: {message}")
    print(f"PASS: {message}")

def create_test_review(project_id, reviewer_id):
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
                    %s, %s, %s, 'manager',
                    %s, %s, %s, '[]'::jsonb, 'BM25',
                    'needs_review', 'user_flagged'
                )
                RETURNING review_id
                """,
                (
                    project_id,
                    "emp2",
                    reviewer_id,
                    "Temporary review service test question?",
                    "Original test answer.",
                    "Original test answer.",
                ),
            )
            review_id = cursor.fetchone()[0]
    created_ids.append(review_id)
    return review_id

try:
    # Test correction storage without modifying the answer.
    review_id = create_test_review("corporate", "emp1")

    feedback = submit_correction_feedback(
        review_id=review_id,
        actor_user_id="emp1",
        corrected_answer="Proposed corrected test answer.",
        reviewer_comment="Temporary integrity test.",
    )

    check(
        feedback["evaluation_status"] == "pending",
        "correction feedback starts pending evaluation",
    )
    check(
        feedback["answer_modified"] is False,
        "correction submission does not claim to modify the answer",
    )

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
                (feedback["feedback_id"],),
            )
            feedback_status = cursor.fetchone()[0]

    check(
        original == "Original test answer."
        and current == "Original test answer.",
        "original and current answer remain unchanged",
    )
    check(
        feedback_status == "pending",
        "database stores correction as pending",
    )

    # The assigned reviewer can request revision, then approve.
    revision = transition_review(
        review_id=review_id,
        actor_user_id="emp1",
        action="revision_requested",
        comment="Temporary revision test.",
    )
    check(
        revision["previous_status"] == "needs_review"
        and revision["status"] == "revision_requested",
        "reviewer can request revision",
    )

    approval = transition_review(
        review_id=review_id,
        actor_user_id="emp1",
        action="approved",
        comment="Temporary approval test.",
    )
    check(
        approval["status"] == "approved",
        "assigned reviewer can approve revision-requested review",
    )

    # An ordinary project member must not take review actions.
    second_id = create_test_review("corporate", "emp1")
    denied = False
    try:
        transition_review(
            review_id=second_id,
            actor_user_id="emp2",
            action="approved",
            comment="Should be denied.",
        )
    except ReviewPermissionError:
        denied = True

    check(denied, "non-assigned project member cannot approve")

    # Assigned reviewer escalates; only the manager resolves escalation.
    escalation_id = create_test_review("corporate", "emp1")
    escalated = transition_review(
        review_id=escalation_id,
        actor_user_id="emp1",
        action="escalated",
        comment="Temporary escalation test.",
    )
    check(
        escalated["status"] == "escalated",
        "assigned reviewer can escalate",
    )

    manager_result = transition_review(
        review_id=escalation_id,
        actor_user_id="manager",
        action="approved",
        comment="Temporary manager resolution test.",
    )
    check(
        manager_result["status"] == "approved",
        "manager can resolve an escalated review",
    )

    # A manager cannot bypass escalation for an ordinary active review.
    bypass_id = create_test_review("corporate", "emp1")
    manager_denied = False
    try:
        transition_review(
            review_id=bypass_id,
            actor_user_id="manager",
            action="approved",
            comment="Should be denied before escalation.",
        )
    except ReviewPermissionError:
        manager_denied = True

    check(
        manager_denied,
        "manager cannot approve an un-escalated review",
    )

    print("\nREVIEW ACTION INTEGRITY CHECKS PASSED")

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
        print("Temporary test reviews and their related records cleaned up.")
