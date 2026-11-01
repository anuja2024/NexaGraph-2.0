from app.database import get_connection


CREATE_REVIEW_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS project_review_assignments (
    project_id TEXT PRIMARY KEY,
    reviewer_user_id TEXT NOT NULL,
    escalation_user_id TEXT NOT NULL DEFAULT 'manager',
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK (project_id IN (
        'corporate',
        'sustainability',
        'industrial_automation'
    ))
);

CREATE TABLE IF NOT EXISTS answer_reviews (
    review_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    project_id TEXT NOT NULL,
    requester_user_id TEXT NOT NULL,
    assigned_reviewer_user_id TEXT NOT NULL,
    escalation_user_id TEXT NOT NULL DEFAULT 'manager',
    question TEXT NOT NULL,
    original_answer TEXT NOT NULL,
    current_answer TEXT NOT NULL,
    citations JSONB NOT NULL DEFAULT '[]'::jsonb,
    retrieval_method TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'needs_review',
    review_reason TEXT NOT NULL,
    request_id TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    reviewed_at TIMESTAMPTZ,
    CHECK (project_id IN (
        'corporate',
        'sustainability',
        'industrial_automation'
    )),
    CHECK (status IN (
        'needs_review',
        'approved',
        'rejected',
        'revision_requested',
        'escalated'
    )),
    CHECK (review_reason IN (
        'high_impact',
        'insufficient_evidence',
        'citation_validation',
        'conflicting_evidence',
        'user_flagged',
        'policy_required'
    ))
);

CREATE INDEX IF NOT EXISTS idx_answer_reviews_project_status
    ON answer_reviews (project_id, status, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_answer_reviews_reviewer_status
    ON answer_reviews (
        assigned_reviewer_user_id,
        status,
        created_at DESC
    );

CREATE INDEX IF NOT EXISTS idx_answer_reviews_requester
    ON answer_reviews (requester_user_id, created_at DESC);

CREATE TABLE IF NOT EXISTS answer_review_events (
    event_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    review_id BIGINT NOT NULL REFERENCES answer_reviews(review_id),
    actor_user_id TEXT NOT NULL,
    action TEXT NOT NULL,
    previous_status TEXT,
    new_status TEXT,
    comment TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK (action IN (
        'assigned',
        'approved',
        'rejected',
        'revision_requested',
        'escalated',
        'correction_submitted',
        'reassigned'
    ))
);

CREATE INDEX IF NOT EXISTS idx_answer_review_events_review
    ON answer_review_events (review_id, created_at);

CREATE TABLE IF NOT EXISTS answer_correction_feedback (
    feedback_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    review_id BIGINT NOT NULL REFERENCES answer_reviews(review_id),
    submitted_by_user_id TEXT NOT NULL,
    original_answer TEXT NOT NULL,
    corrected_answer TEXT NOT NULL,
    reviewer_comment TEXT NOT NULL DEFAULT '',
    evaluation_status TEXT NOT NULL DEFAULT 'pending',
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK (evaluation_status IN (
        'pending',
        'accepted_for_evaluation',
        'rejected_for_evaluation'
    ))
);

CREATE INDEX IF NOT EXISTS idx_answer_correction_feedback_review
    ON answer_correction_feedback (review_id, created_at);
"""


def main() -> None:
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(CREATE_REVIEW_SCHEMA_SQL)

            expected_tables = {
                "project_review_assignments",
                "answer_reviews",
                "answer_review_events",
                "answer_correction_feedback",
            }

            cursor.execute(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                  AND table_name = ANY(%s)
                """,
                (list(expected_tables),),
            )
            found_tables = {row[0] for row in cursor.fetchall()}

            missing = expected_tables - found_tables
            if missing:
                raise RuntimeError(
                    f"Review schema initialization failed: {sorted(missing)}"
                )

    print("HUMAN REVIEW DATABASE INITIALIZATION PASSED")
    for table in sorted(expected_tables):
        print(f"Verified table: {table}")
    print("Existing audit_events table was not modified.")


if __name__ == "__main__":
    main()
