from app.database import get_connection


CREATE_AUDIT_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS audit_events (
    event_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    occurred_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    user_id TEXT,
    event_type TEXT NOT NULL,
    project_id TEXT,
    resource_id TEXT,
    outcome TEXT NOT NULL,
    request_id TEXT,
    details JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_audit_events_occurred_at
    ON audit_events (occurred_at DESC);

CREATE INDEX IF NOT EXISTS idx_audit_events_user_time
    ON audit_events (user_id, occurred_at DESC);

CREATE INDEX IF NOT EXISTS idx_audit_events_project_time
    ON audit_events (project_id, occurred_at DESC);

CREATE INDEX IF NOT EXISTS idx_audit_events_event_type
    ON audit_events (event_type);
"""


def main() -> None:
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(CREATE_AUDIT_TABLE_SQL)

            cursor.execute(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                  AND table_name = 'audit_events'
                """
            )
            result = cursor.fetchone()

            if result is None:
                raise RuntimeError(
                    "Audit table was not found after initialization"
                )

            cursor.execute(
                """
                SELECT COUNT(*)
                FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND table_name = 'audit_events'
                """
            )
            column_count = cursor.fetchone()[0]

    print("AUDIT DATABASE INITIALIZATION PASSED")
    print("Database: nexagraph_next_audit")
    print("Table: audit_events")
    print(f"Columns verified: {column_count}")


if __name__ == "__main__":
    main()
