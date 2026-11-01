import json
from typing import Any

from psycopg.types.json import Jsonb

from app.database import get_connection


ALLOWED_OUTCOMES = {
    "success",
    "denied",
    "failure",
}


def log_event(
    *,
    event_type: str,
    outcome: str,
    user_id: str | None = None,
    project_id: str | None = None,
    resource_id: str | None = None,
    request_id: str | None = None,
    details: dict[str, Any] | None = None,
) -> int:
    """Persist one audit event and return its database event ID."""
    if not event_type or not event_type.strip():
        raise ValueError("event_type must not be empty")

    # Normalize the API's error outcome to the audit schema's failure outcome.
    if outcome == "error":
        outcome = "failure"

    if outcome not in ALLOWED_OUTCOMES:
        raise ValueError("Unsupported audit outcome")

    if details is None:
        details = {}

    if not isinstance(details, dict):
        raise ValueError("details must be a dictionary")

    # Validate JSON serializability before opening a database connection.
    json.dumps(details)

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO audit_events (
                    user_id,
                    event_type,
                    project_id,
                    resource_id,
                    outcome,
                    request_id,
                    details
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING event_id
                """,
                (
                    user_id,
                    event_type.strip(),
                    project_id,
                    resource_id,
                    outcome,
                    request_id,
                    Jsonb(details),
                ),
            )
            row = cursor.fetchone()

        if row is None:
            raise RuntimeError(
                "Audit event insert returned no event ID"
            )

        return int(row[0])
