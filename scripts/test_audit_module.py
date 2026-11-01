from app.audit import log_event
from app.database import get_connection


def main() -> None:
    request_id = "nexagraph-audit-module-test"
    event_id = log_event(
        event_type="audit_module_self_test",
        outcome="success",
        user_id="test-user",
        project_id="corporate",
        request_id=request_id,
        details={
            "test": True,
            "component": "audit",
        },
    )

    print(f"Audit module inserted event: {event_id}")

    try:
        with get_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT event_type, outcome, details
                    FROM audit_events
                    WHERE event_id = %s
                    """,
                    (event_id,),
                )
                row = cursor.fetchone()

        if row is None:
            raise RuntimeError("Audit event was not found")

        if (
            row[0] != "audit_module_self_test"
            or row[1] != "success"
            or row[2].get("component") != "audit"
        ):
            raise RuntimeError("Audit event verification failed")

        print("Audit module event verified.")

    finally:
        with get_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "DELETE FROM audit_events WHERE event_id = %s",
                    (event_id,),
                )

        print("Temporary test event removed.")

    print("AUDIT MODULE TEST PASSED")


if __name__ == "__main__":
    main()
