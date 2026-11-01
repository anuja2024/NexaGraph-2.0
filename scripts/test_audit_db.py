from app.database import get_connection


def main() -> None:
    test_request_id = "nexagraph-audit-self-test"

    event_id = None

    with get_connection() as connection:
        try:
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
                    VALUES (
                        %s, %s, %s, %s, %s, %s, %s::jsonb
                    )
                    RETURNING event_id
                    """,
                    (
                        "test-user",
                        "database_self_test",
                        "corporate",
                        None,
                        "success",
                        test_request_id,
                        '{"test": true, "sensitive_data": false}',
                    ),
                )
                event_id = cursor.fetchone()[0]

            connection.commit()
            print(f"Test event inserted: {event_id}")

            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT
                        event_id,
                        user_id,
                        event_type,
                        project_id,
                        outcome,
                        request_id,
                        details
                    FROM audit_events
                    WHERE event_id = %s
                    """,
                    (event_id,),
                )
                row = cursor.fetchone()

            if row is None:
                raise RuntimeError(
                    "Inserted audit event could not be retrieved"
                )

            if (
                row[1] != "test-user"
                or row[2] != "database_self_test"
                or row[5] != test_request_id
                or row[6].get("test") is not True
            ):
                raise RuntimeError(
                    "Retrieved audit event did not match inserted data"
                )

            print("Test event retrieved and verified.")

        finally:
            if event_id is not None:
                with connection.cursor() as cursor:
                    cursor.execute(
                        """
                        DELETE FROM audit_events
                        WHERE event_id = %s
                        """,
                        (event_id,),
                    )
                connection.commit()
                print("Temporary test event removed.")

    print("AUDIT WRITE/READ TEST PASSED")


if __name__ == "__main__":
    main()
