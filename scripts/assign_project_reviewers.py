from app.database import get_connection

EXPECTED = {
    "corporate": ("emp1", "manager"),
    "sustainability": ("emp4", "manager"),
    "industrial_automation": ("emp7", "manager"),
}

with get_connection() as conn:
    with conn.cursor() as cur:
        for project_id, (reviewer_id, escalation_id) in EXPECTED.items():
            cur.execute(
                """
                INSERT INTO project_review_assignments
                    (project_id, reviewer_user_id, escalation_user_id, updated_at)
                VALUES (%s, %s, %s, NOW())
                ON CONFLICT (project_id)
                DO UPDATE SET
                    reviewer_user_id = EXCLUDED.reviewer_user_id,
                    escalation_user_id = EXCLUDED.escalation_user_id,
                    updated_at = NOW()
                """,
                (project_id, reviewer_id, escalation_id),
            )

        cur.execute(
            """
            SELECT project_id, reviewer_user_id, escalation_user_id
            FROM project_review_assignments
            ORDER BY project_id
            """
        )
        rows = cur.fetchall()

actual = {
    project_id: (reviewer_id, escalation_id)
    for project_id, reviewer_id, escalation_id in rows
}

for project_id, expected in EXPECTED.items():
    if actual.get(project_id) != expected:
        raise RuntimeError(
            f"Assignment verification failed for {project_id}: "
            f"expected {expected}, got {actual.get(project_id)}"
        )
    print(
        f"Verified: {project_id} -> reviewer={expected[0]}, "
        f"escalation={expected[1]}"
    )

print("\nPROJECT REVIEWER ASSIGNMENTS PASSED")
