from app.workflows.review_service import (
    ReviewPermissionError,
    can_access_review,
    get_project_review_assignment,
    list_review_queue,
)

checks = 0

def check(condition, description):
    global checks
    if not condition:
        raise AssertionError(f"FAILED: {description}")
    checks += 1
    print(f"PASS: {description}")

# Reviewer assignments must match the confirmed project configuration.
expected = {
    "corporate": ("emp1", "manager"),
    "sustainability": ("emp4", "manager"),
    "industrial_automation": ("emp7", "manager"),
}

for project_id, (reviewer, manager) in expected.items():
    assignment = get_project_review_assignment(project_id)
    check(
        assignment is not None
        and assignment["reviewer_user_id"] == reviewer
        and assignment["escalation_user_id"] == manager,
        f"{project_id} reviewer assignment",
    )

# Assigned reviewers can access their own project reviews.
check(
    can_access_review(
        user_id="emp1",
        project_id="corporate",
        assigned_reviewer_user_id="emp1",
        escalation_user_id="manager",
    ),
    "assigned reviewer can access review",
)

# Other project members must not inherit review permissions.
check(
    not can_access_review(
        user_id="emp2",
        project_id="corporate",
        assigned_reviewer_user_id="emp1",
        escalation_user_id="manager",
    ),
    "ordinary project member cannot access another reviewer's review",
)

# Cross-project access must be denied.
check(
    not can_access_review(
        user_id="emp1",
        project_id="sustainability",
        assigned_reviewer_user_id="emp4",
        escalation_user_id="manager",
    ),
    "reviewer cannot access another project's review",
)

# Manager escalation must work.
check(
    can_access_review(
        user_id="manager",
        project_id="corporate",
        assigned_reviewer_user_id="emp1",
        escalation_user_id="manager",
    ),
    "manager can access escalated review",
)

# Unknown users must not gain review access.
check(
    not can_access_review(
        user_id="unknown-user",
        project_id="corporate",
        assigned_reviewer_user_id="emp1",
        escalation_user_id="manager",
    ),
    "unknown user denied",
)

# Read the assigned review queue; an empty queue is valid at this stage.
for reviewer, project_id in [
    ("emp1", "corporate"),
    ("emp4", "sustainability"),
    ("emp7", "industrial_automation"),
    ("manager", None),
]:
    queue = list_review_queue(user_id=reviewer, project_id=project_id)
    check(isinstance(queue, list), f"queue read for {reviewer}")

print(f"\nREVIEW SERVICE CHECKS PASSED: {checks}")
