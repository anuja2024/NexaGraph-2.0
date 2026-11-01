from typing import Literal

from fastapi import APIRouter, Header, HTTPException, Query
from pydantic import BaseModel, Field

from app.main import (
    get_authorized_project,
    get_current_user,
    record_audit,
)
from app.workflows.review_service import (
    ReviewNotFoundError,
    ReviewPermissionError,
    ReviewTransitionError,
    get_review_for_user,
    list_review_queue,
    submit_correction_feedback,
    transition_review,
)


router = APIRouter(tags=["human-review"])


class ReviewActionRequest(BaseModel):
    action: Literal[
        "approved",
        "rejected",
        "revision_requested",
        "escalated",
    ]
    comment: str = Field(default="", max_length=2000)


class ReviewCorrectionRequest(BaseModel):
    corrected_answer: str = Field(min_length=1, max_length=20000)
    reviewer_comment: str = Field(default="", max_length=2000)


@router.get("/reviews")
def get_review_queue(
    project_id: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=100),
    x_dev_user: str = Header(...),
    x_request_id: str | None = Header(default=None, alias="X-Request-ID"),
):
    """List reviews visible to the assigned reviewer or manager."""
    user = get_current_user(x_dev_user, x_request_id)

    if project_id is not None:
        get_authorized_project(
            user.user_id,
            project_id,
            event_type="review_queue_access",
            request_id=x_request_id,
        )

    try:
        reviews = list_review_queue(
            user_id=user.user_id,
            project_id=project_id,
            limit=limit,
        )
    except ReviewPermissionError as exc:
        record_audit(
            event_type="review_queue_access",
            outcome="denied",
            user_id=user.user_id,
            project_id=project_id,
            request_id=x_request_id,
            details={"reason": "review_permission_denied"},
        )
        raise HTTPException(
            status_code=403,
            detail="You are not authorized to access this review queue",
        ) from exc

    record_audit(
        event_type="review_queue_access",
        outcome="success",
        user_id=user.user_id,
        project_id=project_id,
        request_id=x_request_id,
        details={"result_count": len(reviews)},
    )

    return {"reviews": reviews, "count": len(reviews)}


@router.get("/reviews/{review_id}")
def get_review_detail(
    review_id: int,
    x_dev_user: str = Header(...),
    x_request_id: str | None = Header(default=None, alias="X-Request-ID"),
):
    """Read a single review after verifying reviewer permissions."""
    user = get_current_user(x_dev_user, x_request_id)

    try:
        review = get_review_for_user(
            review_id=review_id,
            user_id=user.user_id,
        )
    except ReviewNotFoundError as exc:
        record_audit(
            event_type="review_detail_access",
            outcome="denied",
            user_id=user.user_id,
            resource_id=str(review_id),
            request_id=x_request_id,
            details={"reason": "review_not_found"},
        )
        raise HTTPException(
            status_code=404,
            detail="Review not found",
        ) from exc
    except ReviewPermissionError as exc:
        record_audit(
            event_type="review_detail_access",
            outcome="denied",
            user_id=user.user_id,
            resource_id=str(review_id),
            request_id=x_request_id,
            details={"reason": "review_permission_denied"},
        )
        raise HTTPException(
            status_code=403,
            detail="You are not authorized to access this review",
        ) from exc

    record_audit(
        event_type="review_detail_access",
        outcome="success",
        user_id=user.user_id,
        project_id=review["project_id"],
        resource_id=str(review_id),
        request_id=x_request_id,
    )

    return {"review": review}


@router.post("/reviews/{review_id}/action")
def perform_review_action(
    review_id: int,
    request_body: ReviewActionRequest,
    x_dev_user: str = Header(...),
    x_request_id: str | None = Header(default=None, alias="X-Request-ID"),
):
    """Apply an authorized review transition and record an audit event."""
    user = get_current_user(x_dev_user, x_request_id)

    try:
        result = transition_review(
            review_id=review_id,
            actor_user_id=user.user_id,
            action=request_body.action,
            comment=request_body.comment,
        )
    except ReviewNotFoundError as exc:
        record_audit(
            event_type="review_action",
            outcome="denied",
            user_id=user.user_id,
            resource_id=str(review_id),
            request_id=x_request_id,
            details={"action": request_body.action, "reason": "review_not_found"},
        )
        raise HTTPException(
            status_code=404,
            detail="Review not found",
        ) from exc
    except ReviewPermissionError as exc:
        record_audit(
            event_type="review_action",
            outcome="denied",
            user_id=user.user_id,
            resource_id=str(review_id),
            request_id=x_request_id,
            details={
                "action": request_body.action,
                "reason": "review_permission_denied",
            },
        )
        raise HTTPException(
            status_code=403,
            detail="You are not authorized to perform this review action",
        ) from exc
    except ReviewTransitionError as exc:
        record_audit(
            event_type="review_action",
            outcome="denied",
            user_id=user.user_id,
            resource_id=str(review_id),
            request_id=x_request_id,
            details={
                "action": request_body.action,
                "reason": "invalid_status_transition",
            },
        )
        raise HTTPException(
            status_code=409,
            detail=str(exc),
        ) from exc

    record_audit(
        event_type="review_action",
        outcome="success",
        user_id=user.user_id,
        resource_id=str(review_id),
        request_id=x_request_id,
        details={
            "action": result["status"],
            "previous_status": result["previous_status"],
            "new_status": result["status"],
        },
    )

    return result


@router.post("/reviews/{review_id}/correction")
def submit_review_correction(
    review_id: int,
    request_body: ReviewCorrectionRequest,
    x_dev_user: str = Header(...),
    x_request_id: str | None = Header(default=None, alias="X-Request-ID"),
):
    """Store proposed corrections as evaluation feedback, not verified facts."""
    user = get_current_user(x_dev_user, x_request_id)

    try:
        result = submit_correction_feedback(
            review_id=review_id,
            actor_user_id=user.user_id,
            corrected_answer=request_body.corrected_answer,
            reviewer_comment=request_body.reviewer_comment,
        )
    except ReviewNotFoundError as exc:
        record_audit(
            event_type="review_correction_submission",
            outcome="denied",
            user_id=user.user_id,
            resource_id=str(review_id),
            request_id=x_request_id,
            details={"reason": "review_not_found"},
        )
        raise HTTPException(
            status_code=404,
            detail="Review not found",
        ) from exc
    except ReviewPermissionError as exc:
        record_audit(
            event_type="review_correction_submission",
            outcome="denied",
            user_id=user.user_id,
            resource_id=str(review_id),
            request_id=x_request_id,
            details={"reason": "review_permission_denied"},
        )
        raise HTTPException(
            status_code=403,
            detail="You are not authorized to submit correction feedback",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    record_audit(
        event_type="review_correction_submission",
        outcome="success",
        user_id=user.user_id,
        resource_id=str(review_id),
        request_id=x_request_id,
        details={
            "feedback_id": result["feedback_id"],
            "evaluation_status": result["evaluation_status"],
            "answer_modified": False,
        },
    )

    return result
