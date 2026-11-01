from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from app.auth.service import (
    AuthenticationError,
    TokenValidationError,
    authenticate_employee,
    validate_access_token,
)

router = APIRouter(prefix="/auth", tags=["authentication"])
bearer_scheme = HTTPBearer(auto_error=False)


class LoginRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=128)
    password: str = Field(min_length=1, max_length=1024)


def require_authenticated_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> dict[str, str]:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=401,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        return validate_access_token(credentials.credentials)
    except TokenValidationError as exc:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired access token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


@router.post("/login")
def login(request: LoginRequest):
    try:
        return authenticate_employee(
            user_id=request.user_id,
            password=request.password,
        )
    except AuthenticationError as exc:
        raise HTTPException(
            status_code=401,
            detail="Invalid employee ID or password",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


@router.get("/me")
def get_my_account(
    user: dict[str, str] = Depends(require_authenticated_user),
):
    return {"user": user}
