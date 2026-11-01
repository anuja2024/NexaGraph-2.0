import os
from datetime import datetime, timedelta, timezone
from typing import Any

import jwt
from pwdlib import PasswordHash
from psycopg.errors import UniqueViolation

from app.database import get_connection


_password_hasher = PasswordHash.recommended()

TOKEN_ALGORITHM = "HS256"
TOKEN_ISSUER = "nexagraph-next"
TOKEN_AUDIENCE = "nexagraph-employee"
DEFAULT_TOKEN_MINUTES = 30


class AuthenticationError(Exception):
    """Raised when employee authentication fails."""


class AccountProvisioningError(Exception):
    """Raised when an employee account cannot be provisioned."""


class TokenValidationError(Exception):
    """Raised when an authentication token is invalid or expired."""


def _token_secret() -> str:
    secret = os.getenv("NEXAGRAPH_AUTH_SECRET", "")
    if len(secret) < 32:
        raise RuntimeError(
            "NEXAGRAPH_AUTH_SECRET must contain at least 32 characters"
        )
    return secret


def hash_password(password: str) -> str:
    if len(password) < 12:
        raise ValueError("Password must contain at least 12 characters")
    if len(password) > 1024:
        raise ValueError("Password is too long")
    return _password_hasher.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return _password_hasher.verify(password, password_hash)
    except (ValueError, TypeError):
        return False


def provision_employee(
    *,
    user_id: str,
    password: str,
    role: str = "employee",
) -> dict[str, Any]:
    """Create an account without returning or logging its password hash."""
    normalized_user_id = user_id.strip().lower()

    if not normalized_user_id or len(normalized_user_id) > 128:
        raise ValueError("Employee ID must contain 1–128 characters")

    if any(char.isspace() for char in normalized_user_id):
        raise ValueError("Employee ID must not contain whitespace")

    if role not in {"employee", "manager"}:
        raise ValueError("Role must be employee or manager")

    password_hash = hash_password(password)

    try:
        with get_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO employee_accounts
                        (user_id, password_hash, role, is_active)
                    VALUES (%s, %s, %s, TRUE)
                    RETURNING user_id, role, is_active, created_at
                    """,
                    (normalized_user_id, password_hash, role),
                )
                row = cursor.fetchone()
    except UniqueViolation as exc:
        raise AccountProvisioningError(
            "An account with that employee ID already exists"
        ) from exc

    return {
        "user_id": row[0],
        "role": row[1],
        "is_active": row[2],
        "created_at": row[3].isoformat() if row[3] else None,
    }


def authenticate_employee(
    *,
    user_id: str,
    password: str,
) -> dict[str, Any]:
    """Verify credentials and issue a short-lived signed access token."""
    normalized_user_id = user_id.strip().lower()

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT user_id, password_hash, role, is_active
                FROM employee_accounts
                WHERE user_id = %s
                """,
                (normalized_user_id,),
            )
            row = cursor.fetchone()

            if row is None:
                raise AuthenticationError("Invalid employee ID or password")

            stored_user_id, stored_hash, role, is_active = row

            if not verify_password(password, stored_hash) or not is_active:
                raise AuthenticationError(
                    "Invalid employee ID or password"
                )

            now = datetime.now(timezone.utc)
            expires = now + timedelta(minutes=DEFAULT_TOKEN_MINUTES)

            token = jwt.encode(
                {
                    "sub": stored_user_id,
                    "role": role,
                    "iss": TOKEN_ISSUER,
                    "aud": TOKEN_AUDIENCE,
                    "iat": now,
                    "exp": expires,
                },
                _token_secret(),
                algorithm=TOKEN_ALGORITHM,
            )

            cursor.execute(
                """
                UPDATE employee_accounts
                SET last_login_at = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP
                WHERE user_id = %s
                """,
                (stored_user_id,),
            )

    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_in": DEFAULT_TOKEN_MINUTES * 60,
        "user": {
            "user_id": stored_user_id,
            "role": role,
        },
    }


def validate_access_token(token: str) -> dict[str, str]:
    """Validate signature, issuer, audience, and expiration."""
    try:
        claims = jwt.decode(
            token,
            _token_secret(),
            algorithms=[TOKEN_ALGORITHM],
            issuer=TOKEN_ISSUER,
            audience=TOKEN_AUDIENCE,
            options={"require": ["sub", "exp", "iat", "iss", "aud", "role"]},
        )
    except (jwt.InvalidTokenError, RuntimeError) as exc:
        raise TokenValidationError("Invalid or expired access token") from exc

    user_id = claims.get("sub")
    role = claims.get("role")

    if not isinstance(user_id, str) or role not in {"employee", "manager"}:
        raise TokenValidationError("Invalid access token claims")

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT role, is_active
                FROM employee_accounts
                WHERE user_id = %s
                """,
                (user_id,),
            )
            row = cursor.fetchone()

    if row is None or not row[1] or row[0] != role:
        raise TokenValidationError("Employee account is inactive or unavailable")

    return {"user_id": user_id, "role": role}
