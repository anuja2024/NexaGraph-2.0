from starlette.responses import JSONResponse

from app.main import app
from app.access_control import USERS
from app.auth.routes import router as auth_router
from app.auth.service import TokenValidationError, validate_access_token
from app.api.review_routes import router as review_router


app.include_router(auth_router)
app.include_router(review_router)


class BearerAuthenticationMiddleware:
    """Require database-validated bearer tokens for protected HTTP routes."""

    PUBLIC_PATHS = {
        "/health",
        "/auth/login",
        "/docs",
        "/openapi.json",
        "/redoc",
    }

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "").rstrip("/") or "/"

        if path in self.PUBLIC_PATHS:
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        authorization = headers.get(b"authorization", b"").decode(
            "latin-1"
        )

        if not authorization.lower().startswith("bearer "):
            response = JSONResponse(
                status_code=401,
                content={"detail": "Authentication required"},
                headers={"WWW-Authenticate": "Bearer"},
            )
            await response(scope, receive, send)
            return

        token = authorization[7:].strip()
        if not token:
            response = JSONResponse(
                status_code=401,
                content={"detail": "Authentication required"},
                headers={"WWW-Authenticate": "Bearer"},
            )
            await response(scope, receive, send)
            return

        try:
            identity = validate_access_token(token)
        except TokenValidationError:
            response = JSONResponse(
                status_code=401,
                content={"detail": "Invalid or expired access token"},
                headers={"WWW-Authenticate": "Bearer"},
            )
            await response(scope, receive, send)
            return
        except Exception:
            response = JSONResponse(
                status_code=503,
                content={"detail": "Authentication service unavailable"},
            )
            await response(scope, receive, send)
            return

        # Existing authorization rules are defined for these known users.
        # Do not allow the database token role to contradict those rules.
        user = USERS.get(identity["user_id"])
        if user is None or user.role != identity["role"]:
            response = JSONResponse(
                status_code=403,
                content={
                    "detail": "This account is not configured for application access"
                },
            )
            await response(scope, receive, send)
            return

        # Never trust a client-supplied X-Dev-User identity.
        updated_headers = [
            (name, value)
            for name, value in scope.get("headers", [])
            if name.lower() != b"x-dev-user"
        ]
        updated_headers.append(
            (b"x-dev-user", identity["user_id"].encode("utf-8"))
        )

        authenticated_scope = dict(scope)
        authenticated_scope["headers"] = updated_headers

        await self.app(authenticated_scope, receive, send)


app.add_middleware(BearerAuthenticationMiddleware)
