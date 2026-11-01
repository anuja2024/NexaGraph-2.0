from fastapi.testclient import TestClient

from app.review_app import app


def main():
    client = TestClient(app)

    # Public health check must remain available.
    health = client.get("/health")
    print("Public health endpoint:", health.status_code == 200)

    # Protected routes must reject requests without a bearer token.
    projects = client.get(
        "/projects",
        headers={"X-Dev-User": "manager"},
    )
    print("Projects reject missing token:", projects.status_code == 401)

    search = client.get(
        "/projects/corporate/search",
        params={"q": "Siemens"},
        headers={"X-Dev-User": "emp1"},
    )
    print("Search rejects missing token:", search.status_code == 401)

    reviews = client.get(
        "/reviews",
        headers={"X-Dev-User": "manager"},
    )
    print("Reviews reject missing token:", reviews.status_code == 401)

    # Invalid tokens must not authenticate a request.
    invalid = client.get(
        "/projects",
        headers={"Authorization": "Bearer invalid-token"},
    )
    print("Invalid token rejected:", invalid.status_code == 401)

    # Login endpoint should validate input and handle incorrect credentials
    # without returning a server error.
    login = client.post(
        "/auth/login",
        json={
            "user_id": "nonexistent_security_test_user",
            "password": "Incorrect-Test-Password-2026!",
        },
    )
    print("Invalid login safely rejected:", login.status_code == 401)

    assert health.status_code == 200
    assert projects.status_code == 401
    assert search.status_code == 401
    assert reviews.status_code == 401
    assert invalid.status_code == 401
    assert login.status_code == 401

    print("SECURITY BOUNDARY TESTS PASSED")


if __name__ == "__main__":
    main()
