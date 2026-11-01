import asyncio
import httpx

from app.review_app import app


async def main():
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://nexagraph.test",
    ) as client:
        tests = [
            (
                "Assigned corporate reviewer",
                "/reviews?project_id=corporate",
                "emp1",
                200,
            ),
            (
                "Other corporate member denied",
                "/reviews?project_id=corporate",
                "emp2",
                403,
            ),
            (
                "Sustainability reviewer",
                "/reviews?project_id=sustainability",
                "emp4",
                200,
            ),
            (
                "Manager can view all review queues",
                "/reviews",
                "manager",
                200,
            ),
            (
                "Unknown user denied",
                "/reviews",
                "unknown-user",
                401,
            ),
        ]

        for label, path, user_id, expected_status in tests:
            response = await client.get(
                path,
                headers={"X-Dev-User": user_id},
            )

            if response.status_code != expected_status:
                raise AssertionError(
                    f"{label}: expected HTTP {expected_status}, "
                    f"got {response.status_code}: {response.text}"
                )

            print(f"PASS: {label} -> HTTP {response.status_code}")

            if response.status_code == 200:
                payload = response.json()
                if not isinstance(payload.get("reviews"), list):
                    raise AssertionError(
                        f"{label}: response has no reviews list"
                    )
                if payload.get("count") != len(payload["reviews"]):
                    raise AssertionError(
                        f"{label}: response count does not match reviews"
                    )

        print("\nREVIEW API HTTP CHECKS PASSED: 5")


asyncio.run(main())
