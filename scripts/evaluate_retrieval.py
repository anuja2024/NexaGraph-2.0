import getpass
import json
import urllib.error
import urllib.parse
import urllib.request


BASE_URL = "http://127.0.0.1:8001"
TIMEOUT = 30

CASES = [
    {
        "user": "emp1",
        "project": "corporate",
        "query": "revenue",
        "expected_terms": ["revenue"],
    },
    {
        "user": "emp1",
        "project": "corporate",
        "query": "business segments",
        "expected_terms": ["business", "segment"],
    },
    {
        "user": "emp4",
        "project": "sustainability",
        "query": "decarbonization",
        "expected_terms": ["decarbonization"],
    },
    {
        "user": "emp4",
        "project": "sustainability",
        "query": "carbon emissions",
        "expected_terms": ["carbon", "emission"],
    },
    {
        "user": "emp7",
        "project": "industrial_automation",
        "query": "cybersecurity vulnerabilities",
        "expected_terms": ["cybersecurity", "vulnerabilit"],
    },
    {
        "user": "emp7",
        "project": "industrial_automation",
        "query": "firewall security measures",
        "expected_terms": ["firewall", "security"],
    },
]

_TOKEN_CACHE = {}


def post_json(path: str, payload: dict, headers: dict | None = None) -> dict:
    body = json.dumps(payload).encode("utf-8")
    request_headers = {"Content-Type": "application/json"}
    if headers:
        request_headers.update(headers)

    request = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=body,
        headers=request_headers,
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        return json.loads(response.read().decode("utf-8"))


def get_token(user_id: str) -> str:
    if user_id in _TOKEN_CACHE:
        return _TOKEN_CACHE[user_id]

    print(f"\nAuthentication required for {user_id}.")
    password = getpass.getpass(f"Password for {user_id}: ")

    try:
        data = post_json(
            "/auth/login",
            {"user_id": user_id, "password": password},
        )
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Login failed for {user_id} (HTTP {error.code}): {detail}"
        ) from error
    except (OSError, urllib.error.URLError, ValueError) as error:
        raise RuntimeError(
            f"Could not authenticate {user_id}: {error}"
        ) from error

    token = data.get("access_token")
    if not isinstance(token, str) or not token:
        raise RuntimeError(
            f"Login response for {user_id} did not contain an access token."
        )

    _TOKEN_CACHE[user_id] = token
    return token


def search(case: dict) -> dict:
    token = get_token(case["user"])
    params = urllib.parse.urlencode({
        "q": case["query"],
        "limit": 3,
    })
    url = (
        f"{BASE_URL}/projects/{case['project']}/search?"
        f"{params}"
    )

    request = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {token}"},
    )

    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        if response.status != 200:
            raise RuntimeError(
                f"Unexpected HTTP status: {response.status}"
            )
        return json.loads(response.read().decode("utf-8"))


def evaluate_case(case: dict) -> tuple[int, int]:
    print(f"\nQUERY: {case['query']} [{case['project']}] (user: {case['user']})")

    try:
        data = search(case)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        print(f"  FAIL: HTTP {error.code}: {detail}")
        return 0, 1
    except (OSError, urllib.error.URLError, ValueError, RuntimeError) as error:
        print(f"  FAIL: API request failed: {error}")
        return 0, 1

    results = data.get("results", [])
    if not isinstance(results, list):
        print("  FAIL: API results field is not a list")
        return 0, 1

    combined_text = " ".join(
        result.get("text", "").lower()
        for result in results
        if isinstance(result, dict)
        and isinstance(result.get("text"), str)
    )

    checks = {
        "BM25 method": data.get("retrieval_method") == "BM25",
        "has results": len(results) > 0,
        "source metadata": bool(results) and all(
            isinstance(result, dict)
            and isinstance(result.get("source_file"), str)
            and bool(result["source_file"])
            and isinstance(result.get("page_number"), int)
            and not isinstance(result.get("page_number"), bool)
            and result["page_number"] >= 1
            for result in results
        ),
        "expected terms found": all(
            term in combined_text
            for term in case["expected_terms"]
        ),
        "numeric scores": bool(results) and all(
            isinstance(result, dict)
            and isinstance(result.get("score"), (int, float))
            and not isinstance(result.get("score"), bool)
            for result in results
        ),
    }

    passed = 0
    failed = 0

    for name, success in checks.items():
        print(f"  {'PASS' if success else 'FAIL'}: {name}")
        if success:
            passed += 1
        else:
            failed += 1

    for rank, result in enumerate(results, start=1):
        print(
            f"  #{rank}: page {result.get('page_number')} "
            f"| score {result.get('score')} "
            f"| {result.get('source_file')}"
        )

    return passed, failed


def main() -> int:
    total_passed = 0
    total_failed = 0

    print("NexaGraph BM25 Retrieval Evaluation")
    print(f"API: {BASE_URL}")
    print("Passwords are entered privately and are not stored by this script.")

    for case in CASES:
        passed, failed = evaluate_case(case)
        total_passed += passed
        total_failed += failed

    print("\n" + "=" * 55)
    print(f"Checks passed: {total_passed}")
    print(f"Checks failed: {total_failed}")

    if total_failed:
        print("EVALUATION FAILED")
        return 1

    print("EVALUATION PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
