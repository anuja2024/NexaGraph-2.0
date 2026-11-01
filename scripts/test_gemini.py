import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from google import genai


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(ENV_PATH, override=False)

api_key = os.getenv("GEMINI_API_KEY", "").strip()
model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()

if not api_key or api_key == "PASTE_YOUR_REAL_KEY_HERE":
    print("FAIL: GEMINI_API_KEY is missing from .env")
    raise SystemExit(1)

if not model:
    print("FAIL: GEMINI_MODEL is empty")
    raise SystemExit(1)

print(f"API key loaded: yes (value hidden)")
print(f"Model configured: {model}")

try:
    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model=model,
        contents=(
            "Reply with exactly this text and nothing else: "
            "NEXAGRAPH_GEMINI_CONNECTION_OK"
        ),
    )

    answer = (response.text or "").strip()

    if not answer:
        print("FAIL: Gemini returned an empty response")
        raise SystemExit(1)

    print(f"Gemini response: {answer}")

    if "NEXAGRAPH_GEMINI_CONNECTION_OK" in answer:
        print("PASS: Gemini API connection and generation work")
    else:
        print("WARNING: Gemini responded, but not with the expected text")
        raise SystemExit(1)

except Exception as exc:
    # Print the exception type and message, never the API key.
    print(f"FAIL: {type(exc).__name__}: {exc}")
    raise SystemExit(1)
