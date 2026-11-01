
import json
import logging
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


logger = logging.getLogger(__name__)


def normalize_model_text(text: str) -> str:
    """Correct the specific euro-symbol mojibake seen in model output."""
    return text.replace("â¬", "€")


def generate_local_answer(
    prompt: str,
    response_format: str | None = None,
) -> str:
    """Generate an answer using a locally running Ollama model."""

    base_url = os.getenv(
        "OLLAMA_BASE_URL",
        "http://127.0.0.1:11434",
    ).rstrip("/")

    model = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
    timeout = int(os.getenv("OLLAMA_TIMEOUT", "180"))

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_ctx": 8192,
        },
    }

    if response_format == "json":
        payload["format"] = "json"

    request = Request(
        url=f"{base_url}/api/generate",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))

        answer = result.get("response")

        if not isinstance(answer, str) or not answer.strip():
            raise RuntimeError("Ollama returned an empty response")

        return normalize_model_text(answer.strip())

    except HTTPError as exc:
        logger.exception("Ollama HTTP request failed")
        raise RuntimeError(
            f"Ollama returned HTTP {exc.code}"
        ) from exc

    except (URLError, TimeoutError, json.JSONDecodeError) as exc:
        logger.exception("Could not communicate with Ollama")
        raise RuntimeError(
            "Could not communicate with the local Ollama service"
        ) from exc
