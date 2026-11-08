"""Call an OpenAI-compatible Chat Completions endpoint."""
import os
from urllib.parse import urlsplit

import httpx


def complete_chat(payload: dict) -> str:
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").strip().rstrip("/")
    model = os.environ.get("OPENAI_MODEL", "").strip()
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    endpoint = urlsplit(base_url)
    if endpoint.scheme not in ("http", "https") or not endpoint.netloc:
        raise ValueError("OPENAI_BASE_URL must be an http(s) API base URL, normally ending in /v1")
    if not model:
        raise ValueError("Set OPENAI_MODEL to a model name supported by your API provider")
    if endpoint.hostname == "api.openai.com" and not api_key:
        raise ValueError("OPENAI_API_KEY is required for the OpenAI API")

    body = dict(payload)
    body["model"] = model
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    # Local endpoints bypass proxies; remote endpoints use the configured network environment.
    trust_env = endpoint.hostname not in {"localhost", "127.0.0.1", "::1"}
    with httpx.Client(timeout=180.0, trust_env=trust_env) as client:
        response = client.post(base_url + "/chat/completions", json=body, headers=headers)
        response.raise_for_status()
        data = response.json()

    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("API response is missing choices[0].message.content") from exc
    if not isinstance(content, str):
        raise ValueError("Expected a text response in choices[0].message.content")
    return content
