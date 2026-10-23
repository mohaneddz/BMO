import json
from pathlib import Path

import httpx
import pytest

import LLM.core as llm_core
from conftest import api
from constants import settings


@pytest.fixture
def groq_requests(monkeypatch) -> list[dict]:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append({"url": str(request.url), "auth": request.headers["authorization"], "body": body})
        reply = f"BMO heard {len(body['messages'])} messages."
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": reply}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17},
            },
        )

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        llm_core.httpx,
        "AsyncClient",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs),
    )
    monkeypatch.setattr(settings, "GROQ_API_KEY", "test-key")
    monkeypatch.setattr(llm_core, "_CONVERSATIONS", {})
    return seen


def test_chat_calls_groq_and_keeps_history(client, groq_requests):
    url = api(settings.LLM_ROUTE_PREFIX, settings.LLM_CHAT_ROUTE)

    first = client.post(url, json={"message": "Hi BMO"})
    assert first.status_code == 200, first.text
    first_body = first.json()
    assert first_body["reply"] == "BMO heard 2 messages."
    assert first_body["turns"] == 1
    assert first_body["usage"]["total_tokens"] == 17
    assert json.loads(Path(first_body["saved_to"]).read_text(encoding="utf-8"))["reply"] == first_body["reply"]

    second = client.post(url, json={"message": "Remember me?", "conversation_id": first_body["conversation_id"]})
    assert second.status_code == 200, second.text
    assert second.json()["turns"] == 2

    sent = groq_requests[1]
    assert sent["url"] == f"{settings.GROQ_BASE_URL}{settings.LLM_ENDPOINT}"
    assert sent["auth"] == "Bearer test-key"
    assert sent["body"]["model"] == settings.LLM_MODEL_NAME
    assert [m["role"] for m in sent["body"]["messages"]] == ["system", "user", "assistant", "user"]
    assert sent["body"]["messages"][-1]["content"] == "Remember me?"


def test_missing_api_key_is_explicit(client, monkeypatch):
    monkeypatch.setattr(settings, "GROQ_API_KEY", "")

    response = client.post(api(settings.LLM_ROUTE_PREFIX, settings.LLM_CHAT_ROUTE), json={"message": "Hi"})

    assert response.status_code == 503
    assert "GROQ_API_KEY" in response.json()["detail"]


def test_upstream_error_is_reported(client, monkeypatch):
    real_client = httpx.AsyncClient
    transport = httpx.MockTransport(lambda request: httpx.Response(401, json={"error": "invalid key"}))
    monkeypatch.setattr(llm_core.httpx, "AsyncClient", lambda **kwargs: real_client(transport=transport, **kwargs))
    monkeypatch.setattr(settings, "GROQ_API_KEY", "bad-key")

    response = client.post(api(settings.LLM_ROUTE_PREFIX, settings.LLM_CHAT_ROUTE), json={"message": "Hi"})

    assert response.status_code == 502
    assert "401" in response.json()["detail"]
