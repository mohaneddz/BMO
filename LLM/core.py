import json
from datetime import datetime, timezone
from uuid import uuid4

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from constants import settings


router = APIRouter(prefix=settings.LLM_ROUTE_PREFIX, tags=["LLM"])

# In-memory chat history per conversation_id; cleared on server restart.
_CONVERSATIONS: dict[str, list[dict[str, str]]] = {}


class LLMChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    system_prompt: str | None = "You are BMO from Adventure Time. Keep responses concise and playful."
    temperature: float | None = None
    max_tokens: int | None = None
    conversation_id: str | None = None


class LLMChatResponse(BaseModel):
    provider: str
    model: str
    reply: str
    conversation_id: str
    turns: int
    saved_to: str
    usage: dict | None = None


def _persist_chat(conversation_id: str, payload: dict) -> str:
    out_file = settings.ASSETS_LLM_DIR / f"llm_{conversation_id}_{uuid4().hex[:8]}.json"
    out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(out_file)


def _build_messages(req: LLMChatRequest, history: list[dict[str, str]]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if req.system_prompt:
        messages.append({"role": "system", "content": req.system_prompt})
    messages.extend(history[-settings.LLM_HISTORY_MESSAGES :])
    messages.append({"role": "user", "content": req.message})
    return messages


async def _groq_chat(req: LLMChatRequest, messages: list[dict[str, str]]) -> tuple[str, dict | None]:
    if not settings.GROQ_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="LLM is not configured: set GROQ_API_KEY in .env (create a key at https://console.groq.com/keys).",
        )

    endpoint = f"{settings.GROQ_BASE_URL}{settings.LLM_ENDPOINT}"
    headers = {"Authorization": f"Bearer {settings.GROQ_API_KEY}"}
    body = {
        "model": settings.LLM_MODEL_NAME,
        "messages": messages,
        "temperature": req.temperature if req.temperature is not None else settings.LLM_TEMPERATURE,
        "max_tokens": req.max_tokens if req.max_tokens is not None else settings.LLM_MAX_TOKENS,
    }

    try:
        async with httpx.AsyncClient(timeout=settings.GROQ_TIMEOUT_SECONDS) as client:
            response = await client.post(endpoint, headers=headers, json=body)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Could not reach {endpoint}: {exc}") from exc

    if response.status_code >= 400:
        raise HTTPException(
            status_code=502,
            detail=f"Groq LLM request failed ({response.status_code}): {response.text}",
        )

    data = response.json()
    try:
        reply = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise HTTPException(status_code=502, detail=f"Unexpected Groq response shape: {data}") from exc
    return reply, data.get("usage")


@router.get(settings.LLM_HEALTH_ROUTE)
async def health() -> dict:
    return {
        "status": "ok",
        "module": "llm",
        "provider": settings.LLM_PROVIDER,
        "api_key_configured": bool(settings.GROQ_API_KEY),
    }


@router.get(settings.LLM_CONFIG_ROUTE)
async def config() -> dict:
    return {
        "provider": settings.LLM_PROVIDER,
        "model": settings.LLM_MODEL_NAME,
        "base_url": settings.GROQ_BASE_URL,
        "endpoint": settings.LLM_ENDPOINT,
        "history_messages": settings.LLM_HISTORY_MESSAGES,
        "route_prefix": settings.LLM_ROUTE_PREFIX,
    }


@router.post(settings.LLM_CHAT_ROUTE, response_model=LLMChatResponse)
async def chat(req: LLMChatRequest) -> LLMChatResponse:
    convo_id = req.conversation_id or uuid4().hex
    history = _CONVERSATIONS.get(convo_id, [])
    messages = _build_messages(req, history)

    reply, usage = await _groq_chat(req, messages)

    history = [*history, {"role": "user", "content": req.message}, {"role": "assistant", "content": reply}]
    _CONVERSATIONS[convo_id] = history

    persisted = _persist_chat(
        convo_id,
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "provider": settings.LLM_PROVIDER,
            "model": settings.LLM_MODEL_NAME,
            "conversation_id": convo_id,
            "request": req.model_dump(),
            "reply": reply,
            "usage": usage,
        },
    )

    return LLMChatResponse(
        provider=settings.LLM_PROVIDER,
        model=settings.LLM_MODEL_NAME,
        reply=reply,
        conversation_id=convo_id,
        turns=len(history) // 2,
        saved_to=persisted,
        usage=usage,
    )
