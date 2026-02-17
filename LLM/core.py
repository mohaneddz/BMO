import json
from datetime import datetime, timezone
from uuid import uuid4

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from constants import settings


router = APIRouter(prefix=settings.LLM_ROUTE_PREFIX, tags=["LLM"])


class LLMChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    system_prompt: str | None = "You are BMO from Adventure Time. Keep responses concise and playful."
    temperature: float | None = None
    max_tokens: int | None = None
    use_remote: bool | None = None
    conversation_id: str | None = None


class LLMChatResponse(BaseModel):
    provider: str
    model: str
    reply: str
    conversation_id: str
    used_remote: bool
    placeholder: bool
    saved_to: str
    usage: dict | None = None


def _conversation_id(value: str | None) -> str:
    return value or uuid4().hex


def _persist_chat(conversation_id: str, payload: dict) -> str:
    out_file = settings.ASSETS_LLM_DIR / f"llm_{conversation_id}_{uuid4().hex[:8]}.json"
    out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(out_file)


async def _remote_groq_chat(req: LLMChatRequest) -> tuple[str, dict | None]:
    if not settings.GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is missing.")

    endpoint = f"{settings.GROQ_BASE_URL}{settings.LLM_ENDPOINT}"
    headers = {"Authorization": f"Bearer {settings.GROQ_API_KEY}"}

    messages: list[dict[str, str]] = []
    if req.system_prompt:
        messages.append({"role": "system", "content": req.system_prompt})
    messages.append({"role": "user", "content": req.message})

    body = {
        "model": settings.LLM_MODEL_NAME,
        "messages": messages,
        "temperature": req.temperature if req.temperature is not None else settings.LLM_TEMPERATURE,
        "max_tokens": req.max_tokens if req.max_tokens is not None else settings.LLM_MAX_TOKENS,
    }

    async with httpx.AsyncClient(timeout=settings.GROQ_TIMEOUT_SECONDS) as client:
        response = await client.post(endpoint, headers=headers, json=body)

    if response.status_code >= 400:
        raise HTTPException(
            status_code=502,
            detail=f"GROQ LLM request failed ({response.status_code}): {response.text}",
        )

    data = response.json()
    reply = data["choices"][0]["message"]["content"]
    usage = data.get("usage")
    return reply, usage


def _placeholder_reply(message: str) -> str:
    return (
        "BMO says: I am in starter mode and ready to help. "
        f"Your message was: '{message[:120]}'."
    )


@router.get(settings.LLM_HEALTH_ROUTE)
async def health() -> dict:
    return {"status": "ok", "module": "llm", "provider": settings.LLM_PROVIDER}


@router.get(settings.LLM_CONFIG_ROUTE)
async def config() -> dict:
    return {
        "provider": settings.LLM_PROVIDER,
        "model": settings.LLM_MODEL_NAME,
        "use_remote": settings.LLM_USE_REMOTE,
        "route_prefix": settings.LLM_ROUTE_PREFIX,
        "endpoint": settings.LLM_ENDPOINT,
    }


@router.post(settings.LLM_CHAT_ROUTE, response_model=LLMChatResponse)
async def chat(req: LLMChatRequest) -> LLMChatResponse:
    use_remote = req.use_remote if req.use_remote is not None else settings.LLM_USE_REMOTE
    convo_id = _conversation_id(req.conversation_id)
    now = datetime.now(timezone.utc).isoformat()

    if use_remote:
        reply, usage = await _remote_groq_chat(req)
        placeholder = False
    else:
        reply = _placeholder_reply(req.message)
        usage = None
        placeholder = True

    persisted = _persist_chat(
        convo_id,
        {
            "timestamp": now,
            "provider": settings.LLM_PROVIDER,
            "model": settings.LLM_MODEL_NAME,
            "conversation_id": convo_id,
            "request": req.model_dump(),
            "reply": reply,
            "usage": usage,
            "used_remote": use_remote,
            "placeholder": placeholder,
        },
    )

    return LLMChatResponse(
        provider=settings.LLM_PROVIDER,
        model=settings.LLM_MODEL_NAME,
        reply=reply,
        conversation_id=convo_id,
        used_remote=use_remote,
        placeholder=placeholder,
        saved_to=persisted,
        usage=usage,
    )
