from fastapi import FastAPI

from constants import settings
from CV.core import router as cv_router
from LLM.core import router as llm_router
from STT.core import router as stt_router
from TTS.core import router as tts_router


app = FastAPI(
    title=settings.APP_NAME,
    version="0.1.0",
    description="Initial routed skeleton for the BMO multi-module system.",
)


@app.get(settings.ROOT_ROUTE, tags=["Root"])
async def root() -> dict:
    return {
        "name": settings.APP_NAME,
        "env": settings.APP_ENV,
        "api_prefix": settings.API_PREFIX,
        "modules": {
            "stt": settings.API_PREFIX + settings.STT_ROUTE_PREFIX,
            "cv": settings.API_PREFIX + settings.CV_ROUTE_PREFIX,
            "llm": settings.API_PREFIX + settings.LLM_ROUTE_PREFIX,
            "tts": settings.API_PREFIX + settings.TTS_ROUTE_PREFIX,
        },
    }


@app.get(settings.ROOT_HEALTH_ROUTE, tags=["Root"])
async def health() -> dict:
    return {"status": "ok"}


app.include_router(stt_router, prefix=settings.API_PREFIX)
app.include_router(cv_router, prefix=settings.API_PREFIX)
app.include_router(llm_router, prefix=settings.API_PREFIX)
app.include_router(tts_router, prefix=settings.API_PREFIX)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=settings.APP_RELOAD,
    )
