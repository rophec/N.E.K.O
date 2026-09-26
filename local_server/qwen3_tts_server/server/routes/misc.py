"""辅助端点 — /health, /v1/models, /v1/audio/voices"""
from fastapi import APIRouter, Request
from ..models.requests import VoiceInfo, VoicesResponse, ModelInfo, ModelsResponse, HealthResponse

router = APIRouter()


@router.get("/health")
async def health_check(req: Request):
    pool = req.app.state.pool
    config = req.app.state.config
    return HealthResponse(
        model=config.model_id,
        engine_ready=pool.engine is not None and pool.engine.ready,
    )


@router.get("/v1/models")
async def list_models(req: Request):
    config = req.app.state.config
    return ModelsResponse(
        data=[ModelInfo(id=config.model_id)]
    )


@router.get("/v1/audio/voices")
async def list_voices(req: Request):
    pool = req.app.state.pool
    voice_cache = req.app.state.voice_cache
    # 从引擎获取实际可用的音色列表
    voices = []
    if pool.engine and pool.engine.ready:
        for name in pool.engine.available_voices:
            voices.append(VoiceInfo(voice_id=name, name=name, type="preset"))
    for entry in voice_cache.list_registered_voices():
        voices.append(VoiceInfo(**entry))
    return VoicesResponse(voices=voices)


@router.delete("/v1/audio/voices/{voice_id}")
async def delete_voice(req: Request, voice_id: str):
    removed = req.app.state.voice_cache.remove_registered_voice(voice_id)
    return {"deleted": removed, "voice_id": voice_id}
