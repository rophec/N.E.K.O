"""Qwen3-TTS Server — 主入口"""
import sys
import os
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response
import json


def _configure_console_encoding() -> None:
    """Keep Windows GBK consoles from crashing on Unicode status output."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


_configure_console_encoding()

# 确保项目根目录在 sys.path 中
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from .config_loader import load_config
from .engine_pool import EnginePool
from .voice_cache import VoiceCache
from .routes import http_speech, ws_stream, misc


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时加载引擎，关闭时释放资源"""
    config = load_config()
    pool = EnginePool(config)

    print("=" * 50)
    print("  Qwen3-TTS Server")
    print("=" * 50)
    print(f"  模型目录:  {config.model_dir}")
    print(f"  模型 ID:   {config.model_id}")
    print(f"  ONNX EP:   {config.onnx_provider}")
    print(f"  Seeds:     Talker={config.seed}, Predictor={config.sub_seed}")
    print(f"  监听:      {config.host}:{config.port}")
    print(f"  并发数:    {config.max_concurrent_streams}")
    if config.api_key:
        print(f"  API Key:   ✅ 已启用鉴权")
    else:
        print(f"  API Key:   ❌ 未设置（无鉴权）")
    if config.speaker_map:
        print(f"  配置音色:  {len(config.speaker_map)} 个 (来自 config.yaml)")
    print("=" * 50)

    await pool.initialize()

    app.state.pool = pool
    app.state.config = config
    app.state.voice_cache = VoiceCache(config.voice_cache_dir)
    registered_voices = app.state.voice_cache.list_registered_voices()
    print(
        f"🎙️ [VoiceRegistry] 已注册克隆音色: "
        f"{[(item['voice_id'], item['name']) for item in registered_voices]}",
        flush=True,
    )

    yield

    await pool.shutdown()


app = FastAPI(
    title="Qwen3-TTS Server",
    description="OpenAI 兼容的 TTS 服务，基于 Qwen3-TTS 推理引擎",
    version="0.2.0",
    lifespan=lifespan,
)


# ── API Key 鉴权中间件 ──
@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """如果配置了 api_key，除 /health 外所有端点需携带 Authorization: Bearer <key>"""
    config = getattr(request.app.state, "config", None)
    if config and config.api_key:
        # 放行健康检查
        if request.url.path in ("/health",):
            return await call_next(request)

        auth = request.headers.get("Authorization", "")
        expected = "Bearer " + config.api_key
        if auth != expected:
            body = json.dumps({
                "error": "Unauthorized",
                "message": "需要有效的 API Key (Authorization: Bearer <key>)",
            })
            return Response(
                status_code=401,
                content=body,
                media_type="application/json",
                headers={"WWW-Authenticate": "Bearer"},
            )

    return await call_next(request)


app.include_router(http_speech.router)
app.include_router(ws_stream.router)
app.include_router(misc.router)


def main():
    # 仅在直接运行时 fallback（生产环境用 tts_server.sh）
    config = load_config()
    uvicorn.run(
        "server.main:app",
        host=config.host,
        port=config.port,
        log_level="info",
        # Session events carry liveness; protocol pings can time out while
        # llama.cpp is generating and leave an in-flight request orphaned.
        ws_ping_interval=None,
        factory=False,
    )


if __name__ == "__main__":
    main()
