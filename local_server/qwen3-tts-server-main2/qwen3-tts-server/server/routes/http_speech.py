"""HTTP TTS 端点 — POST /v1/audio/speech (非流式 + chunked 流式)"""
import asyncio
import numpy as np
from fastapi import APIRouter, Request, Response
from fastapi.responses import StreamingResponse

from ..models.requests import SpeechRequest
from ..engine_pool import EnginePool
from ..stream_wrapper import StreamingTTSWrapper, StreamEvent
from ..audio_codec import pcm_to_format, pcm_to_raw, wav_header_placeholder, media_type_for_format
from ..voice_cache import VoiceCache
from inference import TTSConfig

router = APIRouter()


def _resolve_mode(request: SpeechRequest, voice_cache: VoiceCache, default_voice: str):
    """根据请求参数确定合成模式、speaker 和 ref_audio_path"""
    if request.ref_audio:
        return "clone", None, request.ref_audio
    # 使用请求中的 voice，如果未指定或为空则使用引擎默认音色
    speaker = request.voice if request.voice else default_voice
    return "custom", speaker, None


def _build_config(request: SpeechRequest, streaming: bool = False) -> TTSConfig:
    """从请求参数构建 TTSConfig"""
    return TTSConfig(
        streaming=streaming,
        max_steps=300,
    )


def _language_from_request(request: SpeechRequest) -> str:
    """从请求中提取语言"""
    if request.language_type:
        # 映射 OpenAI 风格的语言代码
        lang_map = {
            "zh": "chinese", "en": "english", "ja": "japanese",
            "ko": "korean", "de": "german", "es": "spanish",
            "fr": "french", "ru": "russian", "it": "italian",
            "pt": "portuguese",
        }
        return lang_map.get(request.language_type.lower(), request.language_type.lower())
    return "chinese"


@router.post("/v1/audio/speech")
async def create_speech(request: SpeechRequest, req: Request):
    pool: EnginePool = req.app.state.pool
    voice_cache: VoiceCache = req.app.state.voice_cache
    config: "ServerConfig" = req.app.state.config

    # 验证 model
    if request.model and request.model != config.model_id:
        return Response(
            content=f"模型不匹配: 请求 {request.model}, 服务 {config.model_id}",
            status_code=404,
        )

    default_voice = config.default_voice or "Vivian"
    mode, speaker, ref_audio_raw = _resolve_mode(request, voice_cache, default_voice)
    language = _language_from_request(request)

    # 解析 ref_audio
    ref_audio_path = None
    if ref_audio_raw:
        ref_audio_path = await voice_cache.resolve_ref_audio(ref_audio_raw)

    if request.stream:
        # 流式：chunked binary transfer (NOT SSE)
        return StreamingResponse(
            _stream_audio(pool, voice_cache, request, mode, speaker, ref_audio_path, language),
            media_type=media_type_for_format(request.response_format),
            headers={
                "X-Accel-Buffering": "no",
                "Cache-Control": "no-cache",
            },
        )
    else:
        # 非流式：完整音频
        async with pool.semaphore:
            stream = pool.engine.create_stream()
            wrapper = StreamingTTSWrapper(stream, asyncio.get_running_loop())
            try:
                # 设置音色（克隆模式）
                if ref_audio_path:
                    # 检查缓存
                    cached_json = voice_cache.get(ref_audio_path, request.ref_text or "")
                    if cached_json:
                        await asyncio.get_running_loop().run_in_executor(
                            None, stream.set_voice, cached_json
                        )
                    else:
                        await asyncio.get_running_loop().run_in_executor(
                            None, stream.set_voice, ref_audio_path, request.ref_text or ""
                        )
                        # 缓存 voice anchor
                        if stream.voice:
                            voice_cache.put(ref_audio_path, request.ref_text or "", stream.voice)

                result = await wrapper.synthesize_full(
                    text=request.input,
                    mode=mode,
                    speaker=speaker,
                    language=language,
                    instruct=request.instructions,
                    config=_build_config(request, streaming=False),
                )

                if result is None or result.audio is None:
                    return Response(status_code=500, content="合成失败")

                audio_bytes = pcm_to_format(result.audio, request.response_format)
                return Response(
                    content=audio_bytes,
                    media_type=media_type_for_format(request.response_format),
                    headers={"Content-Disposition": f"attachment; filename=speech.{request.response_format}"},
                )
            finally:
                wrapper.shutdown()


async def _stream_audio(pool, voice_cache, request, mode, speaker, ref_audio_path, language):
    """生成器：yield 音频二进制 chunk"""
    async with pool.semaphore:
        stream = pool.engine.create_stream()
        wrapper = StreamingTTSWrapper(stream, asyncio.get_running_loop())
        try:
            # 设置音色（克隆模式）
            if ref_audio_path:
                cached_json = voice_cache.get(ref_audio_path, request.ref_text or "")
                if cached_json:
                    await asyncio.get_running_loop().run_in_executor(
                        None, stream.set_voice, cached_json
                    )
                else:
                    await asyncio.get_running_loop().run_in_executor(
                        None, stream.set_voice, ref_audio_path, request.ref_text or ""
                    )
                    if stream.voice:
                        voice_cache.put(ref_audio_path, request.ref_text or "", stream.voice)

            # WAV 流式：首包发送 WAV 头
            fmt = request.response_format.lower()
            if fmt == "wav":
                yield wav_header_placeholder()
            elif fmt != "pcm":
                # mp3/opus/aac/flac 不支持流式，回退到非流式
                result = await wrapper.synthesize_full(
                    text=request.input, mode=mode, speaker=speaker,
                    language=language, instruct=request.instructions,
                    config=_build_config(request, streaming=False),
                )
                if result and result.audio is not None:
                    yield pcm_to_format(result.audio, fmt)
                return

            # PCM/WAV 流式
            async for event in wrapper.synthesize_stream(
                text=request.input, mode=mode, speaker=speaker,
                language=language, instruct=request.instructions,
                config=_build_config(request, streaming=True),
                timeout=60.0,
            ):
                if event.type == StreamEvent.AUDIO_CHUNK:
                    # PCM float32 → int16 bytes
                    pcm_int16 = (event.audio * 32767).clip(-32768, 32767).astype(np.int16)
                    yield pcm_int16.tobytes()
                elif event.type == StreamEvent.ERROR:
                    break
                elif event.type == StreamEvent.DONE:
                    break
        finally:
            wrapper.shutdown()
