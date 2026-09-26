"""HTTP TTS 端点 — POST /v1/audio/speech (非流式 + chunked 流式)"""
import asyncio
import time
import numpy as np
from fastapi import APIRouter, Request, Response
from fastapi.responses import StreamingResponse

from ..models.requests import SpeechRequest
from ..engine_pool import EnginePool
from ..stream_wrapper import StreamingTTSWrapper, StreamEvent
from ..audio_codec import pcm_to_format, pcm_to_raw, wav_header_placeholder, media_type_for_format
from ..voice_cache import VoiceCache
from ..voice_selection import resolve_requested_voice
from inference import TTSConfig
from inference.schema.constants import map_speaker

router = APIRouter()


def _resolve_mode(request: SpeechRequest, voice_cache: VoiceCache, default_voice: str):
    """根据请求参数确定合成模式、speaker 和 ref_audio_path"""
    if request.ref_audio:
        return "clone", None, request.ref_audio
    # 使用请求中的 voice，如果未指定或为空则使用引擎默认音色
    speaker = resolve_requested_voice(request.voice, default_voice)
    return "custom", speaker, None


def _build_config(request: SpeechRequest, server_config, streaming: bool = False) -> TTSConfig:
    """从请求参数构建 TTSConfig"""
    max_steps = request.max_steps or getattr(server_config, "max_steps", 300)
    seed = request.seed if request.seed is not None else getattr(server_config, "seed", 42)
    sub_seed = (
        request.sub_seed
        if request.sub_seed is not None
        else getattr(server_config, "sub_seed", 45)
    )
    return TTSConfig(
        streaming=streaming,
        max_steps=max_steps,
        seed=seed,
        sub_seed=sub_seed,
    )


def _timing_headers(result, tts_config: TTSConfig, wall_time: float) -> dict:
    """Expose generation timings as HTTP headers for binary audio responses."""
    headers = {}
    if result is None:
        return headers

    codes_count = len(result.codes) if result.codes is not None else 0
    audio_duration = result.duration if result.audio is not None else 0.0
    hit_max_steps = codes_count >= tts_config.max_steps

    headers.update({
        "X-TTS-Steps": str(codes_count),
        "X-TTS-Max-Steps": str(tts_config.max_steps),
        "X-TTS-Hit-Max-Steps": "true" if hit_max_steps else "false",
        "X-TTS-Audio-Duration": f"{audio_duration:.3f}",
        "X-TTS-Wall-Time": f"{wall_time:.3f}",
    })

    stats = result.stats
    if stats is not None:
        inference_time = stats.inference_only_time
        rtf = inference_time / audio_duration if audio_duration > 0 else 0.0
        headers.update({
            "X-TTS-Prompt-Time": f"{stats.prompt_time:.3f}",
            "X-TTS-Prefill-Time": f"{stats.prefill_time:.3f}",
            "X-TTS-Talker-Time": f"{stats.total_talker_time:.3f}",
            "X-TTS-Predictor-Time": f"{stats.total_predictor_time:.3f}",
            "X-TTS-Decoder-Time": f"{stats.total_decoder_time:.3f}",
            "X-TTS-Inference-Time": f"{inference_time:.3f}",
            "X-TTS-First-Audio-Time": f"{stats.first_audio_latency:.3f}",
            "X-TTS-RTF": f"{rtf:.3f}",
        })
    return headers


def _print_timing_summary(result, tts_config: TTSConfig, wall_time: float) -> None:
    if result is None or result.stats is None:
        return

    stats = result.stats
    audio_duration = result.duration if result.audio is not None else 0.0
    inference_time = stats.inference_only_time
    rtf = inference_time / audio_duration if audio_duration > 0 else 0.0
    tok_per_s = stats.total_steps / inference_time if inference_time > 0 else 0.0
    print(
        f"[TTS] text={result.text!r:.80s}  "
        f"steps={stats.total_steps}  "
        f"audio={audio_duration:.2f}s  "
        f"inference={inference_time:.2f}s  "
        f"RTF={rtf:.2f}x  "
        f"tok/s={tok_per_s:.1f}  "
        f"first_chunk={stats.first_chunk_latency:.3f}s  "
        f"first_audio={stats.first_audio_latency:.3f}s",
        flush=True,
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
    speaker_id = map_speaker(speaker) if speaker is not None else None
    print(
        f"[TTS Request] mode={mode} requested_voice={request.voice!r} "
        f"resolved_voice={speaker!r} speaker_id={speaker_id!r} "
        f"ref_audio={bool(ref_audio_raw)}",
        flush=True,
    )

    # 解析 ref_audio
    ref_audio_path = None
    if ref_audio_raw:
        ref_audio_path = await voice_cache.resolve_ref_audio(ref_audio_raw)

    if request.stream:
        # 流式：chunked binary transfer (NOT SSE)
        return StreamingResponse(
            _stream_audio(pool, voice_cache, request, mode, speaker, ref_audio_path, language, config),
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

                tts_config = _build_config(request, config, streaming=False)
                started_at = time.perf_counter()
                result = await wrapper.synthesize_full(
                    text=request.input,
                    mode=mode,
                    speaker=speaker,
                    language=language,
                    instruct=request.instructions,
                    config=tts_config,
                )
                wall_time = time.perf_counter() - started_at

                if result is None or result.audio is None:
                    return Response(status_code=500, content="合成失败")

                _print_timing_summary(result, tts_config, wall_time)
                audio_bytes = pcm_to_format(result.audio, request.response_format)
                headers = {
                    "Content-Disposition": f"attachment; filename=speech.{request.response_format}",
                    **_timing_headers(result, tts_config, wall_time),
                }
                return Response(
                    content=audio_bytes,
                    media_type=media_type_for_format(request.response_format),
                    headers=headers,
                )
            finally:
                wrapper.shutdown()


async def _stream_audio(pool, voice_cache, request, mode, speaker, ref_audio_path, language, server_config):
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
                    config=_build_config(request, server_config, streaming=False),
                )
                if result and result.audio is not None:
                    yield pcm_to_format(result.audio, fmt)
                return

            # PCM/WAV 流式
            async for event in wrapper.synthesize_stream(
                text=request.input, mode=mode, speaker=speaker,
                language=language, instruct=request.instructions,
                config=_build_config(request, server_config, streaming=True),
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
