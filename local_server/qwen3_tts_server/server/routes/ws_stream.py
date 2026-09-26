"""WebSocket 流式 TTS 端点 — WS /v1/audio/speech/stream

分句流式调度：收到文字按标点分句，凑够一句立即推理+输出，
不等 input.done，降低首 chunk TTFB。

采用串行模式（参考 nori-tts）：收到 input.text 后分句，
立即 await 推理+发送，推理完再回来收下一条消息。
天然避免 Starlette WebSocket 并发 send/receive 冲突。
"""
import asyncio
import json
import os
import time
from pathlib import Path
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..debug_audio_export import DebugAudioExporter
from ..engine_pool import EnginePool
from ..stream_wrapper import StreamingTTSWrapper, StreamEvent
from ..voice_cache import VoiceCache
from ..voice_selection import resolve_requested_voice
from inference import TTSConfig
from inference.schema.constants import map_speaker

router = APIRouter()

# 句末标点集合
_SENTENCE_ENDS = set("。！？；…!?.;\n")


@router.websocket("/v1/audio/speech/stream")
async def ws_stream(websocket: WebSocket):
    pool: EnginePool = websocket.app.state.pool
    voice_cache: VoiceCache = websocket.app.state.voice_cache
    config: "ServerConfig" = websocket.app.state.config

    if config.api_key:
        expected = f"Bearer {config.api_key}"
        if websocket.headers.get("Authorization", "") != expected:
            await websocket.close(code=1008, reason="Unauthorized")
            return

    await websocket.accept()
    print(f"[Qwen3-TTS WS][accepted] client={websocket.client}", flush=True)

    stream = None
    wrapper = None
    semaphore_acquired = False

    try:
        # 等待 session.config
        try:
            config_msg = await asyncio.wait_for(websocket.receive_json(), timeout=config.ws_idle_timeout)
        except asyncio.TimeoutError:
            await websocket.close(code=1000, reason="idle timeout")
            return

        if config_msg.get("type") != "session.config":
            await websocket.send_json({"type": "error", "message": "首条消息必须是 session.config"})
            await websocket.close()
            return

        # 解析配置
        requested_voice = config_msg.get("voice")
        requested_voice_name = str(config_msg.get("voice_name") or requested_voice or "").strip()
        voice = resolve_requested_voice(
            requested_voice,
            config.default_voice,
            getattr(pool.engine, "default_voice", ""),
        )
        language_raw = config_msg.get("language", "chinese")
        ref_audio = config_msg.get("ref_audio")
        ref_text = config_msg.get("ref_text", "")
        response_format = config_msg.get("response_format", "pcm")
        clone_mode_raw = str(
            config_msg.get("clone_mode")
            or websocket.query_params.get("clone_mode")
            or "icl"
        ).strip().lower().replace("-", "_")
        # TEMP EXPERIMENT: fixed_speaker now means a complete ICL anchor
        # (spk_emb + reference codes + decoder continuation). Keep the failed
        # embedding-only path available under x_vector_only for A/B comparison.
        fixed_speaker_codes_mode = clone_mode_raw in {
            "fixed_speaker", "fixed_speaker_codes", "icl_codes",
        }
        x_vector_only_mode = clone_mode_raw in {"x_vector", "x_vector_only"}
        # TEMP DIAGNOSTIC: remove debug_export and DebugAudioExporter before merge.
        debug_export_enabled = str(
            config_msg.get("debug_export")
            or websocket.query_params.get("debug_export")
            or os.environ.get("QWEN3_TTS_DEBUG_EXPORT")
            or ""
        ).strip().lower() in {"1", "true", "yes", "on"}
        debug_exporter = DebugAudioExporter() if debug_export_enabled else None
        max_steps = int(config_msg.get("max_steps") or getattr(config, "max_steps", 300))
        seed_raw = config_msg.get("seed")
        sub_seed_raw = config_msg.get("sub_seed")
        seed = int(seed_raw) if seed_raw is not None else getattr(config, "seed", 42)
        sub_seed = (
            int(sub_seed_raw)
            if sub_seed_raw is not None
            else getattr(config, "sub_seed", 45)
        )
        print(
            f"[Qwen3-TTS WS][session.config] model={config_msg.get('model')!r} "
            f"requested_voice={requested_voice!r} voice_name={requested_voice_name!r} "
            f"ref_audio={bool(ref_audio)} "
            f"ref_text_chars={len(ref_text)} response_format={response_format!r} "
            f"clone_mode={clone_mode_raw!r} "
            f"debug_export={debug_export_enabled} "
            f"seed={seed} sub_seed={sub_seed}",
            flush=True,
        )

        # 语言映射
        lang_map = {
            "zh": "chinese", "en": "english", "ja": "japanese",
            "ko": "korean", "de": "german", "es": "spanish",
            "fr": "french", "ru": "russian", "it": "italian",
            "pt": "portuguese",
        }
        language = lang_map.get(language_raw.lower(), language_raw.lower())

        # 等待信号量
        try:
            await asyncio.wait_for(pool.semaphore.acquire(), timeout=config.request_timeout)
        except asyncio.TimeoutError:
            await websocket.send_json({"type": "error", "message": "服务繁忙，请稍后重试"})
            await websocket.close()
            return
        semaphore_acquired = True

        # 创建 TTSStream
        stream = pool.engine.create_stream()
        wrapper = StreamingTTSWrapper(stream, asyncio.get_running_loop())

        # 设置音色（克隆模式）
        ref_audio_path = None
        if ref_audio:
            voice_started_at = time.perf_counter()
            print("[Qwen3-TTS WS][clone.resolve] reference_audio=start", flush=True)
            ref_audio_path = await voice_cache.resolve_ref_audio(ref_audio)
            cached_json = voice_cache.get(ref_audio_path, ref_text)
            print(
                f"[Qwen3-TTS WS][clone.anchor] cache={'hit' if cached_json else 'miss'} "
                f"audio_key={Path(ref_audio_path).stem[:12]}",
                flush=True,
            )
            if cached_json:
                voice_loaded = await asyncio.get_running_loop().run_in_executor(
                    None, stream.set_voice, cached_json
                )
            else:
                voice_loaded = await asyncio.get_running_loop().run_in_executor(
                    None, stream.set_voice, ref_audio_path, ref_text
                )
                if stream.voice:
                    voice_cache.put(ref_audio_path, ref_text, stream.voice)
            if not voice_loaded:
                raise RuntimeError(
                    f"克隆音色锚点加载失败: voice={requested_voice!r} "
                    f"name={requested_voice_name!r}"
                )
            print(
                f"[Qwen3-TTS WS][clone.ready] cache={'hit' if cached_json else 'miss'} "
                f"elapsed={time.perf_counter() - voice_started_at:.3f}s",
                flush=True,
            )
            registered_voices = voice_cache.register_clone_voice(
                requested_voice,
                requested_voice_name,
                Path(ref_audio_path).stem,
            )
            print(
                f"[Qwen3-TTS Voices][registered] current={requested_voice!r} "
                f"name={requested_voice_name!r} available="
                f"{[(item['voice_id'], item['name']) for item in registered_voices]}",
                flush=True,
            )

        # 完整 ICL 固定锚点保留 spk_emb、reference codes 和由 codes 编译出的
        # decoder continuation state。纯 x-vector 对照模式才销毁锚点流。
        fixed_speaker_embedding = None
        reference_code_frames = 0
        decoder_continuation_ready = False
        if ref_audio and (fixed_speaker_codes_mode or x_vector_only_mode):
            if stream.voice is None or stream.voice.spk_emb is None:
                raise RuntimeError("参考音频没有生成可用的 speaker embedding")
            fixed_speaker_embedding = np.array(
                stream.voice.spk_emb, dtype=np.float32, copy=True
            )

        if ref_audio and fixed_speaker_codes_mode:
            reference_codes = getattr(stream.voice, "codes", None)
            reference_code_frames = len(reference_codes) if reference_codes is not None else 0
            decoder_continuation_ready = getattr(stream.voice, "final_state", None) is not None
            if reference_code_frames <= 0:
                raise RuntimeError("参考音频没有生成可用于 ICL 的 codec codes")
            if pool.engine.decoder and not decoder_continuation_ready:
                raise RuntimeError("参考 codec codes 未成功编译 decoder continuation state")
            print(
                f"[Qwen3-TTS WS][fixed_speaker_codes.ready] "
                f"embedding_dim={fixed_speaker_embedding.size} "
                f"reference_code_frames={reference_code_frames} "
                f"prompt_icl=True decoder_continuation={decoder_continuation_ready}",
                flush=True,
            )
        elif ref_audio and x_vector_only_mode:
            wrapper.shutdown()
            stream = pool.engine.create_stream()
            if stream is None:
                raise RuntimeError("x-vector 对照合成流创建失败")
            wrapper = StreamingTTSWrapper(stream, asyncio.get_running_loop())
            print(
                f"[Qwen3-TTS WS][x_vector_only.ready] "
                f"embedding_dim={fixed_speaker_embedding.size} "
                f"reference_codes=False decoder_continuation=False",
                flush=True,
            )

        # fixed_speaker_codes 走完整 clone prompt；x_vector_only 仍走 custom。
        if ref_audio and fixed_speaker_codes_mode:
            session_mode = "fixed_speaker_codes"
            mode = "clone"
            speaker = None
        elif ref_audio and x_vector_only_mode:
            session_mode = "x_vector_only"
            mode = "custom"
            speaker = fixed_speaker_embedding
        elif ref_audio:
            session_mode = "clone"
            mode = "clone"
            speaker = None
        else:
            session_mode = "custom"
            mode = "custom"
            speaker = voice
        speaker_id = (
            map_speaker(speaker)
            if speaker is not None and not isinstance(speaker, np.ndarray)
            else None
        )
        if session_mode == "fixed_speaker_codes":
            resolved_voice_log = (
                f"external_voice_anchor(speaker_dim={fixed_speaker_embedding.size},"
                f"reference_code_frames={reference_code_frames})"
            )
        elif isinstance(speaker, np.ndarray):
            resolved_voice_log = f"external_speaker_embedding(dim={speaker.size})"
        else:
            resolved_voice_log = repr(speaker)
        print(
            f"[TTS Session] mode={session_mode} synthesis_mode={mode} "
            f"requested_voice={requested_voice!r} "
            f"voice_name={requested_voice_name!r} resolved_voice={resolved_voice_log} "
            f"speaker_id={speaker_id!r} "
            f"ref_audio={bool(ref_audio)}",
            flush=True,
        )
        if debug_exporter:
            diagnostic_embedding = fixed_speaker_embedding
            if diagnostic_embedding is None and getattr(stream, "voice", None):
                diagnostic_embedding = stream.voice.spk_emb
            debug_exporter.write_session(
                {
                    "model": config_msg.get("model"),
                    "requested_voice": requested_voice,
                    "voice_name": requested_voice_name,
                    "session_mode": session_mode,
                    "synthesis_mode": mode,
                    "clone_mode": clone_mode_raw,
                    "ref_audio": bool(ref_audio),
                    "ref_text_chars": len(ref_text),
                    "language": language,
                    "response_format": response_format,
                    "seed": seed,
                    "sub_seed": sub_seed,
                    "speaker_id": speaker_id,
                    "reference_code_frames": reference_code_frames,
                    "prompt_icl": mode == "clone",
                    "decoder_continuation": decoder_continuation_ready,
                },
                ref_audio_path=ref_audio_path,
                speaker_embedding=diagnostic_embedding,
            )

        # ── 文本调度（串行模式） ──
        # 克隆模式必须把一个 utterance 当作参考音频后的同一次续读。
        # 如果按标点拆成多次 clone，每一段都会重新采样一套声学轨迹，
        # 即使引用同一个 anchor，也会在句界处产生明显的音色跳变。
        clone_continuation = mode == "clone"
        pending_text = ""
        sentence_index = 0

        def _extract_complete_sentences():
            """从 pending_text 中切出所有完整句子（以句末标点结尾），
            剩余不完整片段保留在缓冲区。"""
            nonlocal pending_text
            sentences = []
            last_cut = 0
            for i, ch in enumerate(pending_text):
                if ch in _SENTENCE_ENDS:
                    segment = pending_text[last_cut:i + 1].strip()
                    if segment:
                        sentences.append(segment)
                    last_cut = i + 1
            pending_text = pending_text[last_cut:]
            return sentences

        async def _synthesize_and_send(text: str):
            """对一段文本执行流式合成，并通过 WS 发送 PCM 音频帧。"""
            nonlocal stream, wrapper, sentence_index
            if not text.strip():
                return

            # 首段直接使用 session.config 阶段准备好的 reference anchor。
            # 只有预置音色的后续分句才需要清理上一句的推理状态。
            if sentence_index > 0 and wrapper:
                wrapper._uninstall_hook()
                stream.reset()
                wrapper = StreamingTTSWrapper(stream, asyncio.get_running_loop())
            elif wrapper is None:
                stream = pool.engine.create_stream()
                wrapper = StreamingTTSWrapper(stream, asyncio.get_running_loop())

            if clone_continuation:
                print(
                    f"[Qwen3-TTS WS][clone.continuation] anchor_once=True "
                    f"text_chars={len(text)} ref_text_chars={len(ref_text)}",
                    flush=True,
                )

            # audio.start
            await websocket.send_json({
                "type": "audio.start",
                "sentence_index": sentence_index,
                "sentence_text": text,
                "format": response_format,
                "sample_rate": 24000,
            })

            # 流式合成 + 立即输出
            debug_pcm = bytearray() if debug_exporter else None
            synthesis_error = None
            async for event in wrapper.synthesize_stream(
                text=text,
                mode=mode,
                speaker=speaker,
                language=language,
                config=TTSConfig(
                    streaming=True,
                    max_steps=max_steps,
                    seed=seed,
                    sub_seed=sub_seed,
                ),
                timeout=60.0,
            ):
                if event.type == StreamEvent.AUDIO_CHUNK:
                    pcm_int16 = (event.audio * 32767).clip(-32768, 32767).astype(np.int16)
                    pcm_bytes = pcm_int16.tobytes()
                    if debug_pcm is not None:
                        debug_pcm.extend(pcm_bytes)
                    await websocket.send_bytes(pcm_bytes)
                elif event.type == StreamEvent.DONE:
                    break
                elif event.type == StreamEvent.ERROR:
                    synthesis_error = event.error or "合成失败"
                    await websocket.send_json({
                        "type": "error",
                        "message": synthesis_error,
                    })
                    break

            if debug_exporter and debug_pcm is not None:
                debug_exporter.write_sentence(
                    sentence_index=sentence_index,
                    text=text,
                    pcm_bytes=bytes(debug_pcm),
                    sample_rate=24000,
                    metadata={
                        "requested_voice": requested_voice,
                        "voice_name": requested_voice_name,
                        "session_mode": session_mode,
                        "synthesis_mode": mode,
                        "seed": seed,
                        "sub_seed": sub_seed,
                        "error": synthesis_error,
                    },
                )

            # audio.done
            await websocket.send_json({
                "type": "audio.done",
                "sentence_index": sentence_index,
            })
            sentence_index += 1

        # ── 主消息循环 ──
        while True:
            raw = await websocket.receive()

            if raw.get("type") == "websocket.disconnect":
                break

            if "text" in raw and raw["text"] is not None:
                try:
                    msg = json.loads(raw["text"])
                except json.JSONDecodeError:
                    await websocket.send_json({"type": "error", "message": "无效的 JSON"})
                    continue

                msg_type = msg.get("type", "")

                if msg_type == "input.text":
                    incoming_text = msg.get("text", "")
                    pending_text += incoming_text
                    print(
                        f"[Qwen3-TTS WS][input.text] chars={len(incoming_text)} "
                        f"pending_chars={len(pending_text)}",
                        flush=True,
                    )
                    if clone_continuation:
                        print(
                            f"[Qwen3-TTS WS][clone.buffer] "
                            f"utterance_chars={len(pending_text)}",
                            flush=True,
                        )
                    else:
                        # 预置音色不依赖参考续读，继续按句低延迟输出。
                        for sentence in _extract_complete_sentences():
                            await _synthesize_and_send(sentence)

                elif msg_type == "input.done":
                    print(
                        f"[Qwen3-TTS WS][input.done] pending_chars={len(pending_text)}",
                        flush=True,
                    )
                    # 克隆模式中 pending_text 是完整 utterance；预置音色模式中
                    # 它只是尚未以标点结尾的最后一句。
                    tail = pending_text.strip()
                    pending_text = ""
                    if tail:
                        await _synthesize_and_send(tail)
                    # session.done
                    await websocket.send_json({
                        "type": "session.done",
                        "total_sentences": sentence_index,
                    })
                    break

                else:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"未知的消息类型: {msg_type}",
                    })

            elif "bytes" in raw and raw["bytes"] is not None:
                await websocket.send_json({"type": "error", "message": "不支持二进制输入"})

    except WebSocketDisconnect:
        print("[Qwen3-TTS WS][disconnect] client_closed", flush=True)
    except Exception as e:
        import traceback
        traceback.print_exc()
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        if wrapper:
            wrapper.shutdown()
        if semaphore_acquired:
            pool.semaphore.release()
