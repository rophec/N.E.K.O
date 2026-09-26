"""WebSocket 流式 TTS 端点 — WS /v1/audio/speech/stream

分句流式调度：收到文字按标点分句，凑够一句立即推理+输出，
不等 input.done，降低首 chunk TTFB。

采用串行模式（参考 nori-tts）：收到 input.text 后分句，
立即 await 推理+发送，推理完再回来收下一条消息。
天然避免 Starlette WebSocket 并发 send/receive 冲突。
"""
import asyncio
import json
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..engine_pool import EnginePool
from ..stream_wrapper import StreamingTTSWrapper, StreamEvent
from ..voice_cache import VoiceCache
from inference import TTSConfig

router = APIRouter()

# 句末标点集合
_SENTENCE_ENDS = set("。！？；…!?.;\n")


@router.websocket("/v1/audio/speech/stream")
async def ws_stream(websocket: WebSocket):
    await websocket.accept()
    pool: EnginePool = websocket.app.state.pool
    voice_cache: VoiceCache = websocket.app.state.voice_cache
    config: "ServerConfig" = websocket.app.state.config

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
        voice = config_msg.get("voice", "Vivian")
        language_raw = config_msg.get("language", "chinese")
        ref_audio = config_msg.get("ref_audio")
        ref_text = config_msg.get("ref_text", "")
        response_format = config_msg.get("response_format", "pcm")

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
            ref_audio_path = await voice_cache.resolve_ref_audio(ref_audio)
            cached_json = voice_cache.get(ref_audio_path, ref_text)
            if cached_json:
                await asyncio.get_running_loop().run_in_executor(
                    None, stream.set_voice, cached_json
                )
            else:
                await asyncio.get_running_loop().run_in_executor(
                    None, stream.set_voice, ref_audio_path, ref_text
                )
                if stream.voice:
                    voice_cache.put(ref_audio_path, ref_text, stream.voice)

        # 确定合成模式
        mode = "clone" if ref_audio else "custom"
        speaker = voice if not ref_audio else None

        # ── 分句流式调度（串行模式） ──
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

            # 复用 TTSStream：reset() 清空 KV cache + state，避免重建 LlamaContext 的 CUDA 分配开销
            # wrapper 在 session init 时创建，每句重建，不应为 None
            if wrapper is None:
                print("⚠️ [WS] wrapper 意外为 None，重建 stream（音色可能丢失）！", flush=True)
                stream = pool.engine.create_stream()
            else:
                wrapper._uninstall_hook()
                stream.reset()
            wrapper = StreamingTTSWrapper(stream, asyncio.get_running_loop())

            # 克隆模式: 仅首次需要 set_voice, 后续句复用已有 voice (reset 不再清除 voice)
            if ref_audio and ref_audio_path and not stream.voice:
                cached_json = voice_cache.get(ref_audio_path, ref_text)
                if cached_json:
                    await asyncio.get_running_loop().run_in_executor(
                        None, stream.set_voice, cached_json
                    )
                else:
                    await asyncio.get_running_loop().run_in_executor(
                        None, stream.set_voice, ref_audio_path, ref_text
                    )
                    if stream.voice:
                        voice_cache.put(ref_audio_path, ref_text, stream.voice)

            # 音色状态诊断：记录当前 voice 信息，帮助定位音色回退问题
            voice_info = "N/A"
            if stream.voice:
                voice_info = f"voice={stream.voice.text[:30]}, codes={stream.voice.codes.shape}, spk_emb={'OK' if stream.voice.spk_emb is not None else 'MISSING'}"
            elif mode == "clone":
                voice_info = "ERROR: clone mode but stream.voice is None"
            else:
                voice_info = f"custom mode, speaker={speaker}"
            print(f"🎤 [WS] 句{sentence_index} 音色状态: {voice_info}", flush=True)

            # audio.start
            await websocket.send_json({
                "type": "audio.start",
                "sentence_index": sentence_index,
                "sentence_text": text,
                "format": response_format,
                "sample_rate": 24000,
            })

            # 流式合成 + 立即输出
            async for event in wrapper.synthesize_stream(
                text=text,
                mode=mode,
                speaker=speaker,
                language=language,
                config=TTSConfig(streaming=True),
                timeout=60.0,
            ):
                if event.type == StreamEvent.AUDIO_CHUNK:
                    pcm_int16 = (event.audio * 32767).clip(-32768, 32767).astype(np.int16)
                    await websocket.send_bytes(pcm_int16.tobytes())
                elif event.type == StreamEvent.DONE:
                    break
                elif event.type == StreamEvent.ERROR:
                    await websocket.send_json({
                        "type": "error",
                        "message": event.error or "合成失败",
                    })
                    break

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
                    pending_text += msg.get("text", "")
                    # 分句后立即推理+发送
                    for sentence in _extract_complete_sentences():
                        await _synthesize_and_send(sentence)

                elif msg_type == "input.done":
                    # 剩余未成句的文字作为最后一句
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
        pass
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
