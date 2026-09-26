"""流式拦截层 — 拦截 DecoderProxy 的 chunk 输出
通过 monkey-patch _handle_decoder_msg 实现流式音频传输
（参考 Qwen3-TTS-GGUF 的同名文件）

对于 ONNX 解码器子进程：monkey-patch 拦截 AUDIO message，推入 asyncio.Queue
对于 PyTorch 解码器（无子进程）：直接使用 on_audio_chunk 回调

DONE 事件在推理完成后由 _sync_synthesize 发送。
"""
import asyncio
import numpy as np
from typing import AsyncIterator, Optional
from inference import TTSConfig, TTSResult
from inference.stream import TTSStream


class StreamEvent:
    """流式事件"""
    AUDIO_CHUNK = "audio_chunk"
    DONE = "done"
    ERROR = "error"

    __slots__ = ("type", "audio", "result", "error")

    def __init__(self, type: str, audio: Optional[np.ndarray] = None,
                 result: Optional[TTSResult] = None, error: Optional[str] = None):
        self.type = type
        self.audio = audio
        self.result = result
        self.error = error


class StreamingTTSWrapper:
    """包装 TTSStream，拦截 DecoderProxy 的 chunk 输出。

    原理：DecoderProxy._listen_loop 在后台线程中从 result_q 取消息，
    收到 AUDIO 消息时调用 _handle_decoder_msg。
    我们 monkey-patch _handle_decoder_msg，将当前 task 的 audio chunk
    通过 loop.call_soon_threadsafe 推入 asyncio.Queue。

    推理完成时 _sync_synthesize 发送 DONE 事件。
    """

    def __init__(self, stream: TTSStream, loop: asyncio.AbstractEventLoop):
        self.stream = stream
        self.loop = loop
        self._queue: asyncio.Queue[StreamEvent] = asyncio.Queue()
        self._task_id: Optional[str] = None
        self._original_handler = None
        self._decoder = getattr(stream, 'decoder', None)

    def _install_chunk_hook(self, task_id: str):
        """在 DecoderProxy 上安装 chunk 拦截器"""
        if self._decoder is None:
            return

        self._task_id = task_id
        self._original_handler = self._decoder._handle_decoder_msg
        queue = self._queue
        loop = self.loop
        target_task_id = task_id
        original = self._original_handler

        def patched_handler(msg):
            # 先走原有逻辑（累积到 results）
            original(msg)

            # 拦截：将当前 task 的 AUDIO chunk 推入 asyncio Queue
            if msg.msg_type == "AUDIO" and msg.task_id == target_task_id:
                if msg.audio is not None and len(msg.audio) > 0:
                    loop.call_soon_threadsafe(
                        queue.put_nowait,
                        StreamEvent(StreamEvent.AUDIO_CHUNK, audio=msg.audio.copy())
                    )
            elif msg.msg_type == "FINISH" and msg.task_id == target_task_id:
                # 不在这里发 DONE（由 _sync_synthesize 在推理完成后发）
                pass

        self._decoder._handle_decoder_msg = patched_handler

    def _uninstall_hook(self):
        """恢复原始 handler"""
        if self._decoder is not None and self._original_handler is not None:
            self._decoder._handle_decoder_msg = self._original_handler
            self._original_handler = None

    async def synthesize_stream(
        self,
        text: str,
        mode: str = "custom",
        speaker: Optional[str] = None,
        language: str = "chinese",
        instruct: Optional[str] = None,
        config: Optional[TTSConfig] = None,
        timeout: float = 60.0,
    ) -> AsyncIterator[StreamEvent]:
        """流式合成，yield 每个 audio chunk"""
        cfg = config or TTSConfig(streaming=True)

        # 预分配 task_id（与 stream 内部 task_counter 对齐）
        task_id = f"stream_{self.stream.task_counter}"
        self._install_chunk_hook(task_id)

        # 在 executor 中运行同步推理
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(
            None,
            self._sync_synthesize,
            text, mode, speaker, language, instruct, cfg,
        )

        # 从 queue 中消费 chunk
        try:
            while True:
                try:
                    event = await asyncio.wait_for(self._queue.get(), timeout=timeout)
                except asyncio.TimeoutError:
                    yield StreamEvent(StreamEvent.ERROR, error="合成超时")
                    break

                yield event

                if event.type in (StreamEvent.DONE, StreamEvent.ERROR):
                    break
        finally:
            # 确保推理完成（即使消费者提前断开）
            cancelled = False
            while not future.done():
                try:
                    await asyncio.shield(future)
                except asyncio.CancelledError:
                    cancelled = True
                    current_task = asyncio.current_task()
                    if current_task is not None and hasattr(current_task, "uncancel"):
                        current_task.uncancel()
                except Exception:
                    break
            self._uninstall_hook()
            if cancelled:
                raise asyncio.CancelledError

    def _sync_synthesize(self, text, mode, speaker, language, instruct, config):
        """同步推理（在 executor 线程中执行）"""
        try:
            if mode == "custom":
                result = self.stream.custom(text, speaker, language, instruct, config)
            elif mode == "clone":
                result = self.stream.clone(text, language, config=config)
            elif mode == "design":
                result = self.stream.design(text, instruct, language, config=config)
            else:
                raise ValueError(f"未知模式: {mode}")

            # 推理完成：打印推理摘要
            if result and result.stats:
                s = result.stats
                # 流式模式：result.duration 只含 tail decode，用 codes 估算总音频量
                # 非流式模式：result.duration 是完整音频时长
                if config and config.streaming:
                    audio_dur = len(result.codes) * 1920 / 24000
                else:
                    audio_dur = result.duration if result.duration > 0 else len(result.codes) * 1920 / 24000
                inf_time = s.inference_only_time
                tok_per_s = s.total_steps / inf_time if inf_time > 0 else 0
                rtf = inf_time / audio_dur if audio_dur > 0 else 0
                summary = (
                    f"[TTS] text={text!r:.60s}  "
                    f"steps={s.total_steps}  "
                    f"audio={audio_dur:.2f}s  "
                    f"inference={inf_time:.2f}s  "
                    f"RTF={rtf:.2f}x  "
                    f"tok/s={tok_per_s:.1f}  "
                    f"first_chunk={s.first_chunk_latency:.3f}s  "
                    f"first_audio={s.first_audio_latency:.3f}s"
                )
                print(summary, flush=True)

            # 推理完成后发送 DONE 事件
            self.loop.call_soon_threadsafe(
                self._queue.put_nowait,
                StreamEvent(StreamEvent.DONE, result=result)
            )
            return result
        except Exception as e:
            self.loop.call_soon_threadsafe(
                self._queue.put_nowait,
                StreamEvent(StreamEvent.ERROR, error=str(e))
            )
            return None

    async def synthesize_full(
        self,
        text: str,
        mode: str = "custom",
        speaker: Optional[str] = None,
        language: str = "chinese",
        instruct: Optional[str] = None,
        config: Optional[TTSConfig] = None,
    ) -> Optional[TTSResult]:
        """非流式合成（一次性返回完整结果）"""
        cfg = config or TTSConfig(streaming=False)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._sync_synthesize_full(text, mode, speaker, language, instruct, cfg),
        )

    def _sync_synthesize_full(self, text, mode, speaker, language, instruct, config):
        """同步非流式推理（在 executor 线程中执行）"""
        if mode == "custom":
            return self.stream.custom(text, speaker, language, instruct, config)
        elif mode == "clone":
            return self.stream.clone(text, language, config=config)
        elif mode == "design":
            return self.stream.design(text, instruct, language, config=config)
        else:
            raise ValueError(f"未知模式: {mode}")

    def shutdown(self):
        """释放 TTSStream 资源"""
        self._uninstall_hook()
        if self.stream:
            self.stream.shutdown()
