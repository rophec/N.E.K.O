"""
proxy.py - 解码器多进程代理 (独立部署版)
分离自 decoder.py，负责主进程与 Decoder Worker 之间的协议通信。
已移除 Speaker/Recorder 进程（服务端不需要本地播放/录制）。
"""
import multiprocessing as mp
import atexit
import threading
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass
import queue
import time
import numpy as np
from typing import Optional, Union, Callable
from .schema.result import TTSResult, DecodeResult
from .schema.protocol import DecodeRequest, DecoderResponse

class DecoderProxy:
    """
    解码器多进程代理 (DecoderProxy) — 独立部署版。
    仅保留 Decoder 子进程。
    支持流式回调：on_audio_chunk 回调函数在收到每个 AUDIO chunk 时被调用。
    """
    def __init__(self, onnx_path: str, onnx_provider: str = 'CPU', chunk_size: int = 12,
                 on_audio_chunk: Optional[Callable[[np.ndarray], None]] = None):
        self.onnx_path = onnx_path
        self.onnx_provider = onnx_provider
        self.chunk_size = chunk_size
        self.on_audio_chunk = on_audio_chunk
        
        self.task_counter = 0
        self.active_task_id = 0
        
        self.codes_q = mp.Queue()
        self.result_q = mp.Queue()
        
        self.decoder_proc = None
        
        self.results = {}
        self.events = {}
        self.streaming_results = {}
        self.ready_states = {"decoder": False}
        
        self.active_decoder_tasks = set()
        self.decoder_idle = threading.Event()
        self.decoder_idle.set()
        
        self.stop_listener = False
        self.listener_thread = None
        
        self.start()
        atexit.register(self.shutdown)

    def start(self):
        from .workers.decoder import decoder_worker_proc
        
        self.decoder_proc = mp.Process(
            target=decoder_worker_proc,
            args=(self.codes_q, self.result_q, self.onnx_path, self.onnx_provider, self.chunk_size),
            daemon=True
        )
        self.decoder_proc.start()
        
        self.listener_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.listener_thread.start()

    def _listen_loop(self):
        while not self.stop_listener:
            try:
                msg = self.result_q.get(timeout=0.1)
                if msg is None:
                    break
                
                if isinstance(msg, DecoderResponse):
                    self._handle_decoder_msg(msg)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"[Proxy] 监听异常: {e}")
                import traceback
                traceback.print_exc()
                break

    def _handle_decoder_msg(self, msg: DecoderResponse):
        if msg.msg_type == "READY":
            self.ready_states["decoder"] = True
            return
            
        task_id = msg.task_id
        
        if msg.msg_type == "AUDIO":
            msg.recv_time = time.time()
            if self.on_audio_chunk and msg.audio is not None and len(msg.audio) > 0:
                self.on_audio_chunk(msg.audio)
            if task_id not in self.results:
                self.results[task_id] = []
            self.results[task_id].append(msg)

        elif msg.msg_type == "FINISH":
            if task_id not in self.results:
                self.results[task_id] = []
            self.results[task_id].append(msg)
            
            if task_id in self.events:
                self.events[task_id].set()
            
            if task_id in self.active_decoder_tasks:
                self.active_decoder_tasks.remove(task_id)
                if not self.active_decoder_tasks:
                    self.decoder_idle.set()

    def wait_until_ready(self, timeout=10):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if all(self.ready_states.values()):
                return True
            time.sleep(0.1)
        return False

    def join_decoder(self, timeout: Optional[float] = None):
        self.decoder_idle.wait(timeout=timeout)

    def decode(self, input: Union[np.ndarray, TTSResult], task_id="default", is_final: bool = False,
               stream: bool = False, state: Optional["DecoderState"] = None) -> np.ndarray:
        if isinstance(input, TTSResult):
            if input.ref_codes is not None and len(input.ref_codes) > 0 and input.final_state is None:
                res_ref = self.decode(input.ref_codes, task_id=f"{task_id}_ref_init", is_final=True)
                input.final_state = res_ref.final_state
            state = state or input.final_state
            codes = input.codes
            is_final = True
        else:
            codes = input

        t_start = time.time()
        
        # 流式 final: 清理之前累积的旧 AUDIO 消息，确保 DecodeResult 只含本次响应
        # （流式 chunk 的 AUDIO 已通过 monkey-patch 推到 asyncio.Queue，无需保留）
        if stream and is_final and task_id in self.results:
            self.results[task_id] = []
        
        if task_id not in self.results:
            self.results[task_id] = []
        if task_id not in self.events:
            self.events[task_id] = threading.Event()
        else:
            self.events[task_id].clear()
            
        if stream:
            self.streaming_results[task_id] = True
            
        if is_final or not stream:
            self.active_decoder_tasks.add(task_id)
            self.decoder_idle.clear()

        msg_type = "DECODE_CHUNK" if stream else "DECODE"
        req = DecodeRequest(task_id=task_id, msg_type=msg_type, codes=codes,
                            is_final=is_final, state=state)
        self.codes_q.put(req)
        
        if stream and not is_final:
            return np.array([], dtype=np.float32)
        
        self.events[task_id].wait(timeout=30.0)
        
        responses = self.results.get(task_id, [])
        result = DecodeResult(responses=responses)
            
        if task_id in self.results: del self.results[task_id]
        if task_id in self.events: del self.events[task_id]
        if task_id in self.streaming_results: del self.streaming_results[task_id]

        if isinstance(input, TTSResult):
            input.audio = result.audio
            input.final_state = result.final_state
            if input.stats:
                input.stats.decoder_compute_times = result.chunk_compute_times
        
        return result

    def get_decode_result(self, task_id) -> DecodeResult:
        responses = self.results.get(task_id, [])
        return DecodeResult(responses=responses)

    def stop(self, task_id="default") -> np.ndarray:
        self.codes_q.put(DecodeRequest(task_id=task_id, msg_type="STOP"))
        
        collected_pcm = self.results.get(task_id, [])
        final_pcm = np.concatenate(collected_pcm) if collected_pcm else np.array([], dtype=np.float32)

        if task_id in self.active_decoder_tasks:
            self.active_decoder_tasks.remove(task_id)
            if not self.active_decoder_tasks:
                self.decoder_idle.set()

        if task_id in self.results: del self.results[task_id]
        if task_id in self.events:
            self.events[task_id].set()
            del self.events[task_id]
        if task_id in self.streaming_results: del self.streaming_results[task_id]

        return final_pcm

    def shutdown(self):
        self.stop_listener = True
        
        try:
            if self.decoder_proc and self.decoder_proc.is_alive():
                self.codes_q.put(None)
        except: pass
            
        if self.decoder_proc and self.decoder_proc.is_alive():
            self.decoder_proc.join(timeout=0.3)
            if self.decoder_proc.is_alive():
                try: self.decoder_proc.terminate()
                except: pass
        
        if self.listener_thread:
            self.listener_thread.join(timeout=0.3)
            
        for q in [self.codes_q, self.result_q]:
            try:
                q.cancel_join_thread()
                while not q.empty():
                    q.get_nowait()
                q.close()
            except: pass
