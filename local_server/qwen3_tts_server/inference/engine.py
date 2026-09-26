"""
engine.py - Qwen3-TTS 核心引擎
负责资源管理、模型初始化、音频特征提取及渲染。
"""
import os
import ctypes
from typing import Optional, List, Tuple
import numpy as np
from pathlib import Path
from . import llama, logger
from .assets import AssetsManager
from .stream import TTSStream
from .proxy import DecoderProxy
from .encoder import CodecEncoder, SpeakerEncoder
from .schema.constants import SPEAKER_MAP, MODEL_VOICE_PROFILES


def _default_on_audio_chunk(audio: np.ndarray):
    """默认音频回调（无操作），由 StreamingTTSWrapper 覆盖设置"""
    pass


class TTSEngine:
    """
    Qwen3-TTS 引擎：资源池与 Stream 工厂。
    使用 ONNX Runtime + 独立子进程解码器。
    """
    def __init__(self, model_dir="model", onnx_provider="CUDA", llm_use_gpu=True, chunk_size=12, verbose=True, n_gpu_layers=-1):
        import time
        import numpy as np
        from tokenizers import Tokenizer
        
        t_start = time.time()
        self.ready = False
        self.model_dir = Path(model_dir)
        self.chunk_size = chunk_size
        
        # 路径定义 (全线使用 Path 对象)
        self.paths = {
            "talker_gguf": self.model_dir / "qwen3_tts_talker.q5_k.gguf",
            "predictor_gguf": self.model_dir / "qwen3_tts_predictor.q8_0.gguf",
            "decoder_onnx": self.model_dir / "qwen3_tts_decoder.fp16.onnx",
            "codec_enc_onnx": self.model_dir / "qwen3_tts_codec_encoder.fp16.onnx",
            "spk_enc_onnx": self.model_dir / "qwen3_tts_speaker_encoder.fp16.onnx",
            "tokenizer": self.model_dir / 'tokenizer.json',
        }
        
        # 核心文件预检
        missing = [name for name, p in self.paths.items() 
                  if name in ["talker_gguf", "predictor_gguf", "decoder_onnx", "tokenizer"] 
                  and not p.exists()]
        
        if missing:
            logger.error(f"❌ 引擎初始化失败: 缺少核心模型文件 {missing}")
            return

        try:
            # 1. 资产加载
            t_assets = time.time()
            self.assets = AssetsManager(str(self.model_dir))
            self.tokenizer = Tokenizer.from_file(str(self.paths['tokenizer']))
            if verbose: print(f"📦 [Engine] 资产与词表加载完成 (耗时: {time.time()-t_assets:.2f}s)")
            
            # 1.5 检测模型类型（基于 codec_embedding_0 中说话人嵌入的实际值）
            self._detect_model_type(verbose)
            
            # 2. 音频及说话人编码器 (CPU 轻量型)
            self.codec_encoder = None
            self.speaker_encoder = None
            if self.paths["codec_enc_onnx"].exists() and self.paths["spk_enc_onnx"].exists():
                t_enc = time.time()
                self.codec_encoder = CodecEncoder(str(self.paths["codec_enc_onnx"]))
                self.speaker_encoder = SpeakerEncoder(str(self.paths["spk_enc_onnx"]))
                if verbose: print(f"🎤 [Engine] 编码器加载完成 (耗时: {time.time()-t_enc:.2f}s)")

            # 3. 异步拉起解码器进程 (并行点 1)
            t_parallel = time.time()
            self.decoder = DecoderProxy(
                str(self.paths["decoder_onnx"]), 
                onnx_provider=onnx_provider, 
                chunk_size=self.chunk_size,
                on_audio_chunk=_default_on_audio_chunk,
            )
            if verbose: print("⏳ [Engine] 正在拉起子进程解码器...")

            # 4. 模型引擎初始化 (并行点 2: GGUF 在主进程加载，Decoder 在子进程同时初始化)
            t_gguf = time.time()
            self._init_llama_engines(llm_use_gpu, n_gpu_layers)
            if verbose: print(f"🧠 [Engine] GGUF 推理后端就绪 (耗时: {time.time()-t_gguf:.2f}s)")
            
            # 5. 最后同步阻塞等待解码器信号
            is_decoder_ready = self.decoder.wait_until_ready(timeout=10)
            if not is_decoder_ready:
                logger.warning("⚠️ [Engine] 解码器就绪超时，渲染功能将不可用。")
                self.ready = False
            else:
                if verbose: print(f'✅ [Engine] 解码器就绪: Decoder {self.decoder.ready_states["decoder"]} (总并行初始化耗时: {time.time()-t_parallel:.2f}s)')
                self.ready = True
            
            if self.ready:
                print(f"🚀 [Engine] 引擎全链路初始化完成! 总耗时: {time.time()-t_start:.2f}s")
            else:
                print(f"❌ [Engine] 引擎初始化未完全就绪 (由于解码器超时)。")
            
        except Exception as e:
            logger.error(f"❌ 引擎初始化过程中出现致命异常: {e}", exc_info=True)
            self.shutdown()

    def __bool__(self):
        return self.ready

    def _detect_model_type(self, verbose=True):
        # ... same as current implementation ...
        emb0 = self.assets.emb_tables[0]
        
        custom_voice_names = ["vivian", "serena", "uncle_fu", "ryan", "aiden", 
                              "ono_anna", "sohee", "eric", "dylan"]
        custom_ids = [SPEAKER_MAP[name] for name in custom_voice_names if name in SPEAKER_MAP]
        custom_norms = [np.linalg.norm(emb0[sid]) for sid in custom_ids]
        custom_active = sum(1 for n in custom_norms if n > 1.0)
        
        speaker_norms = np.linalg.norm(emb0[2800:3072], axis=1)
        active_indices = np.where(speaker_norms > 1.0)[0] + 2800
        
        if custom_active >= 5:
            self.model_type = "custom"
            if verbose:
                print(f"🎭 [Engine] 检测到 CustomVoice 模型 ({custom_active}/9 音色有效)")
        elif len(active_indices) > 0:
            self.model_type = "hutao"
            active_info = [(int(idx), f"{speaker_norms[idx-2800]:.2f}") for idx in active_indices]
            if verbose:
                print(f"🎭 [Engine] 检测到微调模型 (有效 speaker rows: {active_info}, CustomVoice 音色均为噪声)")
        else:
            self.model_type = "base"
            if verbose:
                print(f"🎭 [Engine] 检测到 Base 模型 (无内置音色，仅支持克隆模式)")
        
        if self.model_type in MODEL_VOICE_PROFILES:
            profile = MODEL_VOICE_PROFILES[self.model_type]
            self.available_voices = profile["voices"]
            self.default_voice = profile["default"]
        else:
            voice_names = []
            for name, sid in SPEAKER_MAP.items():
                if sid in active_indices:
                    voice_names.append(name)
            self.available_voices = voice_names if voice_names else []
            self.default_voice = self.available_voices[0] if self.available_voices else ""
        
        if verbose:
            print(f"🎭 [Engine] 可用音色: {self.available_voices}, 默认: {self.default_voice}")

    def _init_llama_engines(self, llm_use_gpu, n_gpu_layers=-1):
        """初始化 GGUF 模型（仅加载模型，不创建 Context）"""
        logger.info("[Engine] 正在加载 GGUF 模型...")
        
        try:
            self.talker_model = llama.LlamaModel(self.paths["talker_gguf"], n_gpu_layers=n_gpu_layers, use_gpu=llm_use_gpu)
            self.predictor_model = llama.LlamaModel(self.paths["predictor_gguf"], n_gpu_layers=n_gpu_layers, use_gpu=llm_use_gpu)
            
            logger.info("✅ [Engine] GGUF 模型加载完成。")
        except Exception as e:
            logger.error(f"❌ 加载 GGUF 模型失败 (可能是显存不足/OOM): {e}")
            raise

    def create_stream(self, n_ctx=2048, voice_path: Optional[str] = None) -> Optional[TTSStream]:
        """工厂方法：创建语音流"""
        if not self.ready:
            logger.error("❌ 引擎未就绪，无法创建语音流。")
            return None
        return TTSStream(self, n_ctx=n_ctx, voice_path=voice_path)

    def encode(self, input) -> Optional[np.ndarray]:
        """快捷入口：提取音色特征 (Speaker Embedding)。支持传入 numpy 数组或 TTSResult。"""
        if self.speaker_encoder is None:
            logger.error("❌ 本模型无 SpeakerEncoder")
            return None
        return self.speaker_encoder.encode(input)

    def decode(self, codes, **kwargs) -> np.ndarray:
        """快捷入口：解码渲染音频。支持传入 numpy 数组 (codes) 或 TTSResult。"""
        from .schema.result import TTSResult
        result = self.decoder.decode(codes, **kwargs)
        # 返回 numpy 数组而非 DecodeResult 对象（保持向后兼容）
        if isinstance(result, TTSResult):
            return result.audio if result.audio is not None else np.array([], dtype=np.float32)
        if hasattr(result, 'audio'):
            return result.audio if result.audio is not None else np.array([], dtype=np.float32)
        return np.array(result, dtype=np.float32) if result is not None else np.array([], dtype=np.float32)

    def decode_get_result(self, codes, **kwargs):
        """返回完整的 DecodeResult 对象（包含每块耗时统计）"""
        from .schema.result import TTSResult
        return self.decoder.decode(codes, **kwargs)

    def shutdown(self):
        """释放资源，支持重新开启引擎"""
        if not hasattr(self, '_already_shutdown'):
            logger.info("[Engine] 正在关闭引擎 (清理显存与子进程)...")
            try:
                if hasattr(self, "decoder"):
                    self.decoder.shutdown()
                self.talker_model = None
                self.predictor_model = None
            except Exception as e:
                logger.warning(f"⚠️ 关闭引擎时出现小异常 (忽略): {e}")
            self.ready = False
            self._already_shutdown = True
            logger.info("✅ [Engine] 引擎资源已彻底释放。")

    def __del__(self):
        self.shutdown()
