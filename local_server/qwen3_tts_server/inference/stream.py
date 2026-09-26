"""
stream.py - TTS 语音流
核心逻辑所在，管理单次会话的上下文，支持流式和非流式合成。
"""
from __future__ import annotations

import time
import os
import numpy as np
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union
from .schema.constants import PROTOCOL, map_speaker, map_language
from .schema.result import TTSResult, Timing, LoopOutput, DecodeResult
from .config import TTSConfig
from .talker import TalkerPredictor
from .predictor import Predictor
# ONNX 解码器通过 engine.decoder (DecoderProxy) 使用

from . import llama, logger
from .prompt_builder import PromptBuilder, PromptData
from .utils.audio import load_audio

if TYPE_CHECKING:
    from .engine import TTSEngine

class TTSStream:
    """
    保存 Talker, Predictor, Decoder 记忆的语音流。
    """
    def __init__(self, engine: TTSEngine, n_ctx: int = 2048, voice_path: Optional[str] = None):
        self.engine = engine
        self.assets = engine.assets
        self.tokenizer = engine.tokenizer
        self.n_ctx = n_ctx
        
        # 1. 初始化流独立的 Context 和 Batch
        self._init_contexts()
        
        # 2. 初始化推理核心 (Talker, Predictor & PromptBuilder)
        self.talker = TalkerPredictor(engine.talker_model, self.talker_ctx, self.talker_batch, self.assets)
        self.predictor = Predictor(engine.predictor_model, self.predictor_ctx, self.predictor_batch, self.assets)
        self.prompt_builder = PromptBuilder(self.tokenizer, self.assets)
        
        # 3. 音色锚点 (Voice)
        self.voice: Optional[TTSResult] = None
        if voice_path:
            self.set_voice(voice_path)
            
        # 4. 解码器初始化：使用 engine.decoder (DecoderProxy)
        self.decoder = getattr(engine, 'decoder', None)
        # 流式场景音频回调（由 StreamingTTSWrapper 设置）
        self.on_audio_chunk = None
        self.task_counter = 0 # 任务计数器，用于生成唯一 task_id

    def _init_contexts(self):
        """初始化此语音流专属的推理环境"""
        # engine.talker_model 是 LlamaModel 对象
        self.talker_ctx = llama.LlamaContext(self.engine.talker_model, n_ctx=self.n_ctx, embeddings=True)
        self.predictor_ctx = llama.LlamaContext(self.engine.predictor_model, n_ctx=64, embeddings=False)
        
        # 使用模型自带的 n_embd，兼容 0.6B(1024) 和 1.7B(2048)
        logger.info(f"[Stream] Talker Dim: {self.engine.talker_model.n_embd} | Predictor Dim: {self.engine.predictor_model.n_embd}")
        self.talker_batch = llama.LlamaBatch(self.n_ctx, embd_dim=self.engine.talker_model.n_embd)
        self.predictor_batch = llama.LlamaBatch(2, embd_dim=self.engine.predictor_model.n_embd)

    # =========================================================================
    # 核心推理 API
    # =========================================================================

    def clone(self, 
              text: str, 
              language: str = "chinese",
              zero_shot: bool = False,
              config: Optional[TTSConfig] = None) -> Optional[TTSResult]:
        """
        [克隆模式] 使用当前流中已设定的音色锚点（Voice Anchor）进行语音合成。

        Args:
            text: 待合成的目标文本。
            language: 目标语言。可选:
                - 'chinese' , 'english', 'japanese', 'korean'
                - 'german', 'spanish', 'french', 'russian', 'italian', 'portuguese'
                - 'beijing_dialect' , 'sichuan_dialect' 
            zero_shot: 是否启用零样本克隆模式。
            config: 推理配置对象 (TTSConfig)，可控制 Temperature, Top-P 等采样参数。

        Returns:
            TTSResult 对象，包含完整音频、特征码及性能统计。
        """
        if self.voice is None:
            logger.error("❌ 尚未设定音色锚点，请先调用 set_voice()。")
            return None
            
        cfg = config or TTSConfig()
        self.talker.clear_memory()
        
        try:
            lang_id = map_language(language)
            pdata = self.prompt_builder.build_clone_prompt(text, self.voice, lang_id, zero_shot)
            
            timing = Timing()
            timing.prompt_time = pdata.compile_time
            
            lout = self._run_engine_loop(pdata, timing, cfg)
            
            return self._post_process(text, pdata, lout)
        except Exception as e:
            logger.error(f"❌ Clone 推理失败: {e}", exc_info=True)
            print(f"❌ Clone 推理失败: {e}")
            return None

    def custom(self,
               text: str,
               speaker: Union[str, int, np.ndarray],
               language: str = "chinese",
               instruct: Optional[str] = None,
               config: Optional[TTSConfig] = None) -> Optional[TTSResult]:
        """
        [内置音色模式] 使用官方内置的预设音色进行合成，支持自然语言渲染指令。

        Args:
            text: 待合成的目标文本。
            speaker: 说话人名 (str)、ID (int) 或 嵌入 (np.ndarray)。
                - 字符串名: ['Vivian', 'Serena', 'Ono_Anna', 'Sohee', 'Ryan', 'Aiden', 'Uncle_Fu', 'Eric', 'Dylan']
                - 数字 ID: 2800-3071
                - 嵌入矩阵: np.ndarray (D,)
            language: 目标语言。可选:
                - 'chinese' , 'english', 'japanese', 'korean'
                - 'german', 'spanish', 'french', 'russian', 'italian', 'portuguese'
                - 'beijing_dialect' , 'sichuan_dialect' 
            instruct: 渲染指令，如 "用温柔的语气说" 或 "充满活力的播报"。
            config: 推理配置对象 (TTSConfig)。

        Returns:
            TTSResult 对象。
        """
        cfg = config or TTSConfig()
        self.talker.clear_memory()
        
        try:
            lang_id = map_language(language) if language.lower() != "auto" else None
            pdata = self.prompt_builder.build_custom_prompt(text, speaker, lang_id, instruct)
            
            timing = Timing()
            timing.prompt_time = pdata.compile_time
            
            lout = self._run_engine_loop(pdata, timing, cfg)
            return self._post_process(text, pdata, lout)
        except Exception as e:
            logger.error(f"❌ Custom 推理失败: {e}", exc_info=True)
            return None

    def design(self,
               text: str,
               instruct: str,
               language: str = "chinese",
               config: Optional[TTSConfig] = None) -> Optional[TTSResult]:
        """
        [音色设计模式] 完全通过自然语言描述来设计并生成一个全新的音色。

        Args:
            text: 待合成的目标文本。
            instruct: 音色设计描述。例如："体现撒娇稚嫩的萝莉女声，音调偏高且起伏明显。"
            language: 目标语言。可选:
                - 'chinese' , 'english', 'japanese', 'korean'
                - 'german', 'spanish', 'french', 'russian', 'italian', 'portuguese'
                - 'beijing_dialect' , 'sichuan_dialect' 
            config: 推理配置对象 (TTSConfig)。
            streaming: 是否启用流式推理。

        Returns:
            TTSResult 对象。
        """
        cfg = config or TTSConfig()
        self.talker.clear_memory()
        
        try:
            lang_id = map_language(language) if language.lower() != "auto" else None
            pdata = self.prompt_builder.build_design_prompt(text, instruct, lang_id)
            
            timing = Timing()
            timing.prompt_time = pdata.compile_time
            
            lout = self._run_engine_loop(pdata, timing, cfg)
            return self._post_process(text, pdata, lout)
        except Exception as e:
            logger.error(f"❌ Design 推理失败: {e}", exc_info=True)
            return None

    def tts(self, *args, **kwargs):
        return self.clone(*args, **kwargs)

    def _run_engine_loop(self, pdata: PromptData, timing: Timing, cfg: TTSConfig) -> LoopOutput:
        streaming = cfg.streaming
        chunk_size = self.engine.chunk_size
        all_codes = []
        turn_summed_embeds = []
        chunk_buffer = []
        
        logger.info(f"[Stream] 启动推理循环: max_steps={cfg.max_steps}, streaming={streaming}")
        
        step_count = 0
        last_chunk_time = time.time()
        current_task_id = f"stream_{self.task_counter}"
        self.task_counter += 1
        
        for step_codes, summed_vec in self._run_engine_loop_gen(pdata, cfg, timing):
            step_count += 1
            all_codes.append(step_codes)
            turn_summed_embeds.append(summed_vec)
            
            if step_count % 50 == 0:
                logger.info(f"[Stream] 正在生成... 已完成 {step_count} 步")
            
            chunk_buffer.append(step_codes)

            if not streaming or len(chunk_buffer) < chunk_size:
                continue

            timing.chunk_gen_times.append(time.time() - last_chunk_time); last_chunk_time = time.time()
            
            # 流式场景：将 chunk 推送给 ONNX 解码器子进程（非阻塞）
            # 首 chunk 带上 voice.final_state（ICL 上下文记忆）
            if self.decoder:
                buf_codes = np.array(chunk_buffer)
                state = self.voice.final_state if len(all_codes) == chunk_size and self.voice else None
                self.decoder.decode(buf_codes, task_id=current_task_id, 
                                    stream=True, is_final=False, state=state)
            
            chunk_buffer = []

        logger.info(f"[Stream] 推理循环结束，共 {step_count} 步")

        timing.chunk_gen_times.append(time.time() - last_chunk_time)

        # 最后一步：解码剩余 codes（流式只发 chunk_buffer 剩余，非流式发全部）
        decode_result = None
        if self.decoder:
            if streaming:
                # 流式：只发 chunk_buffer 中剩余未解码的 codes
                # 子进程 session 已有累积 state，不需再传 voice.final_state
                tail_codes = np.array(chunk_buffer, dtype=np.int64) if chunk_buffer else np.zeros((0, 16), dtype=np.int64)
                dr = self.decoder.decode(tail_codes, task_id=current_task_id,
                                         is_final=True, stream=True, state=None)
            else:
                # 非流式：发全部 codes，首包带 voice.final_state（ICL 上下文）
                all_codes_arr = np.concatenate(all_codes) if all_codes else np.zeros((0, 16), dtype=np.int64)
                state = self.voice.final_state if (self.voice and self.voice.final_state) else None
                dr = self.decoder.decode(all_codes_arr, task_id=current_task_id,
                                         is_final=True, state=state)
            decode_result = dr
            timing.decoder_compute_times = decode_result.chunk_compute_times

        return LoopOutput(
            all_codes=all_codes, 
            summed_embeds=turn_summed_embeds, 
            timing=timing,
            decode_result=decode_result
        )

    def _create_sampler(self, do_sample: bool, temperature: float, top_p: float, top_k: int, 
                        min_p: float = 0.0, repeat_penalty: float = 1.0, 
                        frequency_penalty: float = 0.0, presence_penalty: float = 0.0,
                        penalty_last_n: int = 128, seed: Optional[int] = None) -> llama.LlamaSampler:
        """创建原生采样器实例"""
        return llama.LlamaSampler(
            temperature=temperature if do_sample else 0.0,
            top_p=top_p if do_sample else 1.0,
            top_k=top_k if do_sample else 0,
            min_p=min_p if do_sample else 0.0,
            repeat_penalty=repeat_penalty,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            penalty_last_n=penalty_last_n,
            seed=seed
        )

    def _run_engine_loop_gen(self, pdata: PromptData, cfg: TTSConfig, timing: Timing):
        t_pre_s = time.time()
        # Talker 内部现在会处理 pdata 中的 embd 和 trailing_text_pool
        m_hidden = self.talker.prefill(pdata, seq_id=0)
        timing.prefill_time = time.time() - t_pre_s
        
        # 1. 初始化两级原生采样器
        # 大师阶段 (Talker): 需要全套采样增强 (惩罚项 + Min-P)
        talker_sampler = self._create_sampler(
            cfg.do_sample, cfg.temperature, cfg.top_p, cfg.top_k, 
            min_p=cfg.min_p, 
            repeat_penalty=cfg.repeat_penalty,
            frequency_penalty=cfg.frequency_penalty,
            presence_penalty=cfg.presence_penalty,
            penalty_last_n=cfg.penalty_last_n,
            seed=cfg.seed
        )
        # 工匠阶段 (Predictor): 通常使用简洁采样，不应用惩罚项以保持声音稳定
        predictor_sampler = self._create_sampler(
            cfg.sub_do_sample, 
            cfg.sub_temperature, 
            cfg.sub_top_p, 
            cfg.sub_top_k, 
            seed=cfg.sub_seed
        )
        
        # 惩罚项豁免名单：不希望因为生成过 EOS/BOS 而降低结尾概率
        allow_tokens = {PROTOCOL["EOS"], PROTOCOL["PAD"], PROTOCOL["BOS"]}
            
        step_idx = 0
        try:
            for step_idx in range(cfg.max_steps):
                # 采样获取第 0 层码本 (大师决策)
                code_0 = talker_sampler.sample(
                    self.talker.ctx, idx=-1, 
                    limit_start=0, limit_end=2048, 
                    allow_tokens=allow_tokens # 官方豁免权：允许采到 EOS
                )
                
                # 更新历史记录
                talker_sampler.accept(code_0)
                
                if code_0 == PROTOCOL["EOS"]:
                    break
                
                
                # ---------------- Predictor Stage ----------------
                # 根据第 0 层码本和 Talker 隐层，预测完整的 16 层码本
                t_c_s = time.time()
                step_codes, step_embeds_2048 = self.predictor.predict_frame(
                    m_hidden, 
                    code_0, 
                    sampler=predictor_sampler
                )
                timing.predictor_loop_times.append(time.time() - t_c_s)
                
                # ---------------- Feedback Stage ----------------
                t_m_s = time.time()
                # 汇总 16 层音频 Embedding
                audio_summed = np.sum(step_embeds_2048, axis=0) 
                # 反馈给 Talker。Talker 内部会自动执行 [Audio + Text] 融合
                m_hidden = self.talker.decode_step(audio_summed, seq_id=0)
                timing.talker_loop_times.append(time.time() - t_m_s)
                
                yield step_codes, audio_summed
                
        finally:
            talker_sampler.free()
            predictor_sampler.free()
            
        timing.total_steps = len(pdata.embd) + step_idx

    def _post_process(self, 
                     text: str, 
                     pdata: PromptData, 
                     lout: LoopOutput) -> TTSResult:
        # 获取音频：流式场景音频已通过回调推送，非流式场景从 decode_result 获取
        audio = lout.decode_result.audio if lout.decode_result else None
        # PyTorch decoder 无状态，final_state 为 None（保持字段兼容）
        final_state = lout.decode_result.final_state if lout.decode_result else None
        
        return TTSResult(
            audio=audio,
            text=text,
            text_ids=pdata.text_ids,
            spk_emb=pdata.spk_emb,
            codes=np.array(lout.all_codes),
            summed_embeds=lout.summed_embeds,
            stats=lout.timing,
            final_state=final_state,
            ref_codes=self.voice.codes if self.voice else None
        )

    def reset(self):
        self.talker.clear_memory()
        self.predictor_ctx.clear_kv_cache()
        self.voice = None
        logger.info("扫 [Stream] 状态已重置 (talker/predictor KV cache + voice)")

    def join(self, timeout: Optional[float] = None):
        """阻塞直至当前流所有音频全部完毕"""
        if self.decoder:
            self.decoder.join_decoder(timeout)

    def shutdown(self):
        # 让 Python GC 处理内存释放 (_del_ 会调用 free)
        self.talker_batch = None
        self.talker_ctx = None
        self.predictor_batch = None
        self.predictor_ctx = None
        self.decoder = None

    # =========================================================================
    # 音色设置 API (Voice Management)
    # =========================================================================

    def set_voice(self, source: Union[TTSResult, str, Path], text: Optional[str] = None, **kwargs) -> Union[bool, TTSResult]:
        """统一设置当前流的音色锚点。返回生成的 TTSResult 或 False。"""
        try:
            success = False
            if isinstance(source, TTSResult):
                success = self._set_voice_from_result(source)
            else:
                source_p = Path(source)
                if source_p.suffix.lower() == ".json":
                    success = self._set_voice_from_json(source_p)
                elif source_p.suffix.lower() in [".wav", ".mp3", ".flac", ".m4a", ".opus"]:
                    success = self._set_voice_from_audio(source_p, text or "", **kwargs)
                else:
                    # 尝试作为内置说话人处理
                    return self.set_voice_from_speaker(str(source), text or "你好", **kwargs)
            
            return self.voice if success else False
        except Exception as e:
            logger.error(f"❌ 设置音色时出现无法预料的异常: {e}")
            print(f"❌ 设置音色时出现无法预料的异常: {e}")
            return False

    def _set_voice_from_result(self, res: TTSResult) -> bool:
        if not res.is_valid_anchor:
            return False

        # 如果缺少音色向量但存在 codes，解码音频后提取
        if res.spk_emb is None and len(res.codes) > 0:
            logger.info("🎤 [Stream] 音色向量缺失，正在从 codes 解码音频并提取...")
            if res.audio is None:
                res.audio = self.engine.decode(res.codes)
            self.engine.encode(res)

        # 如果维度不匹配，执行重编码转换
        if res.spk_emb is not None and res.spk_emb.shape[-1] != self.engine.talker_model.n_embd:
            logger.info(f"🔄 [Stream] 维度不匹配 ({res.spk_emb.shape[-1]}->{self.engine.talker_model.n_embd})，正在转换...")
            if res.audio is None:
                self.engine.decode(res)
            self.engine.encode(res)
        
        # 如果缺少解码状态记忆（例如从 JSON 加载），把参考 codes 作为
        # 未结束前缀预填充。这里不能走普通 final decode，否则会 flush 掉
        # 参考尾部，后续生成就不再是官方语义上的 continuation。
        if res.final_state is None and len(res.codes) > 0 and self.engine.decoder:
            prepare_state = getattr(self.engine.decoder, "prepare_continuation_state", None)
            if prepare_state is None:
                raise RuntimeError("当前 Decoder 不支持参考声音续读状态")
            res.final_state = prepare_state(res.codes)
            print(
                f"[Qwen3-TTS Clone][reference.continuation_ready] "
                f"frames={len(res.codes)} is_final=False",
                flush=True,
            )

        self.voice = res
        logger.info(f"🎭 音色已切换为: {res.text[:20]}...")
        return True

    def _set_voice_from_json(self, path: Path) -> bool:
        """从 JSON 文件恢复音色锚点"""
        if not path.exists():
            logger.error(f"❌ 未找到音色 JSON 文件: {path}")
            return False
        try:
            res = TTSResult.from_json(str(path))
            return self._set_voice_from_result(res)
        except Exception as e:
            logger.error(f"❌ 解析音色 JSON 失败 ({path.name}): {e}")
            return False

    def _set_voice_from_audio(self, wav_path: Path, text: str) -> bool:
        """从音频文件克隆音色：使用 pydub 标准化输入"""
        if self.engine.codec_encoder is None or self.engine.speaker_encoder is None:
            logger.error("⚠️ 编码器模块未加载，无法执行音色克隆。")
            return False
            
        logger.info(f"🎤 正在从音频提取音色特征: {wav_path.name}")
        
        # 1. 万能格式转换与预处理 (24kHz, Mono, float32)
        samples = load_audio(wav_path)
        if samples is None:
            return False
            
        # 2. 推理提取特征 (直接传入内存 samples，不再依赖临时文件)
        try:
            codes = self.engine.codec_encoder.encode(samples)
            spk_emb = self.engine.speaker_encoder.encode(samples)
            
            res = TTSResult(text=text, text_ids=self.tokenizer.encode(text).ids, spk_emb=spk_emb, codes=codes)
            return self._set_voice_from_result(res)
        except Exception as e:
            logger.error(f"❌ 音声特征提取失败: {e}")
            return False

    def set_voice_from_speaker(self, speaker_id: str, text: str, **kwargs) -> Optional[TTSResult]:
        """从内置说话人生成音色锚点并设置"""
        try:
            logger.info(f"📍 正在从内置说话人初始化音色核心: {speaker_id}")
            # kwargs 包含 language, streaming 等
            res = self.custom(text, speaker_id, **kwargs)
            if res:
                self._set_voice_from_result(res)
                return res
            return None
        except Exception as e:
            logger.error(f"❌ 内置音色初始化失败: {e}")
            return None
