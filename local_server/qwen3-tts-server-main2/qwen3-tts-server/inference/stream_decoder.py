"""
stream_decoder.py - 滑动窗口流式解码器
实现两阶段混合解码策略：累积校准 + 滑动窗口。

Phase 1 (累积校准): samples_per_frame 未知时，每次将全部累积 codes 一起解码，
                     只取增量音频，直到累积足够帧数后校准 samples_per_frame。

Phase 2 (滑动窗口): samples_per_frame 已校准后，取最后 (context_frames + n_new) 帧
                     作为窗口解码，截掉上下文部分，仅输出新增音频。

参考: faster-qwen3-tts 的 generate_voice_clone_streaming 方法
"""
from __future__ import annotations

import time
import numpy as np
from typing import Optional

from . import logger


class StreamDecoder:
    """流式解码器：混合累积解码 + 滑动窗口"""

    def __init__(self, pytorch_decoder, context_frames: int = 25):
        """
        Args:
            pytorch_decoder: PyTorchDecoder 实例（有 decode_window 方法）
            context_frames: 滑动窗口左上下文帧数（默认25，约2秒音频上下文）
        """
        self.decoder = pytorch_decoder              # PyTorchDecoder 实例
        self.context_frames = context_frames         # 滑动窗口左上下文帧数
        self.all_codes: list[np.ndarray] = []        # 累积的所有 codes
        self.prev_audio_len: int = 0                 # 上一次解码后的音频总长度
        self.samples_per_frame: Optional[float] = None  # 每帧对应的音频样本数（校准后固定）
        self.ref_codes: Optional[np.ndarray] = None  # 参考码本（ICL 场景）
        self.min_calibration_frames: int = 0         # 校准所需最少帧数

        # 性能统计
        self._decode_times: list[float] = []         # 每次解码耗时记录

    # =========================================================================
    # 公开接口
    # =========================================================================

    def decode_chunk(self, codes_chunk: np.ndarray) -> np.ndarray:
        """
        流式解码一个 chunk。

        Args:
            codes_chunk: shape [N, 16], 新到达的 N 帧码本

        Returns:
            new_audio: float32 PCM 波形，仅包含本次新增的音频部分
        """
        t0 = time.time()

        # 累积新到达的 codes
        if codes_chunk.ndim == 1:
            codes_chunk = codes_chunk.reshape(-1, 16)
        self.all_codes.append(codes_chunk)

        # 拼接全部累积 codes
        all_codes_arr = np.concatenate(self.all_codes, axis=0)  # [T_total, 16]
        n_total = all_codes_arr.shape[0]
        n_new = codes_chunk.shape[0]

        # 设置校准所需最少帧数 = max(context_frames, chunk_size)
        if self.min_calibration_frames == 0:
            self.min_calibration_frames = max(self.context_frames, n_new)

        if self.samples_per_frame is None:
            # ================================================================
            # Phase 1: 累积解码（校准阶段）
            # ================================================================
            new_audio = self._decode_cumulative(all_codes_arr, n_total)
        else:
            # ================================================================
            # Phase 2: 滑动窗口解码
            # ================================================================
            new_audio = self._decode_sliding_window(all_codes_arr, n_new)

        dt = time.time() - t0
        self._decode_times.append(dt)
        logger.debug(
            f"[StreamDecoder] 解码完成: n_new={n_new}, n_total={n_total}, "
            f"audio_len={len(new_audio)}, 耗时={dt*1000:.1f}ms, "
            f"calibrated={self.samples_per_frame is not None}"
        )

        return new_audio

    def reset(self):
        """重置解码器状态，用于新会话"""
        self.all_codes = []
        self.prev_audio_len = 0
        self.samples_per_frame = None
        self.min_calibration_frames = 0
        self._decode_times = []
        logger.info("[StreamDecoder] 状态已重置")

    @property
    def is_calibrated(self) -> bool:
        """是否已完成 samples_per_frame 校准"""
        return self.samples_per_frame is not None

    def set_ref_codes(self, ref_codes: np.ndarray):
        """
        设置参考码本（用于声音克隆 ICL 场景）。

        Args:
            ref_codes: shape [R, 16], 参考音频的码本序列
        """
        if ref_codes.ndim == 1:
            ref_codes = ref_codes.reshape(-1, 16)
        self.ref_codes = ref_codes.astype(np.int64)
        logger.info(f"[StreamDecoder] 已设置参考码本: {ref_codes.shape[0]} 帧")

    # =========================================================================
    # 内部方法
    # =========================================================================

    def _decode_cumulative(self, all_codes_arr: np.ndarray, n_total: int) -> np.ndarray:
        """
        Phase 1: 累积解码。

        每次将全部累积 codes 一起解码，只取增量音频部分。
        当累积帧数 >= min_calibration_frames 时，校准 samples_per_frame。

        Args:
            all_codes_arr: 全部累积 codes [T_total, 16]
            n_total: 总帧数

        Returns:
            new_audio: 本次新增的音频部分
        """
        # 构建解码输入：如果有 ref_codes，拼在前面
        if self.ref_codes is not None:
            ref_len = self.ref_codes.shape[0]
            decode_input = np.concatenate([self.ref_codes, all_codes_arr], axis=0)
        else:
            ref_len = 0
            decode_input = all_codes_arr

        # 全量解码
        audio = self.decoder.decode_window(decode_input)

        # 如果有 ref_codes，截掉参考音频对应的部分
        if self.ref_codes is not None and ref_len > 0:
            total_len = decode_input.shape[0]
            # 按比例截取参考音频部分
            ref_audio_cut = int(ref_len / max(total_len, 1) * len(audio))
            audio = audio[ref_audio_cut:]

        # 只取增量音频
        new_audio = audio[self.prev_audio_len:]
        self.prev_audio_len = len(audio)

        # 校准判断：累积帧数达到阈值时，计算 samples_per_frame
        if n_total >= self.min_calibration_frames and self.samples_per_frame is None:
            gen_audio_len = len(audio)
            self.samples_per_frame = gen_audio_len / n_total
            logger.info(
                f"[StreamDecoder] 校准完成: samples_per_frame={self.samples_per_frame:.2f} "
                f"(audio_len={gen_audio_len}, n_total={n_total})"
            )

        return new_audio.astype(np.float32) if len(new_audio) > 0 else np.array([], dtype=np.float32)

    def _decode_sliding_window(self, all_codes_arr: np.ndarray, n_new: int) -> np.ndarray:
        """
        Phase 2: 滑动窗口解码。

        取最后 (context_frames + n_new) 帧作为窗口解码，
        截掉上下文部分，仅输出新增音频。

        Args:
            all_codes_arr: 全部累积 codes [T_total, 16]
            n_new: 本次新增帧数

        Returns:
            new_audio: 本次新增的音频部分
        """
        n_ctx = min(self.context_frames, all_codes_arr.shape[0] - n_new)
        window_size = n_ctx + n_new

        # 取窗口：最后 (context_frames + n_new) 帧
        window_codes = all_codes_arr[-window_size:]

        # 解码窗口
        audio = self.decoder.decode_window(window_codes)

        # 计算上下文对应的样本数并截掉
        ctx_samples = int(round(n_ctx * self.samples_per_frame))
        new_audio = audio[ctx_samples:]

        return new_audio.astype(np.float32) if len(new_audio) > 0 else np.array([], dtype=np.float32)

    # =========================================================================
    # 统计与调试
    # =========================================================================

    @property
    def decode_times(self) -> list[float]:
        """获取每次解码的耗时记录（秒）"""
        return self._decode_times

    @property
    def avg_decode_time(self) -> float:
        """平均解码耗时（秒）"""
        if not self._decode_times:
            return 0.0
        return sum(self._decode_times) / len(self._decode_times)

    @property
    def total_frames(self) -> int:
        """当前累积的总帧数"""
        return sum(c.shape[0] for c in self.all_codes)

    def __repr__(self) -> str:
        return (
            f"StreamDecoder(context_frames={self.context_frames}, "
            f"calibrated={self.is_calibrated}, "
            f"samples_per_frame={self.samples_per_frame}, "
            f"total_frames={self.total_frames}, "
            f"has_ref_codes={self.ref_codes is not None})"
        )
