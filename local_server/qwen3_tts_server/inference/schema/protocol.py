"""
protocol.py - 进程间通信协议定义 (独立部署版)
仅保留 Decoder 相关协议，移除 Speaker/Recorder 协议。
"""
from dataclasses import dataclass, field
from typing import Optional, Union, List
import numpy as np

@dataclass
class DecoderState:
    """解码器的核心记忆 (不跨进程传输，仅在 Worker 内部流转)"""
    pre_conv_history: Optional[np.ndarray] = None
    latent_buffer: Optional[np.ndarray] = None
    conv_history: Optional[np.ndarray] = None
    kv_cache: List[np.ndarray] = field(default_factory=list)
    skip_samples: int = 0
    latent_audio: Optional[np.ndarray] = None

@dataclass
class DecoderSession:
    """会话上下文：维护状态与索引"""
    state: Optional[DecoderState] = None
    index: int = 0

@dataclass
class DecodeRequest:
    """主进程 -> DecoderWorker"""
    task_id: Union[str, int]
    msg_type: str = "DECODE"
    codes: Optional[np.ndarray] = None
    is_final: bool = False
    state: Optional["DecoderState"] = None

@dataclass
class DecoderResponse:
    """DecoderWorker -> Proxy"""
    task_id: Union[str, int]
    msg_type: str = "AUDIO"
    index: int = 0
    audio: Optional[np.ndarray] = None
    compute_time: float = 0.0
    state: Optional["DecoderState"] = None
    recv_time: float = 0.0
