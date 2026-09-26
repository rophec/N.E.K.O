"""服务配置 — Pydantic Settings，支持环境变量和 .env 文件"""
import tempfile
from pathlib import Path
from typing import List, Dict, Any

from pydantic_settings import BaseSettings


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class ServerConfig(BaseSettings):
    """Qwen3-TTS-GGUF 服务端配置"""

    # ── 模型 ──
    model_dir: str = str(PROJECT_ROOT / "model" / "Qwen3-TTS-12Hz-1.7B-Base-GGUF")
    model_id: str = "Qwen3-TTS-Base"
    onnx_provider: str = "CUDA"
    llm_use_gpu: bool = True
    chunk_size: int = 3
    verbose: bool = True
    n_gpu_layers: int = -1

    # ── 服务器 ──
    host: str = "0.0.0.0"
    port: int = 8091
    # API Key 鉴权: 设置后请求需携带 Authorization: Bearer <api_key>
    api_key: str = ""

    # ── 并发 ──
    max_concurrent_streams: int = 1  # GB10: 1, 更大显存可调高

    # ── 音色缓存 ──
    voice_cache_dir: str = str(Path(tempfile.gettempdir()) / "neko_qwen3_tts_voice_cache")

    # ── 可用音色列表 (引擎初始化后自动从 engine 同步，无需手动配置) ──
    available_voices: List[str] = []

    # ── 默认音色 (引擎初始化后自动从 engine 同步) ──
    default_voice: str = ""

    # ── 说话人映射 (从 config.yaml 加载，替代 constants.py 中的硬编码) ──
    speaker_map: Dict[str, int] = {}

    # ── 音色配置档案 (模型类型 → 可用音色列表) ──
    voice_profiles: Dict[str, Dict[str, Any]] = {}

    # ── 语言映射 (从 config.yaml 加载) ──
    language_map: Dict[str, int] = {}

    # ── Warmup ──
    warmup_text: str = "你好"
    warmup_language: str = "chinese"
    warmup_max_steps: int = 50
    max_steps: int = 300
    seed: int = 42
    sub_seed: int = 45
    # warmup_speaker 由引擎自动检测后设置

    # ── 日志 ──
    log_dir: str = ""
    log_level: str = "DEBUG"

    # ── 超时 ──
    request_timeout: float = 60.0       # 排队等待超时 (秒)
    ws_idle_timeout: float = 30.0       # WS 空闲超时 (秒)
    decode_timeout: float = 30.0        # 单次解码超时 (秒)

    model_config = {"env_prefix": "QWEN3_TTS_"}
