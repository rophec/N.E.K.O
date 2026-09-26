"""配置文件加载器

从 config.yaml 读取配置，环境变量 QWEN3_TTS_* 覆盖 YAML 值。
优先级: 环境变量 > YAML 文件 > ServerConfig 代码默认值
"""
import os
import yaml
from pathlib import Path
from typing import Any, Dict
from .config import ServerConfig


def _find_config_file() -> str:
    """查找配置文件位置"""
    env_path = os.environ.get("QWEN3_TTS_CONFIG_FILE", "")
    if env_path:
        resolved = Path(env_path)
        if resolved.exists():
            return str(resolved)

    current = Path(__file__).resolve().parent.parent
    for name in ("config.yaml", "config.yml"):
        p = current / name
        if p.exists():
            return str(p)
    return str(current / "config.yaml")


def _flatten_yaml(raw: dict) -> dict:
    """将嵌套的 YAML 结构拍平为 ServerConfig 的字段名"""
    data: Dict[str, Any] = {}

    if "server" in raw:
        data["host"] = raw["server"].get("host", "0.0.0.0")
        data["port"] = raw["server"].get("port", 8091)
        data["api_key"] = raw["server"].get("api_key", "")
        data["model_id"] = raw["server"].get("model_id", "Qwen3-TTS-Base")

    if "model" in raw:
        data["model_dir"] = raw["model"].get("path", "")
        data["onnx_provider"] = raw["model"].get("onnx_provider", "CUDA")
        data["llm_use_gpu"] = raw["model"].get("llm_use_gpu", True)
        data["chunk_size"] = raw["model"].get("chunk_size", 3)
        data["n_gpu_layers"] = raw["model"].get("n_gpu_layers", -1)

    if "voices" in raw:
        data["speaker_map"] = raw["voices"].get("map", {})
        data["default_voice"] = raw["voices"].get("default", "")

    if "languages" in raw:
        data["language_map"] = raw["languages"].get("map", {})
        if "default" in raw["languages"]:
            data["warmup_language"] = raw["languages"]["default"]

    if "inference" in raw:
        data["max_steps"] = raw["inference"].get("max_steps", 300)
        data["seed"] = raw["inference"].get("seed", 42)
        data["sub_seed"] = raw["inference"].get("sub_seed", 45)

    if "logging" in raw:
        data["log_dir"] = raw["logging"].get("dir", "")
        data["log_level"] = raw["logging"].get("level", "DEBUG")

    return data


def _apply_env_overrides(data: dict) -> dict:
    """环境变量覆盖简单类型字段"""
    prefix = "QWEN3_TTS_"
    scalar_fields = [
        "host", "port", "model_dir", "model_id", "onnx_provider",
        "llm_use_gpu", "chunk_size", "verbose", "max_concurrent_streams",
        "api_key", "warmup_text", "warmup_language", "request_timeout",
        "ws_idle_timeout", "decode_timeout", "voice_cache_dir",
        "default_voice", "n_gpu_layers", "max_steps", "seed", "sub_seed",
    ]
    for field in scalar_fields:
        env_key = f"{prefix}{field.upper()}"
        if env_key in os.environ:
            data[field] = os.environ[env_key]

    for bool_field in ("llm_use_gpu", "verbose"):
        env_key = f"{prefix}{bool_field.upper()}"
        if env_key in os.environ:
            val = os.environ[env_key].strip().lower()
            data[bool_field] = val in ("1", "true", "yes", "on")

    for int_field in ("port", "chunk_size", "max_concurrent_streams",
                      "n_gpu_layers", "max_steps", "seed", "sub_seed",
                      "request_timeout", "ws_idle_timeout",
                      "decode_timeout"):
        env_key = f"{prefix}{int_field.upper()}"
        if env_key in os.environ:
            try:
                data[int_field] = int(os.environ[env_key])
            except (ValueError, TypeError):
                pass

    return data


def load_config(yaml_path: str = "") -> ServerConfig:
    """全量加载配置

    Args:
        yaml_path: 配置文件路径，为空则自动查找

    Returns:
        ServerConfig 实例，所有字段已填充
    """
    path = yaml_path or _find_config_file()

    # 1. 从 YAML 加载
    data: Dict[str, Any] = {}
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        data = _flatten_yaml(raw)

    # 2. 环境变量覆盖
    data = _apply_env_overrides(data)

    # 3. 构造 ServerConfig
    config = ServerConfig(**data)

    model_dir = Path(config.model_dir).expanduser()
    if not model_dir.is_absolute():
        model_dir = Path(__file__).resolve().parent.parent / model_dir
    config.model_dir = str(model_dir.resolve())

    voice_cache_dir = Path(config.voice_cache_dir).expanduser()
    if not voice_cache_dir.is_absolute():
        voice_cache_dir = Path(__file__).resolve().parent.parent / voice_cache_dir
    config.voice_cache_dir = str(voice_cache_dir.resolve())

    # 4. 补充复杂类型（无法通过 env var 传递）
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        if "voices" in raw:
            if not config.speaker_map:
                config.speaker_map = raw["voices"].get("map", {})
        if "languages" in raw:
            if not config.language_map:
                config.language_map = raw["languages"].get("map", {})

    return config
