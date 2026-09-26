# Qwen3-TTS 推理核心包 (独立部署版)
# 不再依赖 qwen3_tts_gguf 顶层包

import logging
import os
from datetime import datetime

# 日志目录
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log")
os.makedirs(LOG_DIR, exist_ok=True)

# 统一 Logger
logger = logging.getLogger("qwen3_tts_inference")
logger.setLevel(logging.DEBUG)
logger.propagate = False

if not logger.handlers:
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler = logging.FileHandler(
        os.path.join(LOG_DIR, "latest.log"), mode="a", encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info("日志系统初始化完成")

from .engine import TTSEngine
from .config import TTSConfig
from .schema.result import TTSResult

__all__ = ["logger", "TTSEngine", "TTSConfig", "TTSResult"]
