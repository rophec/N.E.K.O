"""引擎池 — TTSEngine 单例管理 + 信号量并发控制"""
import asyncio
import sys
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

# 确保项目根目录在 sys.path 中
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from inference import TTSEngine, TTSConfig
from inference.schema import constants as schema_constants


class EnginePool:
    """单引擎 + 信号量 + 线程池的并发管理器

    核心约束：TTSStream 创建 GPU LlamaContext，GB10 上仅支持 1 个并发。
    用 asyncio.Semaphore 串行化推理请求，run_in_executor 将同步调用移出事件循环。
    """

    def __init__(self, config: "ServerConfig"):
        self.engine: Optional[TTSEngine] = None
        self.config = config
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="tts-engine")

    async def initialize(self):
        """lifespan 中调用：加载引擎 + 同步模型信息 + warmup"""
        loop = asyncio.get_running_loop()
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_streams)

        # 引擎初始化前：将 config.yaml 中的 voice/language 映射注入 constants
        # 这样 engine._detect_model_type() 在读取 SPEAKER_MAP 时使用的是配置文件中的值
        self._apply_config_to_constants()

        print(f"🔧 [EnginePool] 正在加载引擎: {self.config.model_dir}")
        self.engine = await loop.run_in_executor(
            self._executor,
            self._load_engine,
        )
        if not self.engine or not self.engine.ready:
            raise RuntimeError("引擎初始化失败")

        # 从引擎同步模型类型信息到 config
        self.config.available_voices = self.engine.available_voices
        self.config.default_voice = self.engine.default_voice
        print(f"🎭 [EnginePool] 模型类型: {self.engine.model_type}, 可用音色: {self.engine.available_voices}, 默认音色: {self.engine.default_voice}")

        # Base 没有内置 speaker，必须先收到 ref_audio 才能构建有效 prompt。
        if self.engine.model_type == "base":
            print("✅ [EnginePool] Base 引擎就绪，等待克隆音色请求（跳过预制音色预热）")
        else:
            print("🔥 [EnginePool] 正在预热模型...")
            await loop.run_in_executor(self._executor, self._warmup)
            print("✅ [EnginePool] 引擎就绪，预热完成")

    def _apply_config_to_constants(self):
        """将配置中的 speaker_map / language_map 注入到 constants 模块

        必须在 TTSEngine() 初始化之前调用，因为 engine._detect_model_type()
        需要读取更新后的 SPEAKER_MAP 来判断模型类型（CustomVoice 检测等）。
        """
        if self.config.speaker_map or self.config.voice_profiles:
            schema_constants.apply_voice_config(
                speaker_map=self.config.speaker_map,
                voice_profiles=self.config.voice_profiles or None,
            )
        if self.config.language_map:
            schema_constants.apply_language_config(
                language_map=self.config.language_map,
            )

    def _load_engine(self) -> TTSEngine:
        """同步加载引擎（在 executor 中执行）"""
        return TTSEngine(
            model_dir=self.config.model_dir,
            onnx_provider=self.config.onnx_provider,
            llm_use_gpu=self.config.llm_use_gpu,
            chunk_size=self.config.chunk_size,
            verbose=self.config.verbose,
            n_gpu_layers=self.config.n_gpu_layers,
        )

    def _warmup(self):
        """同步预热（在 executor 中执行）"""
        stream = self.engine.create_stream()
        try:
            result = stream.custom(
                self.config.warmup_text,
                self.engine.default_voice,
                language=self.config.warmup_language,
                config=TTSConfig(max_steps=self.config.warmup_max_steps, streaming=False),
            )
            if result and result.audio is not None:
                print(f"  ✅ Warmup 音频时长: {result.duration:.2f}s")
            else:
                print("  ⚠️ Warmup 未产出音频，但引擎已加载")
        finally:
            stream.shutdown()

    @property
    def semaphore(self) -> asyncio.Semaphore:
        return self._semaphore

    async def shutdown(self):
        """释放引擎资源"""
        if self.engine:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self._executor, self.engine.shutdown)
            self.engine = None
            print("✅ [EnginePool] 引擎已关闭")
        self._executor.shutdown(wait=False)
