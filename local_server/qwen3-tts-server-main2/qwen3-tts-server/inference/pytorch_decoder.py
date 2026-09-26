"""
pytorch_decoder.py - PyTorch 解码器封装 (PyTorchDecoder)
提供与 StatefulDecoder 类似的对外接口，但内部使用 PyTorch 推理。

使用方法:
    decoder = PyTorchDecoder("/data/models/Qwen3-TTS-12Hz-1.7B-Hutao-GGUF/")
    decoder.warmup()
    
    # 完整解码（非流式）
    audio = decoder.decode(codes)  # codes: [T, 16] -> float32 PCM
    
    # 流式滑动窗口解码
    audio = decoder.chunked_decode(codes, chunk_size=300, left_context=25)
"""
import os
import numpy as np
import torch
from . import logger


class PyTorchDecoder:
    """
    PyTorch codec 解码器封装。
    
    特点:
    - 封装 Qwen3TTSTokenizerV2Decoder，提供简洁的解码接口
    - 支持完整解码和流式滑动窗口解码
    - 输入 numpy ndarray，输出 numpy ndarray
    - 使用 torch.inference_mode() 进行推理
    """
    
    # 常量配置
    SAMPLE_RATE = 24000           # 采样率 24kHz
    SAMPLES_PER_FRAME = 1920     # 每帧 1920 样本 (12Hz × 1920 = 24000)
    TOTAL_UPSAMPLE = 1920        # 总上采样因子
    NUM_QUANTIZERS = 16          # 量化器数量（码本层数）
    
    def __init__(self, model_dir: str, device: str = "cuda"):
        """
        初始化 PyTorch 解码器，加载 decoder 权重。
        
        加载优先级:
        1. codec_decoder.pt - 预提取的独立权重文件
        2. 从 speech_tokenizer/ 子目录加载完整模型，提取 decoder 部分
        
        Args:
            model_dir: 模型目录路径，包含权重文件或 speech_tokenizer 子目录
            device: 推理设备，如 "cuda"、"cpu"
        
        Raises:
            FileNotFoundError: 权重文件不存在时抛出
            RuntimeError: 模型加载失败时抛出
        """
        self.model_dir = model_dir
        self.device = device
        
        # 尝试方式一：加载预提取的独立权重文件
        codec_decoder_path = os.path.join(model_dir, "codec_decoder.pt")
        speech_tokenizer_dir = os.path.join(model_dir, "speech_tokenizer")
        
        if os.path.exists(codec_decoder_path):
            logger.info(f"[PyTorchDecoder] 加载预提取权重: {codec_decoder_path}")
            self.decoder = self._load_from_standalone(codec_decoder_path)
        elif os.path.exists(speech_tokenizer_dir):
            logger.info(f"[PyTorchDecoder] 从完整模型提取 decoder: {speech_tokenizer_dir}")
            self.decoder = self._load_from_full_model(speech_tokenizer_dir)
        else:
            raise FileNotFoundError(
                f"未找到 decoder 权重文件。请确保以下路径之一存在:\n"
                f"  - {codec_decoder_path} (预提取的独立权重)\n"
                f"  - {speech_tokenizer_dir}/ (完整 speech_tokenizer 模型目录)"
            )
        
        # 移至指定设备并设为评估模式
        self.decoder.to(device).eval()
        logger.info(f"[PyTorchDecoder] 模型已加载至 {device}")
    
    def _load_config(self, model_dir: str) -> "Qwen3TTSTokenizerV2DecoderConfig":
        """
        从 speech_tokenizer/config.json 中读取 decoder 配置。
        
        优先级:
        1. speech_tokenizer/config.json 中的 decoder_config 字段
        2. codec_decoder_config.json（提取脚本生成的）
        3. 默认配置
        
        Args:
            model_dir: 模型根目录
            
        Returns:
            Qwen3TTSTokenizerV2DecoderConfig 实例
        """
        import json as _json
        from qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import (
            Qwen3TTSTokenizerV2DecoderConfig,
        )
        
        # 策略1: 从 speech_tokenizer/config.json 读取 decoder_config
        speech_tok_config = os.path.join(model_dir, "speech_tokenizer", "config.json")
        if os.path.exists(speech_tok_config):
            with open(speech_tok_config, "r", encoding="utf-8") as f:
                full_config = _json.load(f)
            decoder_config_dict = full_config.get("decoder_config", {})
            if decoder_config_dict:
                logger.info(f"[PyTorchDecoder] 从 speech_tokenizer/config.json 读取 decoder 配置")
                return Qwen3TTSTokenizerV2DecoderConfig(**decoder_config_dict)
        
        # 策略2: 从 codec_decoder_config.json 读取
        extracted_config = os.path.join(model_dir, "codec_decoder_config.json")
        if os.path.exists(extracted_config):
            with open(extracted_config, "r", encoding="utf-8") as f:
                config_dict = _json.load(f)
            # 过滤掉 HuggingFace 基类的无关字段，只保留 DecoderConfig 认识的参数
            valid_keys = set(Qwen3TTSTokenizerV2DecoderConfig().__dict__.keys())
            # 也允许额外的关键字段
            valid_keys.update({"codebook_dim", "head_dim", "num_semantic_quantizers", 
                              "semantic_codebook_size", "vector_quantization_hidden_dimension"})
            filtered = {k: v for k, v in config_dict.items() if k in valid_keys}
            logger.info(f"[PyTorchDecoder] 从 codec_decoder_config.json 读取 decoder 配置")
            return Qwen3TTSTokenizerV2DecoderConfig(**filtered)
        
        # 策略3: 默认配置
        logger.warning("[PyTorchDecoder] 未找到配置文件，使用默认 DecoderConfig")
        return Qwen3TTSTokenizerV2DecoderConfig()
    
    def _load_from_standalone(self, weight_path: str):
        """
        从预提取的独立权重文件加载 decoder。
        
        Args:
            weight_path: codec_decoder.pt 文件路径
            
        Returns:
            加载好权重的 Qwen3TTSTokenizerV2Decoder 实例
        """
        from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
            Qwen3TTSTokenizerV2Decoder,
        )
        
        # 加载 state_dict
        state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)
        
        # 从模型目录读取正确的配置
        config = self._load_config(self.model_dir)
        
        # 构建模型并加载权重
        model = Qwen3TTSTokenizerV2Decoder(config)
        model.load_state_dict(state_dict, strict=False)
        
        return model
    
    def _load_from_full_model(self, model_dir: str):
        """
        从完整 speech_tokenizer 模型目录加载，提取 decoder 部分。
        
        Args:
            model_dir: speech_tokenizer 目录路径
            
        Returns:
            提取出的 Qwen3TTSTokenizerV2Decoder 实例
        """
        try:
            from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
                Qwen3TTSTokenizerV2Model,
            )
        except ImportError as e:
            raise ImportError(
                "无法导入 Qwen3TTSTokenizerV2Model，"
                "请确保 qwen_tts 包已安装 (pip install qwen_tts)"
            ) from e
        
        # 加载完整模型
        full_model = Qwen3TTSTokenizerV2Model.from_pretrained(model_dir)
        
        # 提取 decoder
        decoder = full_model.decoder
        
        # 释放 encoder 内存
        del full_model
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        return decoder
    
    def _codes_to_tensor(self, codes: np.ndarray) -> torch.Tensor:
        """
        将 numpy codes 转换为模型所需的 torch.Tensor 格式。
        
        转换: [T, 16] -> [1, 16, T]
        
        Args:
            codes: numpy ndarray, shape [T, 16], dtype int64
            
        Returns:
            torch.Tensor, shape [1, 16, T], 在 self.device 上
        """
        # 确保输入是 2D
        if codes.ndim == 1:
            codes = codes.reshape(-1, self.NUM_QUANTIZERS)
        
        # 转换为 torch.Tensor 并转置: [T, 16] -> [16, T] -> [1, 16, T]
        tensor = torch.from_numpy(codes.astype(np.int64))
        tensor = tensor.T.unsqueeze(0).to(self.device)
        
        return tensor
    
    def decode(self, codes: np.ndarray) -> np.ndarray:
        """
        一次性完整解码（非流式）。
        
        输入 codes shape [T, 16]，输出 float32 PCM 波形。
        内部调用 decoder.forward(codes) 进行完整推理。
        
        Args:
            codes: 音频码, shape [T, 16], dtype int64, 值范围 [0, 2047]
            
        Returns:
            float32 numpy ndarray, 24kHz PCM 波形
        """
        if codes.shape[0] == 0:
            return np.array([], dtype=np.float32)
        
        # 转换输入: [T, 16] -> [1, 16, T]
        codes_tensor = self._codes_to_tensor(codes)
        
        # 推理
        with torch.inference_mode():
            wav = self.decoder(codes_tensor)  # [1, 1, T*1920]
        
        # 转换输出: [1, 1, T*1920] -> [T*1920]
        pcm = wav.squeeze().cpu().numpy().astype(np.float32)
        
        return pcm
    
    def decode_window(self, codes: np.ndarray) -> np.ndarray:
        """
        解码指定窗口的 codes，用于滑动窗口流式场景。
        
        与 decode() 相同的单次推理逻辑，语义上表示"解码一个窗口"，
        供外部滑动窗口调度器调用。
        
        Args:
            codes: 窗口内的音频码, shape [T_window, 16], dtype int64
            
        Returns:
            float32 numpy ndarray, 对应窗口的 24kHz PCM 波形
        """
        # decode_window 与 decode 的底层逻辑一致，都是单次完整推理
        # 区别在于语义：decode 用于全量，decode_window 用于单窗口
        return self.decode(codes)
    
    def chunked_decode(self, codes: np.ndarray, chunk_size: int = 300, left_context: int = 25) -> np.ndarray:
        """
        分块解码（流式滑动窗口）。
        
        将长序列分成多个 chunk，每个 chunk 带 left_context 帧的左上下文，
        解码后裁掉重叠部分，拼接得到完整音频。
        利用 decoder 内置的 chunked_decode 方法。
        
        Args:
            codes: 音频码, shape [T, 16], dtype int64
            chunk_size: 每个分块的帧数，默认 300
            left_context: 左上下文帧数，用于平滑分块边界，默认 25
            
        Returns:
            float32 numpy ndarray, 24kHz PCM 波形
        """
        if codes.shape[0] == 0:
            return np.array([], dtype=np.float32)
        
        # 转换输入: [T, 16] -> [1, 16, T]
        codes_tensor = self._codes_to_tensor(codes)
        
        # 推理
        with torch.inference_mode():
            wav = self.decoder.chunked_decode(
                codes_tensor,
                chunk_size=chunk_size,
                left_context_size=left_context,
            )  # [1, 1, T*1920]
        
        # 转换输出: [1, 1, T*1920] -> [T*1920]
        pcm = wav.squeeze().cpu().numpy().astype(np.float32)
        
        return pcm
    
    def warmup(self):
        """
        用零张量做一次推理预热。
        
        建议在初始化后、正式推理前调用，确保 CUDA kernel 编译完成，
        避免首次推理的延迟尖峰。
        """
        logger.info("[PyTorchDecoder] 开始预热...")
        
        # 构造零张量: [1, 16, 12] - 12 帧的零码
        dummy_codes = torch.zeros((1, self.NUM_QUANTIZERS, 12), dtype=torch.int64, device=self.device)
        
        with torch.inference_mode():
            _ = self.decoder(dummy_codes)
        
        # 同步确保 kernel 编译完成
        if self.device == "cuda":
            torch.cuda.synchronize()
        
        logger.info("✅ [PyTorchDecoder] 预热完成")
