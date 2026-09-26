#!/usr/bin/env python3
"""
extract_codec_decoder.py - 从 Qwen3-TTS 模型中提取 Codec Decoder 的 PyTorch 权重

功能说明：
    从 HuggingFace 模型（在线或本地）中提取 Qwen3-TTS codec decoder 的权重，
    并保存为独立的 PyTorch 文件 (codec_decoder.pt)，以便后续直接加载到
    Qwen3TTSTokenizerV2Decoder 实例中使用。

使用方式：
    # 方式1：从本地模型目录提取（默认路径）
    python extract_codec_decoder.py

    # 方式2：指定本地模型目录
    python extract_codec_decoder.py --model-dir /path/to/local/model

    # 方式3：从 HuggingFace 在线下载
    python extract_codec_decoder.py --hf-model-id Qwen/Qwen3-TTS-12Hz-1.7B-Base

    # 方式4：同时指定两者（优先使用本地目录）
    python extract_codec_decoder.py --model-dir /data/models/Qwen3-TTS-12Hz-1.7B-Hutao-GGUF --hf-model-id Qwen/Qwen3-TTS-12Hz-1.7B-Base

输入源优先级：
    1. 本地模型目录中的 speech_tokenizer/model.safetensors（如果存在）
    2. HuggingFace 在线加载（如果本地不存在）

输出文件：
    - codec_decoder.pt          : Decoder 权重（去掉 "model.decoder." 前缀后的 state_dict）
    - codec_decoder_config.json : Decoder 配置（从原始 config.json 中提取的 decoder_config 部分）

权重提取逻辑：
    - 使用 safetensors 库读取 model.safetensors
    - 过滤键前缀为 "model.decoder." 的权重
    - 去掉 "model.decoder." 前缀后保存，使权重键与 Qwen3TTSTokenizerV2Decoder 的
      state_dict 键名一致，可直接通过 load_state_dict 加载

备选方案：
    - 如果 safetensors 库不可用，则尝试使用 qwen_tts 包的
      Qwen3TTSTokenizerV2Decoder 来加载模型并提取权重
"""

import os
import sys
import json
import argparse


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="从 Qwen3-TTS 模型中提取 Codec Decoder 权重",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 从本地模型目录提取
  python extract_codec_decoder.py --model-dir /data/models/Qwen3-TTS-12Hz-1.7B-Hutao-GGUF

  # 从 HuggingFace 在线下载
  python extract_codec_decoder.py --hf-model-id Qwen/Qwen3-TTS-12Hz-1.7B-Base
        """,
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="/data/models/Qwen3-TTS-12Hz-1.7B-Hutao-GGUF",
        help="本地模型目录路径（默认: /data/models/Qwen3-TTS-12Hz-1.7B-Hutao-GGUF）",
    )
    parser.add_argument(
        "--hf-model-id",
        type=str,
        default="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        help="HuggingFace 模型 ID，用于在线下载（默认: Qwen/Qwen3-TTS-12Hz-1.7B-Base）",
    )
    return parser.parse_args()


def count_parameters(state_dict):
    """
    计算 state_dict 中的总参数量

    Args:
        state_dict: PyTorch state_dict（字典，值为张量）

    Returns:
        int: 总参数数量
    """
    total = 0
    for tensor in state_dict.values():
        # 兼容 numpy 数组和 PyTorch 张量
        if hasattr(tensor, "numel"):
            total += tensor.numel()
        else:
            total += tensor.size
    return total


def extract_from_safetensors(safetensors_path, config_path, output_dir):
    """
    使用 safetensors 库从本地 model.safetensors 提取 decoder 权重

    这是首选方案，直接读取 safetensors 文件，过滤 "model.decoder." 前缀的权重键，
    去掉前缀后保存为 codec_decoder.pt。

    Args:
        safetensors_path: model.safetensors 文件的完整路径
        config_path: config.json 文件的完整路径
        output_dir: 输出目录路径

    Returns:
        bool: 提取是否成功
    """
    import torch
    from safetensors import safe_open

    # --- 1. 读取并解析配置文件 ---
    print(f"📄 正在读取配置文件: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        full_config = json.load(f)

    # 从完整配置中提取 decoder_config 部分
    # 原始 config.json 结构包含 encoder_config 和 decoder_config 两个子配置
    decoder_config = full_config.get("decoder_config", None)
    if decoder_config is None:
        print("❌ 错误: 配置文件中未找到 'decoder_config' 字段！")
        return False

    # --- 2. 从 safetensors 中提取 decoder 权重 ---
    print(f"📦 正在从 safetensors 文件提取 decoder 权重: {safetensors_path}")

    # 定义需要过滤的前缀 —— decoder 权重在 safetensors 中的键均以 "model.decoder." 开头
    DECODER_PREFIX = "model.decoder."

    decoder_weights = {}
    total_keys_in_file = 0

    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        all_keys = list(f.keys())
        total_keys_in_file = len(all_keys)
        print(f"   safetensors 文件中的总权重键数: {total_keys_in_file}")

        for key in all_keys:
            if key.startswith(DECODER_PREFIX):
                # 去掉 "model.decoder." 前缀
                # 例如: "model.decoder.quantizer.rvq_first.vq.layers.0._codebook.embedding_sum"
                #   ->  "quantizer.rvq_first.vq.layers.0._codebook.embedding_sum"
                new_key = key[len(DECODER_PREFIX):]
                tensor = f.get_tensor(key)
                decoder_weights[new_key] = tensor

    if not decoder_weights:
        print(f"❌ 错误: 未在 safetensors 中找到前缀为 '{DECODER_PREFIX}' 的权重键！")
        print(f"   文件中存在的键前缀示例:")
        # 打印前10个键帮助调试
        with safe_open(safetensors_path, framework="pt", device="cpu") as f:
            sample_keys = list(f.keys())[:10]
            for k in sample_keys:
                print(f"     {k}")
        return False

    # --- 3. 保存 decoder 权重为 codec_decoder.pt ---
    os.makedirs(output_dir, exist_ok=True)
    weights_path = os.path.join(output_dir, "codec_decoder.pt")

    # 使用 torch.save 保存为 PyTorch 格式
    # 保存后的权重键已去掉前缀，可以直接加载到 Qwen3TTSTokenizerV2Decoder 实例
    torch.save(decoder_weights, weights_path)
    print(f"✅ Decoder 权重已保存至: {weights_path}")

    # --- 4. 保存 decoder 配置为 codec_decoder_config.json ---
    config_output_path = os.path.join(output_dir, "codec_decoder_config.json")
    with open(config_output_path, "w", encoding="utf-8") as f:
        json.dump(decoder_config, f, indent=2, ensure_ascii=False)
    print(f"✅ Decoder 配置已保存至: {config_output_path}")

    # --- 5. 打印统计信息 ---
    num_keys = len(decoder_weights)
    total_params = count_parameters(decoder_weights)
    file_size_mb = os.path.getsize(weights_path) / (1024 * 1024)

    print(f"\n📊 提取统计:")
    print(f"   提取的权重键数量: {num_keys}")
    print(f"   总参数量: {total_params:,} ({total_params / 1e6:.2f}M)")
    print(f"   输出文件大小: {file_size_mb:.2f} MB")
    print(f"   原始文件总键数: {total_keys_in_file}")

    # 打印部分键名示例，帮助验证提取结果
    print(f"\n🔑 提取的权重键示例（前10个）:")
    for i, key in enumerate(sorted(decoder_weights.keys())[:10]):
        shape = decoder_weights[key].shape
        print(f"   {key}: {list(shape)}")
    if num_keys > 10:
        print(f"   ... 以及其他 {num_keys - 10} 个键")

    return True


def extract_from_hf_online(hf_model_id, output_dir):
    """
    从 HuggingFace 在线加载模型并提取 decoder 权重

    当本地模型目录不存在时使用此方案。通过 HuggingFace transformers 库
    下载模型，然后从模型实例中提取 decoder 的 state_dict。

    Args:
        hf_model_id: HuggingFace 模型 ID（如 "Qwen/Qwen3-TTS-12Hz-1.7B-Base"）
        output_dir: 输出目录路径

    Returns:
        bool: 提取是否成功
    """
    import torch

    print(f"🌐 正在从 HuggingFace 在线加载模型: {hf_model_id}")
    print("   这可能需要下载数 GB 的模型文件，请耐心等待...")

    try:
        # 尝试使用 qwen_tts 包加载（更轻量，只加载 speech_tokenizer 部分）
        from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
            Qwen3TTSTokenizerV2Model,
        )

        print("   使用 qwen_tts 包加载模型...")
        model = Qwen3TTSTokenizerV2Model.from_pretrained(hf_model_id)
        decoder = model.decoder

    except ImportError:
        # 如果 qwen_tts 不可用，尝试使用项目自带的模型定义
        print("   qwen_tts 包不可用，尝试使用项目自带的模型定义...")

        # 将项目路径加入 sys.path，以便导入项目中的模型定义
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        tokenizer_path = os.path.join(
            project_root, "qwen3_tts_gguf", "export", "tokenizer_12hz"
        )
        if os.path.exists(tokenizer_path):
            sys.path.insert(0, tokenizer_path)
            sys.path.insert(0, os.path.join(project_root, "qwen3_tts_gguf", "export"))

        try:
            from modeling_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Model

            print("   使用项目模型定义加载...")
            model = Qwen3TTSTokenizerV2Model.from_pretrained(hf_model_id)
            decoder = model.decoder
        except ImportError as e:
            print(f"❌ 错误: 无法导入模型定义: {e}")
            print("   请确保 qwen_tts 包已安装，或在项目根目录下运行此脚本")
            return False

    # 提取 decoder 的 state_dict
    decoder_state_dict = decoder.state_dict()

    os.makedirs(output_dir, exist_ok=True)

    # --- 保存权重 ---
    weights_path = os.path.join(output_dir, "codec_decoder.pt")
    torch.save(decoder_state_dict, weights_path)
    print(f"✅ Decoder 权重已保存至: {weights_path}")

    # --- 保存配置 ---
    # 从模型实例获取配置并转为字典
    decoder_config = decoder.config.to_dict()
    config_output_path = os.path.join(output_dir, "codec_decoder_config.json")
    with open(config_output_path, "w", encoding="utf-8") as f:
        json.dump(decoder_config, f, indent=2, ensure_ascii=False)
    print(f"✅ Decoder 配置已保存至: {config_output_path}")

    # --- 打印统计信息 ---
    num_keys = len(decoder_state_dict)
    total_params = count_parameters(decoder_state_dict)
    file_size_mb = os.path.getsize(weights_path) / (1024 * 1024)

    print(f"\n📊 提取统计:")
    print(f"   提取的权重键数量: {num_keys}")
    print(f"   总参数量: {total_params:,} ({total_params / 1e6:.2f}M)")
    print(f"   输出文件大小: {file_size_mb:.2f} MB")

    return True


def extract_from_qwen_tts_package(model_dir, output_dir):
    """
    备选方案：使用 qwen_tts 包的 Qwen3TTSTokenizerV2Decoder 加载模型

    当 safetensors 库不可用时，使用此方案。通过 qwen_tts 包加载完整的
    Qwen3TTSTokenizerV2Model，然后提取其中的 decoder 部分。

    Args:
        model_dir: 本地模型目录路径
        output_dir: 输出目录路径

    Returns:
        bool: 提取是否成功
    """
    import torch

    print("🔄 正在尝试备选方案: 使用 qwen_tts 包加载模型...")

    # 确定 speech_tokenizer 子目录路径
    speech_tokenizer_path = os.path.join(model_dir, "speech_tokenizer")
    if os.path.exists(speech_tokenizer_path):
        load_path = speech_tokenizer_path
    else:
        load_path = model_dir

    print(f"   模型加载路径: {load_path}")

    try:
        # 尝试从 qwen_tts 包导入
        from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
            Qwen3TTSTokenizerV2Model,
        )

        print("   使用 qwen_tts 包加载...")
    except ImportError:
        # 尝试从项目本地路径导入
        print("   qwen_tts 包不可用，尝试使用项目自带的模型定义...")

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        tokenizer_path = os.path.join(
            project_root, "qwen3_tts_gguf", "export", "tokenizer_12hz"
        )
        if os.path.exists(tokenizer_path):
            sys.path.insert(0, tokenizer_path)
            sys.path.insert(0, os.path.join(project_root, "qwen3_tts_gguf", "export"))

        try:
            from modeling_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Model

            print("   使用项目模型定义加载...")
        except ImportError as e:
            print(f"❌ 错误: 无法导入模型定义: {e}")
            print("   请确保 qwen_tts 包已安装，或 safetensors 库可用")
            return False

    # 加载完整模型，然后提取 decoder
    print("   正在加载完整模型（可能需要较长时间）...")
    model = Qwen3TTSTokenizerV2Model.from_pretrained(load_path)
    decoder = model.decoder

    # 提取 decoder 的 state_dict
    decoder_state_dict = decoder.state_dict()

    os.makedirs(output_dir, exist_ok=True)

    # --- 保存权重 ---
    weights_path = os.path.join(output_dir, "codec_decoder.pt")
    torch.save(decoder_state_dict, weights_path)
    print(f"✅ Decoder 权重已保存至: {weights_path}")

    # --- 保存配置 ---
    decoder_config = decoder.config.to_dict()
    config_output_path = os.path.join(output_dir, "codec_decoder_config.json")
    with open(config_output_path, "w", encoding="utf-8") as f:
        json.dump(decoder_config, f, indent=2, ensure_ascii=False)
    print(f"✅ Decoder 配置已保存至: {config_output_path}")

    # --- 打印统计信息 ---
    num_keys = len(decoder_state_dict)
    total_params = count_parameters(decoder_state_dict)
    file_size_mb = os.path.getsize(weights_path) / (1024 * 1024)

    print(f"\n📊 提取统计:")
    print(f"   提取的权重键数量: {num_keys}")
    print(f"   总参数量: {total_params:,} ({total_params / 1e6:.2f}M)")
    print(f"   输出文件大小: {file_size_mb:.2f} MB")

    return True


def main():
    """主函数：根据输入源选择合适的提取方式"""
    args = parse_args()

    print("=" * 60)
    print("Qwen3-TTS Codec Decoder 权重提取工具")
    print("=" * 60)

    # 本地模型目录中 speech_tokenizer 子目录的路径
    speech_tokenizer_dir = os.path.join(args.model_dir, "speech_tokenizer")

    # safetensors 文件路径（权重文件）
    safetensors_path = os.path.join(speech_tokenizer_dir, "model.safetensors")

    # config.json 文件路径（配置文件）
    config_path = os.path.join(speech_tokenizer_dir, "config.json")

    # 输出目录：保存到模型目录下
    output_dir = args.model_dir

    # --- 策略1: 优先从本地模型目录的 safetensors 提取 ---
    if os.path.exists(safetensors_path) and os.path.exists(config_path):
        print(f"\n📁 检测到本地模型目录: {args.model_dir}")
        print(f"   safetensors 文件: {safetensors_path}")
        print(f"   配置文件: {config_path}")

        try:
            success = extract_from_safetensors(safetensors_path, config_path, output_dir)
            if success:
                print("\n🎉 提取完成！")
                return
        except ImportError:
            print("⚠️  safetensors 库不可用，尝试备选方案...")
        except Exception as e:
            print(f"⚠️  safetensors 提取失败: {e}")
            print("   尝试备选方案...")

        # safetensors 方式失败，尝试 qwen_tts 包
        try:
            success = extract_from_qwen_tts_package(args.model_dir, output_dir)
            if success:
                print("\n🎉 提取完成！")
                return
        except Exception as e:
            print(f"❌ 备选方案也失败了: {e}")

    # --- 策略2: 本地目录不存在，从 HuggingFace 在线加载 ---
    elif args.hf_model_id:
        print(f"\n🌐 本地模型目录不存在或不完整: {args.model_dir}")
        print(f"   尝试从 HuggingFace 在线加载: {args.hf_model_id}")

        try:
            success = extract_from_hf_online(args.hf_model_id, output_dir)
            if success:
                print("\n🎉 提取完成！")
                return
        except Exception as e:
            print(f"❌ 在线加载失败: {e}")

    else:
        print(f"❌ 错误: 本地模型目录不存在，且未指定 HuggingFace 模型 ID")
        print(f"   请使用 --model-dir 指定本地路径，或使用 --hf-model-id 指定在线模型")

    print("\n❌ 提取失败，请检查上述错误信息")
    sys.exit(1)


if __name__ == "__main__":
    main()
