"""
constants.py - Qwen3-TTS 常量定义
包含说话人映射、语言映射和官方协议标签。

SPEAKER_MAP / LANGUAGE_MAP / MODEL_VOICE_PROFILES 支持从外部配置覆盖。
调用 apply_voice_config() / apply_language_config() 可在引擎初始化前注入配置。

设计原则：
- SPEAKER_MAP 硬编码了 9 个标准 CustomVoice 音色，确保标准模型用户零配置可用。
- 微调模型用户通过 config.yaml 的 voices.map 合并/覆盖特定条目，不会丢失标准音色。
- engine._detect_model_type() 会自动过滤出当前模型实际存在的音色（L2>1.0）。
"""

# ── 说话人 ID 映射 (Verified from official config.json) ──
# 9 个标准 CustomVoice 音色，确保标准模型开箱即用
SPEAKER_MAP = {
    "vivian": 3065,
    "serena": 3066,
    "uncle_fu": 3010,
    "ryan": 3061,
    "aiden": 2861,
    "ono_anna": 2873,
    "sohee": 2864,
    "eric": 2875,
    "dylan": 2878,
    # 微调模型专属音色（作为默认值，config.yaml 可覆盖/补充）
    "hutao": 3000,
    "yoimiya": 2900,
    "cyrene": 2950,
}

# ── 模型音色配置 ──
MODEL_VOICE_PROFILES = {
    "custom": {
        "voices": ["Vivian", "Serena", "Uncle_Fu", "Ryan", "Aiden",
                    "Ono_Anna", "Sohee", "Eric", "Dylan"],
        "default": "Vivian",
    },
    "hutao": {
        "voices": ["Hutao", "Yoimiya", "Cyrene"],
        "default": "Hutao",
    },
}

# ── 语言 ID 映射 ──
LANGUAGE_MAP = {
    "english": 2050,
    "german": 2053,
    "spanish": 2054,
    "chinese": 2055,
    "japanese": 2058,
    "french": 2061,
    "sichuan_dialect": 2062,
    "korean": 2064,
    "russian": 2069,
    "italian": 2070,
    "portuguese": 2071,
    "beijing_dialect": 2074,
}

# ── 官方流程协议标签 ──
PROTOCOL = {
    "PAD": 2148,
    "BOS": 2149,
    "EOS": 2150,
    "TTS_BOS": 151672,
    "TTS_EOS": 151673,
    "THINK": 2154,
    "NOTHINK": 2155,
    "THINK_BOS": 2156,
    "THINK_EOS": 2157
}

# ── 默认分步码表数量 ──
NUM_QUANTIZERS = 16

# ── 采样率 ──
SAMPLE_RATE = 24000

from . import logger


def apply_voice_config(speaker_map: dict = None, voice_profiles: dict = None):
    """从外部配置（config.yaml）合并到 SPEAKER_MAP 和 MODEL_VOICE_PROFILES

    策略：合并而非替换。
    SPEAKER_MAP 中硬编码了 9 个标准 CustomVoice 音色（vivian/serena/...），
    任何配置只会覆盖/新增条目，不会删除标准音色。
    微调模型用户只需在 config.yaml 中配置自己特有的音色即可。
    标准模型用户无需任何配置，开箱即用。
    """
    global SPEAKER_MAP, MODEL_VOICE_PROFILES

    if speaker_map is not None:
        # 合并到硬编码默认值，不清除
        normalized = {}
        for k, v in speaker_map.items():
            normalized[str(k).lower()] = int(v)
        SPEAKER_MAP.update(normalized)
        logger.info(f"[Config] 已合并说话人映射: {len(normalized)} 个来自配置, 共 {len(SPEAKER_MAP)} 个")

    if voice_profiles:
        MODEL_VOICE_PROFILES.update(voice_profiles)
        logger.info(f"[Config] 已应用音色配置档案: {list(MODEL_VOICE_PROFILES.keys())}")


def apply_language_config(language_map: dict = None):
    """从外部配置（config.yaml）覆盖 LANGUAGE_MAP（合并）"""
    global LANGUAGE_MAP

    if language_map is not None:
        normalized = {}
        for k, v in language_map.items():
            normalized[str(k).lower()] = int(v)
        LANGUAGE_MAP.update(normalized)
        logger.info(f"[Config] 已合并语言映射: {len(normalized)} 种来自配置, 共 {len(LANGUAGE_MAP)} 种")


def map_speaker(spk) -> int:
    """将说话人名称或 ID 映射为官方数值 ID (2800-3071)"""
    if isinstance(spk, int):
        if 2800 <= spk <= 3071:
            return spk
        logger.warning(f"⚠️ 非法的 Speaker ID: {spk}, 必须在 2800-3071 之间。")
        return None

    return SPEAKER_MAP.get(str(spk).lower(), None)


def map_language(lang) -> int:
    """将语言名称或 ID 映射为官方数值 ID (2048-2147)"""
    if isinstance(lang, int):
        if 2048 <= lang <= 2147:
            return lang
        logger.warning(f"⚠️ 非法的 Language ID: {lang}, 必须在 2048-2147 之间。回退到 Chinese (2055)")
        return None

    return LANGUAGE_MAP.get(str(lang).lower(), None)
