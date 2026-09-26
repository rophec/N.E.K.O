"""Deterministic mahjong strategy calculations.

确定性麻将策略计算。视觉模块负责提供牌局事实，本包只处理规则和牌效。
"""

from .efficiency import (
    DiscardEfficiency,
    EffectiveTile,
    HandEfficiencyAnalysis,
    ShantenResult,
    analyze_hand_efficiency,
    calculate_shanten,
)
from .risk import GenbutsuAssessment, HeldTileSafety, assess_genbutsu

__all__ = [
    "DiscardEfficiency",
    "EffectiveTile",
    "HandEfficiencyAnalysis",
    "ShantenResult",
    "analyze_hand_efficiency",
    "calculate_shanten",
    "GenbutsuAssessment",
    "HeldTileSafety",
    "assess_genbutsu",
]
