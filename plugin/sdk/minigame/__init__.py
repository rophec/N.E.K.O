"""MiniGameSDK public helpers for N.E.K.O plugin creators."""

from __future__ import annotations

from .manifest import (
    MiniGameManifest,
    ManifestError,
    is_minigame_plugin,
    load_minigame_manifest,
    normalize_capabilities,
)
from .runtime import (
    MiniGamePluginMixin,
    MiniGameRuntimeHelper,
    MiniGameSession,
    MiniGameScore,
    MiniGameScoreBoard,
    minigame_error,
    minigame_ok,
)
from .theater import TheaterScript, TheaterValidationError, load_theater_script
from .templates import TEMPLATE_IDS, list_templates, write_template

__all__ = [
    "ManifestError",
    "MiniGameManifest",
    "MiniGamePluginMixin",
    "MiniGameRuntimeHelper",
    "MiniGameScore",
    "MiniGameScoreBoard",
    "MiniGameSession",
    "TheaterScript",
    "TheaterValidationError",
    "TEMPLATE_IDS",
    "is_minigame_plugin",
    "list_templates",
    "load_minigame_manifest",
    "load_theater_script",
    "minigame_error",
    "minigame_ok",
    "normalize_capabilities",
    "write_template",
]
