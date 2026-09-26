"""Load Mahjong Coach from its actual installation directory.

N.E.K.O releases that resolve an import conflict by appending ``_1`` or
another numeric suffix cannot import the canonical ``plugins.mahjong_coach``
entry. The plugin's vendor directory is added to ``sys.path`` before entry
loading, so this stable bootstrap can locate the package without relying on
the surrounding directory name.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys
from types import ModuleType


def _plugin_dir() -> Path:
    plugin_dir = Path(__file__).resolve().parents[2]
    if not (plugin_dir / "plugin.toml").is_file() or not (plugin_dir / "__init__.py").is_file():
        raise ImportError(f"Mahjong Coach runtime files are missing from {plugin_dir}")
    return plugin_dir


def _load_runtime() -> ModuleType:
    plugin_dir = _plugin_dir()
    identity = hashlib.sha256(str(plugin_dir).casefold().encode("utf-8")).hexdigest()[:16]
    module_name = f"_neko_mahjong_coach_runtime_{identity}"
    existing = sys.modules.get(module_name)
    if isinstance(existing, ModuleType):
        return existing

    spec = importlib.util.spec_from_file_location(
        module_name,
        plugin_dir / "__init__.py",
        submodule_search_locations=[str(plugin_dir)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create Mahjong Coach module spec for {plugin_dir}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


MahjongCoachPlugin = _load_runtime().MahjongCoachPlugin

__all__ = ["MahjongCoachPlugin"]
