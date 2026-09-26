"""Manifest parsing for N.E.K.O mini game plugins."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping
import json
import re

try:
    import tomllib
except ImportError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]


SUPPORTED_LOCALES: tuple[str, ...] = ("en", "ja", "ko", "zh-CN", "zh-TW", "ru", "pt", "es")
ALLOWED_ORIENTATIONS = {"any", "portrait", "landscape"}
ALLOWED_INPUT_METHODS = {"keyboard", "mouse", "touch", "voice", "gamepad"}
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{1,62}[a-z0-9]$")


class ManifestError(ValueError):
    """Raised when a mini game manifest is malformed."""


def normalize_capabilities(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def is_minigame_plugin(metadata: Mapping[str, Any] | None) -> bool:
    if not isinstance(metadata, Mapping):
        return False
    plugin_type = str(metadata.get("type") or "").strip().lower()
    capabilities = {item.lower() for item in normalize_capabilities(metadata.get("capabilities"))}
    return plugin_type == "mini_game" or "mini_game" in capabilities


@dataclass(frozen=True, slots=True)
class MiniGameManifest:
    game_id: str
    title: dict[str, str]
    entry: str
    assets_dir: str = "static"
    orientation: str = "any"
    input_methods: tuple[str, ...] = ("keyboard", "mouse", "touch")
    character_voice: bool = True
    leaderboard: bool = False
    memory_archive: bool = False
    route_game_type: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MiniGameManifest":
        if not isinstance(data, Mapping):
            raise ManifestError("manifest must be a mapping")
        game_id = _required_id(data.get("game_id"), "game_id")
        title = _normalize_title(data.get("title"), game_id)
        entry = _required_path(data.get("entry"), "entry")
        assets_dir = _required_path(data.get("assets_dir") or "static", "assets_dir")
        orientation = str(data.get("orientation") or "any").strip().lower()
        if orientation not in ALLOWED_ORIENTATIONS:
            raise ManifestError(f"orientation must be one of {sorted(ALLOWED_ORIENTATIONS)}")
        input_methods = _normalize_input_methods(data.get("input_methods"))
        route_game_type = str(data.get("route_game_type") or game_id).strip()
        if not route_game_type:
            raise ManifestError("route_game_type cannot be empty")
        tags = tuple(str(item).strip() for item in data.get("tags") or [] if str(item).strip())
        return cls(
            game_id=game_id,
            title=title,
            entry=entry,
            assets_dir=assets_dir,
            orientation=orientation,
            input_methods=input_methods,
            character_voice=bool(data.get("character_voice", True)),
            leaderboard=bool(data.get("leaderboard", False)),
            memory_archive=bool(data.get("memory_archive", False)),
            route_game_type=route_game_type,
            tags=tags,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "title": dict(self.title),
            "entry": self.entry,
            "assets_dir": self.assets_dir,
            "orientation": self.orientation,
            "input_methods": list(self.input_methods),
            "character_voice": self.character_voice,
            "leaderboard": self.leaderboard,
            "memory_archive": self.memory_archive,
            "route_game_type": self.route_game_type,
            "tags": list(self.tags),
        }

    def localized_title(self, locale: str = "en") -> str:
        return self.title.get(locale) or self.title.get("en") or next(iter(self.title.values()), self.game_id)

    def resolve_entry_path(self, plugin_dir: Path) -> Path:
        return plugin_dir / self.entry


def load_minigame_manifest(plugin_dir: str | Path, *, manifest_name: str = "minigame.json") -> MiniGameManifest:
    plugin_path = Path(plugin_dir)
    manifest_path = plugin_path / manifest_name
    if manifest_path.exists():
        return MiniGameManifest.from_dict(json.loads(manifest_path.read_text(encoding="utf-8")))
    toml_path = plugin_path / "plugin.toml"
    if not toml_path.exists():
        raise ManifestError(f"missing {manifest_name} or plugin.toml in {plugin_path}")
    with toml_path.open("rb") as stream:
        raw = tomllib.load(stream)
    plugin = raw.get("plugin") if isinstance(raw, Mapping) else None
    if not isinstance(plugin, Mapping):
        raise ManifestError("plugin.toml must contain [plugin]")
    mini_game = plugin.get("mini_game")
    if not isinstance(mini_game, Mapping):
        raise ManifestError("plugin.toml must contain [plugin.mini_game] or minigame.json")
    return MiniGameManifest.from_dict(mini_game)


def _required_id(value: Any, field_name: str) -> str:
    text = str(value or "").strip()
    if not _ID_RE.match(text):
        raise ManifestError(f"{field_name} must be 3-64 chars using lowercase letters, digits, '_' or '-'")
    return text


def _required_path(value: Any, field_name: str) -> str:
    text = str(value or "").replace("\\", "/").strip().strip("/")
    if not text or text.startswith("../") or "/../" in text:
        raise ManifestError(f"{field_name} must be a safe relative path")
    return text


def _normalize_title(value: Any, fallback: str) -> dict[str, str]:
    if isinstance(value, str) and value.strip():
        return {"en": value.strip()}
    if isinstance(value, Mapping):
        title = {str(k): str(v).strip() for k, v in value.items() if str(v).strip()}
        if title:
            return title
    return {"en": fallback}


def _normalize_input_methods(value: Any) -> tuple[str, ...]:
    methods = normalize_capabilities(value) or ["keyboard", "mouse", "touch"]
    normalized = tuple(dict.fromkeys(item.lower() for item in methods))
    unknown = sorted(set(normalized) - ALLOWED_INPUT_METHODS)
    if unknown:
        raise ManifestError(f"unsupported input method(s): {', '.join(unknown)}")
    return normalized
