"""Small theater script DSL validator for creator templates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import json

import yaml


class TheaterValidationError(ValueError):
    """Raised when a theater script cannot be used by the V1 runtime."""


@dataclass(frozen=True, slots=True)
class TheaterScript:
    title: str
    characters: tuple[dict[str, Any], ...]
    scenes: tuple[dict[str, Any], ...]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TheaterScript":
        title = str(data.get("title") or "").strip()
        if not title:
            raise TheaterValidationError("theater script requires title")
        characters = _required_list(data.get("characters"), "characters")
        scenes = _required_list(data.get("scenes"), "scenes")
        character_ids = {str(item.get("id") or "").strip() for item in characters}
        if "" in character_ids:
            raise TheaterValidationError("every character requires id")
        for scene in scenes:
            _validate_scene(scene, character_ids)
        return cls(title=title, characters=tuple(characters), scenes=tuple(scenes))

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "characters": [dict(item) for item in self.characters],
            "scenes": [dict(item) for item in self.scenes],
        }


def load_theater_script(path: str | Path) -> TheaterScript:
    script_path = Path(path)
    text = script_path.read_text(encoding="utf-8")
    if script_path.suffix.lower() in {".yaml", ".yml"}:
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, Mapping):
        raise TheaterValidationError("script root must be an object")
    return TheaterScript.from_dict(data)


def _required_list(value: Any, field_name: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise TheaterValidationError(f"{field_name} must be a non-empty list")
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise TheaterValidationError(f"{field_name}[{index}] must be an object")
        normalized.append(dict(item))
    return normalized


def _validate_scene(scene: Mapping[str, Any], character_ids: set[str]) -> None:
    scene_id = str(scene.get("id") or "").strip()
    if not scene_id:
        raise TheaterValidationError("every scene requires id")
    beats = scene.get("beats")
    if not isinstance(beats, list) or not beats:
        raise TheaterValidationError(f"scene {scene_id} requires beats")
    for index, beat in enumerate(beats):
        if not isinstance(beat, Mapping):
            raise TheaterValidationError(f"scene {scene_id} beat {index} must be an object")
        actor = str(beat.get("character") or beat.get("actor") or "").strip()
        if actor and actor not in character_ids:
            raise TheaterValidationError(f"scene {scene_id} beat {index} references unknown character {actor}")
        if not any(key in beat for key in ("dialogue", "actions", "choice", "choices", "condition", "camera")):
            raise TheaterValidationError(f"scene {scene_id} beat {index} has no playable content")
