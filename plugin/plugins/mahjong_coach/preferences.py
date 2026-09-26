from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

from .models import CapturePreferences, PlayerProfile, WindowTargetDescriptor


class PreferencesStore:
    def __init__(self, path: Path) -> None:
        self.path = path.expanduser().resolve()

    def load(self) -> CapturePreferences:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        return CapturePreferences.from_payload(payload if isinstance(payload, dict) else {})

    def save(self, preferences: CapturePreferences) -> CapturePreferences:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(preferences.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, self.path)
        return preferences

    def update(
        self,
        *,
        auto_start_live: bool | None = None,
        target: WindowTargetDescriptor | None = None,
        clear_target: bool = False,
        profile: PlayerProfile | None = None,
    ) -> CapturePreferences:
        current = self.load()
        updated = replace(
            current,
            auto_start_live=(
                current.auto_start_live if auto_start_live is None else bool(auto_start_live)
            ),
            target=(
                WindowTargetDescriptor()
                if clear_target
                else current.target if target is None else target
            ),
            profile=current.profile if profile is None else profile,
        )
        return self.save(updated)


def profile_from_entry_payload(payload: dict[str, Any] | None, *, legacy_style: str = "") -> PlayerProfile:
    return PlayerProfile.from_payload(payload, legacy_play_style=legacy_style)
