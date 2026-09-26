"""Runtime helper classes for mini game plugins."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import Any, Mapping

from .manifest import MiniGameManifest, load_minigame_manifest


def minigame_ok(**data: Any) -> dict[str, Any]:
    return {"ok": True, **data}


def minigame_error(reason: str, **data: Any) -> dict[str, Any]:
    return {"ok": False, "reason": str(reason or "unknown_error"), **data}


@dataclass(slots=True)
class MiniGameSession:
    session_id: str
    lanlan_name: str = ""
    game_id: str = ""
    started_at: float = field(default_factory=time)
    last_seen_at: float = field(default_factory=time)
    score: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def heartbeat(self) -> None:
        self.last_seen_at = time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "lanlan_name": self.lanlan_name,
            "game_id": self.game_id,
            "started_at": self.started_at,
            "last_seen_at": self.last_seen_at,
            "score": self.score,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class MiniGameScore:
    player_id: str
    session_id: str
    score: int
    mode: str = "default"
    metadata: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "player_id": self.player_id,
            "session_id": self.session_id,
            "score": self.score,
            "mode": self.mode,
            "metadata": dict(self.metadata or {}),
        }


class MiniGameScoreBoard:
    """Small in-memory leaderboard helper for plugin templates and tests."""

    def __init__(self, *, limit: int = 100) -> None:
        self.limit = max(1, int(limit))
        self._scores: list[MiniGameScore] = []

    def submit(self, score: MiniGameScore) -> dict[str, Any]:
        self._scores.append(score)
        self._scores.sort(key=lambda item: (-item.score, item.player_id, item.session_id))
        del self._scores[self.limit :]
        rank = next((idx + 1 for idx, item in enumerate(self._scores) if item is score), None)
        return minigame_ok(rank=rank, total_scores=len(self._scores), score=score.to_dict())

    def top(self, *, limit: int = 10, offset: int = 0, mode: str | None = None) -> dict[str, Any]:
        scores = [item for item in self._scores if mode is None or item.mode == mode]
        start = max(0, int(offset))
        end = start + max(1, int(limit))
        return minigame_ok(
            top=[item.to_dict() for item in scores[start:end]],
            total_scores=len(scores),
            limit=max(1, int(limit)),
            offset=start,
            has_more=end < len(scores),
        )

    def clear_session(self, session_id: str) -> int:
        before = len(self._scores)
        self._scores = [item for item in self._scores if item.session_id != session_id]
        return before - len(self._scores)


class MiniGameRuntimeHelper:
    def __init__(self, manifest: MiniGameManifest, *, plugin_id: str = "") -> None:
        self.manifest = manifest
        self.plugin_id = plugin_id
        self.sessions: dict[str, MiniGameSession] = {}
        self.leaderboard = MiniGameScoreBoard()

    def start_session(self, session_id: str, *, lanlan_name: str = "", metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
        session_id = str(session_id or "").strip()
        if not session_id:
            return minigame_error("missing_session_id")
        session = MiniGameSession(
            session_id=session_id,
            lanlan_name=str(lanlan_name or ""),
            game_id=self.manifest.game_id,
            metadata=dict(metadata or {}),
        )
        self.sessions[session_id] = session
        return minigame_ok(session=session.to_dict(), manifest=self.manifest.to_dict())

    def heartbeat(self, session_id: str) -> dict[str, Any]:
        session = self.sessions.get(str(session_id or "").strip())
        if session is None:
            return minigame_error("unknown_session")
        session.heartbeat()
        return minigame_ok(session=session.to_dict())

    def end_session(self, session_id: str, *, reason: str = "game_end") -> dict[str, Any]:
        session = self.sessions.pop(str(session_id or "").strip(), None)
        if session is None:
            return minigame_error("unknown_session", reason_detail=reason)
        return minigame_ok(session=session.to_dict(), ended=True, end_reason=reason)

    def status(self) -> dict[str, Any]:
        return minigame_ok(
            manifest=self.manifest.to_dict(),
            sessions=[session.to_dict() for session in self.sessions.values()],
        )


class MiniGamePluginMixin:
    """Mixin for NekoPluginBase subclasses that ship a MiniGameSDK manifest."""

    _minigame_runtime: MiniGameRuntimeHelper | None = None

    def load_minigame(self, *, manifest_name: str = "minigame.json") -> MiniGameRuntimeHelper:
        plugin_dir = Path(getattr(self, "config_dir"))
        plugin_id = str(getattr(self, "plugin_id", ""))
        manifest = load_minigame_manifest(plugin_dir, manifest_name=manifest_name)
        self._minigame_runtime = MiniGameRuntimeHelper(manifest, plugin_id=plugin_id)
        return self._minigame_runtime

    @property
    def minigame(self) -> MiniGameRuntimeHelper:
        if self._minigame_runtime is None:
            self._minigame_runtime = self.load_minigame()
        return self._minigame_runtime

    def register_minigame_static_ui(self) -> bool:
        manifest = self.minigame.manifest
        register = getattr(self, "register_static_ui")
        return bool(register(manifest.assets_dir, index_file=Path(manifest.entry).name))
