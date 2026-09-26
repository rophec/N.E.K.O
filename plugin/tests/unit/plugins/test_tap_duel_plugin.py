from __future__ import annotations

from pathlib import Path

import pytest

from plugin.plugins.tap_duel import TapDuelPlugin


class _Ctx:
    plugin_id = "tap_duel"
    logger = None
    bus = None

    def __init__(self, config_path: Path) -> None:
        self.config_path = config_path
        self.metadata = {"id": "tap_duel", "type": "plugin", "capabilities": ["mini_game"]}
        self._effective_config = {}
        self.pushed_messages = []
        self.finished = []

    def push_message(self, **kwargs):
        self.pushed_messages.append(kwargs)
        return {"ok": True}

    async def finish(self, **kwargs):
        self.finished.append(kwargs)
        return {"success": True, **kwargs}

    async def run_update_async(self, **kwargs):
        return {"ok": True, **kwargs}

    async def export_push_async(self, **kwargs):
        return {"ok": True, **kwargs}


@pytest.fixture()
def tap_duel_plugin() -> TapDuelPlugin:
    plugin_dir = Path(__file__).resolve().parents[3] / "plugins" / "tap_duel"
    return TapDuelPlugin(_Ctx(plugin_dir / "plugin.toml"))


def test_tap_duel_manifest_loads(tap_duel_plugin: TapDuelPlugin) -> None:
    manifest = tap_duel_plugin.minigame.manifest

    assert manifest.game_id == "tap_duel"
    assert manifest.leaderboard is True
    assert manifest.route_game_type == "tap_duel"


def test_tap_duel_static_entry_loads_sdk() -> None:
    plugin_dir = Path(__file__).resolve().parents[3] / "plugins" / "tap_duel"
    html = (plugin_dir / "static" / "index.html").read_text(encoding="utf-8")

    assert "http://127.0.0.1:48911/static/game/system/neko-minigame-sdk.js" in html
    assert 'apiBase,' in html
    assert 'id="game"' in html
    assert 'id="start"' in html
    assert 'id="end"' in html
    assert 'params.get("surface") !== "game"' in html
    assert 'window.open(buildLaunchUrl(), "tap_duel_game"' in html
    assert "new URL(location.href)" in html
    assert 'launchUrl.searchParams.set("surface", "game")' in html
    assert 'launchUrl.searchParams.set("game_type", "tap_duel")' in html
    assert 'gameType: "tap_duel"' in html
    assert 'game_id: "tap_duel"' in html


@pytest.mark.asyncio
async def test_tap_duel_entries_cover_session_and_scores(tap_duel_plugin: TapDuelPlugin) -> None:
    assert (await tap_duel_plugin.start_session("round-1", lanlan_name="Yui")).value["ok"] is True
    assert (await tap_duel_plugin.submit_score("Yui", "round-1", 99)).value["ok"] is True
    assert (await tap_duel_plugin.end_session("round-1")).value["ended"] is True
