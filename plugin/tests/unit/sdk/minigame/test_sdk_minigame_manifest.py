from __future__ import annotations

import json
from pathlib import Path

import pytest

from plugin.sdk.minigame import (
    ManifestError,
    MiniGameManifest,
    MiniGameRuntimeHelper,
    MiniGameScore,
    TheaterValidationError,
    is_minigame_plugin,
    list_templates,
    load_minigame_manifest,
    load_theater_script,
    write_template,
)


def test_manifest_from_dict_normalizes_defaults() -> None:
    manifest = MiniGameManifest.from_dict({
        "game_id": "tap_duel",
        "title": {"en": "Tap Duel", "zh-CN": "Tap Duel"},
        "entry": "static/index.html",
    })

    assert manifest.game_id == "tap_duel"
    assert manifest.route_game_type == "tap_duel"
    assert manifest.input_methods == ("keyboard", "mouse", "touch")
    assert manifest.localized_title("ja") == "Tap Duel"


def test_manifest_rejects_unsafe_values() -> None:
    with pytest.raises(ManifestError, match="game_id"):
        MiniGameManifest.from_dict({"game_id": "../bad", "entry": "static/index.html"})

    with pytest.raises(ManifestError, match="entry"):
        MiniGameManifest.from_dict({"game_id": "good_game", "entry": "../index.html"})

    with pytest.raises(ManifestError, match="unsupported input"):
        MiniGameManifest.from_dict({"game_id": "good_game", "entry": "static/index.html", "input_methods": ["brain"]})


def test_load_manifest_from_plugin_toml(tmp_path: Path) -> None:
    (tmp_path / "plugin.toml").write_text(
        "\n".join([
            "[plugin]",
            'id = "demo_game"',
            'type = "mini_game"',
            "",
            "[plugin.mini_game]",
            'game_id = "demo_game"',
            'title = "Demo Game"',
            'entry = "static/index.html"',
            'leaderboard = true',
        ]),
        encoding="utf-8",
    )

    manifest = load_minigame_manifest(tmp_path)

    assert manifest.game_id == "demo_game"
    assert manifest.leaderboard is True


def test_load_manifest_prefers_json_file(tmp_path: Path) -> None:
    (tmp_path / "minigame.json").write_text(
        json.dumps({"game_id": "json_game", "entry": "static/index.html"}),
        encoding="utf-8",
    )

    assert load_minigame_manifest(tmp_path).game_id == "json_game"


def test_is_minigame_plugin_accepts_type_or_capability() -> None:
    assert is_minigame_plugin({"type": "mini_game"}) is True
    assert is_minigame_plugin({"capabilities": ["mini_game"]}) is True
    assert is_minigame_plugin({"type": "tool"}) is False


def test_runtime_helper_session_and_leaderboard_flow() -> None:
    helper = MiniGameRuntimeHelper(MiniGameManifest.from_dict({"game_id": "score_game", "entry": "static/index.html"}))

    started = helper.start_session("s1", lanlan_name="Yui")
    assert started["ok"] is True
    assert helper.heartbeat("s1")["ok"] is True

    submitted = helper.leaderboard.submit(MiniGameScore(player_id="Yui", session_id="s1", score=42))
    assert submitted["rank"] == 1
    assert helper.leaderboard.top()["top"][0]["score"] == 42
    assert helper.end_session("s1")["ended"] is True


def test_theater_script_validation(tmp_path: Path) -> None:
    script_path = tmp_path / "script.json"
    script_path.write_text(
        json.dumps({
            "title": "Demo",
            "characters": [{"id": "yui", "name": "Yui"}],
            "scenes": [{"id": "opening", "beats": [{"character": "yui", "dialogue": "Hello"}]}],
        }),
        encoding="utf-8",
    )

    assert load_theater_script(script_path).title == "Demo"

    bad_path = tmp_path / "bad.json"
    bad_path.write_text(
        json.dumps({
            "title": "Bad",
            "characters": [{"id": "yui"}],
            "scenes": [{"id": "opening", "beats": [{"character": "missing", "dialogue": "Hello"}]}],
        }),
        encoding="utf-8",
    )
    with pytest.raises(TheaterValidationError, match="unknown character"):
        load_theater_script(bad_path)


def test_template_writer_creates_creator_ready_files(tmp_path: Path) -> None:
    output = write_template("score-game", tmp_path / "score_game", plugin_id="score_game")

    assert (output / "plugin.toml").exists()
    assert (output / "minigame.json").exists()
    assert (output / "static" / "index.html").exists()
    for locale in ("en", "ja", "ko", "zh-CN", "zh-TW", "ru", "pt", "es"):
        assert (output / "i18n" / f"{locale}.json").exists()
    assert any(item["id"] == "score-game" for item in list_templates())
