"""Creator template catalog for MiniGameSDK."""

from __future__ import annotations

from pathlib import Path
from string import Template
from typing import Any
import json

from .manifest import SUPPORTED_LOCALES


TEMPLATE_IDS = ("basic-canvas-game", "character-reaction-game", "score-game", "theater-scene")

_BASE_PLUGIN = Template(
    """[plugin]
id = "$plugin_id"
name = "$name"
description = "$description"
version = "0.1.0"
type = "plugin"
capabilities = ["mini_game"]
entry = "plugin.plugins.$plugin_id:${class_name}"

[plugin.sdk]
recommended = ">=0.1.0,<0.2.0"
supported = ">=0.1.0,<0.3.0"

[plugin_runtime]
enabled = true
auto_start = true

[plugin.i18n]
default_locale = "en"
locales_dir = "i18n"
"""
)

_BASE_PY = Template(
    """from plugin.sdk.plugin import NekoPluginBase, neko_plugin, plugin_entry, Ok
from plugin.sdk.minigame import MiniGamePluginMixin, MiniGameScore


@neko_plugin
class ${class_name}(MiniGamePluginMixin, NekoPluginBase):
    def __init__(self, ctx):
        super().__init__(ctx)
        self.load_minigame()

    @plugin_entry(id="status", name="Status", description="Return mini game status.")
    async def status(self):
        return Ok(self.minigame.status())

    @plugin_entry(id="start_session", name="Start Session", description="Start a mini game session.")
    async def start_session(self, session_id: str, lanlan_name: str = ""):
        return Ok(self.minigame.start_session(session_id, lanlan_name=lanlan_name))

    @plugin_entry(id="end_session", name="End Session", description="End a mini game session.")
    async def end_session(self, session_id: str, reason: str = "game_end"):
        return Ok(self.minigame.end_session(session_id, reason=reason))

    @plugin_entry(id="submit_score", name="Submit Score", description="Submit a score.")
    async def submit_score(self, player_id: str, session_id: str, score: int, mode: str = "default"):
        return Ok(self.minigame.leaderboard.submit(MiniGameScore(player_id, session_id, int(score), mode=mode)))
"""
)

_BASIC_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>N.E.K.O Mini Game</title>
  <style>
    body { margin: 0; font-family: system-ui, sans-serif; background: #111; color: white; }
    canvas { display: block; width: 100vw; height: 100vh; }
    button { position: fixed; right: 16px; bottom: 16px; }
  </style>
</head>
<body>
  <canvas id="game"></canvas>
  <button id="end">End</button>
  <script src="/static/game/system/neko-minigame-sdk.js"></script>
  <script>
    const params = new URLSearchParams(location.search);
    const api = NekoMiniGameSDK.create({
      gameType: params.get("game_type") || "__GAME_ID__",
      lanlanName: params.get("lanlan_name") || "",
      sessionId: params.get("session_id") || "",
    });
    const canvas = document.getElementById("game");
    const ctx = canvas.getContext("2d");
    let score = 0;
    function resize() { canvas.width = innerWidth; canvas.height = innerHeight; }
    function draw() {
      ctx.fillStyle = "#111"; ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = "#7dd3fc"; ctx.beginPath();
      ctx.arc(canvas.width / 2, canvas.height / 2, 56 + (score % 6) * 6, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = "white"; ctx.font = "24px system-ui"; ctx.fillText(`Score ${score}`, 24, 42);
    }
    addEventListener("resize", () => { resize(); draw(); });
    canvas.addEventListener("pointerdown", async () => {
      score += 1; draw();
      const result = await api.sendEvent({ type: "tap", score });
      if (result.line) await api.mirrorAssistant(result.line);
    });
    document.getElementById("end").addEventListener("click", () => api.endRoute("manual_end"));
    resize(); draw(); api.startRoute().then(() => api.startHeartbeat());
  </script>
</body>
</html>
"""

_THEATER_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>N.E.K.O Theater</title>
  <style>
    body { margin: 0; font-family: system-ui, sans-serif; background: #18181b; color: white; display: grid; place-items: center; min-height: 100vh; }
    main { width: min(760px, calc(100vw - 32px)); }
    button { margin-top: 16px; }
  </style>
</head>
<body>
  <main>
    <h1 id="title">Theater Scene</h1>
    <p id="line">Press next.</p>
    <button id="next">Next</button>
  </main>
  <script src="/static/game/system/neko-theater-runtime.js"></script>
  <script>
    const script = {
      title: "First Scene",
      characters: [{ id: "yui", name: "Yui" }],
      scenes: [{ id: "opening", beats: [{ character: "yui", dialogue: "Welcome to this N.E.K.O theater scene." }] }],
    };
    document.getElementById("title").textContent = script.title;
    const player = NekoTheaterRuntime.createPlayer(script, {
      onBeat(beat) { document.getElementById("line").textContent = `${beat.character || "scene"}: ${beat.dialogue || ""}`; },
      onEnd() { document.getElementById("line").textContent = "The end."; },
    });
    document.getElementById("next").addEventListener("click", () => player.next());
  </script>
</body>
</html>
"""


def list_templates() -> list[dict[str, Any]]:
    return [
        {
            "id": "basic-canvas-game",
            "name": "Basic Canvas Game",
            "description": "Minimal canvas game with route start, event, heartbeat, and cleanup.",
            "leaderboard": False,
            "memory_archive": False,
        },
        {
            "id": "character-reaction-game",
            "name": "Character Reaction Game",
            "description": "Canvas game that mirrors character lines and is ready for TTS integration.",
            "leaderboard": False,
            "memory_archive": True,
        },
        {
            "id": "score-game",
            "name": "Score Game",
            "description": "Score-focused game with leaderboard-ready Python entries.",
            "leaderboard": True,
            "memory_archive": True,
        },
        {
            "id": "theater-scene",
            "name": "Theater Scene",
            "description": "Small theater scene using the V1 theater DSL preview runtime.",
            "leaderboard": False,
            "memory_archive": True,
            "theater": True,
        },
    ]


def write_template(template_id: str, output_dir: str | Path, *, plugin_id: str = "my_mini_game", name: str | None = None) -> Path:
    template = next((item for item in list_templates() if item["id"] == template_id), None)
    if template is None:
        raise ValueError(f"unknown template_id: {template_id}")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    class_name = "".join(part.capitalize() for part in plugin_id.replace("-", "_").split("_")) + "Plugin"
    display_name = name or str(template["name"])

    (output / "plugin.toml").write_text(
        _BASE_PLUGIN.substitute(
            plugin_id=plugin_id,
            name=display_name,
            description=template["description"],
            class_name=class_name,
        ),
        encoding="utf-8",
    )
    (output / "__init__.py").write_text(_BASE_PY.substitute(class_name=class_name), encoding="utf-8")
    manifest = {
        "game_id": plugin_id,
        "title": {locale: display_name for locale in SUPPORTED_LOCALES},
        "entry": "static/index.html",
        "assets_dir": "static",
        "orientation": "landscape",
        "input_methods": ["mouse", "touch", "keyboard"],
        "character_voice": True,
        "leaderboard": bool(template.get("leaderboard")),
        "memory_archive": bool(template.get("memory_archive")),
        "tags": [template_id],
    }
    (output / "minigame.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    static_dir = output / "static"
    static_dir.mkdir(exist_ok=True)
    html = _THEATER_HTML if template.get("theater") else _BASIC_HTML.replace("__GAME_ID__", plugin_id)
    (static_dir / "index.html").write_text(html, encoding="utf-8")
    i18n_dir = output / "i18n"
    i18n_dir.mkdir(exist_ok=True)
    for locale in SUPPORTED_LOCALES:
        (i18n_dir / f"{locale}.json").write_text(
            json.dumps({"plugin.name": display_name, "plugin.description": template["description"]}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return output
