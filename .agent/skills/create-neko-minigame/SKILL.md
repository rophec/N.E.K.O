---
name: create-neko-minigame
description: Create N.E.K.O MiniGameSDK plugins and low-code mini game templates using plugin.sdk.minigame, minigame.json manifests, browser route helpers, character voice integration, scoreboards, and 8-locale creator-ready files. Use when asked to build or teach a N.E.K.O mini game.
---

# Create Neko Minigame

Build mini games as plugin subtypes, not as a separate platform.

## Workflow

1. Start from `plugin.sdk.minigame.write_template()`. Pick `basic-canvas-game`, `character-reaction-game`, `score-game`, or `theater-scene`.
2. Create `plugin/plugins/<game_id>/` and keep the plugin id ASCII.
3. Declare the subtype in `plugin.toml` with `type = "mini_game"` or `capabilities = ["mini_game"]`.
4. Add `minigame.json` with `game_id`, localized `title`, `entry`, `assets_dir`, `orientation`, `input_methods`, `character_voice`, `leaderboard`, `memory_archive`, and optional `launch_path` when the app has a real independent-window page route.
5. In Python, combine `NekoPluginBase` with `MiniGamePluginMixin`; expose status/start/end/score entries with `plugin_entry`.
6. Use `type = "plugin"` plus `capabilities = ["mini_game"]`; do not set `type = "mini_game"` because the current plugin registry schema rejects it.
6. In browser code, use `/static/game/system/neko-minigame-sdk.js` and call `NekoMiniGameSDK.create(...)`.
7. Add or update all 8 locale files.

Example scaffold command:

```bash
uv run python -c "from plugin.sdk.minigame import write_template; write_template('basic-canvas-game', 'plugin/plugins/tap_duel', plugin_id='tap_duel', name='Tap Duel')"
```

## Runtime Calls

Use the browser SDK methods instead of raw fetches:

- `startRoute()`
- `heartbeat()` / `startHeartbeat()` / `stopHeartbeat()`
- `sendEvent(event)`
- `speak(line)`
- `mirrorAssistant(line)`
- `endRoute(reason)`
- `submitScore(score, extra)`
- `loadCharacter()`

## Acceptance Checklist

- The game opens from the plugin manager or a static route.
- The game can start and end a route without leaving stale sessions.
- Character voice integration is optional and disabled cleanly when not needed.
- Score submission is ignored or handled consistently when leaderboard is false.
- Tests run with `uv run pytest`.
- Static resources work in both web dev mode and Electron windows.
