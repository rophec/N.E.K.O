# MiniGameSDK Creator Path

MiniGameSDK is a plugin subtype for low-code games and small theater scenes.
It reuses `plugin.toml`, `plugin_entry`, Hosted UI, the mini game bridge API,
and the 8-locale i18n convention. It does not register built-in app game
routes such as `/soccer_demo` or `/badminton_demo`.

## Creator Tracks

- 10 minutes: create a normal plugin with `.agent/skills/create-neko-plugin`.
- 30 minutes: create a canvas mini game with `.agent/skills/create-neko-minigame`.
- 60 minutes: create a small theater scene with `.agent/skills/create-neko-theater`.

## Mini Game Manifest

Declare the subtype in `plugin.toml`:

```toml
[plugin]
id = "tap_duel"
name = "Tap Duel"
type = "plugin"
capabilities = ["mini_game"]
entry = "plugin.plugins.tap_duel:TapDuelPlugin"

[plugin.mini_game]
game_id = "tap_duel"
title = "Tap Duel"
entry = "static/index.html"
assets_dir = "static"
orientation = "landscape"
input_methods = ["mouse", "touch"]
character_voice = true
leaderboard = true
memory_archive = true
route_game_type = "tap_duel"
```

Or put the same fields in `minigame.json`.

Use `type = "plugin"` plus `capabilities = ["mini_game"]`. The current plugin
registry rejects `type = "mini_game"`.

`entry` is the plugin static UI entry. If the game needs an independent window,
open the same Hosted UI with an explicit game surface such as
`/plugin/tap_duel/ui?surface=game`; do not add a main app page route for SDK
games.

## Browser SDK

Load `/static/game/system/neko-minigame-sdk.js` from the game page:

```html
<script src="/static/game/system/neko-minigame-sdk.js"></script>
```

Use:

```js
const game = NekoMiniGameSDK.create({
  gameType: "tap_duel",
  lanlanName: new URLSearchParams(location.search).get("lanlan_name") || "",
});
await game.startRoute();
await game.sendEvent({ type: "hit", score: 10 });
await game.speak("Nice shot!");
await game.endRoute("game_end");
```

## Templates

Start from `plugin.sdk.minigame.write_template()`:

- `basic-canvas-game`
- `character-reaction-game`
- `score-game`
- `theater-scene`

Every template includes the same contract: plugin metadata, mini game manifest,
static entry, and 8 locale files.

```bash
uv run python -c "from plugin.sdk.minigame import write_template; write_template('basic-canvas-game', 'plugin/plugins/tap_duel', plugin_id='tap_duel', name='Tap Duel')"
```

## Validation

Run:

```bash
uv run pytest plugin/tests/unit/sdk/minigame
```
