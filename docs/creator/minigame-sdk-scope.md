# MiniGameSDK Scope And Boundaries

> Temporary branch note: this document lives in the MiniGameSDK worktree for now.
> Before merging the SDK work, move or fold it into the official creator
> documentation location.

## Why This SDK Exists

MiniGameSDK is meant to help creators build N.E.K.O mini games, small theater
scenes, and related creator plugins with low friction.

The goal is not to add more built-in games to the main app. The goal is to make
creator-made plugins easy to generate, run, test, localize, and share.

## Core Boundary

MiniGameSDK is a plugin creator SDK.

It should reuse the existing N.E.K.O plugin system:

- `plugin.toml`
- plugin capabilities such as `capabilities = ["mini_game"]`
- Hosted UI
- Python `plugin_entry`
- plugin lifecycle and runtime context
- plugin manager installation, enablement, and validation
- 8-locale i18n convention

MiniGameSDK should not become a new built-in game subsystem.

## Non-Goals For V1

Do not register SDK games as main app pages like built-in demos.

Do not wire SDK examples into existing built-in game routes such as
`/soccer_demo` or `/badminton_demo`.

Do not modify the PC shell's built-in mini-game route allowlist just to make SDK
examples look like built-in games.

Do not treat a mini game as `type = "mini_game"` until the plugin registry
explicitly supports that plugin type. V1 uses `type = "plugin"` plus
`capabilities = ["mini_game"]`.

Do not build a general-purpose game engine in V1. The SDK should provide
templates and host bridges, not replace Canvas, Phaser, Three.js, Cocos, Unity,
or other engines.

## Desired Creator Experience

A creator should be able to describe a game idea in one sentence and get:

- a valid plugin directory
- a mini game manifest
- a Hosted UI entry
- a playable game surface
- Python helper entries
- basic tests
- 8 locale files
- clear instructions for what to edit next

The first successful experience should feel like:

1. Generate from template.
2. Run N.E.K.O.
3. Open the plugin panel.
4. Click "Open Game Window".
5. Play a working mini game.
6. Change a few obvious files to customize it.

## Window And Surface Model

The plugin panel and the game surface are separate concerns.

The plugin panel is allowed to be a launcher or configuration page.

The actual game should run as a plugin-owned game surface. In V1, that can be
the same Hosted UI entry with an explicit query mode, for example:

```text
/plugin/tap_duel/ui
/plugin/tap_duel/ui?surface=game&game_type=tap_duel
```

The panel may call `window.open()` to request an independent window, but the
target should remain plugin-owned. It should not require adding a new main app
route per game.

## Manifest Shape

The manifest describes the creator plugin, not a main app page.

Recommended V1 fields:

```json
{
  "game_id": "tap_duel",
  "title": {
    "en": "Tap Duel",
    "zh-CN": "Tap Duel"
  },
  "entry": "static/index.html",
  "assets_dir": "static",
  "orientation": "landscape",
  "input_methods": ["mouse", "touch", "keyboard"],
  "character_voice": true,
  "leaderboard": true,
  "memory_archive": true,
  "route_game_type": "tap_duel",
  "tags": ["score-game", "official-example"]
}
```

Avoid fields that imply built-in app routing, such as `launch_path`, unless the
architecture is later deliberately changed and documented.

## SDK Responsibilities

The browser SDK should hide N.E.K.O host details from creators:

- `startRoute()`
- `heartbeat()`
- `sendEvent()`
- `speak()`
- `mirrorAssistant()`
- `endRoute()`
- `submitScore()`
- `loadCharacter()`
- `loadAudio()`

The Python helper should make common plugin work boring:

- manifest loading and validation
- session state
- leaderboard storage
- session cleanup
- standard result/error shapes
- prompt or event registration hooks where needed

## Template Responsibilities

Every template should be runnable immediately.

Each template should include:

- `plugin.toml`
- `minigame.json`
- `__init__.py`
- `static/index.html`
- 8 locale files
- focused tests or test fixtures
- a short README or generated instructions

V1 template priorities:

- `basic-canvas-game`
- `character-reaction-game`
- `score-game`
- `theater-scene`

## Theater Boundary

Small theater is part of the creator SDK, but V1 should stay lightweight.

Use a JSON/YAML DSL for:

- characters
- scenes
- beats
- dialogue
- actions
- choices
- conditions

V1 should provide a previewer and validator. It should not attempt a complex
director system yet.

## Skill Boundary

The creator skills are the user-facing production workflow:

- `create-neko-plugin`
- `create-neko-minigame`
- `create-neko-theater`

These skills should encode N.E.K.O engineering constraints so Codex does not
wander into built-in app routes, partial i18n, or unsupported plugin types.

## Review Checklist

Before accepting MiniGameSDK changes, check:

- Does this remain plugin-owned?
- Does it avoid `/soccer_demo` and `/badminton_demo` implementation paths?
- Does it avoid adding one main app route per SDK game?
- Does the example use `type = "plugin"` and `capabilities = ["mini_game"]`?
- Can a creator copy the template and run it quickly?
- Are user-visible plugin strings covered by all 8 locales?
- Are Python commands run through `uv run`?
- Are tests focused on manifest, runtime helper, template output, and plugin
  behavior?

## Current North Star

MiniGameSDK should become a creator ecosystem layer:

```text
creator prompt -> Codex skill -> plugin template -> Hosted UI game surface
-> N.E.K.O host bridge -> shareable creator plugin
```

It should not become:

```text
creator prompt -> new built-in app route -> hardcoded PC window rule
```
