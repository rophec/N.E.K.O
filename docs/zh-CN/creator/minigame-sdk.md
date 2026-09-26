# MiniGameSDK 创作者路径

MiniGameSDK 是 N.E.K.O 插件的一种创作者子类型，用来制作低代码小游戏和小剧场。它复用 `plugin.toml`、`plugin_entry`、Hosted UI、小游戏桥接 API 和 8 语言 i18n 约定。它不注册 `/soccer_demo`、`/badminton_demo` 这类主应用内置小游戏路由。

## 创作者路径

- 10 分钟：用 `.agent/skills/create-neko-plugin` 创建普通插件。
- 30 分钟：用 `.agent/skills/create-neko-minigame` 创建 canvas 小游戏。
- 60 分钟：用 `.agent/skills/create-neko-theater` 创建小剧场。

## 小游戏 Manifest

在 `plugin.toml` 中声明：

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

也可以把同样字段放进 `minigame.json`。

注意：当前插件注册器只接受 `type = "plugin"`、`extension`、`script`、`adapter`。小游戏要用 `capabilities = ["mini_game"]` 标记，不要写 `type = "mini_game"`。

`entry` 是插件静态 UI 入口。需要独立窗口时，打开同一个 Hosted UI 的游戏 surface，例如 `/plugin/tap_duel/ui?surface=game`；不要为 SDK 小游戏新增主应用页面路由。

## 浏览器 SDK

在游戏页面加载：

```html
<script src="/static/game/system/neko-minigame-sdk.js"></script>
```

核心方法：

- `startRoute()`
- `heartbeat()`
- `sendEvent(event)`
- `speak(line)`
- `mirrorAssistant(line)`
- `endRoute(reason)`
- `submitScore(score, extra)`
- `loadCharacter()`

## 模板

从 `plugin.sdk.minigame.write_template()` 开始：

- `basic-canvas-game`
- `character-reaction-game`
- `score-game`
- `theater-scene`

每个模板都包含插件元数据、小游戏 manifest、静态入口和 8 个 locale 文件。

```bash
uv run python -c "from plugin.sdk.minigame import write_template; write_template('basic-canvas-game', 'plugin/plugins/tap_duel', plugin_id='tap_duel', name='Tap Duel')"
```

## 验证

运行：

```bash
uv run pytest plugin/tests/unit/sdk/minigame
```
