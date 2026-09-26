# Mahjong Coach

A screenshot-only Mahjong Soul banter companion that reacts to public table events without gameplay recommendations.

## What it does

- Watches the public Mahjong Soul table through Windows window capture.
- Reacts to round starts, calls, riichi pressure, wins, draws, and settlements.
- Sends short companion banter to N.E.K.O without recommending discards, calls, or ranked actions.
- Keeps screenshot processing local. Runtime captures and user preferences are excluded from this repository and release packages.

## Requirements and boundaries

- Windows 10/11 x64.
- A supported Mahjong Soul desktop window titled `雀魂` or `Mahjong Soul`.
- N.E.K.O with the compatible plugin SDK range declared in `plugin.toml`.
- The bundled ONNX models may use GPU acceleration when available and fall back to CPU.

This plugin is intended as a non-directive companion. It does not inject into the game client, read process memory, or provide discard/call instructions.

## Development

This repository is meant to live at:

```text
N.E.K.O/plugin/plugins/mahjong_coach
```

When publishing to the plugin market, use this GitHub repository name:

```text
n.e.k.o_plugin_mahjong_coach
```

From the N.E.K.O repository root:

```bash
uv run neko-plugin check mahjong_coach
uv run neko-plugin check --release mahjong_coach
```

## Market release

Push a tag matching `plugin.toml` version to create a GitHub Release asset:

```bash
git tag v0.3.30
git push origin v0.3.30
```

The generated `.github/workflows/release.yml` uploads `mahjong_coach.neko-plugin`.
Use that GitHub Release URL when publishing a version in the plugin market.

Before pushing a release tag, run the Market-specific check from the N.E.K.O repository root:

```bash
uv run neko-plugin check --release --market-release /path/to/n.e.k.o_plugin_mahjong_coach
```

## Entry

```toml
entry = "plugin.plugins.mahjong_coach:MahjongCoachPlugin"
```
