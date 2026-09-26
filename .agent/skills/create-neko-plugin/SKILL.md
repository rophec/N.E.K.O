---
name: create-neko-plugin
description: Create ordinary N.E.K.O user plugins using the repository plugin SDK, plugin.toml schema, Hosted UI conventions, plugin_entry tools, and 8-locale i18n. Use when asked to build, scaffold, review, or teach a N.E.K.O plugin that is not specifically a mini game or theater scene.
---

# Create Neko Plugin

Use the existing plugin SDK. Do not invent a second plugin framework.

## Workflow

1. Inspect nearby examples in `plugin/plugins/` and docs under `docs/plugins/`.
2. Create the plugin under `plugin/plugins/<plugin_id>/` with `plugin.toml`, a Python entry module, optional Hosted UI, and 8 locale files when user-visible text is added.
3. Import public APIs from `plugin.sdk.plugin`, especially `NekoPluginBase`, `neko_plugin`, `plugin_entry`, `Ok`, `Err`, `SdkError`, `ui`, and `tr`.
4. Use `plugin_entry` for callable actions. Add `input_schema` and `llm_result_fields` for LLM-visible tools.
5. Use Hosted UI for interactive panels; use Markdown guide surfaces for read-only tutorials.
6. Run tests with `uv run`, never direct system Python.

## Required Shape

Use this minimum structure:

```text
plugin/plugins/<plugin_id>/
  plugin.toml
  __init__.py
  i18n/en.json
  i18n/ja.json
  i18n/ko.json
  i18n/zh-CN.json
  i18n/zh-TW.json
  i18n/ru.json
  i18n/pt.json
  i18n/es.json
```

If adding UI, add `ui/panel.tsx` or `docs/quickstart.md` and declare it in `[plugin.ui]`.

## Guardrails

- Keep plugin ids ASCII, lowercase, and stable.
- Do not log private conversation content through `logger`; use `print` for raw private text.
- Keep provider/backend/feature structures symmetric when touching shared systems.
- Preserve all 8 locale files for any user-facing string.
- For chat UI work, edit `frontend/react-neko-chat/`, not legacy `#chat-container`.
