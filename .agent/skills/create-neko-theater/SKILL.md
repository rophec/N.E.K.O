---
name: create-neko-theater
description: Create N.E.K.O small theater scenes using the MiniGameSDK theater DSL with characters, scenes, beats, dialogue, actions, camera cues, choices, conditions, and a lightweight preview/runtime. Use when asked to make scripted skits, story scenes, or interactive theater content for N.E.K.O.
---

# Create Neko Theater

Use the V1 theater DSL. Do not build a complex director engine unless the user explicitly asks.

## Script Shape

Create `theater/script.json` or `theater/script.yaml`:

```json
{
  "title": "Scene title",
  "characters": [{"id": "yui", "name": "Yui"}],
  "scenes": [{
    "id": "opening",
    "beats": [
      {"character": "yui", "dialogue": "Hello.", "actions": ["wave"], "camera": "medium"},
      {"choices": [{"id": "yes", "label": "Continue", "goto": "ending"}]}
    ]
  }]
}
```

## Workflow

1. Turn the user's story idea into characters, scene ids, beats, dialogue, actions, choices, and conditions.
2. Validate the script with `plugin.sdk.minigame.load_theater_script`.
3. Use `/static/game/system/neko-theater-runtime.js` for preview playback.
4. Package the scene as a mini game plugin when it needs distribution or Hosted UI.
5. Keep all user-visible text in 8 locale files when the scene becomes a reusable template.

## Guardrails

- Keep beat content small and explicit; one emotional/action moment per beat.
- Reference characters only by ids declared in `characters`.
- Use `choices` for branches and `condition` for state gates.
- Prefer readable JSON/YAML over custom syntax.
- Preserve route cleanup if the theater runs inside a mini game window.
