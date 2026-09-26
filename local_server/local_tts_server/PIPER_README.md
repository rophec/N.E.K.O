# NEKO Standalone Piper Local TTS

This is a Piper-only local TTS server for testing Piper separately from the
Kokoro/local lightweight TTS bridge. It uses the same NEKO WebSocket protocol:

```text
ws://127.0.0.1:50000/v1/audio/speech/stream
```

In NEKO settings, fill the custom local TTS URL as:

```text
ws://127.0.0.1:50000
```

## One-Step Start

From a terminal, run:

```text
local_server/local_tts_server/start_piper_server.bat
```

or:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File local_server/local_tts_server/start_piper_server.ps1
```

The launcher installs runtime packages into a temporary uv environment with:

```text
fastapi, uvicorn, numpy, piper-tts
```

## Voice Files

Put a Piper voice pair in:

```text
local_server/local_tts_server/piper_models/
```

Example:

```text
local_server/local_tts_server/piper_models/voice-name.onnx
local_server/local_tts_server/piper_models/voice-name.onnx.json
```

If `LOCAL_TTS_PIPER_MODEL` is not set, the server scans `piper_models` and uses
the first `.onnx` file as the default voice.

## Environment Overrides

```text
LOCAL_TTS_PIPER_BIN=piper
LOCAL_TTS_PIPER_MODEL=F:\models\piper\voice-name.onnx
LOCAL_TTS_PIPER_CONFIG=F:\models\piper\voice-name.onnx.json
LOCAL_TTS_PIPER_VOICE_DIR=F:\models\piper
LOCAL_TTS_PIPER_DEFAULT_VOICE=voice-name
LOCAL_TTS_HOST=127.0.0.1
LOCAL_TTS_PORT=50000
```

The WebSocket config `voice` can be either:

```text
piper:voice-name
```

or just:

```text
voice-name
```

If no voice is configured, the server still starts and `/health` reports
`missing_model`; synthesis fails until a `.onnx` Piper voice is present.
