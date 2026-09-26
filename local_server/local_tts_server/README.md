# NEKO Local Lightweight TTS

This service is the first-phase local TTS bridge for NEKO. It deliberately
implements the same WebSocket protocol expected by `local_cosyvoice_worker`, so
NEKO can use it without changing the main TTS pipeline.

## Protocol

Endpoint:

```text
ws://127.0.0.1:50000/v1/audio/speech/stream
```

Client messages:

```json
{"voice":"piper:default","speed":1.0}
{"text":"你好，"}
{"text":"我是 NEKO。"}
{"event":"end"}
```

Server response:

```text
binary PCM s16le chunks, mono, 22050 Hz
```

NEKO's existing `local_cosyvoice_worker` then resamples this audio to 48 kHz.

## Start

From the repository root:

```bash
uv run python local_server/local_tts_server/server.py --host 127.0.0.1 --port 50000
```

In NEKO settings, use the existing local custom TTS path:

```text
ws://127.0.0.1:50000
```

Keep the existing custom/GPT-SoVITS toggle enabled, because the current router
uses that switch to route `ws://` custom TTS URLs into `local_cosyvoice_worker`.

## Voice Selector

The service accepts a model prefix in `voice`:

```text
piper:<voice>
kokoro:<voice>
melotts:<voice>
chattts:<voice>
```

If the prefix is missing, `LOCAL_TTS_DEFAULT_MODEL` is used. The default is
`piper`.

## Piper

Piper is supported through its CLI.

```bash
set LOCAL_TTS_PIPER_MODEL=F:\models\piper\zh_CN-huayan-medium.onnx
set LOCAL_TTS_PIPER_BIN=piper
uv run python local_server/local_tts_server/server.py --port 50000
```

Optional:

```bash
set LOCAL_TTS_PIPER_CONFIG=F:\models\piper\zh_CN-huayan-medium.onnx.json
```

## Kokoro / MeloTTS / ChatTTS

These are exposed through command adapters for now. The command must write a
16-bit WAV file to `{out_file}`.

```bash
set LOCAL_TTS_KOKORO_CMD=python F:\tts_wrappers\kokoro_cli.py --text-file "{text_file}" --out "{out_file}" --voice "{voice}" --speed {speed}
set LOCAL_TTS_MELOTTS_CMD=python F:\tts_wrappers\melotts_cli.py --text-file "{text_file}" --out "{out_file}" --voice "{voice}" --speed {speed}
set LOCAL_TTS_CHATTTS_CMD=python F:\tts_wrappers\chattts_cli.py --text-file "{text_file}" --out "{out_file}" --voice "{voice}" --speed {speed}
```

ChatTTS is AGPL-3.0. Keep it as an optional external backend unless the product
licensing story is settled.

## Smoke Test

For protocol testing without a model:

```bash
set LOCAL_TTS_ENABLE_TONE=1
set LOCAL_TTS_DEFAULT_MODEL=tone
uv run python local_server/local_tts_server/server.py --port 50000
```

Then point NEKO custom TTS URL to `ws://127.0.0.1:50000`. It should emit a short
tone instead of speech, which verifies the WebSocket and audio path.

You can also probe the protocol directly:

```bash
uv run python local_server/local_tts_server/probe.py --voice tone:default
```
