"""Local lightweight TTS server compatible with NEKO local_cosyvoice_worker.

The first integration phase intentionally keeps the wire protocol identical to
``local_cosyvoice_worker``:

1. Client sends config JSON: {"voice": "...", "speed": 1.0}
2. Client streams text JSON chunks: {"text": "..."}
3. Client sends end JSON: {"event": "end"}
4. Server replies with binary PCM s16le chunks at 22050 Hz.

Model support is implemented as optional adapters so this server can start even
when a specific local TTS runtime has not been installed yet.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


TARGET_SAMPLE_RATE = 22050
CHUNK_BYTES = 4096

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("local_tts_server")

app = FastAPI(title="NEKO Local Lightweight TTS")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TTSEngine(Protocol):
    """Small interface implemented by every local model adapter."""

    name: str

    def synthesize(self, text: str, *, voice: str, speed: float) -> tuple[bytes, int]:
        """Return PCM s16le bytes and source sample rate."""


@dataclass(frozen=True)
class VoiceSpec:
    """Parsed voice selector.

    Accepted examples:
    - "piper:zh_CN-huayan-medium"
    - "kokoro:zf_xiaobei"
    - "melotts:zh"
    - "chattts:default"
    - "中文女" -> falls back to LOCAL_TTS_DEFAULT_MODEL
    """

    model: str
    voice: str


def parse_voice(raw_voice: str) -> VoiceSpec:
    default_model = os.getenv("LOCAL_TTS_DEFAULT_MODEL", "piper").strip().lower() or "piper"
    value = (raw_voice or "").strip()
    if ":" not in value:
        return VoiceSpec(default_model, value or "default")
    model, voice = value.split(":", 1)
    model = model.strip().lower() or default_model
    return VoiceSpec(model, voice.strip() or "default")


def resample_pcm_s16le(pcm: bytes, src_rate: int, dst_rate: int = TARGET_SAMPLE_RATE) -> bytes:
    """Resample mono s16le PCM with numpy interpolation.

    This keeps the local server dependency-light. NEKO still performs its own
    final 22050 -> 48000 conversion in ``local_cosyvoice_worker``.
    """

    if not pcm or src_rate == dst_rate:
        return pcm

    audio = np.frombuffer(pcm, dtype=np.int16)
    if audio.size == 0:
        return b""

    duration = audio.size / float(src_rate)
    out_size = max(1, int(round(duration * dst_rate)))
    src_x = np.linspace(0.0, duration, num=audio.size, endpoint=False)
    dst_x = np.linspace(0.0, duration, num=out_size, endpoint=False)
    resampled = np.interp(dst_x, src_x, audio.astype(np.float32))
    return np.clip(resampled, -32768, 32767).astype(np.int16).tobytes()


def read_wav_pcm(path: Path) -> tuple[bytes, int]:
    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())

    if sample_width != 2:
        raise RuntimeError(f"Only 16-bit WAV is supported, got sample_width={sample_width}")

    if channels == 1:
        return frames, sample_rate

    audio = np.frombuffer(frames, dtype=np.int16).reshape(-1, channels)
    mono = audio.mean(axis=1).clip(-32768, 32767).astype(np.int16)
    return mono.tobytes(), sample_rate


class PiperCliEngine:
    """Piper CLI adapter.

    Configure with:
    - LOCAL_TTS_PIPER_BIN: piper executable path, defaults to "piper"
    - LOCAL_TTS_PIPER_MODEL: default .onnx model path
    - LOCAL_TTS_PIPER_CONFIG: optional json config path
    - LOCAL_TTS_PIPER_VOICE_DIR: optional directory containing `{voice}.onnx` models
    - LOCAL_TTS_PIPER_VOICE_MAP_JSON: optional JSON path mapping `"voice"` → `{model, config?}`
    - LOCAL_TTS_PIPER_VOICE_MAP: optional inline JSON mapping `"voice"` → `{model, config?}`
    """

    name = "piper"

    _voice_map_cache: dict[str, Any] | None = None

    @classmethod
    def _load_voice_map(cls) -> dict[str, Any]:
        if cls._voice_map_cache is not None:
            return cls._voice_map_cache

        merged: dict[str, Any] = {}

        raw_inline = os.getenv("LOCAL_TTS_PIPER_VOICE_MAP", "").strip()
        if raw_inline:
            try:
                parsed = json.loads(raw_inline)
                if isinstance(parsed, dict):
                    merged.update(parsed)
                else:
                    logger.warning("LOCAL_TTS_PIPER_VOICE_MAP must be a JSON object")
            except json.JSONDecodeError as exc:
                logger.warning("LOCAL_TTS_PIPER_VOICE_MAP JSON decode failed: %s", exc)

        map_path = os.getenv("LOCAL_TTS_PIPER_VOICE_MAP_JSON", "").strip()
        if map_path:
            try:
                data = Path(map_path).read_text(encoding="utf-8")
                parsed = json.loads(data)
                if isinstance(parsed, dict):
                    merged.update(parsed)
                else:
                    logger.warning(
                        "LOCAL_TTS_PIPER_VOICE_MAP_JSON must contain a JSON object: %s",
                        map_path,
                    )
            except Exception as exc:
                logger.warning("Failed to read LOCAL_TTS_PIPER_VOICE_MAP_JSON: %s", exc)

        cls._voice_map_cache = merged
        return merged

    @staticmethod
    def _looks_like_model_path(token: str) -> bool:
        lower = token.lower()
        return lower.endswith(".onnx") or lower.endswith(".onnx.gz")

    def _resolve_model_config(self, voice_token: str) -> tuple[str, str]:
        """Resolve Piper model/config paths.

        voice_token comes from ``VoiceSpec.voice`` after ``piper:`` splitting.
        """

        env_model = os.getenv("LOCAL_TTS_PIPER_MODEL", "").strip()
        env_config = os.getenv("LOCAL_TTS_PIPER_CONFIG", "").strip()

        token = (voice_token or "").strip()
        if token and self._looks_like_model_path(token):
            path = Path(token)
            if not path.is_file():
                raise RuntimeError(f"Piper model file not found: {token}")
            return str(path.resolve()), env_config

        voice_map = self._load_voice_map()
        if token:
            entry = voice_map.get(token)
            if isinstance(entry, str):
                model_path = entry.strip()
                cfg_path = ""
            elif isinstance(entry, dict):
                model_path = str(entry.get("model") or "").strip()
                cfg_path = str(entry.get("config") or "").strip()
            else:
                model_path = ""
                cfg_path = ""

            if model_path:
                p = Path(model_path)
                if not p.is_file():
                    raise RuntimeError(f"Piper model path from voice map not found: {model_path}")
                return str(p.resolve()), cfg_path or env_config

        voice_dir = os.getenv("LOCAL_TTS_PIPER_VOICE_DIR", "").strip()
        if voice_dir and token:
            candidate = Path(voice_dir) / f"{token}.onnx"
            if candidate.is_file():
                return str(candidate.resolve()), env_config

        if env_model:
            return env_model, env_config

        raise RuntimeError(
            "Piper model not resolved. Set LOCAL_TTS_PIPER_MODEL, "
            "LOCAL_TTS_PIPER_VOICE_MAP(_JSON), LOCAL_TTS_PIPER_VOICE_DIR, "
            "or pass an `.onnx` path after `piper:`."
        )

    def synthesize(self, text: str, *, voice: str, speed: float) -> tuple[bytes, int]:
        piper_bin = os.getenv("LOCAL_TTS_PIPER_BIN", "piper")
        model, config = self._resolve_model_config(voice)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            out_path = Path(tmp.name)

        cmd = [piper_bin, "--model", model, "--output_file", str(out_path)]
        if config:
            cmd.extend(["--config", config])
        if speed and speed > 0:
            # Piper uses length_scale: larger means slower.
            cmd.extend(["--length_scale", f"{1.0 / speed:.3f}"])

        try:
            proc = subprocess.run(
                cmd,
                input=text,
                text=True,
                capture_output=True,
                check=False,
                timeout=float(os.getenv("LOCAL_TTS_ENGINE_TIMEOUT", "60")),
            )
            if proc.returncode != 0:
                err = (proc.stderr or proc.stdout or "").strip()
                raise RuntimeError(f"Piper failed: {err[:500]}")
            return read_wav_pcm(out_path)
        finally:
            try:
                out_path.unlink(missing_ok=True)
            except Exception:
                pass


class CommandWavEngine:
    """Generic command adapter for Kokoro/MeloTTS/ChatTTS wrappers.

    The command must accept:
      {text_file} {out_file} {voice} {speed}

    Configure per model with:
    - LOCAL_TTS_KOKORO_CMD
    - LOCAL_TTS_MELOTTS_CMD
    - LOCAL_TTS_CHATTTS_CMD
    """

    def __init__(self, name: str, env_var: str):
        self.name = name
        self._env_var = env_var

    def synthesize(self, text: str, *, voice: str, speed: float) -> tuple[bytes, int]:
        template = os.getenv(self._env_var, "").strip()
        if not template:
            raise RuntimeError(f"{self._env_var} is required for {self.name}")

        with tempfile.NamedTemporaryFile("w", suffix=".txt", encoding="utf-8", delete=False) as text_file:
            text_path = Path(text_file.name)
            text_file.write(text)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_file:
            out_path = Path(wav_file.name)

        try:
            cmd = template.format(
                text_file=str(text_path),
                out_file=str(out_path),
                voice=voice,
                speed=speed,
            )
            proc = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                check=False,
                timeout=float(os.getenv("LOCAL_TTS_ENGINE_TIMEOUT", "120")),
            )
            if proc.returncode != 0:
                err = (proc.stderr or proc.stdout or "").strip()
                raise RuntimeError(f"{self.name} command failed: {err[:500]}")
            return read_wav_pcm(out_path)
        finally:
            for path in (text_path, out_path):
                try:
                    path.unlink(missing_ok=True)
                except Exception:
                    pass


class ToneEngine:
    """Smoke-test engine used only when LOCAL_TTS_ENABLE_TONE=1."""

    name = "tone"

    def synthesize(self, text: str, *, voice: str, speed: float) -> tuple[bytes, int]:
        duration = min(2.0, max(0.25, len(text) * 0.04))
        samples = int(TARGET_SAMPLE_RATE * duration)
        freq = 440.0 if "high" not in voice else 660.0
        t = np.arange(samples, dtype=np.float32) / TARGET_SAMPLE_RATE
        audio = 0.2 * np.sin(2.0 * math.pi * freq * t)
        return (audio * 32767.0).astype(np.int16).tobytes(), TARGET_SAMPLE_RATE


def build_engines() -> dict[str, TTSEngine]:
    engines: dict[str, TTSEngine] = {
        "piper": PiperCliEngine(),
        "kokoro": CommandWavEngine("kokoro", "LOCAL_TTS_KOKORO_CMD"),
        "melotts": CommandWavEngine("melotts", "LOCAL_TTS_MELOTTS_CMD"),
        "melo": CommandWavEngine("melotts", "LOCAL_TTS_MELOTTS_CMD"),
        "chattts": CommandWavEngine("chattts", "LOCAL_TTS_CHATTTS_CMD"),
    }
    if os.getenv("LOCAL_TTS_ENABLE_TONE", "").strip().lower() in {"1", "true", "yes", "on"}:
        engines["tone"] = ToneEngine()
    return engines


ENGINES = build_engines()


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "target_sample_rate": TARGET_SAMPLE_RATE,
        "engines": sorted(ENGINES.keys()),
    }


@app.get("/v1/models")
async def list_models():
    return JSONResponse(
        content={
            "object": "list",
            "data": [
                {"id": name, "object": "model", "owned_by": "local_tts"}
                for name in sorted(ENGINES.keys())
            ],
        }
    )


async def send_pcm_chunks(websocket: WebSocket, pcm: bytes) -> None:
    for offset in range(0, len(pcm), CHUNK_BYTES):
        await websocket.send_bytes(pcm[offset : offset + CHUNK_BYTES])
        await asyncio.sleep(0)


@app.websocket("/v1/audio/speech/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    text_parts: list[str] = []
    voice = ""
    speed = 1.0

    try:
        config_msg = await websocket.receive_text()
        config = json.loads(config_msg)
        voice = str(config.get("voice") or "").strip()
        try:
            speed = float(config.get("speed") or 1.0)
        except (TypeError, ValueError):
            speed = 1.0

        logger.info("WS connected: voice=%s speed=%s", voice or "<default>", speed)

        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            text_chunk = msg.get("text")
            if isinstance(text_chunk, str) and text_chunk:
                text_parts.append(text_chunk)

            if msg.get("event") == "end":
                break

        text = "".join(text_parts).strip()
        if not text:
            await websocket.close()
            return

        spec = parse_voice(voice)
        engine = ENGINES.get(spec.model)
        if engine is None:
            raise RuntimeError(f"Unsupported local TTS model: {spec.model}")

        logger.info(
            "Synthesizing via %s: voice=%s chars=%d",
            engine.name,
            spec.voice,
            len(text),
        )
        loop = asyncio.get_running_loop()
        pcm, src_rate = await loop.run_in_executor(
            None,
            lambda: engine.synthesize(text, voice=spec.voice, speed=speed),
        )
        pcm = resample_pcm_s16le(pcm, src_rate, TARGET_SAMPLE_RATE)
        await send_pcm_chunks(websocket, pcm)
        await websocket.close()
    except WebSocketDisconnect:
        logger.info("WS disconnected")
    except Exception as exc:
        logger.error("WS TTS error: %s", exc, exc_info=True)
        try:
            await websocket.close(code=1011)
        except Exception:
            pass


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="NEKO local lightweight TTS server")
    parser.add_argument("--host", default=os.getenv("LOCAL_TTS_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("LOCAL_TTS_PORT", "50000")))
    parser.add_argument("--log-level", default=os.getenv("LOCAL_TTS_LOG_LEVEL", "info"))
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
