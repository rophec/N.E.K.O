"""Standalone Piper TTS server for NEKO local_cosyvoice_worker protocol.

This intentionally mirrors the Kokoro local server launch shape while keeping
Piper isolated. It exposes the same WebSocket endpoint NEKO already knows:

    ws://127.0.0.1:50000/v1/audio/speech/stream

Protocol:
1. Client sends config JSON: {"voice": "piper:<voice>", "speed": 1.0}
2. Client streams text JSON chunks: {"text": "..."}
3. Client sends end JSON: {"event": "end"}
4. Server replies with binary PCM s16le chunks at 22050 Hz.
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
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


TARGET_SAMPLE_RATE = 22050
CHUNK_BYTES = 4096

SERVER_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = SERVER_DIR / "piper_models"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("piper_tts_server")

app = FastAPI(title="NEKO Piper Local TTS")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def resample_pcm_s16le(pcm: bytes, src_rate: int, dst_rate: int = TARGET_SAMPLE_RATE) -> bytes:
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


def normalize_voice(raw_voice: str) -> str:
    voice = (raw_voice or "").strip()
    if ":" in voice:
        provider, value = voice.split(":", 1)
        if provider.strip().lower() == "piper":
            return value.strip() or "default"
    return voice or os.getenv("LOCAL_TTS_PIPER_DEFAULT_VOICE", "default")


def _looks_like_model_path(token: str) -> bool:
    lower = token.lower()
    return lower.endswith(".onnx") or lower.endswith(".onnx.gz")


def _load_voice_map() -> dict[str, object]:
    merged: dict[str, object] = {}

    inline = os.getenv("LOCAL_TTS_PIPER_VOICE_MAP", "").strip()
    if inline:
        try:
            parsed = json.loads(inline)
            if isinstance(parsed, dict):
                merged.update(parsed)
            else:
                logger.warning("LOCAL_TTS_PIPER_VOICE_MAP must be a JSON object")
        except json.JSONDecodeError as exc:
            logger.warning("LOCAL_TTS_PIPER_VOICE_MAP JSON decode failed: %s", exc)

    map_path = os.getenv("LOCAL_TTS_PIPER_VOICE_MAP_JSON", "").strip()
    if map_path:
        try:
            parsed = json.loads(Path(map_path).read_text(encoding="utf-8"))
            if isinstance(parsed, dict):
                merged.update(parsed)
            else:
                logger.warning("Voice map file must contain a JSON object: %s", map_path)
        except Exception as exc:
            logger.warning("Failed to read Piper voice map %s: %s", map_path, exc)

    return merged


def _first_model_in_dir(model_dir: Path) -> Path | None:
    if not model_dir.is_dir():
        return None
    models = sorted(
        path for path in model_dir.rglob("*.onnx") if path.is_file() and not path.name.endswith(".onnx.json")
    )
    return models[0] if models else None


def list_local_voices() -> list[dict[str, str]]:
    voice_dir = Path(os.getenv("LOCAL_TTS_PIPER_VOICE_DIR", str(DEFAULT_MODEL_DIR))).resolve()
    voices: list[dict[str, str]] = []
    if voice_dir.is_dir():
        for model_path in sorted(voice_dir.rglob("*.onnx")):
            if model_path.is_file() and not model_path.name.endswith(".onnx.json"):
                voices.append(
                    {
                        "id": model_path.stem,
                        "provider": "piper",
                        "model": str(model_path),
                        "config": str(model_path.with_suffix(model_path.suffix + ".json")),
                    }
                )
    return voices


def resolve_model_config(voice: str) -> tuple[str, str]:
    env_model = os.getenv("LOCAL_TTS_PIPER_MODEL", "").strip()
    env_config = os.getenv("LOCAL_TTS_PIPER_CONFIG", "").strip()
    token = (voice or "").strip()

    if token and _looks_like_model_path(token):
        model_path = Path(token)
        if not model_path.is_file():
            raise RuntimeError(f"Piper model file not found: {token}")
        return str(model_path.resolve()), env_config

    voice_map = _load_voice_map()
    if token:
        entry = voice_map.get(token)
        model = ""
        config = ""
        if isinstance(entry, str):
            model = entry.strip()
        elif isinstance(entry, dict):
            model = str(entry.get("model") or "").strip()
            config = str(entry.get("config") or "").strip()
        if model:
            model_path = Path(model)
            if not model_path.is_file():
                raise RuntimeError(f"Piper voice map model not found: {model}")
            return str(model_path.resolve()), config or env_config

    voice_dir = Path(os.getenv("LOCAL_TTS_PIPER_VOICE_DIR", str(DEFAULT_MODEL_DIR))).resolve()
    if token and token != "default":
        candidates = sorted(voice_dir.rglob(f"{token}.onnx")) if voice_dir.is_dir() else []
        if candidates:
            return str(candidates[0]), env_config

    if env_model:
        model_path = Path(env_model)
        if not model_path.is_file():
            raise RuntimeError(f"LOCAL_TTS_PIPER_MODEL not found: {env_model}")
        return str(model_path.resolve()), env_config

    first_model = _first_model_in_dir(voice_dir)
    if first_model is not None:
        return str(first_model), env_config

    raise RuntimeError(
        "No Piper model found. Put a .onnx voice file in "
        f"{DEFAULT_MODEL_DIR}, or set LOCAL_TTS_PIPER_MODEL / LOCAL_TTS_PIPER_VOICE_DIR."
    )


def synthesize_piper(text: str, *, voice: str, speed: float) -> tuple[bytes, int]:
    text = text.strip()
    if not text:
        raise RuntimeError("Cannot synthesize empty text")

    piper_bin = os.getenv("LOCAL_TTS_PIPER_BIN", "piper")
    model, config = resolve_model_config(voice)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        out_path = Path(tmp.name)

    cmd = [piper_bin, "--model", model, "--output_file", str(out_path)]
    if config:
        cmd.extend(["--config", config])
    if speed and speed > 0:
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


async def send_pcm_chunks(websocket: WebSocket, pcm: bytes) -> None:
    for offset in range(0, len(pcm), CHUNK_BYTES):
        await websocket.send_bytes(pcm[offset : offset + CHUNK_BYTES])
        await asyncio.sleep(0)


@app.get("/health")
async def health():
    model_ready = True
    model_error = ""
    try:
        model, _ = resolve_model_config(os.getenv("LOCAL_TTS_PIPER_DEFAULT_VOICE", "default"))
    except Exception as exc:
        model_ready = False
        model = ""
        model_error = str(exc)

    return {
        "status": "ok" if model_ready else "missing_model",
        "provider": "piper",
        "target_sample_rate": TARGET_SAMPLE_RATE,
        "model": model,
        "model_error": model_error,
        "voices": list_local_voices(),
    }


@app.get("/v1/models")
async def list_models():
    return JSONResponse(
        content={
            "object": "list",
            "data": [{"id": "piper", "object": "model", "owned_by": "local_tts"}],
        }
    )


@app.get("/v1/audio/voices")
async def list_voices():
    return {"provider": "piper", "voices": list_local_voices()}


@app.websocket("/v1/audio/speech/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    text_parts: list[str] = []
    voice = ""
    speed = 1.0

    try:
        config_msg = await websocket.receive_text()
        config = json.loads(config_msg)
        voice = normalize_voice(str(config.get("voice") or ""))
        try:
            speed = float(config.get("speed") or 1.0)
        except (TypeError, ValueError):
            speed = 1.0

        logger.info("WS connected: voice=%s speed=%s", voice, speed)

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

        loop = asyncio.get_running_loop()
        pcm, src_rate = await loop.run_in_executor(
            None,
            lambda: synthesize_piper(text, voice=voice, speed=speed),
        )
        pcm = resample_pcm_s16le(pcm, src_rate, TARGET_SAMPLE_RATE)
        await send_pcm_chunks(websocket, pcm)
        await websocket.close()
    except WebSocketDisconnect:
        logger.info("WS disconnected")
    except Exception as exc:
        logger.error("Piper WS TTS error: %s", exc, exc_info=True)
        try:
            await websocket.close(code=1011)
        except Exception:
            pass


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="NEKO standalone Piper TTS server")
    parser.add_argument("--host", default=os.getenv("LOCAL_TTS_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("LOCAL_TTS_PORT", "50000")))
    parser.add_argument("--log-level", default=os.getenv("LOCAL_TTS_LOG_LEVEL", "info"))
    args = parser.parse_args()

    DEFAULT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
