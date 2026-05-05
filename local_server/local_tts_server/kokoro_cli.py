"""Minimal Kokoro CLI wrapper for local_tts_server (kokoro-onnx backend).

Usage:
    python kokoro_cli.py <text_file> <out_file> <voice> <speed>

Reads text from <text_file>, synthesizes with kokoro-onnx, writes WAV to <out_file>.

Model files expected in ./kokoro_models/:
    kokoro-v1.0.onnx
    voices-v1.0.bin
"""

from __future__ import annotations

import argparse
import os
import sys
import wave
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = SCRIPT_DIR / "kokoro_models"
DEFAULT_VOICE = os.getenv("LOCAL_TTS_KOKORO_DEFAULT_VOICE", "zf_xiaobei").strip() or "zf_xiaobei"
DEFAULT_VOICES = [
    "af_heart",
    "af_bella",
    "af_nicole",
    "af_sarah",
    "af_sky",
    "am_adam",
    "am_michael",
    "bf_emma",
    "bf_isabella",
    "bm_george",
    "bm_lewis",
    "zf_xiaobei",
    "zf_xiaoni",
    "zf_xiaoxiao",
    "zf_xiaoyi",
]


def resolve_model_paths() -> tuple[Path, Path]:
    model_dir = Path(os.getenv("KOKORO_MODEL_DIR", str(DEFAULT_MODEL_DIR)))
    model_path = model_dir / "kokoro-v1.0.onnx"
    voices_path = model_dir / "voices-v1.0.bin"
    return model_path, voices_path


def infer_language(voice: str) -> str:
    normalized = (voice or "").strip().lower()
    if normalized.startswith("z"):
        return "cmn"
    if normalized.startswith("j"):
        return "ja"
    if normalized.startswith("f"):
        return "fr-fr"
    if normalized.startswith("h"):
        return "hi"
    return "en-us"


def normalize_voice(voice: str) -> str:
    value = (voice or "").strip()
    if not value or value == "default":
        return DEFAULT_VOICE
    return value


def synthesize_to_pcm(text: str, voice: str, speed: float) -> tuple[bytes, int]:
    try:
        from kokoro_onnx import Kokoro
    except ImportError:
        raise RuntimeError("kokoro-onnx not installed. Run: uv pip install kokoro-onnx") from None

    model_path, voices_path = resolve_model_paths()

    if not model_path.is_file():
        raise RuntimeError(f"Model not found: {model_path}")
    if not voices_path.is_file():
        raise RuntimeError(f"Voices not found: {voices_path}")

    if not text:
        raise RuntimeError("Empty text")

    kokoro = Kokoro(str(model_path), str(voices_path))
    voice = normalize_voice(voice)
    lang = infer_language(voice)

    samples, sr = kokoro.create(text, voice=voice, speed=speed, lang=lang)

    pcm_int16 = np.clip(samples, -1.0, 1.0)
    pcm_int16 = (pcm_int16 * 32767.0).astype(np.int16)
    return pcm_int16.tobytes(), sr


def synthesize(text_path: str, out_path: str, voice: str, speed: float) -> int:
    text = Path(text_path).read_text(encoding="utf-8").strip()
    pcm_bytes, sr = synthesize_to_pcm(text, voice, speed)

    with wave.open(out_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm_bytes)

    duration = len(pcm_bytes) / 2 / sr
    print(f"Wrote {out_path}: {len(pcm_bytes) // 2} samples @ {sr} Hz ({duration:.2f}s)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Kokoro ONNX CLI wrapper for local_tts")
    parser.add_argument("text_file")
    parser.add_argument("out_file")
    parser.add_argument("voice")
    parser.add_argument("speed", type=float)
    args = parser.parse_args()
    try:
        return synthesize(args.text_file, args.out_file, args.voice, args.speed)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
