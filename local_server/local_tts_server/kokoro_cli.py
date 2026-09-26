"""Minimal Kokoro CLI wrapper for local_tts_server.

Usage:
    python kokoro_cli.py <text_file> <out_file> <voice> <speed>

Reads text from <text_file>, synthesizes with kokoro, writes WAV to <out_file>.
"""

from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

import numpy as np


def synthesize(text_path: str, out_path: str, voice: str, speed: float) -> int:
    try:
        from kokoro import KPipeline
    except ImportError:
        print("kokoro not installed. Run: uv pip install 'kokoro>=0.9.2'", file=sys.stderr)
        return 1

    text = Path(text_path).read_text(encoding="utf-8").strip()
    if not text:
        print("Empty text file", file=sys.stderr)
        return 1

    # kokoro uses single-letter lang codes: z=zh, a=en, etc.
    # Infer from voice prefix if possible, default to 'z' for Chinese.
    lang = "z"
    if voice.startswith(("a", "af", "am")):
        lang = "a"
    elif voice.startswith(("b", "bf")):
        lang = "b"

    pipeline = KPipeline(lang_code=lang)
    generator = pipeline(text, voice=voice, speed=speed)

    chunks: list[np.ndarray] = []
    sr = 24000  # kokoro default sample rate
    for _, _, audio in generator:
        if audio is not None:
            chunks.append(audio)

    if not chunks:
        print("No audio generated", file=sys.stderr)
        return 1

    pcm = np.concatenate(chunks)
    pcm = np.clip(pcm, -1.0, 1.0)
    pcm_int16 = (pcm * 32767.0).astype(np.int16)

    with wave.open(out_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm_int16.tobytes())

    print(f"Wrote {out_path}: {len(pcm_int16)} samples @ {sr} Hz")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Kokoro CLI wrapper for local_tts")
    parser.add_argument("text_file")
    parser.add_argument("out_file")
    parser.add_argument("voice")
    parser.add_argument("speed", type=float)
    args = parser.parse_args()
    return synthesize(args.text_file, args.out_file, args.voice, args.speed)


if __name__ == "__main__":
    raise SystemExit(main())
