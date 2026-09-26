"""Probe the local_tts WebSocket protocol used by NEKO local_cosyvoice_worker."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import websockets


async def run_probe(url: str, voice: str, speed: float, text: str, save_raw: str) -> int:
    total = 0
    chunks: list[bytes] = []

    async with websockets.connect(url, ping_interval=None, max_size=10 * 1024 * 1024) as ws:
        await ws.send(json.dumps({"voice": voice, "speed": speed}, ensure_ascii=False))
        await ws.send(json.dumps({"text": text}, ensure_ascii=False))
        await ws.send(json.dumps({"event": "end"}, ensure_ascii=False))

        while True:
            try:
                msg = await ws.recv()
            except websockets.exceptions.ConnectionClosed:
                break

            if isinstance(msg, bytes):
                total += len(msg)
                if save_raw:
                    chunks.append(msg)

    if save_raw:
        out_path = Path(save_raw)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"".join(chunks))

    print(f"received_pcm_bytes={total}")
    if save_raw:
        print(f"saved_raw_pcm={Path(save_raw).resolve()}")
    return 0 if total > 0 else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe NEKO local_tts WebSocket")
    parser.add_argument("--url", default="ws://127.0.0.1:50000/v1/audio/speech/stream")
    parser.add_argument("--voice", default="tone:default")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--text", default="本地 TTS 链路测试。")
    parser.add_argument("--save-raw", default="", help="Optional output path for raw PCM s16le")
    args = parser.parse_args()

    raise SystemExit(asyncio.run(run_probe(args.url, args.voice, args.speed, args.text, args.save_raw)))


if __name__ == "__main__":
    main()
