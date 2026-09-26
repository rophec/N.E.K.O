"""End-to-end smoke test for local_tts_server WebSocket protocol.

Tests multiple voice_id formats to verify client-side parsing (in NEKO)
aligns with server-side routing.

Usage:
    # 1. Start the server in another terminal:
    python -m local_tts_server.server

    # 2. Run smoke test:
    python smoke_test.py --url ws://127.0.0.1:50000/v1/audio/speech/stream
"""

from __future__ import annotations

import argparse
import asyncio
import json
import urllib.request
from pathlib import Path

import websockets


def fetch_engines(http_base: str) -> set[str]:
    """Query /health to get the list of available engines."""
    try:
        req = urllib.request.Request(f"{http_base}/health", method="GET")
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return set(data.get("engines", []))
    except Exception as exc:
        print(f"[!] Could not query /health: {exc}")
        return set()


async def test_case(
    ws_url: str,
    voice: str,
    speed: float,
    text: str,
) -> dict:
    """Send config→text→end, collect PCM, return result dict."""
    result = {
        "voice": voice,
        "speed": speed,
        "text": text,
        "success": False,
        "pcm_bytes": 0,
        "error": None,
    }
    chunks: list[bytes] = []

    try:
        async with websockets.connect(ws_url, ping_interval=None) as ws:
            await ws.send(json.dumps({"voice": voice, "speed": speed}, ensure_ascii=False))
            await ws.send(json.dumps({"text": text}, ensure_ascii=False))
            await ws.send(json.dumps({"event": "end"}, ensure_ascii=False))

            while True:
                try:
                    msg = await ws.recv()
                except websockets.exceptions.ConnectionClosed:
                    break
                if isinstance(msg, bytes):
                    chunks.append(msg)
    except Exception as exc:
        result["error"] = str(exc)
        return result

    result["pcm_bytes"] = sum(len(c) for c in chunks)
    if result["pcm_bytes"] == 0:
        result["error"] = "expected PCM but received 0 bytes"
        return result

    result["success"] = True
    return result


async def run_all(ws_url: str, http_base: str) -> int:
    # tone engine cases are the protocol baseline (always present)
    baseline_cases = [
        ("tone:default", 1.0, "冒烟测试短句。"),
        ("tone:high", 1.2, "Speed test with high voice."),
    ]

    # extra cases for voice format parsing / routing (skip if engine not installed)
    extra_cases = [
        ("piper:zh_CN-huayan-medium", 1.0, "Piper voice format test."),
        ("kokoro:zf_xiaobei", 1.0, "Kokoro voice format test."),
        ("melotts:zh", 1.0, "MeloTTS voice format test."),
        ("chattts:default", 1.0, "ChatTTS voice format test."),
        # NEKO client strips speed suffix before sending; server receives clean model:voice
        ("piper:zh_CN-huayan-medium", 0.8, "Voice with speed suffix test."),
    ]

    available = fetch_engines(http_base)
    print(f"Target: {ws_url}")
    print(f"Available engines: {sorted(available) or '(unknown)'}")
    print("=" * 50)

    passed = skipped = failed = 0

    async def run_one(voice: str, speed: float, text: str, required: bool) -> None:
        nonlocal passed, skipped, failed
        model = voice.split(":", 1)[0] if ":" in voice else voice
        if not required and model not in available:
            skipped += 1
            print(f"\n[{voice}  speed={speed}]\n  -  SKIPPED (engine '{model}' not available)")
            return

        print(f"\n[{voice}  speed={speed}]")
        res = await test_case(ws_url, voice, speed, text)
        if res["success"]:
            passed += 1
            print(f"  ✓  PCM={res['pcm_bytes']} bytes")
        else:
            failed += 1
            print(f"  ✗  {res['error']}")

    # Baseline (protocol) tests — must pass
    print("\n--- Protocol baseline (tone engine) ---")
    for voice, speed, text in baseline_cases:
        await run_one(voice, speed, text, required=True)

    # Extra routing/format tests — skip if engine missing
    print("\n--- Routing/format (skip if engine not installed) ---")
    for voice, speed, text in extra_cases:
        await run_one(voice, speed, text, required=False)

    print("\n" + "=" * 50)
    print(f"Passed: {passed} / Skipped: {skipped} / Failed: {failed} / Total: {len(baseline_cases) + len(extra_cases)}")
    return 0 if failed == 0 else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test local_tts WebSocket")
    parser.add_argument("--url", default="ws://127.0.0.1:50000/v1/audio/speech/stream")
    parser.add_argument("--http", default="http://127.0.0.1:50000", help="HTTP base URL for /health")
    args = parser.parse_args()

    exit_code = asyncio.run(run_all(args.url, args.http))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
