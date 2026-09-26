import asyncio
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
for console_stream in (sys.stdout, sys.stderr):
    if hasattr(console_stream, "reconfigure"):
        console_stream.reconfigure(encoding="utf-8", errors="replace")

from inference import TTSConfig
from server.config_loader import load_config
from server.engine_pool import EnginePool


async def main() -> None:
    output_dir = Path(__file__).resolve().parent
    anchor_path = output_dir / "client_voice_anchor.json"
    output_wav = output_dir / "anchor_only_synthesis.wav"
    pool = EnginePool(load_config())

    try:
        await pool.initialize()
        stream = pool.engine.create_stream()
        try:
            started = time.perf_counter()
            loaded = stream.set_voice(anchor_path)
            load_seconds = time.perf_counter() - started
            print(
                f"[ANCHOR TEST] loaded={bool(loaded)} anchor={anchor_path} "
                f"load={load_seconds:.3f}s",
                flush=True,
            )
            if not loaded:
                raise RuntimeError("voice anchor load failed")

            started = time.perf_counter()
            result = stream.clone(
                "这是只读取音色文件生成的测试语音。",
                "chinese",
                config=TTSConfig(max_steps=160, streaming=False),
            )
            synth_seconds = time.perf_counter() - started
            if result is None or result.audio is None:
                raise RuntimeError("anchor-only synthesis failed")

            result.save_wav(str(output_wav))
            print(
                f"[ANCHOR TEST] success wav={output_wav} "
                f"audio={result.duration:.2f}s synthesis={synth_seconds:.2f}s",
                flush=True,
            )
        finally:
            stream.shutdown()
    finally:
        await pool.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
