import asyncio
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
for console_stream in (sys.stdout, sys.stderr):
    if hasattr(console_stream, "reconfigure"):
        console_stream.reconfigure(encoding="utf-8", errors="replace")

from inference import TTSConfig
from server.clone_sampling import clone_sampling_options
from server.config_loader import load_config
from server.engine_pool import EnginePool


def cosine(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.dot(first, second) / (np.linalg.norm(first) * np.linalg.norm(second)))


async def main() -> None:
    output_dir = Path(__file__).resolve().parent
    anchor_path = output_dir / "client_voice_anchor.json"
    pool = EnginePool(load_config())

    try:
        await pool.initialize()
        stream = pool.engine.create_stream()
        try:
            loaded = stream.set_voice(anchor_path)
            if not loaded:
                raise RuntimeError("voice anchor load failed")
            reference_embedding = stream.voice.spk_emb.copy()
            sampling = clone_sampling_options()
            output_embeddings = []
            texts = [
                "今天天气很好，我们出去走走吧。",
                "回来以后，再一起喝一杯热茶。",
            ]

            for index, text in enumerate(texts, start=1):
                if index > 1:
                    stream.reset(preserve_voice=True)
                started = time.perf_counter()
                result = stream.clone(
                    text,
                    "chinese",
                    config=TTSConfig(
                        max_steps=160,
                        streaming=False,
                        **sampling,
                    ),
                )
                if result is None or result.audio is None:
                    raise RuntimeError(f"sentence {index} synthesis failed")
                output_wav = output_dir / f"stable_sentence_{index}.wav"
                result.save_wav(str(output_wav))
                embedding = pool.engine.speaker_encoder.encode(result.audio)
                output_embeddings.append(embedding)
                print(
                    f"[STABILITY TEST] sentence={index} audio={result.duration:.2f}s "
                    f"elapsed={time.perf_counter() - started:.2f}s "
                    f"reference_similarity={cosine(reference_embedding, embedding):.6f}",
                    flush=True,
                )

            print(
                "[STABILITY TEST] sentence_pair_similarity="
                f"{cosine(output_embeddings[0], output_embeddings[1]):.6f}",
                flush=True,
            )
        finally:
            stream.shutdown()
    finally:
        await pool.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
