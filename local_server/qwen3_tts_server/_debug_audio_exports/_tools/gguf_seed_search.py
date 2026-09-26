"""Temporary GGUF seed search for voice-similarity diagnosis. Do not merge."""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf


SERVER_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SERVER_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from inference import TTSEngine, TTSConfig
from inference.utils.audio import load_audio


ROOT = Path(__file__).resolve().parents[4]
EXPORTS = ROOT / "local_server" / "qwen3_tts_server" / "_debug_audio_exports"
MODEL_DIR = (
    ROOT
    / "modelsprepare"
    / "Qwen3-TTS-GGUF"
    / "model"
    / "Qwen3-TTS-12Hz-1.7B-Base-GGUF"
)
ANCHOR_JSON = (
    Path(os.environ["TEMP"])
    / "neko_qwen3_tts_voice_cache"
    / "02232bf9aa235002.json"
)
TARGET_TEXT = "好的主人。我现在开始说。"


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right))
    )


def main() -> None:
    output_dir = EXPORTS / "_gguf_seed_search"
    output_dir.mkdir(parents=True, exist_ok=True)
    sessions = sorted(
        [
            path
            for path in EXPORTS.iterdir()
            if path.is_dir() and list(path.glob("reference_audio.*"))
        ],
        key=lambda path: path.stat().st_mtime,
    )
    reference = next(sessions[-1].glob("reference_audio.*"))
    engine = TTSEngine(
        model_dir=str(MODEL_DIR), onnx_provider="DML", verbose=False
    )
    reference_embedding = engine.speaker_encoder.encode_audio(load_audio(reference))
    rows = []
    try:
        for seed in range(24):
            stream = engine.create_stream()
            started_at = time.perf_counter()
            try:
                if not stream.set_voice(str(ANCHOR_JSON)):
                    raise RuntimeError("failed to load reference anchor")
                result = stream.clone(
                    TARGET_TEXT,
                    language="chinese",
                    zero_shot=False,
                    config=TTSConfig(
                        streaming=False,
                        max_steps=160,
                        do_sample=True,
                        temperature=0.6,
                        top_p=1.0,
                        top_k=50,
                        sub_do_sample=False,
                        sub_temperature=0.0,
                        sub_top_p=1.0,
                        sub_top_k=0,
                        seed=seed,
                        sub_seed=45,
                    ),
                )
                if result is None or result.audio is None:
                    raise RuntimeError("clone returned no audio")
                output = output_dir / f"seed_{seed:02d}.wav"
                sf.write(output, np.asarray(result.audio, dtype=np.float32), 24000)
                generated_embedding = engine.speaker_encoder.encode_audio(
                    load_audio(output)
                )
                row = {
                    "seed": seed,
                    "similarity": cosine(reference_embedding, generated_embedding),
                    "duration": len(result.audio) / 24000,
                    "elapsed": time.perf_counter() - started_at,
                    "output": str(output),
                }
                rows.append(row)
                print(
                    f"SEED {seed:02d} similarity={row['similarity']:.6f} "
                    f"duration={row['duration']:.3f}s",
                    flush=True,
                )
            finally:
                stream.shutdown()
    finally:
        engine.shutdown()

    rows.sort(key=lambda row: row["similarity"], reverse=True)
    (output_dir / "results.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("TOP5", flush=True)
    for row in rows[:5]:
        print(
            f"seed={row['seed']:02d} similarity={row['similarity']:.6f} "
            f"output={row['output']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
