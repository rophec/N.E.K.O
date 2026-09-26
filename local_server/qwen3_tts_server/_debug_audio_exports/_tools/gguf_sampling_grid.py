"""Temporary GGUF sampling grid for voice-similarity diagnosis. Do not merge."""

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
VOICE_CACHE = Path(os.environ["TEMP"]) / "neko_qwen3_tts_voice_cache"
ANCHOR_JSON = VOICE_CACHE / "02232bf9aa235002.json"
TARGET_TEXT = "好的主人。我现在开始说。"


CASES = [
    {
        "name": "sample_0.9_0.9",
        "do_sample": True,
        "temperature": 0.9,
        "sub_do_sample": True,
        "sub_temperature": 0.9,
    },
    {
        "name": "sample_0.6_0.6",
        "do_sample": True,
        "temperature": 0.6,
        "sub_do_sample": True,
        "sub_temperature": 0.6,
    },
    {
        "name": "sample_0.3_0.3",
        "do_sample": True,
        "temperature": 0.3,
        "sub_do_sample": True,
        "sub_temperature": 0.3,
    },
    {
        "name": "sample_0.6_sub_greedy",
        "do_sample": True,
        "temperature": 0.6,
        "sub_do_sample": False,
        "sub_temperature": 0.0,
    },
    {
        "name": "greedy",
        "do_sample": False,
        "temperature": 0.0,
        "sub_do_sample": False,
        "sub_temperature": 0.0,
    },
]


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right))
    )


def main() -> None:
    output_dir = EXPORTS / "_gguf_sampling_grid"
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

    print(f"MODEL {MODEL_DIR}", flush=True)
    print(f"ANCHOR {ANCHOR_JSON}", flush=True)
    print(f"REFERENCE {reference}", flush=True)
    engine = TTSEngine(
        model_dir=str(MODEL_DIR), onnx_provider="DML", verbose=False
    )
    reference_embedding = engine.speaker_encoder.encode_audio(load_audio(reference))
    results = []
    try:
        for case in CASES:
            stream = engine.create_stream()
            started_at = time.perf_counter()
            try:
                if not stream.set_voice(str(ANCHOR_JSON)):
                    raise RuntimeError("failed to load reference anchor")
                config = TTSConfig(
                    streaming=False,
                    max_steps=160,
                    do_sample=case["do_sample"],
                    temperature=case["temperature"],
                    top_p=1.0,
                    top_k=50,
                    sub_do_sample=case["sub_do_sample"],
                    sub_temperature=case["sub_temperature"],
                    sub_top_p=1.0,
                    sub_top_k=50,
                    seed=42,
                    sub_seed=45,
                )
                result = stream.clone(
                    TARGET_TEXT,
                    language="chinese",
                    zero_shot=False,
                    config=config,
                )
                if result is None or result.audio is None:
                    raise RuntimeError("clone returned no audio")
                output = output_dir / f"{case['name']}.wav"
                sf.write(output, np.asarray(result.audio, dtype=np.float32), 24000)
                generated_embedding = engine.speaker_encoder.encode_audio(
                    load_audio(output)
                )
                elapsed = time.perf_counter() - started_at
                row = {
                    **case,
                    "text": TARGET_TEXT,
                    "similarity": cosine(reference_embedding, generated_embedding),
                    "duration": len(result.audio) / 24000,
                    "elapsed": elapsed,
                    "output": str(output),
                }
                results.append(row)
                print(
                    f"RESULT {case['name']} similarity={row['similarity']:.6f} "
                    f"duration={row['duration']:.3f}s elapsed={elapsed:.3f}s",
                    flush=True,
                )
            finally:
                stream.shutdown()
    finally:
        engine.shutdown()

    (output_dir / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
