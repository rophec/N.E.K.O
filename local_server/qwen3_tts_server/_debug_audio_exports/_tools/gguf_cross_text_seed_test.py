"""Temporary cross-text seed robustness test. Do not merge."""

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
SEEDS = [11, 9, 14, 21, 5, 42]
TEXTS = [
    "好的主人。我现在开始说。",
    "星沉。月升。",
    "星垂。月隐。",
    "主人，我在。我现在就按你要求的，说两句长一点的话。",
    "我会按你说的每句之间停两秒，第一句说完就立刻准备第二句，不会断也不会卡。只要你觉得这样的节奏和长度可以，我之后都能保持这个状态。",
    "我还能根据你喜欢的风格调整内容，比如多加点和星星月亮相关的描述。",
]


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right))
    )


def main() -> None:
    output_dir = EXPORTS / "_gguf_cross_text_seed_test"
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
        for seed in SEEDS:
            similarities = []
            for text_index, text in enumerate(TEXTS):
                stream = engine.create_stream()
                started_at = time.perf_counter()
                try:
                    if not stream.set_voice(str(ANCHOR_JSON)):
                        raise RuntimeError("failed to load reference anchor")
                    result = stream.clone(
                        text,
                        language="chinese",
                        zero_shot=False,
                        config=TTSConfig(
                            streaming=False,
                            max_steps=300,
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
                    output = output_dir / f"seed_{seed:02d}_text_{text_index}.wav"
                    sf.write(output, np.asarray(result.audio, dtype=np.float32), 24000)
                    generated_embedding = engine.speaker_encoder.encode_audio(
                        load_audio(output)
                    )
                    similarity = cosine(reference_embedding, generated_embedding)
                    similarities.append(similarity)
                    print(
                        f"SEED {seed:02d} TEXT {text_index} similarity={similarity:.6f} "
                        f"elapsed={time.perf_counter() - started_at:.3f}s",
                        flush=True,
                    )
                finally:
                    stream.shutdown()
            rows.append(
                {
                    "seed": seed,
                    "minimum": min(similarities),
                    "mean": float(np.mean(similarities)),
                    "median": float(np.median(similarities)),
                    "maximum": max(similarities),
                    "similarities": similarities,
                }
            )
    finally:
        engine.shutdown()

    rows.sort(key=lambda row: (row["minimum"], row["mean"]), reverse=True)
    (output_dir / "results.json").write_text(
        json.dumps({"texts": TEXTS, "results": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("RANKING", flush=True)
    for row in rows:
        print(
            f"seed={row['seed']:02d} min={row['minimum']:.6f} "
            f"mean={row['mean']:.6f} median={row['median']:.6f} "
            f"max={row['maximum']:.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
