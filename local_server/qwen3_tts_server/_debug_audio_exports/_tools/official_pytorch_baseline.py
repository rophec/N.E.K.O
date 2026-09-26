"""Temporary official PyTorch voice-clone quality baseline. Do not merge."""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


ROOT = Path(__file__).resolve().parents[4]
EXPORTS = ROOT / "local_server" / "qwen3_tts_server" / "_debug_audio_exports"
QWEN_SOURCE = ROOT / "modelsprepare" / "Qwen3-TTS-GGUF" / "Qwen3-TTS-main"
MODEL_DIR = (
    ROOT
    / "modelsprepare"
    / "Qwen3-TTS-GGUF"
    / "source_models"
    / "Qwen3-TTS-12Hz-1.7B-Base"
)

sys.path.insert(0, str(QWEN_SOURCE))
from qwen_tts import Qwen3TTSModel


def main() -> None:
    sessions = sorted(
        [
            path
            for path in EXPORTS.iterdir()
            if path.is_dir() and list(path.glob("reference_audio.*"))
        ],
        key=lambda path: path.stat().st_mtime,
    )
    reference = next(sessions[-1].glob("reference_audio.*"))
    cache = (
        Path(os.environ["TEMP"])
        / "neko_qwen3_tts_voice_cache"
        / "02232bf9aa235002.json"
    )
    ref_text = json.loads(cache.read_text(encoding="utf-8"))["text"]
    target_text = "好的主人。我现在开始说。"
    output_dir = EXPORTS / "_official_pytorch_baseline"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"MODEL {MODEL_DIR}", flush=True)
    print(f"REFERENCE {reference}", flush=True)
    print(f"TARGET {target_text}", flush=True)
    started_at = time.perf_counter()
    model = Qwen3TTSModel.from_pretrained(
        str(MODEL_DIR),
        device_map="cpu",
        dtype=torch.bfloat16,
    )
    loaded_at = time.perf_counter()
    print(f"LOAD_SECONDS {loaded_at - started_at:.3f}", flush=True)

    wavs, sample_rate = model.generate_voice_clone(
        text=target_text,
        language="Chinese",
        ref_audio=str(reference),
        ref_text=ref_text,
        x_vector_only_mode=False,
        do_sample=True,
        top_k=50,
        top_p=1.0,
        temperature=0.9,
        subtalker_dosample=True,
        subtalker_top_k=50,
        subtalker_top_p=1.0,
        subtalker_temperature=0.9,
        max_new_tokens=120,
    )
    completed_at = time.perf_counter()
    output = output_dir / "official_fp_bf16_temp_0.9.wav"
    sf.write(output, np.asarray(wavs[0], dtype=np.float32), sample_rate)
    metadata = {
        "model": str(MODEL_DIR),
        "reference": str(reference),
        "ref_text": ref_text,
        "text": target_text,
        "sample_rate": int(sample_rate),
        "load_seconds": loaded_at - started_at,
        "inference_seconds": completed_at - loaded_at,
        "temperature": 0.9,
        "subtalker_temperature": 0.9,
        "output": str(output),
    }
    (output_dir / "official_fp_bf16_temp_0.9.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"INFERENCE_SECONDS {metadata['inference_seconds']:.3f}", flush=True)
    print(f"OUTPUT {output}", flush=True)


if __name__ == "__main__":
    main()
