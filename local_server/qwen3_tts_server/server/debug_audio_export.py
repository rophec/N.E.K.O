"""Temporary exact-audio export probe for Qwen3-TTS WebSocket debugging.

This module is diagnostic-only and must be removed before the feature is merged.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import wave
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np


DEBUG_EXPORT_ROOT = Path(__file__).resolve().parents[1] / "_debug_audio_exports"


class DebugAudioExporter:
    """Persist the exact PCM frames sent to a WebSocket client."""

    def __init__(self) -> None:
        session_id = f"{datetime.now():%Y%m%d_%H%M%S_%f}_{uuid4().hex[:8]}"
        self.session_dir = DEBUG_EXPORT_ROOT / session_id
        self.session_dir.mkdir(parents=True, exist_ok=False)
        print(
            f"[Qwen3-TTS DEBUG EXPORT][session] path={self.session_dir}",
            flush=True,
        )

    def write_session(
        self,
        metadata: dict,
        *,
        ref_audio_path: str | None,
        speaker_embedding: np.ndarray | None,
    ) -> None:
        payload = dict(metadata)
        if speaker_embedding is not None:
            embedding = np.asarray(speaker_embedding, dtype=np.float32)
            payload["speaker_embedding"] = {
                "shape": list(embedding.shape),
                "dtype": str(embedding.dtype),
                "sha256": hashlib.sha256(embedding.tobytes()).hexdigest(),
            }
        else:
            payload["speaker_embedding"] = None

        if ref_audio_path:
            source = Path(ref_audio_path)
            if source.is_file():
                reference_name = f"reference_audio{source.suffix or '.bin'}"
                shutil.copyfile(source, self.session_dir / reference_name)
                payload["reference_audio_export"] = reference_name

        self._write_json("session.json", payload)

    def write_sentence(
        self,
        *,
        sentence_index: int,
        text: str,
        pcm_bytes: bytes,
        sample_rate: int,
        metadata: dict,
    ) -> Path:
        stem = f"sentence_{sentence_index:03d}"
        wav_path = self.session_dir / f"{stem}.wav"
        with wave.open(str(wav_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm_bytes)

        payload = {
            "text": text,
            "sample_rate": sample_rate,
            "pcm_bytes": len(pcm_bytes),
            "wav_sha256": hashlib.sha256(wav_path.read_bytes()).hexdigest(),
            **metadata,
        }
        self._write_json(f"{stem}.json", payload)
        print(
            f"[Qwen3-TTS DEBUG EXPORT][sentence] index={sentence_index} "
            f"bytes={len(pcm_bytes)} path={wav_path}",
            flush=True,
        )
        return wav_path

    def _write_json(self, filename: str, payload: dict) -> None:
        (self.session_dir / filename).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
