import hashlib
import json
import wave

import numpy as np


def test_debug_export_writes_exact_pcm_reference_and_embedding_fingerprint(
    monkeypatch, tmp_path
):
    from local_server.qwen3_tts_server.server import debug_audio_export as module

    monkeypatch.setattr(module, "DEBUG_EXPORT_ROOT", tmp_path / "exports")
    reference = tmp_path / "reference.wav"
    reference.write_bytes(b"reference-audio")
    embedding = np.arange(8, dtype=np.float32)
    pcm = np.array([0, 100, -100, 32767, -32768], dtype=np.int16).tobytes()

    exporter = module.DebugAudioExporter()
    exporter.write_session(
        {"requested_voice": "test-voice", "session_mode": "fixed_speaker"},
        ref_audio_path=str(reference),
        speaker_embedding=embedding,
    )
    wav_path = exporter.write_sentence(
        sentence_index=0,
        text="测试导出。",
        pcm_bytes=pcm,
        sample_rate=24000,
        metadata={"seed": 42},
    )

    assert (exporter.session_dir / "reference_audio.wav").read_bytes() == b"reference-audio"
    session = json.loads((exporter.session_dir / "session.json").read_text("utf-8"))
    assert session["speaker_embedding"]["sha256"] == hashlib.sha256(
        embedding.tobytes()
    ).hexdigest()

    with wave.open(str(wav_path), "rb") as wav_file:
        assert wav_file.getframerate() == 24000
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.readframes(wav_file.getnframes()) == pcm

    sentence = json.loads(
        (exporter.session_dir / "sentence_000.json").read_text("utf-8")
    )
    assert sentence["text"] == "测试导出。"
    assert sentence["pcm_bytes"] == len(pcm)


def test_ws_debug_export_is_explicit_and_marked_for_removal():
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[2]
        / "local_server"
        / "qwen3_tts_server"
        / "server"
        / "routes"
        / "ws_stream.py"
    ).read_text(encoding="utf-8")

    assert 'websocket.query_params.get("debug_export")' in source
    assert 'os.environ.get("QWEN3_TTS_DEBUG_EXPORT")' in source
    assert "TEMP DIAGNOSTIC" in source
    assert "debug_pcm.extend(pcm_bytes)" in source
