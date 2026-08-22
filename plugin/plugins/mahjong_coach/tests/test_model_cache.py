from __future__ import annotations

import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from plugin.plugins.mahjong_coach.perception import tile_classifier_dispatch as dispatch
from plugin.plugins.mahjong_coach.perception import vit_tile_classifier_onnx as vit


def test_vit_failure_and_session_caches_follow_model_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_dir = tmp_path / "vit-model"
    sessions: list[object] = []

    class FakeSession:
        def get_inputs(self) -> list[object]:
            return [SimpleNamespace(name="input")]

    fake_ort = SimpleNamespace(
        get_available_providers=lambda: ["CPUExecutionProvider"],
        InferenceSession=lambda _path, providers: sessions.append(FakeSession()) or sessions[-1],
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    vit._MODEL_CACHE.clear()
    vit._MODEL_FAILURES.clear()

    assert dispatch.onnx_discard_available(model_dir=model_dir) is False

    model_dir.mkdir()
    model_path = model_dir / "model.onnx"
    model_path.write_bytes(b"model-a")
    (model_dir / "preprocessor.json").write_text('{"size":8}', encoding="utf-8")
    (model_dir / "labels.json").write_text('{"0":"1m"}', encoding="utf-8")
    original_stat = model_path.stat()

    assert dispatch.onnx_discard_available(model_dir=model_dir) is True
    first = vit._load_model(model_dir.resolve())
    assert len(sessions) == 1

    model_path.write_bytes(b"model-b")
    os.utime(model_path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    second = vit._load_model(model_dir.resolve())

    assert second is not first
    assert len(sessions) == 2
    assert len(vit._MODEL_CACHE) == 1
    assert not vit._MODEL_FAILURES

    vit._MODEL_CACHE.clear()
    vit._MODEL_FAILURES.clear()
