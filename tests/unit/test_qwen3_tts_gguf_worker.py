from functools import partial
from types import SimpleNamespace

import pytest

from main_logic import tts_client
from main_logic.tts_client.workers import qwen3_tts_gguf as qwen_worker
from utils.tts import provider_registry


class _ConfigManager:
    def __init__(self, raw):
        self.raw = raw

    def load_json_config(self, _name, _default):
        return self.raw


def _context(raw, *, enabled=True):
    return SimpleNamespace(
        core_config={"ENABLE_CUSTOM_API": enabled},
        cm=_ConfigManager(raw),
    )


def test_qwen3_tts_gguf_provider_is_preset_only_and_has_own_module():
    provider = provider_registry.get("qwen3_tts_gguf")

    assert provider is not None
    assert provider.capabilities == frozenset({"preset"})
    assert provider.priority < provider_registry.get("vllm_omni").priority
    assert provider.default_url == "ws://127.0.0.1:8091/v1"
    assert provider.default_model == "Qwen3-TTS"
    assert provider.default_voice == "default"
    assert provider.editable_endpoint is True
    assert provider.probe_sub_type == "qwen3_tts_gguf"
    assert tts_client.qwen3_tts_gguf_tts_worker.__module__.endswith("qwen3_tts_gguf")


def test_qwen3_tts_gguf_selection_requires_explicit_provider_and_custom_api():
    selected = _context({"ttsModelProvider": "qwen3_tts_gguf"})
    legacy = _context({"ttsModelProvider": "vllm_omni"})
    disabled = _context({"ttsModelProvider": "qwen3_tts_gguf"}, enabled=False)

    assert qwen_worker._qwen3_tts_gguf_is_selected(selected) is True
    assert qwen_worker._qwen3_tts_gguf_is_selected(legacy) is False
    assert qwen_worker._qwen3_tts_gguf_is_selected(disabled) is False


def test_qwen3_tts_gguf_resolve_binds_customvoice_speaker_without_clone_fields():
    ctx = _context({
        "ttsModelProvider": "qwen3_tts_gguf",
        "ttsModelUrl": "ws://server:8091/v1",
        "ttsModelId": "trained-checkpoint",
        "ttsVoiceId": "hutao",
        "ttsModelApiKey": "local-key",
    })

    worker, api_key, provider_key = qwen_worker._qwen3_tts_gguf_resolve(ctx)

    assert isinstance(worker, partial)
    assert worker.func is qwen_worker.qwen3_tts_gguf_tts_worker
    assert worker.keywords == {
        "base_url": "ws://server:8091/v1",
        "model": "trained-checkpoint",
        "voice": "hutao",
    }
    assert "ref_audio" not in worker.keywords
    assert "ref_text" not in worker.keywords
    assert api_key == "local-key"
    assert provider_key == "qwen3_tts_gguf"


def test_qwen3_tts_gguf_worker_forwards_only_preset_speaker(monkeypatch):
    captured = {}

    def fake_transport(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return "ok"

    monkeypatch.setattr(qwen_worker, "vllm_omni_tts_worker", fake_transport)

    result = qwen_worker.qwen3_tts_gguf_tts_worker(
        "requests",
        "responses",
        "key",
        "character-voice",
        base_url="ws://server:8091/v1",
        model="trained-checkpoint",
        voice="hutao",
    )

    assert result == "ok"
    assert captured["args"] == ("requests", "responses", "key", "character-voice")
    assert captured["kwargs"] == {
        "base_url": "ws://server:8091/v1",
        "model": "trained-checkpoint",
        "voice": "hutao",
        "provider_key": "qwen3_tts_gguf",
        "prefer_bound_voice": True,
    }


def test_qwen3_tts_gguf_defaults_match_local_sidecar():
    ctx = _context({"ttsModelProvider": "qwen3_tts_gguf"})

    worker, api_key, provider_key = qwen_worker._qwen3_tts_gguf_resolve(ctx)

    assert worker.keywords["base_url"] == "ws://127.0.0.1:8091/v1"
    assert worker.keywords["model"] == "Qwen3-TTS"
    assert worker.keywords["voice"] == "default"
    assert api_key == ""
    assert provider_key == "qwen3_tts_gguf"


def test_get_tts_worker_routes_explicit_qwen3_tts_gguf_without_clone_lookup(monkeypatch):
    class _CM:
        def get_core_config(self):
            return {
                "assistApi": "mimo",
                "TTS_PROVIDER": "",
                "ENABLE_CUSTOM_API": True,
                "GPTSOVITS_ENABLED": False,
            }

        def load_json_config(self, filename, default):
            assert filename == "core_config.json"
            return {
                "ttsModelProvider": "qwen3_tts_gguf",
                "ttsModelUrl": "ws://127.0.0.1:8091/v1",
                "ttsModelId": "trained-checkpoint",
                "ttsVoiceId": "hutao",
                "ttsModelApiKey": "",
            }

        def get_model_api_config(self, model_type):
            pytest.fail("explicit Qwen3-TTS GGUF must bypass fallback TTS config")

        def get_tts_api_key(self, provider):
            pytest.fail("explicit Qwen3-TTS GGUF must bypass assist provider")

    monkeypatch.setattr(tts_client, "get_config_manager", lambda: _CM())
    monkeypatch.setattr(
        tts_client,
        "_get_voice_meta",
        lambda _voice_id: pytest.fail("CustomVoice preset must not load clone metadata"),
    )

    worker, api_key, provider_key = tts_client.get_tts_worker(
        core_api_type="qwen",
        has_custom_voice=True,
        voice_id="character-voice-id",
    )

    assert worker.func is qwen_worker.qwen3_tts_gguf_tts_worker
    assert worker.keywords["voice"] == "hutao"
    assert api_key == ""
    assert provider_key == "qwen3_tts_gguf"
