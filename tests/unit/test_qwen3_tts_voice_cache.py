import base64
from pathlib import Path

import pytest

from local_server.qwen3_tts_server.server.voice_cache import VoiceCache


def _data_uri(payload: bytes) -> str:
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:audio/wav;base64,{encoded}"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_inline_reference_audio_uses_stable_content_addressed_path(tmp_path):
    cache = VoiceCache(str(tmp_path))
    payload = b"RIFF-identical-reference-audio"

    first = await cache.resolve_ref_audio(_data_uri(payload))
    second = await cache.resolve_ref_audio(_data_uri(payload))

    assert first == second
    assert Path(first).read_bytes() == payload
    assert len(list(cache.reference_audio_dir.glob("*"))) == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_different_reference_audio_uses_different_cache_identity(tmp_path):
    cache = VoiceCache(str(tmp_path))

    first = await cache.resolve_ref_audio(_data_uri(b"sample-one"))
    second = await cache.resolve_ref_audio(_data_uri(b"sample-two"))

    assert first != second
    assert cache._cache_key(first, "same transcript") != cache._cache_key(
        second, "same transcript"
    )


@pytest.mark.unit
def test_clone_voice_registry_persists_display_name_and_can_delete(tmp_path):
    cache = VoiceCache(str(tmp_path))
    clone_id = "qwen3-tts-gguf-clone-ch-abc"

    voices = cache.register_clone_voice(clone_id, "我的克隆音色", "d885c4a1f897")

    assert voices == [{
        "voice_id": clone_id,
        "name": "我的克隆音色",
        "type": "clone",
        "audio_key": "d885c4a1f897",
        "updated_at": voices[0]["updated_at"],
    }]
    reloaded = VoiceCache(str(tmp_path))
    assert reloaded.list_registered_voices()[0]["name"] == "我的克隆音色"
    assert reloaded.remove_registered_voice(clone_id) is True
    assert VoiceCache(str(tmp_path)).list_registered_voices() == []
