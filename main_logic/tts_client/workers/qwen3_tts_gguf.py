# Copyright 2025-2026 Project N.E.K.O. Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Qwen3-TTS GGUF CustomVoice provider.

This provider intentionally supports trained, preset speaker names only.  It
does not route reference audio or Base-model ICL voice-clone metadata.
"""

from functools import partial

from utils.config_manager import _as_bool

from .vllm_omni import (
    _vllm_omni_normalize_ws_endpoint,
    vllm_omni_tts_worker,
)


QWEN3_TTS_GGUF_PROVIDER_KEY = "qwen3_tts_gguf"
QWEN3_TTS_GGUF_DEFAULT_BASE_URL = "ws://127.0.0.1:8091/v1"
QWEN3_TTS_GGUF_DEFAULT_MODEL = "Qwen3-TTS"
QWEN3_TTS_GGUF_DEFAULT_VOICE = "default"


def qwen3_tts_gguf_tts_worker(
    request_queue,
    response_queue,
    audio_api_key,
    voice_id,
    *,
    base_url=QWEN3_TTS_GGUF_DEFAULT_BASE_URL,
    model=QWEN3_TTS_GGUF_DEFAULT_MODEL,
    voice=QWEN3_TTS_GGUF_DEFAULT_VOICE,
):
    """Run the shared streaming protocol without any voice-clone fields."""
    return vllm_omni_tts_worker(
        request_queue,
        response_queue,
        audio_api_key,
        voice_id,
        base_url=base_url,
        model=model,
        voice=voice,
        provider_key=QWEN3_TTS_GGUF_PROVIDER_KEY,
        prefer_bound_voice=True,
    )


def _qwen3_tts_gguf_normalize_ws_endpoint(base_url: str) -> str:
    return _vllm_omni_normalize_ws_endpoint(base_url)


def _qwen3_tts_gguf_is_selected(ctx) -> bool:
    if not _as_bool(ctx.core_config.get("ENABLE_CUSTOM_API"), False):
        return False
    try:
        raw = ctx.cm.load_json_config("core_config.json", {})
    except Exception:
        raw = {}
    return str(raw.get("ttsModelProvider") or "").strip() == QWEN3_TTS_GGUF_PROVIDER_KEY


def _qwen3_tts_gguf_resolve(ctx):
    try:
        raw = ctx.cm.load_json_config("core_config.json", {})
    except Exception:
        raw = {}

    base_url = (
        str(raw.get("ttsModelUrl") or "").strip()
        or QWEN3_TTS_GGUF_DEFAULT_BASE_URL
    )
    model = (
        str(raw.get("ttsModelId") or "").strip()
        or QWEN3_TTS_GGUF_DEFAULT_MODEL
    )
    voice = (
        str(raw.get("ttsVoiceId") or "").strip()
        or QWEN3_TTS_GGUF_DEFAULT_VOICE
    )
    api_key = str(raw.get("ttsModelApiKey") or "").strip()
    worker = partial(
        qwen3_tts_gguf_tts_worker,
        base_url=base_url,
        model=model,
        voice=voice,
    )
    return worker, api_key, QWEN3_TTS_GGUF_PROVIDER_KEY
