"""音色锚点缓存 — 缓存 voice anchor JSON，避免重复的 set_voice 开销"""
import base64
import hashlib
import json
import tempfile
import os
import threading
import time
from pathlib import Path
from typing import Optional


class VoiceCache:
    """缓存 voice anchor (TTSResult.json)，避免重复的 set_voice 开销。

    ref_audio + ref_text → cache_key → .json 文件路径
    如果命中缓存，set_voice 直接从 JSON 加载，跳过编码器提取（~5ms vs ~200ms）。
    """

    def __init__(self, cache_dir: str = "/tmp/qwen3_tts_voice_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.reference_audio_dir = self.cache_dir / "reference_audio"
        self.reference_audio_dir.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.cache_dir / "voices.json"
        self._registry_lock = threading.Lock()
        self._registered_voices = self._load_voice_registry()

    def _cache_key(self, ref_audio_path: str, ref_text: str) -> str:
        content = f"{ref_audio_path}:{ref_text}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def get(self, ref_audio_path: str, ref_text: str) -> Optional[str]:
        """返回缓存的 JSON 路径，未命中返回 None"""
        key = self._cache_key(ref_audio_path, ref_text)
        json_path = self.cache_dir / f"{key}.json"
        if json_path.exists():
            return str(json_path)
        return None

    def put(self, ref_audio_path: str, ref_text: str, voice_result):
        """缓存 voice anchor JSON"""
        key = self._cache_key(ref_audio_path, ref_text)
        json_path = self.cache_dir / f"{key}.json"
        try:
            voice_result.save_json(str(json_path), light=True)
        except Exception as e:
            print(f"⚠️ [VoiceCache] 缓存写入失败: {e}")

    def list_registered_voices(self) -> list[dict]:
        """Return persisted clone identities without reference text or audio."""
        with self._registry_lock:
            return [dict(entry) for entry in self._registered_voices.values()]

    def register_clone_voice(self, voice_id: str, name: str, audio_key: str) -> list[dict]:
        """Persist the N.E.K.O. clone identity associated with a prepared anchor."""
        normalized_id = str(voice_id or "").strip()[:256]
        if not normalized_id or normalized_id.casefold() == "default":
            return self.list_registered_voices()
        normalized_name = str(name or normalized_id).strip()[:128] or normalized_id
        entry = {
            "voice_id": normalized_id,
            "name": normalized_name,
            "type": "clone",
            "audio_key": str(audio_key or "").strip()[:64],
            "updated_at": int(time.time()),
        }
        with self._registry_lock:
            self._registered_voices[normalized_id] = entry
            self._save_voice_registry_locked()
            return [dict(item) for item in self._registered_voices.values()]

    def remove_registered_voice(self, voice_id: str) -> bool:
        normalized_id = str(voice_id or "").strip()
        with self._registry_lock:
            removed = self._registered_voices.pop(normalized_id, None) is not None
            if removed:
                self._save_voice_registry_locked()
            return removed

    def _load_voice_registry(self) -> dict[str, dict]:
        if not self.registry_path.exists():
            return {}
        try:
            raw = json.loads(self.registry_path.read_text(encoding="utf-8"))
            voices = raw.get("voices", {}) if isinstance(raw, dict) else {}
            return {
                str(voice_id): dict(entry)
                for voice_id, entry in voices.items()
                if isinstance(entry, dict)
            }
        except Exception as exc:
            print(f"⚠️ [VoiceCache] 音色注册表读取失败: {exc}", flush=True)
            return {}

    def _save_voice_registry_locked(self) -> None:
        payload = {"version": 1, "voices": self._registered_voices}
        tmp_path = self.registry_path.with_suffix(".json.tmp")
        tmp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp_path, self.registry_path)

    async def resolve_ref_audio(self, ref_audio: str) -> str:
        """解析 ref_audio 来源，返回本地文件路径。

        支持:
        - data:audio/...;base64,... → 解码到临时文件
        - http(s)://... → 下载到临时文件
        - file:///... → 直接使用
        - 其他 → 当作本地路径
        """
        if ref_audio.startswith("data:"):
            return await self._resolve_data_url(ref_audio)
        elif ref_audio.startswith("http://") or ref_audio.startswith("https://"):
            return await self._resolve_http_url(ref_audio)
        elif ref_audio.startswith("file://"):
            return ref_audio[7:]
        else:
            return ref_audio

    async def _resolve_data_url(self, data_url: str) -> str:
        """解析 data URL，并按内容寻址落盘以复用跨 WebSocket 的音色缓存。"""
        header, encoded = data_url.split(",", 1)
        audio_bytes = base64.b64decode(encoded, validate=True)

        # 提取扩展名
        ext = "wav"
        if "mp3" in header:
            ext = "mp3"
        elif "ogg" in header:
            ext = "ogg"
        elif "flac" in header:
            ext = "flac"
        elif "m4a" in header:
            ext = "m4a"

        return self._materialize_reference_audio(audio_bytes, f".{ext}")

    async def _resolve_http_url(self, url: str) -> str:
        """下载 HTTP URL 到临时文件"""
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                resp.raise_for_status()
                audio_bytes = await resp.read()

        ext = Path(url.split("?")[0]).suffix or ".wav"
        return self._materialize_reference_audio(audio_bytes, ext)

    def _materialize_reference_audio(self, audio_bytes: bytes, suffix: str) -> str:
        """Persist one reference sample at a deterministic, content-addressed path.

        ``VoiceCache._cache_key`` includes the reference path. A random temporary
        filename therefore made identical inline samples miss the anchor cache on
        every new WebSocket. The digest path keeps the identity stable across
        requests and also prevents unbounded duplicate temp files.
        """
        digest = hashlib.sha256(audio_bytes).hexdigest()
        safe_suffix = suffix if suffix.startswith(".") else f".{suffix}"
        target = self.reference_audio_dir / f"{digest}{safe_suffix.lower()}"
        if target.exists():
            return str(target)

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                suffix=".tmp",
                prefix=f"{digest}.",
                dir=self.reference_audio_dir,
                delete=False,
            ) as tmp:
                tmp.write(audio_bytes)
                tmp.flush()
                os.fsync(tmp.fileno())
                tmp_path = Path(tmp.name)
            os.replace(tmp_path, target)
        finally:
            if tmp_path is not None and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
        return str(target)
