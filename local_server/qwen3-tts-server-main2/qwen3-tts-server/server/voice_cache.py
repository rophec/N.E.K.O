"""音色锚点缓存 — 缓存 voice anchor JSON，避免重复的 set_voice 开销"""
import asyncio
import base64
import hashlib
import tempfile
import os
from pathlib import Path
from typing import Optional

import numpy as np


class VoiceCache:
    """缓存 voice anchor (TTSResult.json)，避免重复的 set_voice 开销。

    ref_audio + ref_text → cache_key → .json 文件路径
    如果命中缓存，set_voice 直接从 JSON 加载，跳过编码器提取（~5ms vs ~200ms）。
    """

    def __init__(self, cache_dir: str = ""):
        self.cache_dir = Path(cache_dir) if cache_dir else Path(tempfile.gettempdir()) / "qwen3_tts_voice_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

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
        """解析 data URL"""
        header, encoded = data_url.split(",", 1)
        audio_bytes = base64.b64decode(encoded)

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

        tmp = tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False)
        tmp.write(audio_bytes)
        tmp.close()
        return tmp.name

    async def _resolve_http_url(self, url: str) -> str:
        """下载 HTTP URL 到临时文件"""
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                resp.raise_for_status()
                audio_bytes = await resp.read()

        ext = Path(url.split("?")[0]).suffix or ".wav"
        tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
        tmp.write(audio_bytes)
        tmp.close()
        return tmp.name
