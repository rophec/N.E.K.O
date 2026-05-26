# -*- coding: utf-8 -*-
"""Forged-card inventory storage abstraction.

当前阶段 NEKO 还没有部署用户云端账户服务器，所以铸造卡仓库以本地 JSON 文件
落盘（每个 character 一份 ``forged_cards.json``，放在 NEKO 记忆目录下角色子目
录里，跟 ``facts.json`` 同级）。

将来云端搭好之后，只需要新增一个实现同 ``InventoryStorage`` Protocol 的
``CloudStorage`` 类，把模块底部的 ``STORAGE`` 单例替换掉，server.py、HTTP
接口契约和前端代码都不动。
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Protocol

from active_neko_context import safe_character_segment

logger = logging.getLogger("forge_server.inventory")


INVENTORY_FILENAME = "forged_cards.json"


class InventoryStorageError(RuntimeError):
    """Raised when the inventory backend cannot satisfy a request."""


class InventoryStorage(Protocol):
    def list(self, character: str) -> list[dict[str, Any]]: ...
    def add(self, character: str, card: dict[str, Any]) -> dict[str, Any]: ...
    def delete(self, character: str, card_id: str) -> bool: ...


def _resolve_memory_dir() -> Path | None:
    env_memory_dir = os.environ.get("NEKO_MEMORY_DIR", "").strip()
    if env_memory_dir:
        return Path(env_memory_dir)

    try:
        from utils.config_manager import get_config_manager

        memory_dir = getattr(get_config_manager(), "memory_dir", None)
        return Path(memory_dir) if memory_dir else None
    except Exception as exc:
        logger.warning("inventory: failed to resolve memory_dir: %s", type(exc).__name__)
        return None


class LocalFileStorage:
    """Filesystem-backed inventory: one ``forged_cards.json`` per character.

    Writes are serialized per-character and atomic (write to temp file, rename).
    """

    def __init__(self) -> None:
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    def _lock_for(self, character: str) -> threading.Lock:
        with self._locks_guard:
            lock = self._locks.get(character)
            if lock is None:
                lock = threading.Lock()
                self._locks[character] = lock
            return lock

    def _resolve_path(self, character: str) -> Path:
        safe = safe_character_segment(character)
        if not safe:
            raise InventoryStorageError("invalid_character")
        memory_dir = _resolve_memory_dir()
        if memory_dir is None:
            raise InventoryStorageError("memory_dir_unresolved")
        return memory_dir / safe / INVENTORY_FILENAME

    def _read(self, path: Path) -> list[dict[str, Any]]:
        if not path.is_file():
            return []
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("inventory: failed to read %s: %s", path, type(exc).__name__)
            return []
        if not isinstance(data, list):
            return []
        return [item for item in data if isinstance(item, dict)]

    def _write(self, path: Path, cards: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write: temp file in the same directory, then os.replace.
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(path.parent),
            prefix=path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            json.dump(cards, tmp, ensure_ascii=False, indent=2)
            tmp_path = tmp.name
        os.replace(tmp_path, path)

    def list(self, character: str) -> list[dict[str, Any]]:
        path = self._resolve_path(character)
        with self._lock_for(character):
            return self._read(path)

    def add(self, character: str, card: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(card, dict) or not card.get("id"):
            raise InventoryStorageError("card_missing_id")
        path = self._resolve_path(character)
        with self._lock_for(character):
            cards = self._read(path)
            # 同 id 视为覆盖更新（例如前端 retry）
            cards = [c for c in cards if c.get("id") != card["id"]]
            cards.append(card)
            self._write(path, cards)
            return card

    def delete(self, character: str, card_id: str) -> bool:
        if not card_id:
            return False
        path = self._resolve_path(character)
        with self._lock_for(character):
            cards = self._read(path)
            remaining = [c for c in cards if c.get("id") != card_id]
            if len(remaining) == len(cards):
                return False
            self._write(path, remaining)
            return True


# 模块级单例。将来云端化只需要把这一行换成 ``STORAGE = CloudStorage(...)``。
STORAGE: InventoryStorage = LocalFileStorage()
