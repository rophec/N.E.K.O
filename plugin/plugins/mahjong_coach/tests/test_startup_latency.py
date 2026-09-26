from __future__ import annotations

import asyncio
import inspect

import pytest

from plugin.plugins.mahjong_coach import MahjongCoachPlugin
from plugin.plugins.mahjong_coach.models import MahjongCoachConfig


def test_startup_defers_background_work_out_of_ephemeral_lifecycle_loop() -> None:
    source = inspect.getsource(MahjongCoachPlugin.startup)

    assert "self._start_yolo26_warmup" not in source
    assert "asyncio.create_task" not in source
    assert "self._runtime_warmup_pending" in source
    assert "self._auto_start_live_pending" in source


@pytest.mark.asyncio
async def test_first_entry_activates_deferred_warmup_and_auto_start() -> None:
    plugin = MahjongCoachPlugin.__new__(MahjongCoachPlugin)
    plugin._cfg = MahjongCoachConfig(
        tile_recognition_mode="yolo26",
        inference_provider="speed",
    )
    plugin._engine = object()
    plugin._runtime_warmup_pending = True
    plugin._auto_start_live_pending = True
    plugin._deferred_auto_start_task = None

    warmup_calls: list[str] = []
    auto_start_calls: list[dict[str, object]] = []
    plugin._start_yolo26_warmup = lambda provider: warmup_calls.append(provider)

    async def start_live(**kwargs: object) -> str:
        auto_start_calls.append(dict(kwargs))
        return "started"

    plugin._overlay_start_live = start_live
    plugin._activate_deferred_runtime_work()
    await asyncio.sleep(0)

    assert warmup_calls == ["speed"]
    assert auto_start_calls == [{"overlay": True, "inference_provider": "speed"}]
    assert plugin._runtime_warmup_pending is False
    assert plugin._auto_start_live_pending is False
    assert plugin._deferred_auto_start_task is not None
    assert plugin._deferred_auto_start_task.done()
