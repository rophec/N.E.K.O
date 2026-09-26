from __future__ import annotations

import time as _real_time

import pytest

import plugin.plugins.mahjong_coach as mahjong_plugin_module


class _ScopedTime:
    def __init__(self, **overrides):
        self._overrides = overrides

    def __getattr__(self, name):
        try:
            return self._overrides[name]
        except KeyError:
            return getattr(_real_time, name)


@pytest.fixture
def patch_module_clock():
    """Patch one importing module without mutating the stdlib time module."""

    def patch(monkeypatch: pytest.MonkeyPatch, module, **overrides) -> None:
        monkeypatch.setattr(module, "time", _ScopedTime(**overrides))

    return patch


@pytest.fixture(autouse=True)
def _avoid_real_model_warmup(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep unit tests independent from host GPU/CPU inference availability."""

    monkeypatch.setattr(
        mahjong_plugin_module,
        "warmup_yolo26_runtime",
        lambda *, inference_provider="speed", **_kwargs: {
            "status": "unavailable",
            "reason": "unit_test_stub",
            "elapsed_ms": 0.0,
            "providers": [],
            "inference_provider": inference_provider,
        },
    )
