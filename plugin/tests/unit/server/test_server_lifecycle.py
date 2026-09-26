from __future__ import annotations

import copy

import pytest

from plugin.server import lifecycle as module


pytestmark = pytest.mark.plugin_unit


def test_message_plane_runner_is_recreated_when_previous_runner_is_unhealthy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class _Runner:
        def __init__(self, name: str, healthy: bool) -> None:
            self.name = name
            self.healthy = healthy

        def health_check(self, *, timeout_s: float = 1.0) -> bool:
            calls.append(f"health:{self.name}:{timeout_s}")
            return self.healthy

        def start(self) -> None:
            calls.append(f"start:{self.name}")

        def stop(self) -> None:
            calls.append(f"stop:{self.name}")

    service = module.ServerLifecycleService()
    stale = _Runner("stale", False)
    fresh = _Runner("fresh", True)
    service._message_plane_runner = stale
    monkeypatch.setattr(module, "build_message_plane_runner", lambda: fresh)

    assert service._ensure_message_plane_started_sync() is fresh
    assert service._message_plane_runner is fresh
    assert calls == ["health:stale:0.2", "stop:stale", "start:fresh"]


@pytest.mark.asyncio
async def test_ensure_plugin_messaging_started_initializes_response_map_and_router(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class _State:
        @property
        def plugin_response_map(self) -> dict[str, object]:
            calls.append("response_map")
            return {}

    async def _start_router() -> None:
        calls.append("router_start")

    async def _start_plane() -> None:
        calls.append("message_plane_start")

    monkeypatch.setattr(module, "state", _State())
    monkeypatch.setattr(module.plugin_router, "start", _start_router)
    monkeypatch.setattr(module._service, "_start_message_plane", _start_plane)
    monkeypatch.setattr(module, "start_bridge", lambda: calls.append("bridge_start"))
    monkeypatch.setattr(module, "start_proactive_bridge", lambda: calls.append("proactive_bridge_start"))

    ensure = getattr(module, "ensure_plugin_messaging_started", None)
    assert callable(ensure)

    await ensure()

    assert calls == [
        "response_map",
        "router_start",
        "message_plane_start",
        "bridge_start",
        "proactive_bridge_start",
    ]


@pytest.mark.asyncio
async def test_ensure_plugin_messaging_started_starts_router_when_response_map_init_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class _State:
        @property
        def plugin_response_map(self) -> dict[str, object]:
            calls.append("response_map")
            raise RuntimeError("response map unavailable")

    async def _start_router() -> None:
        calls.append("router_start")

    async def _start_plane() -> None:
        calls.append("message_plane_start")

    warnings: list[tuple[str, str, str]] = []

    class _Logger:
        def warning(self, message: str, err_type: str, err: str) -> None:
            warnings.append((message, err_type, err))

        def debug(self, *_args: object, **_kwargs: object) -> None:
            return None

    monkeypatch.setattr(module, "state", _State())
    monkeypatch.setattr(module.plugin_router, "start", _start_router)
    monkeypatch.setattr(module._service, "_start_message_plane", _start_plane)
    monkeypatch.setattr(module, "start_bridge", lambda: calls.append("bridge_start"))
    monkeypatch.setattr(module, "start_proactive_bridge", lambda: calls.append("proactive_bridge_start"))
    monkeypatch.setattr(module, "logger", _Logger())

    await module.ensure_plugin_messaging_started()

    assert calls == [
        "response_map",
        "router_start",
        "message_plane_start",
        "bridge_start",
        "proactive_bridge_start",
    ]
    assert warnings == [
        (
            "failed to initialize plugin response map early: err_type={}, err={}",
            "RuntimeError",
            "response map unavailable",
        )
    ]


@pytest.mark.asyncio
async def test_startup_uses_registry_refresh_then_autostart(monkeypatch: pytest.MonkeyPatch) -> None:
    plugins_backup = copy.deepcopy(module.state.plugins)
    hosts_backup = dict(module.state.plugin_hosts)
    handlers_backup = dict(module.state.event_handlers)
    cache_backup = copy.deepcopy(module.state._snapshot_cache)
    calls: list[tuple[str, str]] = []

    async def _noop_async(*args, **kwargs):
        return None

    try:
        service = module.ServerLifecycleService()

        monkeypatch.setattr(module.ServerLifecycleService, "_clear_runtime_state", staticmethod(lambda: None))
        monkeypatch.setattr(module, "emit_lifecycle_event", lambda event: None)
        monkeypatch.setattr(module.plugin_router, "start", _noop_async)
        monkeypatch.setattr(service, "_start_message_plane", _noop_async)
        monkeypatch.setattr(module.bus_subscription_manager, "start", _noop_async)
        monkeypatch.setattr(module.status_manager, "start_status_consumer", _noop_async)
        monkeypatch.setattr(module.metrics_collector, "start", _noop_async)
        monkeypatch.setattr(module, "start_bridge", lambda: None)
        monkeypatch.setattr(module, "start_proactive_bridge", lambda: None)

        async def _refresh_registry() -> dict[str, object]:
            calls.append(("registry", "refresh"))
            with module.state.acquire_plugins_write_lock():
                module.state.plugins.clear()
                module.state.plugins.update(
                    {
                        "auto_plugin": {
                            "id": "auto_plugin",
                            "type": "plugin",
                            "runtime_enabled": True,
                            "runtime_auto_start": True,
                        },
                        "manual_plugin": {
                            "id": "manual_plugin",
                            "type": "plugin",
                            "runtime_enabled": True,
                            "runtime_auto_start": False,
                        },
                        "failed_plugin": {
                            "id": "failed_plugin",
                            "type": "plugin",
                            "runtime_enabled": True,
                            "runtime_auto_start": True,
                            "runtime_load_state": "failed",
                        },
                    }
                )
            return {"success": True, "added": ["auto_plugin"], "updated": [], "removed": [], "failed": []}

        async def _start_plugin(plugin_id: str, restore_state: bool = False, *, refresh_registry: bool = True) -> dict[str, object]:
            _ = restore_state
            calls.append(("start", f"{plugin_id}:{refresh_registry}"))
            return {"success": True, "plugin_id": plugin_id}

        monkeypatch.setattr(service._plugin_registry_service, "refresh_registry", _refresh_registry)
        monkeypatch.setattr(service._plugin_lifecycle_service, "start_plugin", _start_plugin)

        await service.startup()

        assert calls == [
            ("registry", "refresh"),
            ("start", "auto_plugin:False"),
        ]
    finally:
        with module.state.acquire_plugins_write_lock():
            module.state.plugins.clear()
            module.state.plugins.update(plugins_backup)
        with module.state.acquire_plugin_hosts_write_lock():
            module.state.plugin_hosts.clear()
            module.state.plugin_hosts.update(hosts_backup)
        with module.state.acquire_event_handlers_write_lock():
            module.state.event_handlers.clear()
            module.state.event_handlers.update(handlers_backup)
        with module.state._snapshot_cache_lock:
            module.state._snapshot_cache = cache_backup
