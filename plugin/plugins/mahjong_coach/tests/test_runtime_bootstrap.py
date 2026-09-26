from __future__ import annotations

import importlib
from importlib.util import spec_from_file_location
from pathlib import Path
import sys
from types import ModuleType

from plugin.plugins.mahjong_coach.runtime_bootstrap import prioritize_plugin_vendor_path


def test_prioritize_plugin_vendor_path_moves_existing_entry_to_front(
    tmp_path: Path,
    monkeypatch,
) -> None:
    plugin_dir = tmp_path / "mahjong_coach"
    vendor_dir = plugin_dir / "vendor"
    (vendor_dir / "onnxruntime").mkdir(parents=True)
    vendor_value = str(vendor_dir)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(sys, "path", ["host-runtime", vendor_value, "project"])

    selected = prioritize_plugin_vendor_path(plugin_dir)

    assert selected == vendor_value
    assert sys.path == [vendor_value, "host-runtime", "project"]


def test_prioritize_plugin_vendor_path_ignores_missing_private_runtime(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(sys, "path", ["host-runtime"])

    assert prioritize_plugin_vendor_path(tmp_path / "mahjong_coach") == ""
    assert sys.path == ["host-runtime"]


def test_prioritize_plugin_vendor_path_releases_host_onnxruntime_in_child(
    tmp_path: Path,
    monkeypatch,
) -> None:
    plugin_dir = tmp_path / "mahjong_coach"
    vendor_dir = plugin_dir / "vendor"
    (vendor_dir / "onnxruntime" / "capi").mkdir(parents=True)
    host_module = ModuleType("onnxruntime")
    host_module.__file__ = str(tmp_path / "host" / "onnxruntime" / "__init__.py")
    host_capi = ModuleType("onnxruntime.capi")
    dll_directories: list[str] = []
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(sys, "path", ["host-runtime"])
    monkeypatch.setenv("NEKO_PLUGIN_SERVICE_NAME", "Plugin_mahjong_coach")
    monkeypatch.setitem(sys.modules, "onnxruntime", host_module)
    monkeypatch.setitem(sys.modules, "onnxruntime.capi", host_capi)
    monkeypatch.setattr(
        "plugin.plugins.mahjong_coach.runtime_bootstrap.os.add_dll_directory",
        lambda value: dll_directories.append(value) or object(),
    )

    prioritize_plugin_vendor_path(plugin_dir)

    assert "onnxruntime" not in sys.modules
    assert "onnxruntime.capi" not in sys.modules
    assert dll_directories == [str(vendor_dir / "onnxruntime" / "capi")]


def test_prioritize_plugin_vendor_path_keeps_host_module_during_parent_scan(
    tmp_path: Path,
    monkeypatch,
) -> None:
    plugin_dir = tmp_path / "mahjong_coach"
    vendor_dir = plugin_dir / "vendor"
    (vendor_dir / "onnxruntime").mkdir(parents=True)
    host_module = ModuleType("onnxruntime")
    host_module.__file__ = str(tmp_path / "host" / "onnxruntime" / "__init__.py")
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(sys, "path", ["host-runtime"])
    monkeypatch.delenv("NEKO_PLUGIN_SERVICE_NAME", raising=False)
    monkeypatch.setitem(sys.modules, "onnxruntime", host_module)

    prioritize_plugin_vendor_path(plugin_dir)

    assert sys.modules["onnxruntime"] is host_module


def test_private_runtime_finder_beats_host_frozen_finder(
    tmp_path: Path,
    monkeypatch,
) -> None:
    plugin_dir = tmp_path / "mahjong_coach"
    vendor_package = plugin_dir / "vendor" / "onnxruntime"
    host_package = tmp_path / "host" / "onnxruntime"
    (vendor_package / "capi").mkdir(parents=True)
    host_package.mkdir(parents=True)
    (vendor_package / "__init__.py").write_text("SOURCE = 'vendor'\n", encoding="utf-8")
    (host_package / "__init__.py").write_text("SOURCE = 'host'\n", encoding="utf-8")

    class HostFrozenFinder:
        @staticmethod
        def find_spec(fullname, path=None, target=None):
            del path, target
            if fullname == "onnxruntime":
                return spec_from_file_location(fullname, host_package / "__init__.py")
            return None

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(sys, "path", ["host-runtime"])
    monkeypatch.setattr(sys, "meta_path", [HostFrozenFinder()])
    monkeypatch.setenv("NEKO_PLUGIN_SERVICE_NAME", "Plugin_mahjong_coach")
    monkeypatch.delitem(sys.modules, "onnxruntime", raising=False)
    monkeypatch.setattr(
        "plugin.plugins.mahjong_coach.runtime_bootstrap.os.add_dll_directory",
        lambda value: object(),
    )

    prioritize_plugin_vendor_path(plugin_dir)
    loaded = importlib.import_module("onnxruntime")

    assert loaded.SOURCE == "vendor"
    assert Path(loaded.__file__).resolve() == (vendor_package / "__init__.py").resolve()
