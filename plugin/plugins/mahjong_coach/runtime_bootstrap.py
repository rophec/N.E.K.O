from __future__ import annotations

import importlib
from importlib.machinery import PathFinder
import os
from pathlib import Path
import sys
from typing import Any


_DLL_DIRECTORY_HANDLES: list[object] = []
_VENDOR_RUNTIME_FINDER: object | None = None


class _VendorOnnxRuntimeFinder:
    """Resolve only the private onnxruntime namespace before Nuitka's finder."""

    def __init__(self, vendor_dir: Path) -> None:
        self.vendor_dir = vendor_dir.resolve()

    def find_spec(
        self,
        fullname: str,
        path: Any = None,
        target: Any = None,
    ) -> Any:
        del target
        if fullname == "onnxruntime":
            return PathFinder.find_spec(fullname, [str(self.vendor_dir)])
        if fullname.startswith("onnxruntime."):
            return PathFinder.find_spec(fullname, path)
        return None


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except (OSError, ValueError):
        return False
    return True


def prioritize_plugin_vendor_path(plugin_dir: Path | None = None) -> str:
    """Put this plugin's private dependencies ahead of the host runtime.

    N.E.K.O's isolated plugin process can inherit a ``sys.path`` that already
    contains the plugin vendor directory, but behind the application's bundled
    packages.  Merely checking that the path exists is therefore insufficient:
    the host CPU-only ONNX Runtime can shadow the DirectML runtime distributed
    with Mahjong Coach.
    """

    if sys.platform != "win32":
        return ""

    root = Path(plugin_dir) if plugin_dir is not None else Path(__file__).resolve().parent
    vendor_dir = root / "vendor"
    if not (vendor_dir / "onnxruntime").is_dir():
        return ""

    value = str(vendor_dir)
    sys.path[:] = [entry for entry in sys.path if entry != value]
    sys.path.insert(0, value)

    # Nuitka's plugin host imports its bundled CPU ONNX Runtime before loading
    # the plugin entry module.  In the isolated child process no inference
    # session exists yet, so discard that Python module namespace and let the
    # first perception request import the plugin-private DirectML build.
    service_name = str(os.environ.get("NEKO_PLUGIN_SERVICE_NAME") or "")
    is_plugin_child = service_name.startswith("Plugin_")
    global _VENDOR_RUNTIME_FINDER
    if is_plugin_child:
        if _VENDOR_RUNTIME_FINDER is not None:
            with_context = _VENDOR_RUNTIME_FINDER
            sys.meta_path[:] = [finder for finder in sys.meta_path if finder is not with_context]
        _VENDOR_RUNTIME_FINDER = _VendorOnnxRuntimeFinder(vendor_dir)
        sys.meta_path.insert(0, _VENDOR_RUNTIME_FINDER)

    loaded = sys.modules.get("onnxruntime")
    loaded_origin = Path(str(getattr(loaded, "__file__", "") or ""))
    if (
        is_plugin_child
        and loaded is not None
        and not _is_within(loaded_origin, vendor_dir)
    ):
        for module_name in tuple(sys.modules):
            if module_name == "onnxruntime" or module_name.startswith("onnxruntime."):
                sys.modules.pop(module_name, None)

    capi_dir = vendor_dir / "onnxruntime" / "capi"
    add_dll_directory = getattr(os, "add_dll_directory", None)
    if is_plugin_child and capi_dir.is_dir() and callable(add_dll_directory):
        _DLL_DIRECTORY_HANDLES.append(add_dll_directory(str(capi_dir)))

    importlib.invalidate_caches()
    return value
