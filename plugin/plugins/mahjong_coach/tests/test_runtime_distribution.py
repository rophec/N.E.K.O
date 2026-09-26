from pathlib import Path
import tomllib

from scripts.configure_windows_onnxruntime import (
    ORT_VERSION,
    command_plan,
    verify_command,
)


def test_cuda_component_replaces_conflicting_runtimes_with_abi_aligned_gpu_package() -> None:
    python = Path("C:/build/.venv/Scripts/python.exe")

    plan = command_plan(python=python, provider="cuda", uv_executable="uv")

    assert plan[0][-3:] == ["onnxruntime", "onnxruntime-directml", "onnxruntime-gpu"]
    assert plan[1][-1] == f"onnxruntime-gpu=={ORT_VERSION}"
    assert "--only-binary=:all:" in plan[1]
    assert "CUDAExecutionProvider" in verify_command(python=python, provider="cuda")[-1]


def test_default_windows_component_selects_directml() -> None:
    python = Path("C:/build/.venv/Scripts/python.exe")

    plan = command_plan(python=python, provider="directml", uv_executable="uv")

    assert plan[1][-1] == f"onnxruntime-directml=={ORT_VERSION}"
    assert "DmlExecutionProvider" in verify_command(python=python, provider="directml")[-1]


def test_mahjong_plugin_carries_its_own_windows_directml_runtime() -> None:
    plugin_project = Path(__file__).resolve().parents[1] / "pyproject.toml"
    metadata = tomllib.loads(plugin_project.read_text(encoding="utf-8"))

    assert (
        f"onnxruntime-directml=={ORT_VERSION}; sys_platform == 'win32'"
        in metadata["project"]["dependencies"]
    )
