from pathlib import Path
import tomllib

from tools.configure_windows_onnxruntime import (
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


def test_distribution_excludes_runtime_captures_and_local_preferences() -> None:
    plugin_root = Path(__file__).resolve().parents[1]
    with (plugin_root / "pyproject.toml").open("rb") as handle:
        config = tomllib.load(handle)

    for section_name in ("build", "pack"):
        rules = config["tool"]["neko"][section_name]
        assert "live_frames" in rules["exclude_dirs"]
        assert "tools" in rules["exclude_dirs"]
        assert {
            ".overlay-preview.json",
            "coach_preferences.json",
            "overlay_prefs.json",
        }.issubset(rules["exclude_files"])
