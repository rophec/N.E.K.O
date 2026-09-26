"""Select one ABI-aligned ONNX Runtime provider for a Windows build environment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys


ORT_VERSION = "1.24.4"
RUNTIME_PACKAGES = ("onnxruntime", "onnxruntime-directml", "onnxruntime-gpu")
PROVIDER_PACKAGES = {
    "directml": "onnxruntime-directml",
    "cuda": "onnxruntime-gpu",
}
EXPECTED_PROVIDERS = {
    "directml": "DmlExecutionProvider",
    "cuda": "CUDAExecutionProvider",
}


def command_plan(*, python: Path, provider: str, uv_executable: str = "uv") -> list[list[str]]:
    normalized = str(provider).strip().lower()
    if normalized not in PROVIDER_PACKAGES:
        raise ValueError(f"unsupported provider: {provider}")
    target = PROVIDER_PACKAGES[normalized]
    return [
        [uv_executable, "pip", "uninstall", "--python", str(python), *RUNTIME_PACKAGES],
        [
            uv_executable,
            "pip",
            "install",
            "--python",
            str(python),
            "--only-binary=:all:",
            f"{target}=={ORT_VERSION}",
        ],
    ]


def verify_command(*, python: Path, provider: str) -> list[str]:
    expected = EXPECTED_PROVIDERS[str(provider).strip().lower()]
    probe = (
        "import json, onnxruntime as ort; "
        "providers=list(ort.get_available_providers()); "
        f"assert {expected!r} in providers, providers; "
        "print(json.dumps({'version': ort.__version__, 'providers': providers}))"
    )
    return [str(python), "-c", probe]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True, help="Target build Python executable")
    parser.add_argument("--provider", choices=sorted(PROVIDER_PACKAGES), required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    python = args.python.resolve()
    if not python.is_file():
        parser.error(f"target Python does not exist: {python}")
    uv_executable = shutil.which("uv")
    if not uv_executable:
        parser.error("uv executable is unavailable")
    plan = command_plan(python=python, provider=args.provider, uv_executable=uv_executable)
    if args.dry_run:
        print(json.dumps({"provider": args.provider, "commands": plan}, ensure_ascii=False))
        return 0

    for command in plan:
        completed = subprocess.run(command, check=False)
        # `uv pip uninstall` returns success when one or more listed packages
        # are absent on current uv versions, but tolerate only that first step.
        if completed.returncode and command is not plan[0]:
            return int(completed.returncode)
    verified = subprocess.run(verify_command(python=python, provider=args.provider), check=False)
    return int(verified.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
