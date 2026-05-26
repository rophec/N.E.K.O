"""N.E.K.O 卡牌铸造功能一键启动脚本。

仿 ``start_battle_arena.py`` 的三窗口模型：
  [1/3] N.E.K.O 主应用              (uv run launcher.py，端口 48911)
  [2/3] forge_server 铸造子服务     (uv run server.py，端口 3002)
  [3/3] frontend/card-forge dev     (npm run dev，端口 5174)

用户访问入口：
  - dev 模式直接打开 http://localhost:5174
  - 主应用菜单 -> "卡牌铸造"      → http://localhost:48911/card_forge
    （主应用页面会通过 IIFE 加载 static/react/card-forge/card-forge.iife.js；
     如果尚未构建过，请先运行 ./build_frontend.sh 或 build_frontend.bat。）

如果只想在已有 N.E.K.O 进程上加铸造服务，可以单独 ``uv run server.py``
（位于 local_server/forge_server/）。
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
FORGE_SERVER_ROOT = PROJECT_ROOT / "local_server" / "forge_server"
CARD_FORGE_FRONTEND_ROOT = PROJECT_ROOT / "frontend" / "card-forge"


def ps_quote(path: Path) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def launch_window(title: str, cwd: Path, command: str) -> None:
    safe_title = title.replace("'", "''")
    ps_command = (
        f"$Host.UI.RawUI.WindowTitle = '{safe_title}'; "
        f"Set-Location -LiteralPath {ps_quote(cwd)}; "
        f"{command}"
    )
    subprocess.Popen(
        ["powershell.exe", "-NoExit", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps_command],
        creationflags=subprocess.CREATE_NEW_CONSOLE,
    )


def ensure_path(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def main() -> int:
    ensure_path(PROJECT_ROOT / "launcher.py", "N.E.K.O launcher")
    ensure_path(FORGE_SERVER_ROOT / "server.py", "Forge server")
    ensure_path(CARD_FORGE_FRONTEND_ROOT / "package.json", "card-forge frontend")

    print("=" * 52)
    print("   N.E.K.O Card Forge - One Click Startup")
    print("=" * 52)
    print(f"Project root: {PROJECT_ROOT}")
    print()

    print("[1/3] Opening N.E.K.O main server window (port 48911)...")
    launch_window(
        "N.E.K.O Main Server - 48911",
        PROJECT_ROOT,
        "uv run .\\launcher.py",
    )

    time.sleep(3)

    print("[2/3] Opening forge server window (port 3002)...")
    launch_window(
        "N.E.K.O Forge Server - 3002",
        FORGE_SERVER_ROOT,
        "uv run server.py",
    )

    time.sleep(2)

    print("[3/3] Opening card-forge frontend window (port 5174)...")
    launch_window(
        "N.E.K.O Card Forge Frontend - 5174",
        CARD_FORGE_FRONTEND_ROOT,
        "npm run dev",
    )

    print()
    print("=" * 52)
    print("   Startup commands have been sent to 3 windows.")
    print("=" * 52)
    print("URLs:")
    print("  card-forge dev:   http://localhost:5174")
    print("  main app entry:   http://localhost:48911/card_forge")
    print("  forge API health: http://localhost:3002/health")
    print()
    print("Keep the three opened command windows running while testing.")
    print("Press Enter to close this launcher window...")
    input()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[startup error] {exc}", file=sys.stderr)
        print("Press Enter to close this launcher window...")
        input()
        raise
