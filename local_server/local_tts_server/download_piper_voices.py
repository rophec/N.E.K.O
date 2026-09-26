"""Download Piper voices from rhasspy/piper-voices.

By default this downloads every voice listed in voices.json into the local
``piper_models`` directory, flattened as:

    piper_models/<voice-key>.onnx
    piper_models/<voice-key>.onnx.json

The downloaded model files are runtime assets and are ignored by git.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


SERVER_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SERVER_DIR / "piper_models"
DEFAULT_REPO = "rhasspy/piper-voices"
DEFAULT_REVISION = "main"
HF_BASE = "https://huggingface.co"


def hf_resolve_url(repo: str, revision: str, path: str) -> str:
    return f"{HF_BASE}/{repo}/resolve/{revision}/{path}"


def read_url(url: str, *, timeout: int = 120) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "NEKO-piper-voice-downloader/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def download_file(url: str, dest: Path, *, expected_md5: str = "", retries: int = 3) -> None:
    tmp = dest.with_suffix(dest.suffix + ".download")
    dest.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "NEKO-piper-voice-downloader/1.0"})
            with urllib.request.urlopen(request, timeout=120) as response, tmp.open("wb") as out:
                total = int(response.headers.get("Content-Length") or "0")
                seen = 0
                last_log = time.monotonic()
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
                    seen += len(chunk)
                    now = time.monotonic()
                    if total and now - last_log >= 2:
                        pct = seen * 100.0 / total
                        print(f"    {pct:5.1f}% {seen / 1024 / 1024:.1f}/{total / 1024 / 1024:.1f} MB")
                        last_log = now

            if expected_md5:
                digest = hashlib.md5(tmp.read_bytes()).hexdigest()
                if digest.lower() != expected_md5.lower():
                    raise RuntimeError(
                        f"MD5 mismatch for {dest.name}: expected {expected_md5}, got {digest}"
                    )

            tmp.replace(dest)
            return
        except Exception as exc:
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass
            if attempt == retries:
                raise
            print(f"    retry {attempt}/{retries} after error: {exc}")
            time.sleep(2 * attempt)


def parse_languages(raw: str) -> set[str]:
    return {part.strip() for part in raw.split(",") if part.strip()}


def voice_files(entry: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    files = entry.get("files") or {}
    selected: list[tuple[str, dict[str, Any]]] = []
    if not isinstance(files, dict):
        return selected

    for remote_path, meta in files.items():
        if remote_path.endswith(".onnx") or remote_path.endswith(".onnx.json"):
            selected.append((remote_path, meta if isinstance(meta, dict) else {}))
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Piper TTS voice models")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--languages",
        default="",
        help="Optional comma-separated language codes such as zh_CN,en_US. Empty means all.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional max voices for testing.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    voices_url = hf_resolve_url(args.repo, args.revision, "voices.json")
    print(f"Fetching voice index: {voices_url}")
    voices = json.loads(read_url(voices_url).decode("utf-8"))
    if not isinstance(voices, dict):
        raise RuntimeError("voices.json did not contain an object")

    allowed_languages = parse_languages(args.languages)
    selected: list[tuple[str, dict[str, Any]]] = []
    for key, entry in sorted(voices.items()):
        if not isinstance(entry, dict):
            continue
        language = entry.get("language") or {}
        lang_code = language.get("code") if isinstance(language, dict) else ""
        if allowed_languages and lang_code not in allowed_languages:
            continue
        selected.append((key, entry))
        if args.limit and len(selected) >= args.limit:
            break

    total_bytes = 0
    for _, entry in selected:
        for _, meta in voice_files(entry):
            total_bytes += int(meta.get("size_bytes") or 0)

    print(f"Selected voices : {len(selected)}")
    print(f"Estimated size  : {total_bytes / 1024 / 1024 / 1024:.2f} GB")
    print(f"Output dir      : {output_dir}")

    failures: list[str] = []
    downloaded = 0
    skipped = 0

    for index, (key, entry) in enumerate(selected, start=1):
        files = voice_files(entry)
        print(f"[{index}/{len(selected)}] {key}")
        for remote_path, meta in files:
            dest = output_dir / Path(remote_path).name
            if dest.exists() and dest.stat().st_size == int(meta.get("size_bytes") or dest.stat().st_size):
                print(f"  skip {dest.name}")
                skipped += 1
                continue

            url = hf_resolve_url(args.repo, args.revision, remote_path)
            print(f"  get  {dest.name}")
            try:
                download_file(url, dest, expected_md5=str(meta.get("md5_digest") or ""))
                downloaded += 1
            except (urllib.error.URLError, TimeoutError, RuntimeError, OSError) as exc:
                print(f"  FAILED {dest.name}: {exc}", file=sys.stderr)
                failures.append(f"{key}: {remote_path}: {exc}")

    print("")
    print(f"Downloaded files: {downloaded}")
    print(f"Skipped files   : {skipped}")
    print(f"Failures        : {len(failures)}")
    if failures:
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
