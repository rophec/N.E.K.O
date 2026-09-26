from __future__ import annotations

import hashlib
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, TypeAlias

from PIL import Image


ImageSource: TypeAlias = str | Path | Image.Image


def source_exists(source: ImageSource | None) -> bool:
    if isinstance(source, Image.Image):
        return source.width > 0 and source.height > 0
    if source is None:
        return False
    try:
        return Path(source).exists()
    except (OSError, TypeError, ValueError):
        return False


@contextmanager
def open_rgb(source: ImageSource) -> Iterator[Image.Image]:
    if isinstance(source, Image.Image):
        # Live capture already supplies RGB frames. Reuse that immutable input
        # instead of allocating another full-resolution PIL buffer in every
        # detector participating in the same analysis pass.
        image = source if source.mode == "RGB" else source.convert("RGB")
        try:
            yield image
        finally:
            if image is not source:
                image.close()
        return
    with Image.open(Path(source)) as opened:
        image = opened.convert("RGB")
        try:
            yield image
        finally:
            image.close()


def source_stem(source: ImageSource) -> str:
    if isinstance(source, Image.Image):
        return f"memory-{int(getattr(source, '_neko_frame_id', id(source)))}"
    return Path(source).stem


def source_identity(source: ImageSource | None) -> tuple[str, int, int, str] | None:
    if isinstance(source, Image.Image):
        frame_id = int(getattr(source, "_neko_frame_id", id(source)))
        rgb = source if source.mode == "RGB" else source.convert("RGB")
        thumb = rgb.resize((32, 18), Image.Resampling.BILINEAR)
        try:
            digest = hashlib.blake2s(thumb.tobytes(), digest_size=12).hexdigest()
        finally:
            thumb.close()
            if rgb is not source:
                rgb.close()
        return "memory", int(source.width * source.height * 3), frame_id, digest
    if source is None:
        return None
    path = Path(source)
    try:
        resolved = path.expanduser().resolve()
        stat = resolved.stat()
        with resolved.open("rb") as frame_file:
            digest = hashlib.file_digest(frame_file, "blake2s").hexdigest()
    except OSError:
        return None
    return str(resolved), int(stat.st_size), int(stat.st_mtime_ns), digest
