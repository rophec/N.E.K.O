from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


DEFAULT_INPUT_DIR = Path("C:/Users/19079/Desktop") / "\u8bc6\u522b\u56fe"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "automajhong_reference_output"
DEFAULT_AUTOMAJSOUL_DIR = Path(__file__).resolve().parent / "vendor" / "AutoMajsoul"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the original AutoMajsoul table detector on Mahjong Soul screenshots."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--automajsoul-dir", type=Path, default=DEFAULT_AUTOMAJSOUL_DIR)
    args = parser.parse_args()

    recognizer_cls = _load_original_automajsoul_recognizer(args.automajsoul_dir)
    images = _collect_images(args.input_dir)
    if not images:
        print(f"No images found in {args.input_dir}")
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: list[dict[str, Any]] = []
    for index, image_path in enumerate(images, start=1):
        summary.append(_run_one(image_path, args.output_dir, recognizer_cls, index))

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote summary: {summary_path}")
    for item in summary:
        print(
            f"{item['image_name']}: screen={item['screen']} method={item['method']} "
            f"quad={item['quad']} proof={item['proof_sheet']}"
        )
    return 0


def _load_original_automajsoul_recognizer(automajsoul_dir: Path) -> Any:
    android_dir = automajsoul_dir / "Android"
    recognizer_path = android_dir / "ingame_recognizer.py"
    if not recognizer_path.exists():
        raise FileNotFoundError(f"AutoMajsoul recognizer not found: {recognizer_path}")
    sys.path.insert(0, str(android_dir))
    from ingame_recognizer import IngameRecognizer  # type: ignore[import-not-found]

    return IngameRecognizer


def _collect_images(input_dir: Path) -> list[Path]:
    images: list[Path] = []
    for pattern in ("*.png", "*.jpg", "*.jpeg"):
        images.extend(input_dir.glob(pattern))
    return sorted(images)


def _run_one(image_path: Path, output_dir: Path, recognizer_cls: Any, index: int) -> dict[str, Any]:
    image = Image.open(image_path).convert("RGB")
    stem = image_path.stem
    case_id = _safe_case_id(stem, index)
    case_dir = output_dir / case_id
    debug_dir = case_dir / "automajsoul_debug"
    case_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    source_copy = case_dir / f"{case_id}-source.png"
    image.save(source_copy)

    recognizer = recognizer_cls(debug=True, debug_dir=debug_dir)
    recognizer._debug_timestamp = case_id
    screen, detection = recognizer.detect(image)

    quad: list[list[float]] = []
    warp_path = ""
    review_overlay_path = ""
    if detection is not None and "quad" in detection:
        quad_raw = detection["quad"]
        quad = [[round(float(x), 2), round(float(y), 2)] for x, y in quad_raw.tolist()]
        review_overlay_path = str(case_dir / f"{case_id}-review-overlay.png")
        _write_review_overlay(image, quad, Path(review_overlay_path))
        warp = recognizer.warp_board(image, quad_raw)
        warp_path = str(case_dir / f"{case_id}-automajsoul-warp.png")
        warp.save(warp_path)

    method = _detect_method(debug_dir, case_id)
    proof_sheet = case_dir / f"{case_id}-proof-sheet.png"
    _write_proof_sheet(
        proof_sheet,
        [
            ("source", source_copy),
            ("mask before", debug_dir / f"{case_id}_02_color_mask_before.png"),
            ("mask after", debug_dir / f"{case_id}_02_color_mask_after.png"),
            ("edges", debug_dir / f"{case_id}_02_edges.png"),
            ("quad", debug_dir / f"{case_id}_02_quad_detection_{method}.png"),
            ("review overlay", Path(review_overlay_path) if review_overlay_path else None),
            ("warp", Path(warp_path) if warp_path else None),
        ],
    )

    quad_info_path = case_dir / f"{case_id}-quad.json"
    quad_info = {
        "image": str(image_path),
        "case_id": case_id,
        "screen": getattr(screen, "name", str(screen)),
        "method": method,
        "quad_order": ["top_left", "top_right", "bottom_left", "bottom_right"],
        "quad": quad,
        "automajsoul_debug_dir": str(debug_dir),
        "source": str(source_copy),
        "warp": warp_path,
        "review_overlay": review_overlay_path,
        "proof_sheet": str(proof_sheet),
    }
    quad_info_path.write_text(json.dumps(quad_info, ensure_ascii=False, indent=2), encoding="utf-8")
    return {**quad_info, "image_name": image_path.name, "quad_json": str(quad_info_path)}


def _safe_case_id(stem: str, index: int) -> str:
    if stem and all(char.isascii() and (char.isalnum() or char in "-_") for char in stem):
        return stem
    return f"case_{index:03d}"


def _detect_method(debug_dir: Path, stem: str) -> str:
    if (debug_dir / f"{stem}_02_quad_detection_color.png").exists():
        return "color"
    if (debug_dir / f"{stem}_02_quad_detection_edges.png").exists():
        return "edges"
    return "unknown"


def _write_proof_sheet(output_path: Path, entries: list[tuple[str, Path | None]]) -> None:
    tiles: list[tuple[str, Image.Image]] = []
    for label, path in entries:
        if path is None or not path.exists():
            continue
        tiles.append((label, Image.open(path).convert("RGB")))
    if not tiles:
        return

    cell_w, cell_h = 520, 330
    label_h = 34
    cols = 2
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell_w, rows * (cell_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    for index, (label, tile) in enumerate(tiles):
        col = index % cols
        row = index // cols
        x = col * cell_w
        y = row * (cell_h + label_h)
        draw.text((x + 8, y + 8), label, fill="black", font=font)
        preview = _fit_image(tile, (cell_w, cell_h))
        sheet.paste(preview, (x + (cell_w - preview.width) // 2, y + label_h + (cell_h - preview.height) // 2))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def _write_review_overlay(image: Image.Image, quad: list[list[float]], output_path: Path) -> None:
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    top_left, top_right, bottom_left, bottom_right = [tuple(point) for point in quad]
    boundary = [top_left, top_right, bottom_right, bottom_left, top_left]
    draw.line(boundary, fill="yellow", width=5)
    for label, point in zip(("TL", "TR", "BL", "BR"), (top_left, top_right, bottom_left, bottom_right)):
        draw.text((point[0] + 8, point[1] + 8), label, fill="yellow")
    draw.text(
        (16, 16),
        "AutoMajsoul quad boundary only; this is not table seam detection.",
        fill="yellow",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(output_path)


def _fit_image(image: Image.Image, box: tuple[int, int]) -> Image.Image:
    max_w, max_h = box
    scale = min(max_w / image.width, max_h / image.height)
    size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


if __name__ == "__main__":
    raise SystemExit(main())
