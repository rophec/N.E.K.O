from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TILE_ROOT = Path.home() / "Desktop" / "识别图" / "单牌"
DEFAULT_BASE = (
    REPO_ROOT
    / "datasets"
    / "mahjong_yolo_active"
    / "dataset_registry"
    / "10_targeted_error_aug_20260707"
)
DEFAULT_BACKGROUNDS = (
    REPO_ROOT
    / "datasets"
    / "mahjong_yolo_active"
    / "dataset_registry"
    / "09_manual_gtfix_20260706"
    / "train_ready_53"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "datasets"
    / "mahjong_yolo_active"
    / "dataset_registry"
    / "12_synthetic_tile_paste_20260709"
)
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
FOCUS_PAIRS = [("2p", "4p"), ("3m", "2m"), ("5m", "6m")]
EXTRA_PAIRS = [("6p", "9p"), ("9s", "8p"), ("3z", "4m")]


@dataclass(frozen=True)
class Box:
    cls: int
    x: float
    y: float
    w: float
    h: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create synthetic Mahjong tile-paste samples from single-tile assets."
    )
    parser.add_argument("--tile-root", type=Path, default=DEFAULT_TILE_ROOT)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--backgrounds", type=Path, default=DEFAULT_BACKGROUNDS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def imread_unicode(path: Path, flags: int = cv2.IMREAD_UNCHANGED) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite_unicode(path: Path, image: np.ndarray) -> None:
    ext = path.suffix or ".png"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        raise RuntimeError(f"Failed to encode image: {path}")
    encoded.tofile(str(path))


def read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def class_names(dataset_root: Path) -> list[str]:
    names = read_yaml(dataset_root / "data.yaml")["names"]
    if isinstance(names, dict):
        return [names[index] for index in range(len(names))]
    return list(names)


def read_boxes(path: Path) -> list[Box]:
    boxes: list[Box] = []
    if not path.exists():
        return boxes
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls, x, y, w, h = parts
        boxes.append(Box(int(cls), float(x), float(y), float(w), float(h)))
    return boxes


def write_boxes(path: Path, boxes: list[Box]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{box.cls} {box.x:.6f} {box.y:.6f} {box.w:.6f} {box.h:.6f}" for box in boxes]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def image_label_pairs(dataset_root: Path, split: str) -> list[tuple[Path, Path]]:
    image_dir = dataset_root / "images" / split
    label_dir = dataset_root / "labels" / split
    return [
        (image_path, label_dir / f"{image_path.stem}.txt")
        for image_path in sorted(image_dir.iterdir())
        if image_path.suffix.lower() in IMAGE_EXTS
    ]


def copy_tree_dataset(source: Path, target: Path) -> None:
    for split in ("train", "val", "test"):
        for kind in ("images", "labels"):
            src_dir = source / kind / split
            if not src_dir.exists():
                continue
            dst_dir = target / kind / split
            dst_dir.mkdir(parents=True, exist_ok=True)
            for path in src_dir.iterdir():
                if path.is_file() and path.suffix.lower() != ".cache":
                    shutil.copy2(path, dst_dir / path.name)


def write_data_yaml(output: Path, names: list[str]) -> None:
    lines = [
        f"path: {output.as_posix()}",
        "train: images/train",
        "val: images/val",
    ]
    if (output / "images" / "test").exists():
        lines.append("test: images/test")
    lines.extend([f"nc: {len(names)}", "names:"])
    lines.extend(f"  {index}: {name}" for index, name in enumerate(names))
    (output / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "classes.txt").write_text("\n".join(names) + "\n", encoding="utf-8")


def box_to_xyxy(box: Box, width: int, height: int) -> tuple[float, float, float, float]:
    cx = box.x * width
    cy = box.y * height
    bw = box.w * width
    bh = box.h * height
    return cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0


def xyxy_to_box(cls: int, xyxy: tuple[float, float, float, float], width: int, height: int) -> Box:
    x1, y1, x2, y2 = xyxy
    return Box(cls, ((x1 + x2) / 2.0) / width, ((y1 + y2) / 2.0) / height, (x2 - x1) / width, (y2 - y1) / height)


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def load_tile_assets(tile_root: Path, names: list[str]) -> dict[str, list[Path]]:
    assets: dict[str, list[Path]] = {name: [] for name in names}
    for path in tile_root.rglob("*.png"):
        if path.name == "reference_contact_sheet.png":
            continue
        label = path.stem
        if label in assets:
            assets[label].append(path)
    missing = [name for name, paths in assets.items() if not paths]
    if missing:
        raise FileNotFoundError(f"Missing single-tile assets for: {missing}")
    return assets


def blue_table_score(region: np.ndarray) -> float:
    if region.size == 0:
        return 0.0
    b = region[:, :, 0].astype(np.int16)
    g = region[:, :, 1].astype(np.int16)
    r = region[:, :, 2].astype(np.int16)
    mask = (b > 45) & (b > r + 12) & (b >= g - 12)
    return float(mask.mean())


def prepare_tile(asset_path: Path, rng: random.Random) -> np.ndarray:
    tile = imread_unicode(asset_path, cv2.IMREAD_UNCHANGED)
    if tile is None:
        raise RuntimeError(f"Failed to read tile asset: {asset_path}")
    if tile.ndim == 2:
        tile = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGRA)
    elif tile.shape[2] == 3:
        alpha = np.full(tile.shape[:2] + (1,), 255, dtype=np.uint8)
        tile = np.concatenate([tile, alpha], axis=2)
    target_long = rng.randint(44, 78)
    h, w = tile.shape[:2]
    scale = target_long / max(h, w)
    new_w = max(12, int(round(w * scale)))
    new_h = max(12, int(round(h * scale)))
    tile = cv2.resize(tile, (new_w, new_h), interpolation=cv2.INTER_AREA)
    if rng.random() < 0.35:
        angle = rng.uniform(-4.0, 4.0)
        matrix = cv2.getRotationMatrix2D((new_w / 2.0, new_h / 2.0), angle, 1.0)
        cos = abs(matrix[0, 0])
        sin = abs(matrix[0, 1])
        bound_w = int(new_h * sin + new_w * cos)
        bound_h = int(new_h * cos + new_w * sin)
        matrix[0, 2] += bound_w / 2.0 - new_w / 2.0
        matrix[1, 2] += bound_h / 2.0 - new_h / 2.0
        tile = cv2.warpAffine(tile, matrix, (bound_w, bound_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    bgr = tile[:, :, :3].astype(np.float32)
    bgr *= rng.uniform(0.94, 1.06)
    bgr += rng.uniform(-5.0, 5.0)
    tile[:, :, :3] = np.clip(bgr, 0, 255).astype(np.uint8)
    return tile


def paste_tile(background: np.ndarray, tile: np.ndarray, left: int, top: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    out = background.copy()
    h, w = tile.shape[:2]
    x1, y1, x2, y2 = left, top, left + w, top + h
    alpha = tile[:, :, 3:4].astype(np.float32) / 255.0
    shadow_alpha = cv2.GaussianBlur(alpha[:, :, 0], (0, 0), 2.2)[:, :, None] * 0.22
    sx1, sy1 = min(out.shape[1], x1 + 3), min(out.shape[0], y1 + 4)
    sx2, sy2 = min(out.shape[1], x2 + 3), min(out.shape[0], y2 + 4)
    if sx2 > sx1 and sy2 > sy1:
        sh = sy2 - sy1
        sw = sx2 - sx1
        roi = out[sy1:sy2, sx1:sx2].astype(np.float32)
        shadow = shadow_alpha[:sh, :sw]
        out[sy1:sy2, sx1:sx2] = np.clip(roi * (1.0 - shadow), 0, 255).astype(np.uint8)
    roi = out[y1:y2, x1:x2].astype(np.float32)
    fg = tile[:, :, :3].astype(np.float32)
    out[y1:y2, x1:x2] = np.clip(fg * alpha + roi * (1.0 - alpha), 0, 255).astype(np.uint8)
    return out, (x1, y1, x2, y2)


def find_position(
    image: np.ndarray,
    tile_shape: tuple[int, int],
    occupied: list[tuple[float, float, float, float]],
    rng: random.Random,
) -> tuple[int, int] | None:
    h, w = tile_shape
    height, width = image.shape[:2]
    if w >= width or h >= height:
        return None
    for _ in range(700):
        left = rng.randint(10, width - w - 10)
        top = rng.randint(10, height - h - 10)
        candidate = (left, top, left + w, top + h)
        if any(iou(candidate, box) > 0.01 for box in occupied):
            continue
        pad = 4
        region = image[max(0, top - pad) : min(height, top + h + pad), max(0, left - pad) : min(width, left + w + pad)]
        if blue_table_score(region) < 0.45:
            continue
        return left, top
    return None


def make_xany_json(image_name: str, width: int, height: int, boxes: list[Box], names: list[str]) -> dict:
    shapes = []
    for box in boxes:
        x1, y1, x2, y2 = box_to_xyxy(box, width, height)
        shapes.append(
            {
                "label": names[box.cls],
                "score": None,
                "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                "group_id": None,
                "description": "",
                "difficult": False,
                "shape_type": "rectangle",
                "flags": {},
                "attributes": {},
                "kie_linking": [],
            }
        )
    return {
        "version": "4.0.0-beta.11",
        "flags": {},
        "checked": False,
        "shapes": shapes,
        "imagePath": image_name,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
    }


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    base = args.base.resolve()
    backgrounds = args.backgrounds.resolve()
    output = args.output.resolve()
    tile_root = args.tile_root.resolve()
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists, pass --overwrite: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    names = class_names(base)
    name_to_id = {name: index for index, name in enumerate(names)}
    assets = load_tile_assets(tile_root, names)
    copy_tree_dataset(base, output)

    background_pairs = image_label_pairs(backgrounds, "train")
    synthetic_dir = output / "xanylabeling_review_synthetic_paste"
    synthetic_dir.mkdir(parents=True, exist_ok=True)
    pasted_counts: Counter[str] = Counter()
    generated = 0
    skipped = 0

    class_plan = []
    for pair in FOCUS_PAIRS:
        class_plan.extend(pair)
    for pair in EXTRA_PAIRS:
        class_plan.extend(pair)

    for index in range(args.count):
        image_path, label_path = rng.choice(background_pairs)
        image = imread_unicode(image_path, cv2.IMREAD_COLOR)
        if image is None:
            skipped += 1
            continue
        height, width = image.shape[:2]
        original_boxes = read_boxes(label_path)
        occupied = [box_to_xyxy(box, width, height) for box in original_boxes]
        out_image = image.copy()
        boxes = list(original_boxes)

        focus_pairs = list(FOCUS_PAIRS)
        rng.shuffle(focus_pairs)
        selected_pairs = focus_pairs[: rng.randint(1, 3)]
        if rng.random() < 0.40:
            selected_pairs.append(rng.choice(EXTRA_PAIRS))

        pasted_this_image = 0
        for left_name, right_name in selected_pairs:
            for class_name in (left_name, right_name):
                repeat = 2 if class_name in {"2p", "4p"} and rng.random() < 0.55 else 1
                for _ in range(repeat):
                    tile = prepare_tile(rng.choice(assets[class_name]), rng)
                    th, tw = tile.shape[:2]
                    position = find_position(out_image, (th, tw), occupied, rng)
                    if position is None:
                        continue
                    left, top = position
                    out_image, xyxy = paste_tile(out_image, tile, left, top)
                    occupied.append(xyxy)
                    boxes.append(xyxy_to_box(name_to_id[class_name], xyxy, width, height))
                    pasted_counts[class_name] += 1
                    pasted_this_image += 1

        if pasted_this_image < 2:
            skipped += 1
            continue
        stem = f"{image_path.stem}_synthetic_paste_{generated:04d}"
        image_name = f"{stem}.png"
        out_img_path = output / "images" / "train" / image_name
        out_label_path = output / "labels" / "train" / f"{stem}.txt"
        imwrite_unicode(out_img_path, out_image)
        write_boxes(out_label_path, boxes)
        shutil.copy2(out_img_path, synthetic_dir / image_name)
        (synthetic_dir / f"{stem}.json").write_text(
            json.dumps(make_xany_json(image_name, width, height, boxes, names), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        generated += 1

    write_data_yaml(output, names)
    manifest = {
        "base_dataset": str(base),
        "background_dataset": str(backgrounds),
        "tile_root": str(tile_root),
        "output": str(output),
        "synthetic_review_dir": str(synthetic_dir),
        "requested_count": args.count,
        "generated_images": generated,
        "skipped_backgrounds": skipped,
        "focus_pairs": FOCUS_PAIRS,
        "extra_pairs": EXTRA_PAIRS,
        "pasted_counts": dict(pasted_counts.most_common()),
        "notes": [
            "Uses real manually labeled train images as backgrounds.",
            "Pasted tiles avoid existing YOLO boxes and prefer blue table regions.",
            "Validation split is copied unchanged from the base dataset.",
            "X-AnyLabeling JSON for synthetic images is exported beside copied synthetic PNGs.",
        ],
    }
    (output / "synthetic_paste_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (synthetic_dir / "_export_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
