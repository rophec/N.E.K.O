from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE = (
    REPO_ROOT
    / "datasets"
    / "mahjong_yolo_active"
    / "dataset_registry"
    / "09_manual_gtfix_20260706"
    / "safe_aug_221"
)
DEFAULT_ORIGINAL = (
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
    / "10_targeted_error_aug_20260707"
)
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


@dataclass(frozen=True)
class Box:
    cls: int
    x: float
    y: float
    w: float
    h: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a targeted YOLO HBB dataset for recurring Mahjong class errors."
    )
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--original", type=Path, default=DEFAULT_ORIGINAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--crops-per-image", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def class_names(dataset_root: Path) -> list[str]:
    data = read_yaml(dataset_root / "data.yaml")
    names = data["names"]
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
    for split in ("train", "val"):
        for kind in ("images", "labels"):
            src_dir = source / kind / split
            dst_dir = target / kind / split
            dst_dir.mkdir(parents=True, exist_ok=True)
            for path in src_dir.iterdir():
                if path.is_file():
                    shutil.copy2(path, dst_dir / path.name)


def box_to_xyxy(box: Box, width: int, height: int) -> tuple[float, float, float, float]:
    cx = box.x * width
    cy = box.y * height
    bw = box.w * width
    bh = box.h * height
    return cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0


def xyxy_to_box(cls: int, xyxy: tuple[float, float, float, float], width: int, height: int) -> Box | None:
    x1, y1, x2, y2 = xyxy
    x1 = min(max(x1, 0.0), width - 1.0)
    y1 = min(max(y1, 0.0), height - 1.0)
    x2 = min(max(x2, 0.0), width - 1.0)
    y2 = min(max(y2, 0.0), height - 1.0)
    bw = x2 - x1
    bh = y2 - y1
    if bw < 4 or bh < 4:
        return None
    return Box(cls=cls, x=((x1 + x2) / 2.0) / width, y=((y1 + y2) / 2.0) / height, w=bw / width, h=bh / height)


def crop_boxes(
    boxes: list[Box],
    crop_xyxy: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
) -> list[Box]:
    crop_x1, crop_y1, crop_x2, crop_y2 = crop_xyxy
    crop_width = crop_x2 - crop_x1
    crop_height = crop_y2 - crop_y1
    out: list[Box] = []
    for box in boxes:
        x1, y1, x2, y2 = box_to_xyxy(box, image_width, image_height)
        ix1 = max(x1, crop_x1)
        iy1 = max(y1, crop_y1)
        ix2 = min(x2, crop_x2)
        iy2 = min(y2, crop_y2)
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        visible = ((ix2 - ix1) * (iy2 - iy1)) / max((x2 - x1) * (y2 - y1), 1.0)
        if visible < 0.55:
            continue
        shifted = (ix1 - crop_x1, iy1 - crop_y1, ix2 - crop_x1, iy2 - crop_y1)
        new_box = xyxy_to_box(box.cls, shifted, crop_width, crop_height)
        if new_box is not None:
            out.append(new_box)
    return out


def choose_crop(
    target_box: Box,
    width: int,
    height: int,
    rng: random.Random,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box_to_xyxy(target_box, width, height)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    tile_size = max(x2 - x1, y2 - y1)
    crop_size = int(rng.uniform(5.0, 7.2) * tile_size)
    crop_size = max(240, min(crop_size, min(width, height)))
    cx += rng.uniform(-0.45, 0.45) * tile_size
    cy += rng.uniform(-0.45, 0.45) * tile_size
    left = int(round(cx - crop_size / 2.0))
    top = int(round(cy - crop_size / 2.0))
    left = max(0, min(left, width - crop_size))
    top = max(0, min(top, height - crop_size))
    return left, top, left + crop_size, top + crop_size


def light_color_jitter(image: np.ndarray, rng: random.Random) -> np.ndarray:
    # EN: Keep glyph identity intact; only simulate tiny capture color shifts.
    # ZH: 保留牌面字形，只模拟很小的截图色彩波动。
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-1.5, 1.5)) % 180.0
    hsv[..., 1] *= rng.uniform(0.96, 1.04)
    hsv[..., 2] *= rng.uniform(0.96, 1.04)
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def write_data_yaml(output: Path, names: list[str]) -> None:
    lines = [
        f"path: {output.as_posix()}",
        "train: images/train",
        "val: images/val",
        f"nc: {len(names)}",
        "names:",
    ]
    lines.extend(f"  {index}: {name}" for index, name in enumerate(names))
    (output / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "classes.txt").write_text("\n".join(names) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    base = args.base.resolve()
    original = args.original.resolve()
    output = args.output.resolve()
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists, pass --overwrite: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    names = class_names(base)
    hard_class_names = {
        "1m",
        "2m",
        "3m",
        "4m",
        "5m",
        "6m",
        "8m",
        "9m",
        "2p",
        "4p",
        "5p",
        "6p",
        "8p",
        "9p",
        "5s",
        "6s",
        "7s",
        "1z",
        "2z",
        "4z",
        "5z",
        "6z",
        "7z",
    }
    hard_class_ids = {names.index(name) for name in hard_class_names if name in names}

    copy_tree_dataset(base, output)
    targeted_written = 0
    source_image_count = 0
    hard_box_count = 0
    per_class_written: dict[str, int] = {name: 0 for name in names}

    for image_path, label_path in image_label_pairs(original, "train"):
        boxes = read_boxes(label_path)
        hard_boxes = [box for box in boxes if box.cls in hard_class_ids]
        if not hard_boxes:
            continue
        source_image_count += 1
        hard_box_count += len(hard_boxes)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        height, width = image.shape[:2]

        rng.shuffle(hard_boxes)
        for index, target_box in enumerate(hard_boxes[: args.crops_per_image], start=1):
            crop_xyxy = choose_crop(target_box, width, height, rng)
            crop_x1, crop_y1, crop_x2, crop_y2 = crop_xyxy
            crop = image[crop_y1:crop_y2, crop_x1:crop_x2].copy()
            crop = light_color_jitter(crop, rng)
            cropped_boxes = crop_boxes(boxes, crop_xyxy, width, height)
            if not any(box.cls == target_box.cls for box in cropped_boxes):
                continue
            stem = f"{image_path.stem}_target_{index:02d}_{names[target_box.cls]}"
            out_image = output / "images" / "train" / f"{stem}.png"
            out_label = output / "labels" / "train" / f"{stem}.txt"
            cv2.imwrite(str(out_image), crop)
            write_boxes(out_label, cropped_boxes)
            targeted_written += 1
            for box in cropped_boxes:
                per_class_written[names[box.cls]] += 1

    write_data_yaml(output, names)
    manifest = {
        "base_dataset": str(base),
        "original_dataset": str(original),
        "output": str(output),
        "source_train_images_with_hard_classes": source_image_count,
        "hard_boxes_seen": hard_box_count,
        "targeted_crop_images_written": targeted_written,
        "crops_per_image_limit": args.crops_per_image,
        "hard_class_names": sorted(hard_class_names),
        "targeted_labels_written_by_class": {
            name: count for name, count in per_class_written.items() if count
        },
        "notes": [
            "Base safe_aug_221 train/val is copied first.",
            "Extra crops are generated only from original train split, never from fixed test.",
            "Crops enlarge hard visible tiles while preserving adjacent-tile context.",
        ],
    }
    (output / "targeted_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
