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
DEFAULT_BASE = (
    REPO_ROOT
    / "datasets"
    / "mahjong_yolo_active"
    / "dataset_registry"
    / "10_targeted_error_aug_20260707"
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
    / "11_hard_negative_pairs_20260708"
)
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
PRIMARY_PAIRS = [("2p", "4p"), ("3m", "2m"), ("5m", "6m")]
SECONDARY_PAIRS = [
    ("9s", "6m"),
    ("3z", "4m"),
    ("9s", "8p"),
    ("6m", "7p"),
    ("9p", "8s"),
    ("6p", "3m"),
    ("3p", "1m"),
    ("6p", "9p"),
]


@dataclass(frozen=True)
class Box:
    cls: int
    x: float
    y: float
    w: float
    h: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a context-preserving hard-negative dataset from recurring YOLO class errors."
    )
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--original", type=Path, default=DEFAULT_ORIGINAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--primary-variants", type=int, default=6)
    parser.add_argument("--secondary-variants", type=int, default=3)
    parser.add_argument("--max-crops-per-image", type=int, default=18)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


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
    for split in ("train", "val"):
        for kind in ("images", "labels"):
            src_dir = source / kind / split
            dst_dir = target / kind / split
            dst_dir.mkdir(parents=True, exist_ok=True)
            for path in src_dir.iterdir():
                if path.is_file() and path.suffix.lower() != ".cache":
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
    return Box(cls, ((x1 + x2) / 2.0) / width, ((y1 + y2) / 2.0) / height, bw / width, bh / height)


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
        if visible < 0.62:
            continue
        shifted = (ix1 - crop_x1, iy1 - crop_y1, ix2 - crop_x1, iy2 - crop_y1)
        new_box = xyxy_to_box(box.cls, shifted, crop_width, crop_height)
        if new_box is not None:
            out.append(new_box)
    return out


def union_xyxy(items: list[Box], width: int, height: int) -> tuple[float, float, float, float]:
    boxes = [box_to_xyxy(item, width, height) for item in items]
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def choose_pair_crop(
    boxes: list[Box],
    target: Box,
    partner: Box | None,
    width: int,
    height: int,
    rng: random.Random,
) -> tuple[int, int, int, int]:
    focus = [target] + ([partner] if partner is not None else [])
    x1, y1, x2, y2 = union_xyxy(focus, width, height)
    tile_size = max(x2 - x1, y2 - y1)
    center_x = (x1 + x2) / 2.0 + rng.uniform(-0.4, 0.4) * tile_size
    center_y = (y1 + y2) / 2.0 + rng.uniform(-0.4, 0.4) * tile_size
    span = max(x2 - x1, y2 - y1, tile_size * rng.uniform(5.2, 8.0))
    crop_size = int(max(240, min(span, min(width, height))))
    left = int(round(center_x - crop_size / 2.0))
    top = int(round(center_y - crop_size / 2.0))
    left = max(0, min(left, width - crop_size))
    top = max(0, min(top, height - crop_size))
    return left, top, left + crop_size, top + crop_size


def closest_partner(
    target: Box,
    candidates: list[Box],
    width: int,
    height: int,
) -> Box | None:
    if not candidates:
        return None
    tx1, ty1, tx2, ty2 = box_to_xyxy(target, width, height)
    tcx = (tx1 + tx2) / 2.0
    tcy = (ty1 + ty2) / 2.0
    return min(
        candidates,
        key=lambda box: (
            ((box.x * width) - tcx) ** 2 + ((box.y * height) - tcy) ** 2,
            -box.w * box.h,
        ),
    )


def light_jitter(image: np.ndarray, rng: random.Random) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-1.0, 1.0)) % 180.0
    hsv[..., 1] *= rng.uniform(0.98, 1.03)
    hsv[..., 2] *= rng.uniform(0.97, 1.04)
    out = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    if rng.random() < 0.35:
        blur = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.25, 0.55))
        out = cv2.addWeighted(out, 1.25, blur, -0.25, 0)
    return out


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
    name_to_id = {name: index for index, name in enumerate(names)}
    primary_pairs = [(name_to_id[left], name_to_id[right]) for left, right in PRIMARY_PAIRS]
    secondary_pairs = [(name_to_id[left], name_to_id[right]) for left, right in SECONDARY_PAIRS]

    copy_tree_dataset(base, output)

    written = 0
    per_class_written: Counter[str] = Counter()
    per_pair_written: Counter[str] = Counter()
    source_image_count = 0

    for image_path, label_path in image_label_pairs(original, "train"):
        boxes = read_boxes(label_path)
        if not boxes:
            continue
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        height, width = image.shape[:2]

        planned: list[tuple[Box, Box | None, str, int]] = []
        for left_id, right_id in primary_pairs:
            left_boxes = [box for box in boxes if box.cls == left_id]
            right_boxes = [box for box in boxes if box.cls == right_id]
            for target in left_boxes + right_boxes:
                candidates = right_boxes if target.cls == left_id else left_boxes
                planned.append((target, closest_partner(target, candidates, width, height), f"{names[left_id]}_{names[right_id]}", args.primary_variants))

        for left_id, right_id in secondary_pairs:
            left_boxes = [box for box in boxes if box.cls == left_id]
            right_boxes = [box for box in boxes if box.cls == right_id]
            for target in left_boxes + right_boxes:
                candidates = right_boxes if target.cls == left_id else left_boxes
                planned.append((target, closest_partner(target, candidates, width, height), f"{names[left_id]}_{names[right_id]}", args.secondary_variants))

        if not planned:
            continue
        source_image_count += 1
        rng.shuffle(planned)
        crops_this_image = 0
        for target, partner, pair_name, variant_count in planned:
            if crops_this_image >= args.max_crops_per_image:
                break
            for variant in range(variant_count):
                if crops_this_image >= args.max_crops_per_image:
                    break
                crop_xyxy = choose_pair_crop(boxes, target, partner, width, height, rng)
                crop_x1, crop_y1, crop_x2, crop_y2 = crop_xyxy
                crop = image[crop_y1:crop_y2, crop_x1:crop_x2].copy()
                crop = light_jitter(crop, rng)
                cropped_boxes = crop_boxes(boxes, crop_xyxy, width, height)
                if not any(box.cls == target.cls for box in cropped_boxes):
                    continue
                if partner is not None and not any(box.cls == partner.cls for box in cropped_boxes):
                    continue
                stem = f"{image_path.stem}_hn_{written:04d}_{pair_name}_{names[target.cls]}_v{variant + 1:02d}"
                out_image = output / "images" / "train" / f"{stem}.png"
                out_label = output / "labels" / "train" / f"{stem}.txt"
                cv2.imwrite(str(out_image), crop)
                write_boxes(out_label, cropped_boxes)
                written += 1
                crops_this_image += 1
                per_pair_written[pair_name] += 1
                for box in cropped_boxes:
                    per_class_written[names[box.cls]] += 1

    write_data_yaml(output, names)
    manifest = {
        "base_dataset": str(base),
        "original_dataset": str(original),
        "output": str(output),
        "source_train_images_with_pairs": source_image_count,
        "hard_negative_crop_images_written": written,
        "primary_pairs": PRIMARY_PAIRS,
        "secondary_pairs": SECONDARY_PAIRS,
        "primary_variants": args.primary_variants,
        "secondary_variants": args.secondary_variants,
        "max_crops_per_image": args.max_crops_per_image,
        "crops_written_by_pair": dict(per_pair_written.most_common()),
        "labels_written_by_class": dict(per_class_written.most_common()),
        "notes": [
            "Base dataset is copied first, then extra hard-negative crops are appended to train only.",
            "Fixed test images are not used.",
            "Crops prefer keeping the confused pair in the same local context when both classes are present.",
        ],
    }
    (output / "hard_negative_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
