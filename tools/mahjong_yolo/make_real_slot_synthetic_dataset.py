from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from make_synthetic_tile_paste_dataset import (
    Box,
    class_names,
    copy_tree_dataset,
    image_label_pairs,
    imread_unicode,
    imwrite_unicode,
    read_boxes,
    write_boxes,
    write_data_yaml,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TILE_ROOT = Path.home() / "Desktop" / "识别图" / "单牌"
DEFAULT_BASE = (
    REPO_ROOT
    / "datasets"
    / "mahjong_yolo_active"
    / "dataset_registry"
    / "14_real_slot_synthetic_20260712"
)
DEFAULT_BACKGROUNDS = (
    REPO_ROOT
    / "datasets"
    / "mahjong_yolo_active"
    / "manual_ground_truth_canonical_20260712"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "datasets"
    / "mahjong_yolo_active"
    / "dataset_registry"
    / "15_real_slot_synthetic_next"
)

# Error-driven groups. Repeating 3m gives the dominant remaining confusion more weight.
TARGET_GROUPS = [
    ("3m", "1m", "2m", "4m"),
    ("3m", "1m", "2m", "4m"),
    ("3m", "1m", "2m", "4m"),
    ("6m", "3m"),
    ("5z", "6z"),
    ("6p", "4m"),
    ("2z", "3z"),
    ("4p", "2p"),
    ("4s", "2s"),
    ("9p", "5p"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replace real labeled tile slots with single-tile assets for targeted YOLO training."
    )
    parser.add_argument("--tile-root", type=Path, default=DEFAULT_TILE_ROOT)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--backgrounds", type=Path, default=DEFAULT_BACKGROUNDS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument(
        "--train-count",
        type=int,
        help="Synthetic images appended to train. Defaults to --count for backward compatibility.",
    )
    parser.add_argument(
        "--val-count",
        type=int,
        default=0,
        help="Synthetic images appended to val using only validation backgrounds.",
    )
    parser.add_argument("--min-replacements", type=int, default=2)
    parser.add_argument("--max-replacements", type=int, default=5)
    parser.add_argument("--preview-count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument(
        "--target-classes",
        nargs="+",
        help="Restrict replacement labels to these classes, for example: 5z 6z.",
    )
    parser.add_argument(
        "--uniform-slots",
        action="store_true",
        help="Sample all labeled tile boxes uniformly instead of using hard-case position weights.",
    )
    parser.add_argument(
        "--slot-orientation",
        choices=("any", "horizontal", "vertical"),
        default="any",
        help="Restrict replaced slots by HBB orientation.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_oriented_assets(tile_root: Path, names: list[str]) -> dict[str, dict[int, list[Path]]]:
    assets = {name: {0: [], 90: [], 180: [], 270: []} for name in names}
    for path in tile_root.rglob("*.png"):
        if path.stem not in assets or path.name == "reference_contact_sheet.png":
            continue
        rotation = int(path.parent.name) if path.parent.name in {"0", "90", "180", "270"} else 0
        assets[path.stem][rotation].append(path)
    missing = [name for name, rotations in assets.items() if not any(rotations.values())]
    if missing:
        raise FileNotFoundError(f"Missing single-tile assets for: {missing}")
    return assets


def box_xyxy(box: Box, width: int, height: int) -> tuple[int, int, int, int]:
    x1 = max(0, int(round((box.x - box.w / 2.0) * width)))
    y1 = max(0, int(round((box.y - box.h / 2.0) * height)))
    x2 = min(width, int(round((box.x + box.w / 2.0) * width)))
    y2 = min(height, int(round((box.y + box.h / 2.0) * height)))
    return x1, y1, x2, y2


def slot_rotation(box: Box) -> int:
    if box.w > box.h:
        return 90 if box.x < 0.5 else 270
    return 180 if box.y < 0.5 else 0


def choose_asset(
    assets: dict[str, dict[int, list[Path]]], class_name: str, rotation: int, rng: random.Random
) -> Path:
    candidates = assets[class_name][rotation]
    if not candidates:
        candidates = [path for paths in assets[class_name].values() for path in paths]
    return rng.choice(candidates)


def slot_weight(box: Box) -> float:
    weight = 1.0
    if box.w > box.h:
        weight *= 4.0
    if box.x >= 0.58:
        weight *= 3.0
    if box.w * box.h < 0.006:
        weight *= 2.0
    return weight


def prepare_replacement(asset_path: Path, width: int, height: int, rng: random.Random) -> np.ndarray:
    tile = imread_unicode(asset_path, cv2.IMREAD_COLOR)
    if tile is None:
        raise RuntimeError(f"Failed to read tile asset: {asset_path}")
    tile = cv2.resize(tile, (width, height), interpolation=cv2.INTER_AREA)
    tile = tile.astype(np.float32)
    tile *= rng.uniform(0.92, 1.08)
    tile += rng.uniform(-7.0, 7.0)
    if rng.random() < 0.30:
        tile = cv2.GaussianBlur(tile, (3, 3), rng.uniform(0.2, 0.7))
    return np.clip(tile, 0, 255).astype(np.uint8)


def replace_slot(image: np.ndarray, tile: np.ndarray, xyxy: tuple[int, int, int, int]) -> None:
    x1, y1, x2, y2 = xyxy
    region = image[y1:y2, x1:x2]
    if region.shape[:2] != tile.shape[:2]:
        raise ValueError("Replacement tile does not match the target slot")
    mask = np.ones(tile.shape[:2], dtype=np.float32)
    if min(tile.shape[:2]) >= 8:
        mask[[0, -1], :] = 0.45
        mask[:, [0, -1]] = 0.45
        mask = cv2.GaussianBlur(mask, (3, 3), 0.6)
    alpha = mask[:, :, None]
    image[y1:y2, x1:x2] = np.clip(tile * alpha + region * (1.0 - alpha), 0, 255).astype(np.uint8)


def choose_target(old_name: str, rng: random.Random, target_classes: tuple[str, ...] | None = None) -> str:
    if target_classes:
        candidates = [name for name in target_classes if name != old_name]
        return rng.choice(candidates or list(target_classes))
    for _ in range(30):
        group = rng.choice(TARGET_GROUPS)
        target = rng.choice(group)
        if target != old_name:
            return target
    return "3m" if old_name != "3m" else "2m"


def draw_preview(image: np.ndarray, changes: list[dict[str, object]]) -> np.ndarray:
    canvas = image.copy()
    for change in changes:
        x1, y1, x2, y2 = change["xyxy"]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 230, 255), 2)
        text = f"{change['old']}->{change['new']}"
        cv2.putText(canvas, text, (x1, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, text, (x1, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 230, 255), 1, cv2.LINE_AA)
    return canvas


def write_contact_sheet(images: list[np.ndarray], path: Path) -> None:
    if not images:
        return
    thumb_size = 400
    columns = 4
    rows = (len(images) + columns - 1) // columns
    sheet = np.full((rows * thumb_size, columns * thumb_size, 3), 245, dtype=np.uint8)
    for index, image in enumerate(images):
        thumb = cv2.resize(image, (thumb_size, thumb_size), interpolation=cv2.INTER_AREA)
        row, column = divmod(index, columns)
        sheet[row * thumb_size : (row + 1) * thumb_size, column * thumb_size : (column + 1) * thumb_size] = thumb
    imwrite_unicode(path, sheet)


def main() -> None:
    args = parse_args()
    if args.min_replacements < 1 or args.max_replacements < args.min_replacements:
        raise ValueError("Invalid replacement range")
    train_count = args.count if args.train_count is None else args.train_count
    if train_count < 0 or args.val_count < 0:
        raise ValueError("Synthetic image counts cannot be negative")
    rng = random.Random(args.seed)
    base = args.base.resolve()
    backgrounds = args.backgrounds.resolve()
    output = args.output.resolve()
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists, pass --overwrite: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    names = class_names(base)
    name_to_id = {name: index for index, name in enumerate(names)}
    target_classes = tuple(args.target_classes) if args.target_classes else None
    if target_classes:
        unknown = sorted(set(target_classes) - set(names))
        if unknown:
            raise ValueError(f"Unknown target classes: {unknown}")
    assets = load_oriented_assets(args.tile_root.resolve(), names)
    copy_tree_dataset(base, output)
    split_stats: dict[str, dict[str, object]] = {}

    for split, count in (("train", train_count), ("val", args.val_count)):
        background_pairs = image_label_pairs(backgrounds, split)
        if args.slot_orientation != "any":
            # EN: Orientation filtering guarantees targeted samples use the weak slot type.
            # ZH: 方向过滤保证定向增强确实使用模型薄弱的牌槽类型。
            background_pairs = [
                pair
                for pair in background_pairs
                if any(
                    (box.w > box.h) == (args.slot_orientation == "horizontal")
                    for box in read_boxes(pair[1])
                )
            ]
        if count and not background_pairs:
            raise RuntimeError(f"No {args.slot_orientation} slots found in {backgrounds / split}")
        replacements_by_class: Counter[str] = Counter()
        replacement_pairs: Counter[str] = Counter()
        orientation_counts: Counter[str] = Counter()
        source_counts: Counter[str] = Counter()
        previews: list[np.ndarray] = []

        for index in range(count):
            image_path, label_path = rng.choice(background_pairs)
            image = imread_unicode(image_path, cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Failed to read background: {image_path}")
            boxes = read_boxes(label_path)
            if not boxes:
                raise RuntimeError(f"Background has no labels: {label_path}")
            height, width = image.shape[:2]
            out_image = image.copy()
            out_boxes = list(boxes)
            available = [
                index
                for index, box in enumerate(boxes)
                if args.slot_orientation == "any"
                or ((box.w > box.h) == (args.slot_orientation == "horizontal"))
            ]
            replacement_count = min(rng.randint(args.min_replacements, args.max_replacements), len(available))
            selected: list[int] = []
            for _ in range(replacement_count):
                weights = None if args.uniform_slots else [slot_weight(boxes[i]) for i in available]
                chosen = rng.choices(available, weights=weights, k=1)[0]
                selected.append(chosen)
                available.remove(chosen)

            changes: list[dict[str, object]] = []
            for box_index in selected:
                old_box = boxes[box_index]
                old_name = names[old_box.cls]
                new_name = choose_target(old_name, rng, target_classes)
                xyxy = box_xyxy(old_box, width, height)
                x1, y1, x2, y2 = xyxy
                if x2 - x1 < 8 or y2 - y1 < 8:
                    continue
                rotation = slot_rotation(old_box)
                asset = choose_asset(assets, new_name, rotation, rng)
                tile = prepare_replacement(asset, x2 - x1, y2 - y1, rng)
                replace_slot(out_image, tile, xyxy)
                out_boxes[box_index] = Box(name_to_id[new_name], old_box.x, old_box.y, old_box.w, old_box.h)
                replacements_by_class[new_name] += 1
                replacement_pairs[f"{old_name}->{new_name}"] += 1
                orientation_counts[str(rotation)] += 1
                changes.append({"old": old_name, "new": new_name, "xyxy": xyxy})

            if not changes:
                raise RuntimeError(f"Could not replace a valid slot in {image_path}")
            stem = f"{image_path.stem}_{split}_real_slot_{index:04d}"
            imwrite_unicode(output / "images" / split / f"{stem}.png", out_image)
            write_boxes(output / "labels" / split / f"{stem}.txt", out_boxes)
            source_counts[image_path.name] += 1
            if len(previews) < args.preview_count:
                previews.append(draw_preview(out_image, changes))

        write_contact_sheet(previews, output / f"real_slot_synthetic_{split}_preview.png")
        split_stats[split] = {
            "generated_images": count,
            "replacements_by_class": dict(replacements_by_class.most_common()),
            "replacement_pairs": dict(replacement_pairs.most_common()),
            "orientation_counts": dict(orientation_counts),
            "source_image_usage": dict(source_counts.most_common()),
        }

    write_data_yaml(output, names)
    manifest = {
        "base_dataset": str(base),
        "background_dataset": str(backgrounds),
        "tile_root": str(args.tile_root.resolve()),
        "output": str(output),
        "generated_images": {"train": train_count, "val": args.val_count},
        "target_groups": [list(target_classes)] if target_classes else TARGET_GROUPS,
        "uniform_slots": args.uniform_slots,
        "slot_orientation": args.slot_orientation,
        "splits": split_stats,
        "notes": [
            "Train synthetic images use only train backgrounds; validation synthetic images use only val backgrounds.",
            "Every synthetic tile replaces an existing manually labeled tile slot.",
            "Position, HBB size, and orientation are inherited from the real slot.",
            "Slot sampling is uniform when --uniform-slots is enabled; otherwise hard-case weights are used.",
            "Fixed test images are never used as source pixels or training labels.",
        ],
    }
    (output / "real_slot_synthetic_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
