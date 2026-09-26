"""Build a leakage-safe Mahjong dataset for tile-face fine-tuning."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
from collections import Counter
from pathlib import Path

import cv2

from make_real_slot_synthetic_dataset import (
    box_xyxy,
    choose_asset,
    draw_preview,
    load_oriented_assets,
    prepare_replacement,
    replace_slot,
    slot_rotation,
    write_contact_sheet,
)
from make_synthetic_tile_paste_dataset import (
    Box,
    class_names,
    imread_unicode,
    imwrite_unicode,
    read_boxes,
    write_boxes,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = (
    REPO_ROOT
    / "datasets"
    / "mahjong_yolo_active"
    / "dataset_registry"
    / "15_balanced_dragon_features_20260713"
)
DEFAULT_CANONICAL = (
    REPO_ROOT / "datasets" / "mahjong_yolo_active" / "manual_ground_truth_canonical_20260712"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "datasets"
    / "mahjong_yolo_active"
    / "dataset_registry"
    / "16_real_val_face_detail_20260714"
)
DEFAULT_TILE_ROOT = Path.home() / "Desktop" / "识别图" / "单牌"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

# EN: Repeated groups intentionally weight the high-confidence 7z/7p and
# white/green-dragon confusions more heavily.
# 中文：重复分组用于提高高置信度 7z/7p 与白板/发财混淆样本的权重。
TARGET_GROUPS = [
    ("7z", "7p"),
    ("7z", "7p"),
    ("7z", "7p"),
    ("5z", "6z", "1m"),
    ("5z", "6z", "1m"),
    ("3m", "5m", "6m"),
    ("3m", "5m", "6m"),
    ("3p", "3m"),
    ("2z", "3z"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--canonical", type=Path, default=DEFAULT_CANONICAL)
    parser.add_argument("--tile-root", type=Path, default=DEFAULT_TILE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--extra-real-val", type=int, default=11)
    parser.add_argument("--synthetic-train", type=int, default=450)
    parser.add_argument("--min-replacements", type=int, default=2)
    parser.add_argument("--max-replacements", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def image_files(directory: Path) -> list[Path]:
    return sorted(path for path in directory.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)


def link_or_copy(source: Path, target: Path) -> str:
    """Reuse immutable dataset files with hard links when possible.

    EN: Hard links avoid another full copy of the historical datasets.
    中文：优先硬链接，避免历史数据集再次完整占用磁盘空间。
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def read_feature_counts(label_path: Path) -> Counter[tuple[int, str]]:
    features: Counter[tuple[int, str]] = Counter()
    for box in read_boxes(label_path):
        orientation = "horizontal" if box.w > box.h else "vertical"
        features[(box.cls, orientation)] += 1
    return features


def choose_extra_validation(
    canonical: Path, existing_val: list[Path], count: int,
) -> list[Path]:
    """Greedily improve class/orientation coverage using real source images."""
    current: Counter[tuple[int, str]] = Counter()
    for image_path in existing_val:
        current.update(read_feature_counts(canonical / "labels" / "val" / f"{image_path.stem}.txt"))

    remaining = image_files(canonical / "images" / "train")
    selected: list[Path] = []
    for _ in range(min(count, len(remaining))):
        def score(path: Path) -> tuple[float, str]:
            features = read_feature_counts(canonical / "labels" / "train" / f"{path.stem}.txt")
            coverage = sum(amount / (1.0 + current[key]) for key, amount in features.items())
            return coverage, path.name

        best = max(remaining, key=score)
        selected.append(best)
        current.update(read_feature_counts(canonical / "labels" / "train" / f"{best.stem}.txt"))
        remaining.remove(best)
    return selected


def family_for(stem: str, canonical_stems: list[str]) -> str | None:
    for source_stem in canonical_stems:
        if stem == source_stem or stem.startswith(f"{source_stem}_"):
            return source_stem
    return None


def write_data_yaml(output: Path, names: list[str]) -> None:
    lines = [
        f"path: {output.as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        f"nc: {len(names)}",
        "names:",
    ]
    lines.extend(f"  {index}: {name}" for index, name in enumerate(names))
    (output / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "classes.txt").write_text("\n".join(names) + "\n", encoding="utf-8")


def choose_target(old_name: str, rng: random.Random) -> str:
    for _ in range(20):
        group = rng.choice(TARGET_GROUPS)
        choices = [name for name in group if name != old_name]
        if choices:
            return rng.choice(choices)
    raise RuntimeError(f"Could not choose a target class for {old_name}")


def append_hard_samples(
    output: Path,
    canonical: Path,
    retained_train_stems: set[str],
    tile_root: Path,
    names: list[str],
    count: int,
    min_replacements: int,
    max_replacements: int,
    rng: random.Random,
) -> dict[str, object]:
    name_to_id = {name: index for index, name in enumerate(names)}
    assets = load_oriented_assets(tile_root, names)
    backgrounds = [
        path for path in image_files(canonical / "images" / "train")
        if path.stem in retained_train_stems
    ]
    replacements_by_class: Counter[str] = Counter()
    orientation_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    previews = []

    for index in range(count):
        image_path = rng.choice(backgrounds)
        label_path = canonical / "labels" / "train" / f"{image_path.stem}.txt"
        image = imread_unicode(image_path, cv2.IMREAD_COLOR)
        boxes = read_boxes(label_path)
        if image is None or not boxes:
            raise RuntimeError(f"Invalid real background: {image_path}")
        height, width = image.shape[:2]
        output_image = image.copy()
        output_boxes = list(boxes)
        replacement_count = min(rng.randint(min_replacements, max_replacements), len(boxes))
        available = list(range(len(boxes)))
        selected = rng.sample(available, replacement_count)
        changes: list[dict[str, object]] = []

        for box_index in selected:
            old_box = boxes[box_index]
            old_name = names[old_box.cls]
            new_name = choose_target(old_name, rng)
            xyxy = box_xyxy(old_box, width, height)
            x1, y1, x2, y2 = xyxy
            if x2 - x1 < 8 or y2 - y1 < 8:
                continue
            rotation = slot_rotation(old_box)
            asset = choose_asset(assets, new_name, rotation, rng)
            tile = prepare_replacement(asset, x2 - x1, y2 - y1, rng)
            replace_slot(output_image, tile, xyxy)
            output_boxes[box_index] = Box(name_to_id[new_name], old_box.x, old_box.y, old_box.w, old_box.h)
            replacements_by_class[new_name] += 1
            orientation_counts[str(rotation)] += 1
            changes.append({"old": old_name, "new": new_name, "xyxy": xyxy})

        if not changes:
            raise RuntimeError(f"No valid replacement in {image_path}")
        stem = f"{image_path.stem}_face_detail_{index:04d}"
        imwrite_unicode(output / "images" / "train" / f"{stem}.png", output_image)
        write_boxes(output / "labels" / "train" / f"{stem}.txt", output_boxes)
        source_counts[image_path.name] += 1
        if len(previews) < 16:
            previews.append(draw_preview(output_image, changes))

    write_contact_sheet(previews, output / "face_detail_train_preview.png")
    return {
        "generated_images": count,
        "replacements_by_class": dict(replacements_by_class.most_common()),
        "orientation_counts": dict(orientation_counts),
        "source_image_usage": dict(source_counts.most_common()),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_dataset(output: Path, names: list[str], selected_val_stems: set[str]) -> dict[str, object]:
    splits: dict[str, object] = {}
    hashes: dict[str, set[str]] = {}
    for split in ("train", "val", "test"):
        images = image_files(output / "images" / split)
        labels = list((output / "labels" / split).glob("*.txt"))
        class_counts: Counter[int] = Counter()
        boxes = 0
        for label in labels:
            parsed = read_boxes(label)
            boxes += len(parsed)
            class_counts.update(box.cls for box in parsed)
        hashes[split] = {sha256(path) for path in images}
        splits[split] = {
            "images": len(images),
            "labels": len(labels),
            "boxes": boxes,
            "class_counts": {names[index]: class_counts[index] for index in range(len(names))},
            "missing_labels": [
                path.name for path in images
                if not (output / "labels" / split / f"{path.stem}.txt").exists()
            ],
        }

    leaked_families = sorted(
        stem for stem in selected_val_stems
        if any(family_for(path.stem, [stem]) == stem for path in image_files(output / "images" / "train"))
    )
    return {
        "splits": splits,
        "selected_val_family_leakage": leaked_families,
        "exact_hash_overlap": {
            "train_val": sorted(hashes["train"] & hashes["val"]),
            "train_test": sorted(hashes["train"] & hashes["test"]),
            "val_test": sorted(hashes["val"] & hashes["test"]),
        },
    }


def main() -> None:
    args = parse_args()
    if args.min_replacements < 1 or args.max_replacements < args.min_replacements:
        raise ValueError("Invalid replacement count range")
    source = args.source.resolve()
    canonical = args.canonical.resolve()
    output = args.output.resolve()
    tile_root = args.tile_root.resolve()
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists, pass --overwrite: {output}")
        shutil.rmtree(output)
    for split in ("train", "val", "test"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    names = class_names(source)
    existing_val = image_files(canonical / "images" / "val")
    extra_val = choose_extra_validation(canonical, existing_val, args.extra_real_val)
    selected_val_stems = {path.stem for path in existing_val + extra_val}
    canonical_train_stems = sorted(
        (path.stem for path in image_files(canonical / "images" / "train")),
        key=len,
        reverse=True,
    )
    retained_train_stems = set(canonical_train_stems) - {path.stem for path in extra_val}
    link_stats: Counter[str] = Counter()
    excluded_train_files: list[str] = []

    for image_path in image_files(source / "images" / "train"):
        family = family_for(image_path.stem, canonical_train_stems)
        if family in selected_val_stems:
            excluded_train_files.append(image_path.name)
            continue
        label_path = source / "labels" / "train" / f"{image_path.stem}.txt"
        link_stats[link_or_copy(image_path, output / "images" / "train" / image_path.name)] += 1
        link_stats[link_or_copy(label_path, output / "labels" / "train" / label_path.name)] += 1

    for split, paths in (("val", existing_val), ("val", extra_val)):
        source_split = "val" if paths is existing_val else "train"
        for image_path in paths:
            label_path = canonical / "labels" / source_split / f"{image_path.stem}.txt"
            link_stats[link_or_copy(image_path, output / "images" / split / image_path.name)] += 1
            link_stats[link_or_copy(label_path, output / "labels" / split / label_path.name)] += 1

    for image_path in image_files(canonical / "images" / "test"):
        label_path = canonical / "labels" / "test" / f"{image_path.stem}.txt"
        link_stats[link_or_copy(image_path, output / "images" / "test" / image_path.name)] += 1
        link_stats[link_or_copy(label_path, output / "labels" / "test" / label_path.name)] += 1

    rng = random.Random(args.seed)
    synthetic = append_hard_samples(
        output=output,
        canonical=canonical,
        retained_train_stems=retained_train_stems,
        tile_root=tile_root,
        names=names,
        count=args.synthetic_train,
        min_replacements=args.min_replacements,
        max_replacements=args.max_replacements,
        rng=rng,
    )
    write_data_yaml(output, names)
    audit = audit_dataset(output, names, selected_val_stems)
    manifest = {
        "source_dataset": str(source),
        "canonical_dataset": str(canonical),
        "output": str(output),
        "seed": args.seed,
        "existing_real_val": [path.name for path in existing_val],
        "extra_real_val_from_old_train": [path.name for path in extra_val],
        "retained_real_train_families": sorted(retained_train_stems),
        "excluded_train_derivatives": len(excluded_train_files),
        "link_stats": dict(link_stats),
        "synthetic_train": synthetic,
        "target_groups": [list(group) for group in TARGET_GROUPS],
        "audit": audit,
        "notes": [
            "Validation contains only manually labeled real images.",
            "Synthetic images are appended to train only.",
            "All derivatives of validation source families are excluded from train.",
            "The fixed 54-image test split is hard-linked unchanged and never used for training.",
        ],
    }
    (output / "DATASET_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if audit["selected_val_family_leakage"] or any(audit["exact_hash_overlap"].values()):
        raise RuntimeError("Dataset leakage audit failed; inspect DATASET_MANIFEST.json")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
