from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN_VAL = (
    REPO_ROOT
    / "datasets"
    / "mahjong_yolo_active"
    / "dataset_registry"
    / "09_manual_gtfix_20260706"
    / "train_ready_53"
)
DEFAULT_TEST = (
    REPO_ROOT
    / "datasets"
    / "mahjong_yolo_active"
    / "dataset_registry"
    / "13_combined_fixed_test_54_20260710"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "datasets"
    / "mahjong_yolo_active"
    / "manual_ground_truth_canonical_20260712"
)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assemble the manually verified Mahjong ground truth into one canonical dataset."
    )
    parser.add_argument("--train-val", type=Path, default=DEFAULT_TRAIN_VAL)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_names(data_yaml: Path) -> list[str]:
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = data["names"]
    if isinstance(names, dict):
        return [names[index] for index in sorted(names)]
    return list(names)


def read_image(path: Path) -> np.ndarray:
    encoded = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")
    return image


def image_files(directory: Path) -> list[Path]:
    return sorted(path for path in directory.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def yolo_to_xany(
    image_name: str,
    width: int,
    height: int,
    label_lines: list[str],
    names: list[str],
) -> dict:
    shapes = []
    for line_number, line in enumerate(label_lines, start=1):
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Invalid YOLO label at {image_name}:{line_number}: {line}")
        class_id = int(parts[0])
        if not 0 <= class_id < len(names):
            raise ValueError(f"Invalid class id at {image_name}:{line_number}: {class_id}")
        cx, cy, box_width, box_height = map(float, parts[1:])
        if not all(0.0 <= value <= 1.0 for value in (cx, cy, box_width, box_height)):
            raise ValueError(f"Out-of-range YOLO box at {image_name}:{line_number}: {line}")
        x1 = (cx - box_width / 2.0) * width
        y1 = (cy - box_height / 2.0) * height
        x2 = (cx + box_width / 2.0) * width
        y2 = (cy + box_height / 2.0) * height
        shapes.append(
            {
                "label": names[class_id],
                "score": None,
                "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                "group_id": None,
                "description": "manually verified ground truth",
                "difficult": False,
                "shape_type": "rectangle",
                "flags": {},
                "attributes": {},
                "kie_linking": [],
            }
        )
    return {
        "version": "4.0.0-beta.11",
        "flags": {"canonical_manual_ground_truth": True},
        "checked": True,
        "shapes": shapes,
        "imagePath": image_name,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_split(
    source_root: Path,
    source_split: str,
    output: Path,
    output_split: str,
    names: list[str],
) -> dict:
    source_images = source_root / "images" / source_split
    source_labels = source_root / "labels" / source_split
    target_images = output / "images" / output_split
    target_labels = output / "labels" / output_split
    target_images.mkdir(parents=True, exist_ok=True)
    target_labels.mkdir(parents=True, exist_ok=True)

    class_counts: Counter[str] = Counter()
    hashes: dict[str, str] = {}
    total_boxes = 0
    images = image_files(source_images)
    for image_path in images:
        label_path = source_labels / f"{image_path.stem}.txt"
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label for {image_path}: {label_path}")
        lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        image = read_image(image_path)
        height, width = image.shape[:2]

        target_image = target_images / image_path.name
        target_label = target_labels / label_path.name
        shutil.copy2(image_path, target_image)
        shutil.copy2(label_path, target_label)
        xany = yolo_to_xany(image_path.name, width, height, lines, names)
        (target_images / f"{image_path.stem}.json").write_text(
            json.dumps(xany, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        hashes[image_path.name] = sha256(target_image)
        total_boxes += len(lines)
        for line in lines:
            class_counts[names[int(line.split()[0])]] += 1

    return {
        "images": len(images),
        "yolo_labels": len(images),
        "xany_json": len(images),
        "boxes": total_boxes,
        "class_counts": dict(class_counts),
        "image_sha256": hashes,
    }


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


def main() -> None:
    args = parse_args()
    train_val = args.train_val.resolve()
    test = args.test.resolve()
    output = args.output.resolve()
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists, pass --overwrite: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    names = read_names(train_val / "data.yaml")
    if read_names(test / "data.yaml") != names:
        raise ValueError("Train/val and test class mappings differ")
    splits = {
        "train": copy_split(train_val, "train", output, "train", names),
        "val": copy_split(train_val, "val", output, "val", names),
        "test": copy_split(test, "test", output, "test", names),
    }

    hash_to_splits: dict[str, set[str]] = {}
    for split, summary in splits.items():
        for digest in summary["image_sha256"].values():
            hash_to_splits.setdefault(digest, set()).add(split)
    cross_split_duplicates = {
        digest: sorted(found_splits)
        for digest, found_splits in hash_to_splits.items()
        if len(found_splits) > 1
    }
    if cross_split_duplicates:
        raise ValueError(f"Cross-split duplicate images found: {cross_split_duplicates}")

    manifest = {
        "description": "Canonical manually verified Mahjong HBB ground truth. No augmented or synthetic images.",
        "train_val_source": str(train_val),
        "test_source": str(test),
        "output": str(output),
        "class_names": names,
        "splits": splits,
        "cross_split_duplicate_hashes": cross_split_duplicates,
        "xany_json_policy": (
            "JSON files are normalized from the final manually verified YOLO TXT labels so both formats are identical."
        ),
    }
    (output / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "README.md").write_text(
        "# Canonical Manual Ground Truth\n\n"
        "This directory contains only manually verified Mahjong labels. It contains no augmented, synthetic, "
        "or model-predicted samples.\n\n"
        "- `images/<split>/`: image and matching X-AnyLabeling JSON. Open this directory directly in X-AnyLabeling.\n"
        "- `labels/<split>/`: matching YOLO HBB TXT labels.\n"
        "- `data.yaml`: Ultralytics train/val/test definition.\n"
        "- `MANIFEST.json`: counts, provenance, class counts, and image SHA-256 values.\n",
        encoding="utf-8",
    )
    write_data_yaml(output, names)
    compact = {
        split: {key: value for key, value in summary.items() if key not in {"class_counts", "image_sha256"}}
        for split, summary in splits.items()
    }
    print(json.dumps({"output": str(output), "splits": compact}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
