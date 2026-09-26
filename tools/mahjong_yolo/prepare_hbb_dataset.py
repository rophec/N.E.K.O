from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = (
    REPO_ROOT
    / "datasets"
    / "mahjong_yolo_active"
    / "manual_labeling_round2"
    / "exports"
    / "yolo_hbb_export"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "datasets"
    / "mahjong_yolo_active"
    / "train_ready"
    / "hbb_v1"
)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


@dataclass(frozen=True)
class DatasetItem:
    image: Path
    label: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the Mahjong HBB YOLO export for YOLO26 training."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def read_classes(source: Path) -> list[str]:
    classes_path = source / "classes.txt"
    if not classes_path.exists():
        raise FileNotFoundError(f"Missing classes.txt: {classes_path}")
    classes = [
        line.strip()
        for line in classes_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not classes:
        raise ValueError(f"No classes found in {classes_path}")
    return classes


def collect_items(source: Path) -> list[DatasetItem]:
    image_dir = source / "images"
    label_dir = source / "labels"
    if not image_dir.exists():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"Missing label directory: {label_dir}")

    items: list[DatasetItem] = []
    missing_labels: list[str] = []
    for image in sorted(image_dir.iterdir()):
        if not image.is_file() or image.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label = label_dir / f"{image.stem}.txt"
        if not label.exists():
            missing_labels.append(image.name)
            continue
        items.append(DatasetItem(image=image, label=label))

    if missing_labels:
        preview = ", ".join(missing_labels[:10])
        raise ValueError(f"Images without labels: {preview}")
    if not items:
        raise ValueError(f"No image/label pairs found in {source}")
    return items


def validate_label_file(label: Path, class_count: int) -> list[str]:
    errors: list[str] = []
    for line_number, line in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) != 5:
            errors.append(f"{label.name}:{line_number}: expected 5 fields, got {len(parts)}")
            continue
        try:
            class_id = int(parts[0])
            coords = [float(value) for value in parts[1:]]
        except ValueError:
            errors.append(f"{label.name}:{line_number}: non-numeric field")
            continue
        if not 0 <= class_id < class_count:
            errors.append(f"{label.name}:{line_number}: class id {class_id} out of range")
        x_center, y_center, width, height = coords
        if not 0.0 <= x_center <= 1.0 or not 0.0 <= y_center <= 1.0:
            errors.append(f"{label.name}:{line_number}: center outside normalized range")
        if not 0.0 < width <= 1.0 or not 0.0 < height <= 1.0:
            errors.append(f"{label.name}:{line_number}: invalid box size")
    return errors


def validate_dataset(items: list[DatasetItem], class_count: int) -> None:
    errors: list[str] = []
    for item in items:
        errors.extend(validate_label_file(item.label, class_count))
    if errors:
        preview = "\n".join(errors[:30])
        raise ValueError(f"Label validation failed:\n{preview}")


def split_items(
    items: list[DatasetItem], val_ratio: float, seed: int
) -> tuple[list[DatasetItem], list[DatasetItem]]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("--val-ratio must be between 0 and 1")
    shuffled = items[:]
    random.Random(seed).shuffle(shuffled)
    val_count = max(1, round(len(shuffled) * val_ratio))
    val_items = sorted(shuffled[:val_count], key=lambda item: item.image.name)
    train_items = sorted(shuffled[val_count:], key=lambda item: item.image.name)
    if not train_items:
        raise ValueError("Validation split consumed the whole dataset")
    return train_items, val_items


def reset_output(output: Path, clean: bool) -> None:
    if output.exists() and clean:
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)


def copy_items(items: list[DatasetItem], output: Path, split: str) -> None:
    for item in items:
        shutil.copy2(item.image, output / "images" / split / item.image.name)
        shutil.copy2(item.label, output / "labels" / split / item.label.name)


def write_data_yaml(output: Path, classes: list[str]) -> None:
    names = "\n".join(f"  {index}: {name}" for index, name in enumerate(classes))
    data_yaml = (
        f"path: {output.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {len(classes)}\n"
        "names:\n"
        f"{names}\n"
    )
    (output / "data.yaml").write_text(data_yaml, encoding="utf-8")
    (output / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")


def write_manifest(
    output: Path,
    source: Path,
    train_items: list[DatasetItem],
    val_items: list[DatasetItem],
    classes: list[str],
    seed: int,
    val_ratio: float,
) -> None:
    manifest = {
        "source": str(source),
        "output": str(output),
        "task": "detect",
        "format": "YOLO HBB",
        "seed": seed,
        "val_ratio": val_ratio,
        "class_count": len(classes),
        "train_count": len(train_items),
        "val_count": len(val_items),
        "train_images": [item.image.name for item in train_items],
        "val_images": [item.image.name for item in val_items],
    }
    (output / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()

    classes = read_classes(source)
    items = collect_items(source)
    validate_dataset(items, len(classes))
    train_items, val_items = split_items(items, args.val_ratio, args.seed)
    reset_output(output, args.clean)
    copy_items(train_items, output, "train")
    copy_items(val_items, output, "val")
    write_data_yaml(output, classes)
    write_manifest(output, source, train_items, val_items, classes, args.seed, args.val_ratio)

    print(
        json.dumps(
            {
                "source": str(source),
                "output": str(output),
                "classes": len(classes),
                "total": len(items),
                "train": len(train_items),
                "val": len(val_items),
                "data_yaml": str(output / "data.yaml"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
