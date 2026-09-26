from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
WHITE_DRAGON_CLASS_ID = 31
GREEN_DRAGON_CLASS_ID = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repair propagated 37.png white-dragon labels stored as green dragons."
    )
    parser.add_argument("--datasets", type=Path, nargs="+", required=True)
    parser.add_argument("--source-prefix", default="37")
    parser.add_argument("--max-green-ratio", type=float, default=0.02)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


def find_image(dataset: Path, split: str, stem: str) -> Path:
    for suffix in IMAGE_SUFFIXES:
        candidate = dataset / "images" / split / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing image for {dataset.name}/{split}/{stem}")


def green_pixel_ratio(
    image: np.ndarray, fields: list[str]
) -> tuple[float, tuple[int, int, int, int]]:
    height, width = image.shape[:2]
    center_x, center_y, box_width, box_height = map(float, fields[1:5])
    left = max(0, round((center_x - box_width / 2) * width))
    right = min(width, round((center_x + box_width / 2) * width))
    top = max(0, round((center_y - box_height / 2) * height))
    bottom = min(height, round((center_y + box_height / 2) * height))
    crop = image[top:bottom, left:right].astype(np.float32)
    red, green, blue = crop[:, :, 0], crop[:, :, 1], crop[:, :, 2]
    green_pixels = (green > 70) & (green > red * 1.18) & (green > blue * 1.08)
    return float(green_pixels.mean()), (left, top, right, bottom)


def collect_corrections(
    dataset: Path, source_prefix: str, max_green_ratio: float
) -> list[dict[str, object]]:
    corrections: list[dict[str, object]] = []
    for split in ("train", "val", "test"):
        labels_dir = dataset / "labels" / split
        if not labels_dir.exists():
            continue
        for label_path in sorted(labels_dir.glob(f"{source_prefix}*.txt")):
            image_path = find_image(dataset, split, label_path.stem)
            image = np.asarray(Image.open(image_path).convert("RGB"))
            lines = label_path.read_text(encoding="utf-8").splitlines()
            for line_number, line in enumerate(lines, start=1):
                fields = line.split()
                if not fields or int(fields[0]) != GREEN_DRAGON_CLASS_ID:
                    continue
                ratio, box = green_pixel_ratio(image, fields)
                if ratio >= max_green_ratio:
                    continue
                corrections.append(
                    {
                        "dataset": str(dataset.resolve()),
                        "split": split,
                        "image": str(image_path.resolve()),
                        "label": str(label_path.resolve()),
                        "label_line": line_number,
                        "old_class_id": GREEN_DRAGON_CLASS_ID,
                        "old_class_name": "6z",
                        "new_class_id": WHITE_DRAGON_CLASS_ID,
                        "new_class_name": "5z",
                        "green_pixel_ratio": ratio,
                        "box_xyxy": list(box),
                    }
                )
    return corrections


def apply_corrections(corrections: list[dict[str, object]]) -> None:
    by_label: dict[Path, dict[int, int]] = {}
    for row in corrections:
        by_label.setdefault(Path(str(row["label"])), {})[int(row["label_line"])] = int(
            row["new_class_id"]
        )

    for label_path, changes in by_label.items():
        lines = label_path.read_text(encoding="utf-8").splitlines()
        for line_number, new_class_id in changes.items():
            fields = lines[line_number - 1].split()
            current_class_id = int(fields[0])
            if current_class_id == new_class_id:
                # Derived datasets can share labels through Windows hard links.
                # A correction applied through one path may already be visible here.
                continue
            if current_class_id != GREEN_DRAGON_CLASS_ID:
                raise ValueError(
                    f"Expected class {GREEN_DRAGON_CLASS_ID} at {label_path}:{line_number}"
                )
            fields[0] = str(new_class_id)
            lines[line_number - 1] = " ".join(fields)
        label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def class_counts(dataset: Path, split: str) -> Counter[int]:
    counts: Counter[int] = Counter()
    labels_dir = dataset / "labels" / split
    if not labels_dir.exists():
        return counts
    for label_path in labels_dir.glob("*.txt"):
        for line in label_path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if fields:
                counts[int(fields[0])] += 1
    return counts


def update_metadata(dataset: Path) -> None:
    audit_path = dataset / "DATASET_AUDIT.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if "class_counts" in audit:
            for split in audit["class_counts"]:
                counts = class_counts(dataset, split)
                audit["class_counts"][split] = {
                    str(class_id): counts[class_id] for class_id in range(34)
                }
        audit_path.write_text(
            json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    manifest_path = dataset / "DATASET_MANIFEST.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        splits = manifest.get("audit", {}).get("splits", {})
        names = [
            *(f"{rank}m" for rank in range(1, 10)),
            *(f"{rank}p" for rank in range(1, 10)),
            *(f"{rank}s" for rank in range(1, 10)),
            *(f"{rank}z" for rank in range(1, 8)),
        ]
        for split, split_meta in splits.items():
            counts = class_counts(dataset, split)
            split_meta["class_counts"] = {
                name: counts[class_id] for class_id, name in enumerate(names)
            }
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def main() -> None:
    args = parse_args()
    datasets = [path.resolve() for path in args.datasets]
    corrections = [
        row
        for dataset in datasets
        for row in collect_corrections(
            dataset, args.source_prefix, args.max_green_ratio
        )
    ]
    report = {
        "applied": bool(args.apply),
        "source_prefix": args.source_prefix,
        "max_green_ratio": args.max_green_ratio,
        "datasets": [str(path) for path in datasets],
        "correction_count": len(corrections),
        "counts_by_dataset": dict(
            Counter(Path(str(row["dataset"])).name for row in corrections)
        ),
        "corrections": corrections,
    }
    if args.apply:
        apply_corrections(corrections)
        for dataset in datasets:
            update_metadata(dataset)
    args.report.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.report.resolve().write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "corrections"}, indent=2))


if __name__ == "__main__":
    main()
