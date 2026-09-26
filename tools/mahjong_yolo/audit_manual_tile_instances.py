from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


@dataclass(frozen=True)
class TileInstance:
    instance_id: str
    split: str
    image_path: Path
    label_path: Path
    line_number: int
    class_id: int
    class_name: str
    x_center: float
    y_center: float
    width: float
    height: float
    left: int
    top: int
    right: int
    bottom: int
    image_width: int
    image_height: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit every instance of one tile class in canonical manual ground truth."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--class-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--page-size", type=int, default=30)
    parser.add_argument("--columns", type=int, default=5)
    return parser.parse_args()


def load_class_names(dataset: Path) -> list[str]:
    manifest_path = dataset / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return [str(name) for name in manifest["class_names"]]


def find_image(images_dir: Path, stem: str) -> Path:
    for suffix in IMAGE_SUFFIXES:
        candidate = images_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No image found for label stem: {stem}")


def yolo_box_to_pixels(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    left = round((x_center - width / 2) * image_width)
    top = round((y_center - height / 2) * image_height)
    right = round((x_center + width / 2) * image_width)
    bottom = round((y_center + height / 2) * image_height)
    return (
        max(0, min(image_width - 1, left)),
        max(0, min(image_height - 1, top)),
        max(1, min(image_width, right)),
        max(1, min(image_height, bottom)),
    )


def collect_instances(
    dataset: Path, class_id: int, class_name: str
) -> list[TileInstance]:
    instances: list[TileInstance] = []
    split_counters = {"train": 0, "val": 0, "test": 0}
    for split in ("train", "val", "test"):
        labels_dir = dataset / "labels" / split
        images_dir = dataset / "images" / split
        for label_path in sorted(labels_dir.glob("*.txt")):
            matching_lines: list[tuple[int, list[str]]] = []
            for line_number, line in enumerate(
                label_path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                fields = line.split()
                if fields and int(fields[0]) == class_id:
                    matching_lines.append((line_number, fields))
            if not matching_lines:
                continue

            image_path = find_image(images_dir, label_path.stem)
            with Image.open(image_path) as image:
                image_width, image_height = image.size

            for line_number, fields in matching_lines:
                split_counters[split] += 1
                x_center, y_center, width, height = map(float, fields[1:5])
                left, top, right, bottom = yolo_box_to_pixels(
                    x_center,
                    y_center,
                    width,
                    height,
                    image_width,
                    image_height,
                )
                instances.append(
                    TileInstance(
                        instance_id=f"{split}_{split_counters[split]:03d}",
                        split=split,
                        image_path=image_path.resolve(),
                        label_path=label_path.resolve(),
                        line_number=line_number,
                        class_id=class_id,
                        class_name=class_name,
                        x_center=x_center,
                        y_center=y_center,
                        width=width,
                        height=height,
                        left=left,
                        top=top,
                        right=right,
                        bottom=bottom,
                        image_width=image_width,
                        image_height=image_height,
                    )
                )
    return instances


def context_bounds(instance: TileInstance) -> tuple[int, int, int, int]:
    box_width = instance.right - instance.left
    box_height = instance.bottom - instance.top
    pad_x = max(8, round(box_width * 0.65))
    pad_y = max(8, round(box_height * 0.40))
    return (
        max(0, instance.left - pad_x),
        max(0, instance.top - pad_y),
        min(instance.image_width, instance.right + pad_x),
        min(instance.image_height, instance.bottom + pad_y),
    )


def render_instance(instance: TileInstance, output_path: Path) -> Image.Image:
    with Image.open(instance.image_path) as source:
        source = source.convert("RGB")
        crop_left, crop_top, crop_right, crop_bottom = context_bounds(instance)
        crop = source.crop((crop_left, crop_top, crop_right, crop_bottom))

    draw = ImageDraw.Draw(crop)
    box = (
        instance.left - crop_left,
        instance.top - crop_top,
        instance.right - crop_left,
        instance.bottom - crop_top,
    )
    line_width = max(2, round(min(crop.size) / 80))
    draw.rectangle(box, outline=(255, 32, 32), width=line_width)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(output_path, quality=95)
    return crop


def fit_thumbnail(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    result = Image.new("RGB", size, "white")
    thumbnail = image.copy()
    thumbnail.thumbnail((size[0] - 8, size[1] - 8), Image.Resampling.LANCZOS)
    x = (size[0] - thumbnail.width) // 2
    y = (size[1] - thumbnail.height) // 2
    result.paste(thumbnail, (x, y))
    return result


def render_contact_sheets(
    instances: list[TileInstance],
    crops: dict[str, Image.Image],
    output_dir: Path,
    page_size: int,
    columns: int,
) -> list[Path]:
    font = ImageFont.load_default()
    cell_width, cell_height = 260, 230
    image_height = 176
    rows = (page_size + columns - 1) // columns
    paths: list[Path] = []
    for page_index, start in enumerate(range(0, len(instances), page_size), start=1):
        page_items = instances[start : start + page_size]
        sheet = Image.new(
            "RGB", (columns * cell_width, rows * cell_height), (238, 238, 238)
        )
        draw = ImageDraw.Draw(sheet)
        for index, instance in enumerate(page_items):
            row, column = divmod(index, columns)
            x, y = column * cell_width, row * cell_height
            thumb = fit_thumbnail(crops[instance.instance_id], (cell_width, image_height))
            sheet.paste(thumb, (x, y))
            source_name = instance.image_path.name
            if len(source_name) > 30:
                source_name = f"{source_name[:27]}..."
            draw.text((x + 5, y + image_height + 5), instance.instance_id, fill="black", font=font)
            draw.text((x + 5, y + image_height + 20), source_name, fill="black", font=font)
            draw.text(
                (x + 5, y + image_height + 35),
                f"line {instance.line_number} box {instance.right-instance.left}x{instance.bottom-instance.top}",
                fill="black",
                font=font,
            )
        page_path = output_dir / f"contact_sheet_{page_index:02d}.jpg"
        page_path.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(page_path, quality=94)
        paths.append(page_path)
    return paths


def write_csv(instances: list[TileInstance], output_path: Path) -> None:
    fieldnames = [
        "instance_id",
        "split",
        "source_image",
        "source_label",
        "label_line",
        "class_id",
        "class_name",
        "x_center_norm",
        "y_center_norm",
        "width_norm",
        "height_norm",
        "left_px",
        "top_px",
        "right_px",
        "bottom_px",
        "box_width_px",
        "box_height_px",
        "box_aspect_width_over_height",
    ]
    with output_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for item in instances:
            box_width = item.right - item.left
            box_height = item.bottom - item.top
            writer.writerow(
                {
                    "instance_id": item.instance_id,
                    "split": item.split,
                    "source_image": str(item.image_path),
                    "source_label": str(item.label_path),
                    "label_line": item.line_number,
                    "class_id": item.class_id,
                    "class_name": item.class_name,
                    "x_center_norm": item.x_center,
                    "y_center_norm": item.y_center,
                    "width_norm": item.width,
                    "height_norm": item.height,
                    "left_px": item.left,
                    "top_px": item.top,
                    "right_px": item.right,
                    "bottom_px": item.bottom,
                    "box_width_px": box_width,
                    "box_height_px": box_height,
                    "box_aspect_width_over_height": round(box_width / box_height, 6),
                }
            )


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve()
    output = args.output.resolve()
    class_names = load_class_names(dataset)
    if args.class_name not in class_names:
        raise ValueError(f"Unknown class {args.class_name!r}; choices: {class_names}")
    class_id = class_names.index(args.class_name)
    instances = collect_instances(dataset, class_id, args.class_name)
    output.mkdir(parents=True, exist_ok=True)

    crops: dict[str, Image.Image] = {}
    for instance in instances:
        crop_path = output / "instances" / instance.split / f"{instance.instance_id}.png"
        crops[instance.instance_id] = render_instance(instance, crop_path)

    sheets: dict[str, list[str]] = {}
    for split in ("train", "val", "test"):
        split_instances = [item for item in instances if item.split == split]
        split_paths = render_contact_sheets(
            split_instances,
            crops,
            output / "contact_sheets" / split,
            args.page_size,
            args.columns,
        )
        sheets[split] = [str(path) for path in split_paths]

    write_csv(instances, output / "instances.csv")
    counts = {
        split: sum(item.split == split for item in instances)
        for split in ("train", "val", "test")
    }
    source_image_counts = {
        split: len({item.image_path for item in instances if item.split == split})
        for split in ("train", "val", "test")
    }
    summary = {
        "dataset": str(dataset),
        "class_id": class_id,
        "class_name": args.class_name,
        "instance_count": len(instances),
        "instances_by_split": counts,
        "source_images_by_split": source_image_counts,
        "contact_sheets": sheets,
        "notes": [
            "Only canonical manually verified labels are read.",
            "Red rectangles show the exact HBB ground-truth boxes.",
            "Context outside each box is retained only for human review.",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
