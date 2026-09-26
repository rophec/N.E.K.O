from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import yaml


def load_names(data_yaml: Path) -> list[str]:
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = data["names"]
    if isinstance(names, dict):
        return [names[index] for index in sorted(names)]
    return list(names)


def image_size(image_path: Path) -> tuple[int, int]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    height, width = image.shape[:2]
    return width, height


def shape_to_yolo_line(shape: dict, label_to_id: dict[str, int], width: int, height: int) -> str | None:
    label = str(shape.get("label", "")).strip()
    if label not in label_to_id:
        raise ValueError(f"Unknown label: {label}")
    points = shape.get("points") or []
    if len(points) < 2:
        return None

    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    x1, x2 = max(0.0, min(xs)), min(float(width), max(xs))
    y1, y2 = max(0.0, min(ys)), min(float(height), max(ys))
    box_w = x2 - x1
    box_h = y2 - y1
    if box_w <= 0 or box_h <= 0:
        return None

    x_center = (x1 + x2) / 2.0 / width
    y_center = (y1 + y2) / 2.0 / height
    norm_w = box_w / width
    norm_h = box_h / height
    return f"{label_to_id[label]} {x_center:.8f} {y_center:.8f} {norm_w:.8f} {norm_h:.8f}"


def convert_one(json_path: Path, image_path: Path, output_path: Path, label_to_id: dict[str, int]) -> int:
    width, height = image_size(image_path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    lines: list[str] = []
    for shape in payload.get("shapes", []):
        line = shape_to_yolo_line(shape, label_to_id, width, height)
        if line is not None:
            lines.append(line)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert X-AnyLabeling HBB rectangle JSON files to YOLO HBB labels.")
    parser.add_argument("--json-dir", required=True, type=Path)
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--data-yaml", required=True, type=Path)
    parser.add_argument("--output-label-dir", required=True, type=Path)
    args = parser.parse_args()

    names = load_names(args.data_yaml)
    label_to_id = {name: index for index, name in enumerate(names)}

    total = 0
    converted = 0
    for json_path in sorted(args.json_dir.glob("*.json")):
        image_path = args.image_dir / f"{json_path.stem}.png"
        if not image_path.exists():
            image_path = args.image_dir / f"{json_path.stem}.jpg"
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image for {json_path.name}")
        count = convert_one(json_path, image_path, args.output_label_dir / f"{json_path.stem}.txt", label_to_id)
        converted += 1
        total += count
        print(f"converted {json_path.name}: labels={count}")
    print(f"done files={converted} labels={total}")


if __name__ == "__main__":
    main()
