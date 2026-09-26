from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = (
    REPO_ROOT
    / "datasets"
    / "mahjong_yolo_active"
    / "augmented"
    / "hbb_v1_safe_aug"
)
DEFAULT_OUTPUT = REPO_ROOT / "runs" / "mahjong_yolo26_hbb" / "label_audit"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render YOLO label audit images.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_classes(dataset: Path) -> list[str]:
    return [line.strip() for line in (dataset / "classes.txt").read_text(encoding="utf-8").splitlines() if line.strip()]


def read_yolo_label(label_path: Path, width: int, height: int, classes: list[str]) -> list[dict[str, object]]:
    labels: list[dict[str, object]] = []
    if not label_path.exists():
        return labels
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        class_id = int(float(parts[0]))
        cx, cy, bw, bh = [float(v) for v in parts[1:]]
        x1 = (cx - bw / 2.0) * width
        y1 = (cy - bh / 2.0) * height
        x2 = (cx + bw / 2.0) * width
        y2 = (cy + bh / 2.0) * height
        labels.append(
            {
                "class_id": class_id,
                "class_name": classes[class_id] if 0 <= class_id < len(classes) else str(class_id),
                "xyxy": [x1, y1, x2, y2],
            }
        )
    return sorted(labels, key=lambda p: (round(float(p["xyxy"][1]) / 20.0), float(p["xyxy"][0])))


def color_for_index(idx: int) -> tuple[int, int, int]:
    palette = [
        (255, 80, 80),
        (80, 220, 255),
        (120, 255, 120),
        (255, 180, 80),
        (220, 120, 255),
        (80, 160, 255),
        (255, 255, 80),
        (80, 255, 200),
    ]
    return palette[idx % len(palette)]


def clamp_box(xyxy: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = xyxy
    return (
        max(0, min(width - 1, int(round(x1)))),
        max(0, min(height - 1, int(round(y1)))),
        max(0, min(width - 1, int(round(x2)))),
        max(0, min(height - 1, int(round(y2)))),
    )


def draw_numbered(image: np.ndarray, labels: list[dict[str, object]]) -> np.ndarray:
    output = image.copy()
    height, width = output.shape[:2]
    for idx, label in enumerate(labels, start=1):
        x1, y1, x2, y2 = clamp_box(label["xyxy"], width, height)
        color = color_for_index(idx)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        text = str(idx)
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        tag_y1 = max(0, y1 - th - baseline - 4)
        if tag_y1 == 0:
            tag_y1 = min(height - th - baseline - 5, y2 + 3)
        cv2.rectangle(output, (x1, tag_y1), (x1 + tw + 8, tag_y1 + th + baseline + 6), color, -1)
        cv2.putText(output, text, (x1 + 4, tag_y1 + th + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
    return output


def make_crop_sheet(image: np.ndarray, labels: list[dict[str, object]]) -> np.ndarray:
    height, width = image.shape[:2]
    cell_w, cell_h = 160, 190
    cols = 5
    rows = max(1, math.ceil(len(labels) / cols))
    sheet = np.full((rows * cell_h, cols * cell_w, 3), 245, dtype=np.uint8)
    for idx, label in enumerate(labels, start=1):
        row = (idx - 1) // cols
        col = (idx - 1) % cols
        x1, y1, x2, y2 = clamp_box(label["xyxy"], width, height)
        pad = 8
        crop = image[max(0, y1 - pad) : min(height, y2 + pad), max(0, x1 - pad) : min(width, x2 + pad)]
        if crop.size == 0:
            continue
        crop_h, crop_w = crop.shape[:2]
        scale = min((cell_w - 18) / crop_w, (cell_h - 55) / crop_h)
        resized = cv2.resize(crop, (max(1, int(crop_w * scale)), max(1, int(crop_h * scale))), interpolation=cv2.INTER_CUBIC)
        ox = col * cell_w + (cell_w - resized.shape[1]) // 2
        oy = row * cell_h + 10
        sheet[oy : oy + resized.shape[0], ox : ox + resized.shape[1]] = resized
        text = f"#{idx} {label['class_name']}"
        cv2.putText(sheet, text, (col * cell_w + 8, row * cell_h + cell_h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
    return sheet


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve()
    output = args.output_root.resolve() / time.strftime("%Y%m%d-%H%M%S") / args.split
    classes = load_classes(dataset)
    image_dir = dataset / "images" / args.split
    label_dir = dataset / "labels" / args.split
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for image_path in sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        height, width = image.shape[:2]
        labels = read_yolo_label(label_dir / f"{image_path.stem}.txt", width, height, classes)
        case_dir = output / image_path.stem
        case_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(case_dir / "01-source.png"), image)
        cv2.imwrite(str(case_dir / "02-label-numbered.png"), draw_numbered(image, labels))
        cv2.imwrite(str(case_dir / "03-label-crop-sheet.png"), make_crop_sheet(image, labels))
        (case_dir / "labels.json").write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")
        with (case_dir / "labels.csv").open("w", encoding="utf-8-sig", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow(["id", "class_id", "class_name", "x1", "y1", "x2", "y2"])
            for idx, label in enumerate(labels, start=1):
                writer.writerow([idx, label["class_id"], label["class_name"], *[f"{float(v):.2f}" for v in label["xyxy"]]])
        rows.append({"image": image_path.name, "label_count": len(labels), "audit_dir": str(case_dir)})
        print(f"ok {image_path.name}: labels={len(labels)}")

    (output / "label_audit_summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"output={output}")


if __name__ == "__main__":
    main()
