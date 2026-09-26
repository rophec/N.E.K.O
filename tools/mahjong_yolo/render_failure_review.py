from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = REPO_ROOT / "datasets" / "mahjong_yolo_active" / "augmented" / "hbb_v1_safe_aug"
DEFAULT_MODEL = (
    REPO_ROOT
    / "runs"
    / "detect"
    / "runs"
    / "mahjong_yolo26_hbb"
    / "hbb_v2_official_safe_aug"
    / "weights"
    / "best.pt"
)
DEFAULT_OUTPUT = REPO_ROOT / "runs" / "mahjong_yolo26_hbb" / "failure_review"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review ground-truth labels missed by the model.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--split", default="val")
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--match-iou", type=float, default=0.35)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_classes(dataset: Path) -> list[str]:
    return [line.strip() for line in (dataset / "classes.txt").read_text(encoding="utf-8").splitlines() if line.strip()]


def read_labels(path: Path, width: int, height: int, classes: list[str]) -> list[dict[str, object]]:
    labels: list[dict[str, object]] = []
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        class_id = int(float(parts[0]))
        cx, cy, bw, bh = map(float, parts[1:])
        labels.append(
            {
                "id": idx,
                "class_id": class_id,
                "class_name": classes[class_id],
                "xyxy": [
                    (cx - bw / 2.0) * width,
                    (cy - bh / 2.0) * height,
                    (cx + bw / 2.0) * width,
                    (cy + bh / 2.0) * height,
                ],
            }
        )
    return labels


def iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def predict(model: YOLO, image_path: Path, conf: float, iou_threshold: float, device: str) -> list[dict[str, object]]:
    result = model.predict(str(image_path), conf=conf, iou=iou_threshold, device=device, verbose=False)[0]
    predictions: list[dict[str, object]] = []
    for box in result.boxes:
        class_id = int(box.cls.item())
        predictions.append(
            {
                "class_id": class_id,
                "class_name": result.names[class_id],
                "confidence": float(box.conf.item()),
                "xyxy": [float(v) for v in box.xyxy[0].tolist()],
            }
        )
    return predictions


def classify_label_match(label: dict[str, object], predictions: list[dict[str, object]], match_iou: float) -> dict[str, object]:
    same_class = [p for p in predictions if p["class_id"] == label["class_id"]]
    best_any = max(predictions, key=lambda p: iou(label["xyxy"], p["xyxy"]), default=None)
    best_same = max(same_class, key=lambda p: iou(label["xyxy"], p["xyxy"]), default=None)
    best_any_iou = iou(label["xyxy"], best_any["xyxy"]) if best_any else 0.0
    best_same_iou = iou(label["xyxy"], best_same["xyxy"]) if best_same else 0.0
    if best_same_iou >= match_iou:
        reason = "matched"
    elif best_any_iou >= match_iou:
        reason = "class_mismatch"
    elif best_same_iou > 0.05:
        reason = "box_offset_same_class"
    elif best_any_iou > 0.05:
        reason = "box_offset_or_nearby_other_class"
    else:
        reason = "missed_or_suppressed"
    return {
        **label,
        "match_reason": reason,
        "best_any_iou": round(best_any_iou, 4),
        "best_same_iou": round(best_same_iou, 4),
        "best_any_class": best_any["class_name"] if best_any else "",
        "best_any_confidence": round(float(best_any["confidence"]), 4) if best_any else 0.0,
        "best_same_confidence": round(float(best_same["confidence"]), 4) if best_same else 0.0,
    }


def crop_sheet(image: np.ndarray, rows: list[dict[str, object]]) -> np.ndarray:
    cell_w, cell_h = 190, 210
    cols = 4
    sheet = np.full((max(1, math.ceil(len(rows) / cols)) * cell_h, cols * cell_w, 3), 245, dtype=np.uint8)
    height, width = image.shape[:2]
    for idx, row in enumerate(rows, start=1):
        col = (idx - 1) % cols
        line = (idx - 1) // cols
        x1, y1, x2, y2 = [int(round(v)) for v in row["xyxy"]]
        pad = 14
        crop = image[max(0, y1 - pad) : min(height, y2 + pad), max(0, x1 - pad) : min(width, x2 + pad)]
        if crop.size == 0:
            continue
        scale = min((cell_w - 20) / crop.shape[1], (cell_h - 70) / crop.shape[0])
        resized = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), max(1, int(crop.shape[0] * scale))), interpolation=cv2.INTER_CUBIC)
        ox = col * cell_w + (cell_w - resized.shape[1]) // 2
        oy = line * cell_h + 8
        sheet[oy : oy + resized.shape[0], ox : ox + resized.shape[1]] = resized
        text1 = f"#{row['id']} {row['class_name']} {row['match_reason']}"
        text2 = f"same {row['best_same_iou']:.2f} any {row['best_any_iou']:.2f}"
        cv2.putText(sheet, text1, (col * cell_w + 6, line * cell_h + cell_h - 42), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1, cv2.LINE_AA)
        cv2.putText(sheet, text2, (col * cell_w + 6, line * cell_h + cell_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1, cv2.LINE_AA)
    return sheet


def draw_failures(image: np.ndarray, failures: list[dict[str, object]]) -> np.ndarray:
    out = image.copy()
    for row in failures:
        x1, y1, x2, y2 = [int(round(v)) for v in row["xyxy"]]
        color = (60, 60, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, f"#{row['id']} {row['class_name']}", (x1, max(16, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2, cv2.LINE_AA)
    return out


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve()
    output = args.output_root.resolve() / time.strftime("%Y%m%d-%H%M%S")
    output.mkdir(parents=True, exist_ok=True)
    classes = read_classes(dataset)
    model = YOLO(str(args.model.resolve()))
    all_rows: list[dict[str, object]] = []
    summary: list[dict[str, object]] = []

    for image_path in sorted((dataset / "images" / args.split).iterdir()):
        if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        labels = read_labels(dataset / "labels" / args.split / f"{image_path.stem}.txt", width, height, classes)
        predictions = predict(model, image_path, args.conf, args.iou, args.device)
        reviewed = [classify_label_match(label, predictions, args.match_iou) for label in labels]
        failures = [row for row in reviewed if row["match_reason"] != "matched"]
        all_rows.extend({"image": image_path.name, **row} for row in failures)
        summary.append(
            {
                "image": image_path.name,
                "labels": len(labels),
                "predictions": len(predictions),
                "failures": len(failures),
                "failure_reasons": {
                    reason: sum(1 for row in failures if row["match_reason"] == reason)
                    for reason in sorted({row["match_reason"] for row in failures})
                },
            }
        )
        if failures:
            case_dir = output / image_path.stem
            case_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(case_dir / "01-failure-numbered.png"), draw_failures(image, failures))
            cv2.imwrite(str(case_dir / "02-failure-crops.png"), crop_sheet(image, failures))
            (case_dir / "failures.json").write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"ok {image_path.name}: failures={len(failures)}")

    with (output / "failures.csv").open("w", encoding="utf-8-sig", newline="") as fp:
        fieldnames = [
            "image",
            "id",
            "class_id",
            "class_name",
            "match_reason",
            "best_any_iou",
            "best_same_iou",
            "best_any_class",
            "best_any_confidence",
            "best_same_confidence",
            "xyxy",
        ]
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"output={output}")


if __name__ == "__main__":
    main()
