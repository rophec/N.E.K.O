"""Evaluate dense Mahjong detections with confidence-first one-to-one matching.

The stock error analyzer matches each GT to the highest-IoU prediction first.
That can create false class errors when adjacent Mahjong tiles overlap. This
tool assigns predictions from highest to lowest confidence to the best remaining
GT box, then reports class errors, low-confidence detections, misses, and false
positives.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


@dataclass(frozen=True)
class Box:
    index: int
    class_id: int
    class_name: str
    xyxy: tuple[float, float, float, float]
    confidence: float = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--models", type=Path, nargs="+", required=True)
    parser.add_argument("--tags", nargs="+")
    parser.add_argument("--split", default="test")
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--probe-conf", type=float, default=0.01)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--agnostic-nms", action="store_true")
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--dedup-iou", type=float, default=0.0)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_names(data_yaml: Path) -> list[str]:
    names: dict[int, str] = {}
    in_names = False
    for raw in data_yaml.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("names:"):
            in_names = True
            continue
        if in_names and ":" in line:
            key, value = line.split(":", 1)
            if key.strip().isdigit():
                names[int(key)] = value.strip().strip("'\"")
    return [names[index] for index in sorted(names)]


def read_gt(path: Path, width: int, height: int, names: list[str]) -> list[Box]:
    boxes: list[Box] = []
    for index, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = raw.split()
        if len(parts) != 5:
            continue
        class_id = int(float(parts[0]))
        cx, cy, bw, bh = map(float, parts[1:])
        boxes.append(
            Box(
                index=index,
                class_id=class_id,
                class_name=names[class_id],
                xyxy=((cx - bw / 2) * width, (cy - bh / 2) * height,
                      (cx + bw / 2) * width, (cy + bh / 2) * height),
            )
        )
    return boxes


def predict(model: YOLO, image: Path, args: argparse.Namespace) -> list[Box]:
    result = model.predict(
        str(image), conf=args.probe_conf, iou=args.nms_iou, imgsz=args.imgsz,
        device=args.device, agnostic_nms=args.agnostic_nms, verbose=False,
    )[0]
    boxes = [
        Box(
            index=index,
            class_id=int(raw.cls.item()),
            class_name=result.names[int(raw.cls.item())],
            confidence=float(raw.conf.item()),
            xyxy=tuple(float(value) for value in raw.xyxy[0].tolist()),
        )
        for index, raw in enumerate(result.boxes, start=1)
    ]
    if args.dedup_iou <= 0:
        return boxes
    # EN: A physical tile may only produce one final detection.
    # 中文：一张实体麻将牌最终只能保留一个检测框。
    kept: list[Box] = []
    for candidate in sorted(boxes, key=lambda box: box.confidence, reverse=True):
        if all(iou(candidate, existing) < args.dedup_iou for existing in kept):
            kept.append(candidate)
    return kept


def iou(first: Box, second: Box) -> float:
    ax1, ay1, ax2, ay2 = first.xyxy
    bx1, by1, bx2, by2 = second.xyxy
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return intersection / union if union > 0 else 0.0


def confidence_first_pairs(
    gt: list[Box], predictions: list[Box], gt_ids: set[int], pred_ids: set[int], threshold: float,
) -> list[tuple[int, int, float]]:
    """Assign high-confidence predictions to their best remaining GT box."""
    pairs: list[tuple[int, int, float]] = []
    remaining_gt = set(gt_ids)
    ordered_predictions = sorted(pred_ids, key=lambda index: predictions[index].confidence, reverse=True)
    for pred_index in ordered_predictions:
        candidates = [
            (iou(gt[gt_index], predictions[pred_index]), gt_index)
            for gt_index in remaining_gt
        ]
        if not candidates:
            continue
        overlap, gt_index = max(candidates)
        if overlap < threshold:
            continue
        remaining_gt.remove(gt_index)
        pairs.append((gt_index, pred_index, overlap))
    return pairs


def evaluate_image(gt: list[Box], predictions: list[Box], args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[int]]:
    high_ids = {index for index, box in enumerate(predictions) if box.confidence >= args.conf}
    all_gt = set(range(len(gt)))
    rows: list[dict[str, Any]] = []
    used_gt: set[int] = set()
    used_pred: set[int] = set()

    def add_pair(gt_index: int, pred_index: int, overlap: float, status: str) -> None:
        truth, pred = gt[gt_index], predictions[pred_index]
        rows.append({
            "gt_id": truth.index, "gt_class": truth.class_name, "status": status,
            "pred_class": pred.class_name, "pred_conf": round(pred.confidence, 6),
            "iou": round(overlap, 6), "gt_xyxy": list(map(lambda v: round(v, 2), truth.xyxy)),
            "pred_xyxy": list(map(lambda v: round(v, 2), pred.xyxy)),
        })
        used_gt.add(gt_index)
        used_pred.add(pred_index)

    # EN: Prediction confidence decides which class wins at an occupied tile.
    # 中文：同一张牌存在多个候选框时，先由预测置信度决定哪个类别获胜。
    for gt_index, pred_index, overlap in confidence_first_pairs(gt, predictions, all_gt, high_ids, args.match_iou):
        status = "matched" if gt[gt_index].class_id == predictions[pred_index].class_id else "class_error"
        add_pair(gt_index, pred_index, overlap, status)

    # Probe detections below the reporting threshold without letting them steal high-confidence pairs.
    low_ids = set(range(len(predictions))) - high_ids - used_pred
    remaining_gt = all_gt - used_gt
    for gt_index, pred_index, overlap in confidence_first_pairs(gt, predictions, remaining_gt, low_ids, args.match_iou):
        add_pair(gt_index, pred_index, overlap, "low_conf_candidate")

    for gt_index in sorted(all_gt - used_gt):
        truth = gt[gt_index]
        rows.append({
            "gt_id": truth.index, "gt_class": truth.class_name, "status": "missed",
            "pred_class": "", "pred_conf": "", "iou": 0.0,
            "gt_xyxy": list(map(lambda v: round(v, 2), truth.xyxy)), "pred_xyxy": "",
        })
    false_positive_ids = sorted(high_ids - used_pred)
    return rows, false_positive_ids


def draw_bad_case(image: Any, rows: list[dict[str, Any]], false_predictions: list[Box]) -> Any:
    canvas = image.copy()
    for row in rows:
        if row["status"] == "matched":
            continue
        gx1, gy1, gx2, gy2 = map(int, row["gt_xyxy"])
        cv2.rectangle(canvas, (gx1, gy1), (gx2, gy2), (0, 0, 255), 2)
        label = f"GT {row['gt_class']} {row['status']}"
        cv2.putText(canvas, label, (gx1, max(18, gy1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        if row["pred_xyxy"]:
            px1, py1, px2, py2 = map(int, row["pred_xyxy"])
            cv2.rectangle(canvas, (px1, py1), (px2, py2), (255, 255, 0), 2)
            cv2.putText(canvas, f"P {row['pred_class']} {row['pred_conf']:.3f}", (px1, py2 + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    for pred in false_predictions:
        x1, y1, x2, y2 = map(int, pred.xyxy)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.putText(canvas, f"FP {pred.class_name} {pred.confidence:.3f}", (x1, max(18, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
    return canvas


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_class_metrics(
    rows: list[dict[str, Any]], false_rows: list[dict[str, Any]], names: list[str],
) -> list[dict[str, Any]]:
    """Build per-tile face metrics from the final one-box-per-tile decisions.

    EN: A wrong class is both a false negative for its GT class and a false
    positive for its predicted class. Duplicate/background boxes only add a
    false positive to the predicted class.
    中文：错分类同时计入真值类别的假阴性和预测类别的假阳性；重复框或
    背景框只计入预测类别的假阳性。
    """
    metrics: list[dict[str, Any]] = []
    for name in names:
        gt_rows = [row for row in rows if row["gt_class"] == name]
        correct_rows = [row for row in gt_rows if row["status"] == "matched"]
        class_error_rows = [row for row in gt_rows if row["status"] == "class_error"]
        missed_rows = [
            row for row in gt_rows
            if row["status"] in {"missed", "low_conf_candidate"}
        ]
        predicted_class_errors = [
            row for row in rows
            if row["status"] == "class_error" and row["pred_class"] == name
        ]
        extra_predictions = [row for row in false_rows if row["pred_class"] == name]

        true_positive = len(correct_rows)
        false_positive = len(predicted_class_errors) + len(extra_predictions)
        false_negative = len(class_error_rows) + len(missed_rows)
        support = len(gt_rows)
        localized = true_positive + len(class_error_rows)
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        face_accuracy = true_positive / localized if localized else 0.0
        wrong_confidences = [float(row["pred_conf"]) for row in class_error_rows]

        metrics.append({
            "class": name,
            "gt_total": support,
            "localized": localized,
            "correct": true_positive,
            "class_error_as_gt": len(class_error_rows),
            "missed_or_low_conf": len(missed_rows),
            "false_positive_as_prediction": false_positive,
            "precision": round(precision, 6),
            "recall_end_to_end": round(recall, 6),
            "face_accuracy_when_localized": round(face_accuracy, 6),
            "f1": round(f1, 6),
            "max_wrong_confidence": round(max(wrong_confidences), 6) if wrong_confidences else "",
        })
    return metrics


def evaluate_model(model_path: Path, tag: str, args: argparse.Namespace, names: list[str]) -> dict[str, Any]:
    dataset = args.dataset.resolve()
    image_dir = dataset / "images" / args.split
    label_dir = dataset / "labels" / args.split
    output = args.output.resolve() / tag
    bad_dir = output / "bad_cases"
    bad_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(model_path.resolve()))
    all_rows: list[dict[str, Any]] = []
    image_rows: list[dict[str, Any]] = []
    pairs: Counter[tuple[str, str]] = Counter()
    counts: Counter[str] = Counter()
    false_rows: list[dict[str, Any]] = []

    for image_path in sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        gt = read_gt(label_dir / f"{image_path.stem}.txt", width, height, names)
        predictions = predict(model, image_path, args)
        rows, false_ids = evaluate_image(gt, predictions, args)
        for row in rows:
            row["image"] = image_path.name
            all_rows.append(row)
            counts[row["status"]] += 1
            if row["status"] == "class_error":
                pairs[(row["gt_class"], row["pred_class"])] += 1
        counts["false_positive"] += len(false_ids)
        for pred_index in false_ids:
            pred = predictions[pred_index]
            overlaps = [(iou(truth, pred), truth.class_name) for truth in gt]
            max_overlap, nearest_gt = max(overlaps, default=(0.0, ""))
            false_rows.append({
                "image": image_path.name, "pred_class": pred.class_name,
                "pred_conf": round(pred.confidence, 6),
                "nearest_gt": nearest_gt, "max_gt_iou": round(max_overlap, 6),
                "fp_kind": "duplicate" if max_overlap >= args.match_iou else "background",
                "pred_xyxy": list(map(lambda value: round(value, 2), pred.xyxy)),
            })
        bad = sum(row["status"] != "matched" for row in rows) + len(false_ids)
        image_rows.append({"image": image_path.name, "gt": len(gt), "pred": len(predictions), "bad": bad})
        if bad:
            false_predictions = [predictions[index] for index in false_ids]
            cv2.imwrite(str(bad_dir / image_path.name), draw_bad_case(image, rows, false_predictions))

    total = sum(counts[key] for key in ("matched", "class_error", "low_conf_candidate", "missed"))
    summary = {
        "tag": tag, "model": str(model_path.resolve()), "gt_total": total,
        "matched": counts["matched"], "class_error": counts["class_error"],
        "low_conf_candidate": counts["low_conf_candidate"], "missed": counts["missed"],
        "false_positive": counts["false_positive"],
        "exact_gt_accuracy": counts["matched"] / total if total else 0.0,
        "top_class_errors": [
            {"gt": gt_name, "pred": pred_name, "count": count}
            for (gt_name, pred_name), count in pairs.most_common()
        ],
        "bad_images": sum(row["bad"] > 0 for row in image_rows),
    }
    write_csv(output / "gt_results.csv", all_rows)
    write_csv(output / "false_positives.csv", false_rows)
    write_csv(output / "image_summary.csv", image_rows)
    write_csv(output / "per_class_metrics.csv", build_class_metrics(all_rows, false_rows, names))
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    if args.tags and len(args.tags) != len(args.models):
        raise ValueError("--tags must contain one name per model")
    tags = args.tags or [path.stem for path in args.models]
    args.output.resolve().mkdir(parents=True, exist_ok=True)
    names = read_names(args.dataset.resolve() / "data.yaml")
    summaries = [evaluate_model(path, tag, args, names) for path, tag in zip(args.models, tags)]
    (args.output.resolve() / "comparison.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for summary in summaries:
        print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
