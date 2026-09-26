from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = (
    REPO_ROOT
    / "datasets"
    / "mahjong_yolo_active"
    / "dataset_registry"
    / "05_yolo_hbb_fixed_test_21"
)
DEFAULT_OUTPUT = REPO_ROOT / "runs" / "mahjong_yolo26_hbb" / "error_analysis"


@dataclass(frozen=True)
class Label:
    id: int
    class_id: int
    class_name: str
    xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class Prediction:
    id: int
    class_id: int
    class_name: str
    confidence: float
    xyxy: tuple[float, float, float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze YOLO HBB detection errors on a labeled dataset.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--probe-conf", type=float, default=0.01)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument(
        "--agnostic-nms",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Suppress overlapping tile boxes across classes before diagnostic matching.",
    )
    parser.add_argument("--device", default="0")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_names(data_yaml: Path) -> list[str]:
    names: dict[int, str] = {}
    in_names = False
    for raw_line in data_yaml.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        if line.startswith("names:"):
            in_names = True
            continue
        if not in_names:
            continue
        stripped = line.strip()
        if not stripped or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        if key.strip().isdigit():
            names[int(key.strip())] = value.strip().strip("'\"")
    if not names:
        raise ValueError(f"Could not read class names from {data_yaml}")
    return [names[idx] for idx in sorted(names)]


def read_labels(label_path: Path, width: int, height: int, names: list[str]) -> list[Label]:
    labels: list[Label] = []
    if not label_path.exists():
        return labels
    for idx, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        class_id = int(float(parts[0]))
        cx, cy, bw, bh = [float(part) for part in parts[1:]]
        labels.append(
            Label(
                id=idx,
                class_id=class_id,
                class_name=names[class_id],
                xyxy=(
                    (cx - bw / 2.0) * width,
                    (cy - bh / 2.0) * height,
                    (cx + bw / 2.0) * width,
                    (cy + bh / 2.0) * height,
                ),
            )
        )
    return labels


def predict(
    model: YOLO,
    image_path: Path,
    conf: float,
    nms_iou: float,
    device: str,
    agnostic_nms: bool,
) -> list[Prediction]:
    result = model.predict(
        str(image_path),
        conf=conf,
        iou=nms_iou,
        device=device,
        agnostic_nms=agnostic_nms,
        verbose=False,
    )[0]
    predictions: list[Prediction] = []
    for idx, box in enumerate(result.boxes, start=1):
        class_id = int(box.cls.item())
        predictions.append(
            Prediction(
                id=idx,
                class_id=class_id,
                class_name=result.names[class_id],
                confidence=float(box.conf.item()),
                xyxy=tuple(float(v) for v in box.xyxy[0].tolist()),
            )
        )
    return predictions


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


def best_prediction(label: Label, predictions: list[Prediction], used: set[int] | None = None) -> tuple[Prediction | None, float]:
    candidates = predictions if used is None else [pred for pred in predictions if pred.id not in used]
    if not candidates:
        return None, 0.0
    best = max(candidates, key=lambda pred: iou(label.xyxy, pred.xyxy))
    return best, iou(label.xyxy, best.xyxy)


def match_frame(
    labels: list[Label],
    predictions: list[Prediction],
    probe_predictions: list[Prediction],
    match_iou: float,
) -> tuple[list[dict[str, Any]], set[int]]:
    rows: list[dict[str, Any]] = []
    used_predictions: set[int] = set()

    for label in labels:
        pred, overlap = best_prediction(label, predictions, used_predictions)
        probe_pred, probe_overlap = best_prediction(label, probe_predictions)
        if pred is not None and overlap >= match_iou:
            used_predictions.add(pred.id)
            if pred.class_id == label.class_id:
                status = "matched"
            else:
                status = "class_error"
        elif probe_pred is not None and probe_overlap >= match_iou:
            status = "low_conf_candidate"
        elif pred is not None and overlap >= 0.10:
            status = "box_offset"
        else:
            status = "missed"

        rows.append(
            {
                "gt_id": label.id,
                "gt_class": label.class_name,
                "gt_class_id": label.class_id,
                "status": status,
                "pred_id": pred.id if pred else "",
                "pred_class": pred.class_name if pred else "",
                "pred_class_id": pred.class_id if pred else "",
                "pred_conf": round(pred.confidence, 4) if pred else "",
                "iou": round(overlap, 4),
                "probe_class": probe_pred.class_name if probe_pred else "",
                "probe_class_id": probe_pred.class_id if probe_pred else "",
                "probe_conf": round(probe_pred.confidence, 4) if probe_pred else "",
                "probe_iou": round(probe_overlap, 4),
                "probe_xyxy": tuple(round(v, 2) for v in probe_pred.xyxy) if probe_pred else "",
                "gt_xyxy": tuple(round(v, 2) for v in label.xyxy),
                "pred_xyxy": tuple(round(v, 2) for v in pred.xyxy) if pred else "",
            }
        )
    return rows, used_predictions


def class_stats(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_class: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        by_class[str(row["gt_class"])][str(row["status"])] += 1

    out: list[dict[str, Any]] = []
    for class_name in sorted(by_class):
        counts = by_class[class_name]
        total = sum(counts.values())
        bad = total - counts["matched"]
        out.append(
            {
                "class": class_name,
                "total": total,
                "matched": counts["matched"],
                "bad": bad,
                "class_error": counts["class_error"],
                "low_conf_candidate": counts["low_conf_candidate"],
                "box_offset": counts["box_offset"],
                "missed": counts["missed"],
                "match_rate": round(counts["matched"] / total, 4) if total else 0.0,
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    path: Path,
    model_path: Path,
    dataset: Path,
    total_labels: int,
    total_predictions: int,
    rows: list[dict[str, Any]],
    false_positive_rows: list[dict[str, Any]],
    stats: list[dict[str, Any]],
) -> None:
    status_counts = Counter(str(row["status"]) for row in rows)
    lines = [
        "# Mahjong YOLO Error Analysis",
        "",
        f"- Model: `{model_path}`",
        f"- Dataset: `{dataset}`",
        f"- GT boxes: {total_labels}",
        f"- Predictions: {total_predictions}",
        f"- Matched: {status_counts['matched']}",
        f"- Class errors: {status_counts['class_error']}",
        f"- Low-confidence candidates: {status_counts['low_conf_candidate']}",
        f"- Box offsets: {status_counts['box_offset']}",
        f"- Missed: {status_counts['missed']}",
        f"- False positives: {len(false_positive_rows)}",
        "",
        "## Worst Classes",
        "",
        "| class | total | matched | bad | class_error | low_conf | box_offset | missed | match_rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(stats, key=lambda item: (-int(item["bad"]), str(item["class"])))[:12]:
        lines.append(
            "| {class} | {total} | {matched} | {bad} | {class_error} | {low_conf_candidate} | {box_offset} | {missed} | {match_rate:.4f} |".format(
                **row
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve()
    model_path = args.model.resolve()
    output = args.output_root.resolve() / time.strftime("%Y%m%d-%H%M%S")
    output.mkdir(parents=True, exist_ok=True)

    names = read_names(dataset / "data.yaml")
    model = YOLO(str(model_path))
    image_dir = dataset / "images" / args.split
    label_dir = dataset / "labels" / args.split

    gt_rows: list[dict[str, Any]] = []
    false_positive_rows: list[dict[str, Any]] = []
    image_rows: list[dict[str, Any]] = []
    total_labels = 0
    total_predictions = 0

    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        labels = read_labels(label_dir / f"{image_path.stem}.txt", width, height, names)
        predictions = predict(model, image_path, args.conf, args.nms_iou, args.device, args.agnostic_nms)
        probe_predictions = predict(
            model,
            image_path,
            args.probe_conf,
            args.nms_iou,
            args.device,
            args.agnostic_nms,
        )
        frame_rows, used_predictions = match_frame(labels, predictions, probe_predictions, args.match_iou)

        for row in frame_rows:
            row["image"] = image_path.name
            row["image_path"] = str(image_path.resolve())
            row["label_path"] = str((label_dir / f"{image_path.stem}.txt").resolve())
        gt_rows.extend(frame_rows)

        for pred in predictions:
            if pred.id in used_predictions:
                continue
            nearest_label = max(labels, key=lambda label: iou(label.xyxy, pred.xyxy), default=None)
            nearest_iou = iou(nearest_label.xyxy, pred.xyxy) if nearest_label else 0.0
            if nearest_iou >= args.match_iou:
                false_positive_reason = "duplicate_prediction"
            elif nearest_iou >= 0.10:
                false_positive_reason = "box_offset_extra"
            else:
                false_positive_reason = "background_false_positive"
            false_positive_rows.append(
                {
                    "image": image_path.name,
                    "image_path": str(image_path.resolve()),
                    "label_path": str((label_dir / f"{image_path.stem}.txt").resolve()),
                    "pred_id": pred.id,
                    "pred_class": pred.class_name,
                    "pred_class_id": pred.class_id,
                    "pred_conf": round(pred.confidence, 4),
                    "pred_xyxy": tuple(round(v, 2) for v in pred.xyxy),
                    "reason": false_positive_reason,
                    "nearest_gt_id": nearest_label.id if nearest_label else "",
                    "nearest_gt_class": nearest_label.class_name if nearest_label else "",
                    "nearest_gt_iou": round(nearest_iou, 4),
                    "nearest_gt_xyxy": tuple(round(v, 2) for v in nearest_label.xyxy) if nearest_label else "",
                }
            )

        counts = Counter(str(row["status"]) for row in frame_rows)
        image_rows.append(
            {
                "image": image_path.name,
                "gt": len(labels),
                "predictions": len(predictions),
                "matched": counts["matched"],
                "class_error": counts["class_error"],
                "low_conf_candidate": counts["low_conf_candidate"],
                "box_offset": counts["box_offset"],
                "missed": counts["missed"],
                "false_positive": len(predictions) - len(used_predictions),
            }
        )
        total_labels += len(labels)
        total_predictions += len(predictions)

    stats = class_stats(gt_rows)
    write_csv(output / "gt_error_rows.csv", gt_rows)
    write_csv(output / "false_positive_rows.csv", false_positive_rows)
    write_csv(output / "image_summary.csv", image_rows)
    write_csv(output / "class_summary.csv", stats)
    (output / "summary.json").write_text(
        json.dumps(
            {
                "model": str(model_path),
                "dataset": str(dataset),
                "split": args.split,
                "conf": args.conf,
                "probe_conf": args.probe_conf,
                "nms_iou": args.nms_iou,
                "agnostic_nms": args.agnostic_nms,
                "match_iou": args.match_iou,
                "total_labels": total_labels,
                "total_predictions": total_predictions,
                "status_counts": Counter(str(row["status"]) for row in gt_rows),
                "false_positives": len(false_positive_rows),
                "false_positive_reasons": Counter(str(row["reason"]) for row in false_positive_rows),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_report(
        output / "report.md",
        model_path,
        dataset,
        total_labels,
        total_predictions,
        gt_rows,
        false_positive_rows,
        stats,
    )
    print(f"output={output}")


if __name__ == "__main__":
    main()
