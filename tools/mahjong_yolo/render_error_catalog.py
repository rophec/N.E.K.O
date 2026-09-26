from __future__ import annotations

import argparse
import ast
import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = (
    REPO_ROOT
    / "datasets"
    / "mahjong_yolo_active"
    / "dataset_registry"
    / "05_yolo_hbb_fixed_test_21"
)
DEFAULT_OUTPUT = REPO_ROOT / "runs" / "mahjong_yolo26_hbb" / "error_catalog"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render compact error catalog from analyze_detection_errors output.")
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-crops", type=int, default=120)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fp:
        return list(csv.DictReader(fp))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_box(value: str) -> tuple[float, float, float, float] | None:
    if not value:
        return None
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, (list, tuple)) or len(parsed) != 4:
        return None
    return tuple(float(v) for v in parsed)


def union_box(*boxes: tuple[float, float, float, float] | None) -> tuple[float, float, float, float] | None:
    kept = [box for box in boxes if box is not None]
    if not kept:
        return None
    return (
        min(box[0] for box in kept),
        min(box[1] for box in kept),
        max(box[2] for box in kept),
        max(box[3] for box in kept),
    )


def crop_with_pad(
    image: np.ndarray,
    box: tuple[float, float, float, float],
    pad: int = 18,
) -> tuple[np.ndarray, tuple[int, int]]:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    left = max(0, x1 - pad)
    top = max(0, y1 - pad)
    right = min(width, x2 + pad)
    bottom = min(height, y2 + pad)
    return image[top:bottom, left:right].copy(), (left, top)


def draw_local_box(
    crop: np.ndarray,
    box: tuple[float, float, float, float] | None,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    label: str,
) -> None:
    if box is None:
        return
    ox, oy = origin
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    cv2.rectangle(crop, (x1 - ox, y1 - oy), (x2 - ox, y2 - oy), color, 2)
    cv2.putText(crop, label, (max(0, x1 - ox), max(14, y1 - oy - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def make_sheet(
    rows: list[dict[str, str]],
    image_dir: Path,
    title: str,
    output_path: Path,
    max_crops: int,
) -> None:
    cell_w, cell_h = 190, 170
    cols = 5
    selected = rows[:max_crops]
    if not selected:
        return
    sheet_h = 38 + ((len(selected) + cols - 1) // cols) * cell_h
    sheet = np.full((sheet_h, cols * cell_w, 3), 245, dtype=np.uint8)
    cv2.putText(sheet, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (20, 20, 20), 2, cv2.LINE_AA)

    cache: dict[str, np.ndarray] = {}
    for idx, row in enumerate(selected):
        image_name = row["image"]
        if image_name not in cache:
            cache[image_name] = cv2.imread(str(image_dir / image_name))
        image = cache[image_name]
        if image is None:
            continue

        status = row.get("status", "")
        gt_box = parse_box(row.get("gt_xyxy", ""))
        if status == "low_conf_candidate":
            pred_box = parse_box(row.get("probe_xyxy", ""))
            pred_class = row.get("probe_class", "")
            pred_conf = row.get("probe_conf", "")
            pred_iou = row.get("probe_iou", "")
        elif status == "missed":
            pred_box = None
            pred_class = ""
            pred_conf = ""
            pred_iou = ""
        else:
            pred_box = parse_box(row.get("pred_xyxy", ""))
            pred_class = row.get("pred_class", "")
            pred_conf = row.get("pred_conf", "")
            pred_iou = row.get("iou", "")
        combined = union_box(gt_box, pred_box)
        if combined is None:
            continue

        crop, origin = crop_with_pad(image, combined)
        draw_local_box(crop, gt_box, origin, (40, 40, 230), f"GT {row.get('gt_class', '')}")
        draw_local_box(crop, pred_box, origin, (230, 180, 20), f"P {pred_class}")

        scale = min((cell_w - 12) / max(1, crop.shape[1]), (cell_h - 60) / max(1, crop.shape[0]))
        crop = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), max(1, int(crop.shape[0] * scale))), interpolation=cv2.INTER_CUBIC)
        col = idx % cols
        line = idx // cols
        x = col * cell_w + (cell_w - crop.shape[1]) // 2
        y = 38 + line * cell_h + 5
        sheet[y : y + crop.shape[0], x : x + crop.shape[1]] = crop

        image_caption = image_name[:24]
        caption = f"{status} {row.get('gt_class', '')}->{pred_class}"
        cv2.putText(sheet, image_caption, (col * cell_w + 6, 38 + line * cell_h + cell_h - 46), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (20, 20, 20), 1, cv2.LINE_AA)
        cv2.putText(sheet, caption[:24], (col * cell_w + 6, 38 + line * cell_h + cell_h - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1, cv2.LINE_AA)
        cv2.putText(sheet, f"conf={pred_conf} iou={pred_iou}", (col * cell_w + 6, 38 + line * cell_h + cell_h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1, cv2.LINE_AA)

    cv2.imwrite(str(output_path), sheet)


def confusion_outputs(rows: list[dict[str, str]], output: Path) -> list[dict[str, Any]]:
    class_errors = [row for row in rows if row["status"] == "class_error"]
    pair_counts = Counter((row["gt_class"], row["pred_class"]) for row in class_errors)
    pair_rows = [
        {"gt_class": gt, "pred_class": pred, "count": count}
        for (gt, pred), count in pair_counts.most_common()
    ]
    write_csv(output / "confusion_pairs.csv", pair_rows)

    classes = sorted({row["gt_class"] for row in rows} | {row["pred_class"] for row in class_errors if row["pred_class"]})
    matrix_rows = []
    for gt in classes:
        matrix_row: dict[str, Any] = {"gt_class": gt}
        for pred in classes:
            matrix_row[pred] = pair_counts[(gt, pred)]
        matrix_rows.append(matrix_row)
    write_csv(output / "confusion_matrix.csv", matrix_rows)
    return pair_rows


def false_positive_image_index(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["image"]].append(row)
    return [
        {
            "image": image,
            "false_positive": len(group),
            "highest_confidence": max(float(row["pred_conf"]) for row in group),
            "image_path": group[0].get("image_path", ""),
            "label_path": group[0].get("label_path", ""),
        }
        for image, group in sorted(
            grouped.items(),
            key=lambda item: (-len(item[1]), -max(float(row["pred_conf"]) for row in item[1]), item[0]),
        )
    ]


def main() -> None:
    args = parse_args()
    analysis_dir = args.analysis_dir.resolve()
    dataset = args.dataset.resolve()
    output = args.output_root.resolve() / time.strftime("%Y%m%d-%H%M%S")
    output.mkdir(parents=True, exist_ok=True)

    gt_rows = read_csv(analysis_dir / "gt_error_rows.csv")
    fp_rows = read_csv(analysis_dir / "false_positive_rows.csv")
    image_dir = dataset / "images" / args.split

    pair_rows = confusion_outputs(gt_rows, output)
    status_counts = Counter(row["status"] for row in gt_rows)
    bad_rows = [row for row in gt_rows if row["status"] != "matched"]
    class_error_rows = [row for row in gt_rows if row["status"] == "class_error"]
    low_or_missed_rows = [row for row in gt_rows if row["status"] in {"low_conf_candidate", "missed", "box_offset"}]
    false_positive_display_rows = [
        {
            **row,
            "status": row.get("reason", "false_positive"),
            "gt_class": row.get("nearest_gt_class", ""),
            "gt_xyxy": row.get("nearest_gt_xyxy", ""),
            "iou": row.get("nearest_gt_iou", ""),
        }
        for row in sorted(fp_rows, key=lambda row: float(row["pred_conf"]), reverse=True)
    ]

    make_sheet(class_error_rows, image_dir, "Class errors: red=GT cyan=prediction", output / "class_error_sheet.png", args.max_crops)
    make_sheet(low_or_missed_rows, image_dir, "Low confidence / missed / offset: red=GT cyan=prediction", output / "low_conf_missed_sheet.png", args.max_crops)
    make_sheet(false_positive_display_rows, image_dir, "False positives: cyan=prediction", output / "false_positive_sheet.png", args.max_crops)
    write_csv(output / "false_positive_image_index.csv", false_positive_image_index(fp_rows))

    lines = [
        "# Mahjong YOLO Error Catalog",
        "",
        f"- Source analysis: `{analysis_dir}`",
        f"- Dataset: `{dataset}`",
        f"- Error rows: {len(bad_rows)}",
        f"- False positives: {len(fp_rows)}",
        "",
        "## Status Counts",
        "",
        "| status | count |",
        "|---|---:|",
    ]
    for status, count in status_counts.most_common():
        lines.append(f"| {status} | {count} |")
    lines.extend(
        [
            "",
            "## Top Confusions",
            "",
            "| gt | predicted | count |",
            "|---|---|---:|",
        ]
    )
    for row in pair_rows[:20]:
        lines.append(f"| {row['gt_class']} | {row['pred_class']} | {row['count']} |")
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `confusion_pairs.csv`: sorted class-to-class mistakes.",
            "- `confusion_matrix.csv`: full class confusion matrix.",
            "- `class_error_sheet.png`: visual catalog of class mistakes.",
            "- `low_conf_missed_sheet.png`: visual catalog of low-confidence and missed cases.",
            "- `false_positive_sheet.png`: highest-confidence extra detections without a matching GT box.",
            "- `false_positive_image_index.csv`: image and GT-label paths grouped by false-positive count.",
        ]
    )
    (output / "catalog.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "analysis_dir": str(analysis_dir),
                "dataset": str(dataset),
                "status_counts": status_counts,
                "top_confusions": pair_rows[:20],
                "false_positives": len(fp_rows),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"output={output}")


if __name__ == "__main__":
    main()
