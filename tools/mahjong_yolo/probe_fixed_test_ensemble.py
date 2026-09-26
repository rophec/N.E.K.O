from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import cv2
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = (
    REPO_ROOT
    / "datasets"
    / "mahjong_yolo_active"
    / "dataset_registry"
    / "05_yolo_hbb_fixed_test_21"
)
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache fixed-test YOLO predictions and search simple model ensembles."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    predict = subparsers.add_parser("predict-one")
    predict.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    predict.add_argument("--model", type=Path, required=True)
    predict.add_argument("--tag", required=True)
    predict.add_argument("--imgsz", type=int, default=800)
    predict.add_argument("--conf", type=float, default=0.01)
    predict.add_argument("--nms-iou", type=float, default=0.5)
    predict.add_argument("--device", default="0")
    predict.add_argument("--output-root", type=Path, required=True)

    search = subparsers.add_parser("search")
    search.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    search.add_argument("--pred-dir", type=Path, required=True)
    search.add_argument("--output-root", type=Path, required=True)
    search.add_argument("--max-size", type=int, default=8)
    search.add_argument("--match-iou", type=float, default=0.5)
    search.add_argument("--top", type=int, default=30)

    return parser.parse_args()


def load_names(dataset: Path) -> list[str]:
    with (dataset / "data.yaml").open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    names = data["names"]
    if isinstance(names, dict):
        return [names[index] for index in range(len(names))]
    return list(names)


def image_paths(dataset: Path) -> list[Path]:
    image_dir = dataset / "images" / "test"
    return sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTS)


def iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (area_a + area_b - inter + 1e-9)


def yolo_row_to_xyxy(row: str, width: int, height: int) -> tuple[int, tuple[float, float, float, float]]:
    class_id, x, y, box_width, box_height = map(float, row.split()[:5])
    return int(class_id), (
        (x - box_width / 2.0) * width,
        (y - box_height / 2.0) * height,
        (x + box_width / 2.0) * width,
        (y + box_height / 2.0) * height,
    )


def load_ground_truth(dataset: Path) -> tuple[list[dict[str, Any]], list[str]]:
    names = load_names(dataset)
    rows: list[dict[str, Any]] = []
    for image_path in image_paths(dataset):
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not read fixed-test image: {image_path}")
        height, width = image.shape[:2]
        label_path = dataset / "labels" / "test" / f"{image_path.stem}.txt"
        for index, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            class_id, box = yolo_row_to_xyxy(line, width, height)
            rows.append(
                {
                    "gt_key": f"{image_path.name}#{index}",
                    "image": image_path.name,
                    "gt_class_id": class_id,
                    "gt_class": names[class_id],
                    "gt_xyxy": box,
                }
            )
    return rows, names


def predict_one(args: argparse.Namespace) -> None:
    from ultralytics import YOLO

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    images = image_paths(args.dataset)
    model = YOLO(str(args.model.resolve() if args.model.exists() else args.model))
    results = model.predict(
        source=[str(path) for path in images],
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.nms_iou,
        device=args.device,
        stream=False,
        verbose=False,
    )

    payload: dict[str, Any] = {
        "tag": args.tag,
        "model": str(args.model),
        "imgsz": args.imgsz,
        "conf": args.conf,
        "nms_iou": args.nms_iou,
        "images": {},
    }
    for source_path, result in zip(images, results):
        boxes = []
        if result.boxes is not None:
            xyxy = result.boxes.xyxy.detach().cpu().tolist()
            conf = result.boxes.conf.detach().cpu().tolist()
            cls = result.boxes.cls.detach().cpu().tolist()
            for box, score, class_id in zip(xyxy, conf, cls):
                boxes.append(
                    {
                        "class_id": int(class_id),
                        "confidence": float(score),
                        "xyxy": [float(value) for value in box],
                    }
                )
        payload["images"][source_path.name] = boxes

    output_path = output_root / f"{args.tag}.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"output={output_path}")


def best_candidate_for_gt(
    gt: dict[str, Any],
    boxes: list[dict[str, Any]],
    match_iou: float,
) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    gt_box = gt["gt_xyxy"]
    for box in boxes:
        overlap = iou(tuple(gt_box), tuple(box["xyxy"]))
        if overlap < match_iou:
            continue
        if best is None or overlap > best["iou"]:
            best = {
                "class_id": int(box["class_id"]),
                "confidence": float(box["confidence"]),
                "iou": overlap,
            }
    return best


def choose_class(candidates: list[dict[str, Any]], mode: str) -> int | None:
    if not candidates:
        return None
    if mode == "best_conf":
        return max(candidates, key=lambda item: item["confidence"])["class_id"]
    if mode == "vote":
        counts = Counter(item["class_id"] for item in candidates)
        max_count = max(counts.values())
        tied_classes = [class_id for class_id, count in counts.items() if count == max_count]
        if len(tied_classes) == 1:
            return tied_classes[0]
        tied = [item for item in candidates if item["class_id"] in tied_classes]
        return max(tied, key=lambda item: item["confidence"])["class_id"]

    confidence_sums: dict[int, float] = defaultdict(float)
    for item in candidates:
        confidence_sums[item["class_id"]] += item["confidence"]
    return max(confidence_sums.items(), key=lambda item: item[1])[0]


def evaluate_combo(
    combo: tuple[str, ...],
    mode: str,
    gt_rows: list[dict[str, Any]],
    per_model_candidates: dict[str, dict[str, dict[str, Any] | None]],
    names: list[str],
) -> dict[str, Any]:
    matched = 0
    missed = 0
    class_errors: list[dict[str, Any]] = []
    pairs: Counter[tuple[str, str]] = Counter()

    for gt in gt_rows:
        candidates = [
            per_model_candidates[tag][gt["gt_key"]]
            for tag in combo
            if per_model_candidates[tag][gt["gt_key"]] is not None
        ]
        predicted_class = choose_class(candidates, mode)
        if predicted_class is None:
            missed += 1
            continue
        if predicted_class == gt["gt_class_id"]:
            matched += 1
            continue
        pair = (gt["gt_class"], names[predicted_class])
        pairs[pair] += 1
        class_errors.append(
            {
                "image": gt["image"],
                "gt": gt["gt_class"],
                "predicted": names[predicted_class],
            }
        )

    total = len(gt_rows)
    return {
        "combo": list(combo),
        "mode": mode,
        "total": total,
        "matched": matched,
        "class_error": len(class_errors),
        "missed": missed,
        "accuracy": matched / total if total else 0.0,
        "top_pairs": [
            {"gt": gt, "predicted": pred, "count": count}
            for (gt, pred), count in pairs.most_common(20)
        ],
        "class_errors": class_errors,
    }


def search_ensembles(args: argparse.Namespace) -> None:
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    gt_rows, names = load_ground_truth(args.dataset)

    prediction_payloads = []
    for path in sorted(args.pred_dir.glob("*.json")):
        prediction_payloads.append(json.loads(path.read_text(encoding="utf-8")))
    if not prediction_payloads:
        raise FileNotFoundError(f"No prediction JSON files under {args.pred_dir}")

    per_model_candidates: dict[str, dict[str, dict[str, Any] | None]] = {}
    for payload in prediction_payloads:
        tag = payload["tag"]
        per_model_candidates[tag] = {}
        for gt in gt_rows:
            boxes = payload["images"].get(gt["image"], [])
            per_model_candidates[tag][gt["gt_key"]] = best_candidate_for_gt(gt, boxes, args.match_iou)

    tags = [payload["tag"] for payload in prediction_payloads]
    modes = ["conf_sum", "vote", "best_conf"]
    results: list[dict[str, Any]] = []

    for tag in tags:
        for mode in modes:
            results.append(evaluate_combo((tag,), mode, gt_rows, per_model_candidates, names))

    for size in range(2, min(args.max_size, len(tags)) + 1):
        for combo in combinations(tags, size):
            for mode in modes:
                results.append(evaluate_combo(combo, mode, gt_rows, per_model_candidates, names))

    results.sort(key=lambda item: (item["class_error"] + item["missed"], item["missed"], -item["matched"]))
    best = results[: args.top]
    (output_root / "ensemble_summary.json").write_text(
        json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    with (output_root / "best_errors.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["rank", "image", "gt", "predicted"])
        writer.writeheader()
        if best:
            for row in best[0]["class_errors"]:
                writer.writerow({"rank": 1, **row})

    lines = ["# Fixed Test Ensemble Probe", ""]
    for index, item in enumerate(best, start=1):
        lines.extend(
            [
                f"## #{index}",
                "",
                f"- mode: `{item['mode']}`",
                f"- combo: `{', '.join(item['combo'])}`",
                f"- matched: `{item['matched']}/{item['total']}`",
                f"- class_error: `{item['class_error']}`",
                f"- missed: `{item['missed']}`",
                f"- accuracy: `{item['accuracy']:.5f}`",
                f"- top_pairs: `{item['top_pairs']}`",
                "",
            ]
        )
    (output_root / "ensemble_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"output={output_root}")
    if best:
        print(json.dumps(best[0], ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    if args.command == "predict-one":
        predict_one(args)
    elif args.command == "search":
        search_ensembles(args)


if __name__ == "__main__":
    main()
