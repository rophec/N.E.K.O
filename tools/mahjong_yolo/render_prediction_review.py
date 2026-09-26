from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np

from postprocess_yolo_predictions import PostprocessConfig, postprocess_predictions


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    REPO_ROOT
    / "runs"
    / "mahjong_yolo26_hbb"
    / "manual_validation"
    / "20260701-125400-v2-official-conf025"
)
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
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "runs"
    / "mahjong_yolo26_hbb"
    / "manual_validation_review"
)
DEFAULT_THRESHOLDS = "0.10,0.25,0.50"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render non-overlapping review artifacts for Mahjong YOLO predictions."
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--thresholds", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--device", default="0")
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument(
        "--end2end",
        choices=["auto", "true", "false"],
        default="auto",
        help="YOLO26 predict mode. Use false to enable traditional NMS-style postprocessing.",
    )
    parser.add_argument("--postprocess", action="store_true")
    parser.add_argument("--post-min-conf", type=float, default=0.05)
    parser.add_argument("--post-duplicate-iou", type=float, default=0.28)
    return parser.parse_args()


def color_for_class(class_id: int) -> tuple[int, int, int]:
    # EN: Stable high-contrast colors per class for visual review.
    # ZH: 每个类别固定高对比色，便于人工检查。
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
    return palette[class_id % len(palette)]


def clamp_box(xyxy: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = xyxy
    return (
        max(0, min(width - 1, int(round(x1)))),
        max(0, min(height - 1, int(round(y1)))),
        max(0, min(width - 1, int(round(x2)))),
        max(0, min(height - 1, int(round(y2)))),
    )


def sorted_predictions(preds: list[dict[str, object]]) -> list[dict[str, object]]:
    # EN: Top-to-bottom then left-to-right ordering gives stable IDs.
    # ZH: 按从上到下、从左到右排序，保证编号稳定。
    return sorted(
        preds,
        key=lambda p: (
            round(float(p["xyxy"][1]) / 20.0),
            float(p["xyxy"][0]),
            -float(p["confidence"]),
        ),
    )


def draw_box_only(image: np.ndarray, preds: list[dict[str, object]]) -> np.ndarray:
    out = image.copy()
    height, width = out.shape[:2]
    for pred in preds:
        class_id = int(pred["class_id"])
        x1, y1, x2, y2 = clamp_box(pred["xyxy"], width, height)
        cv2.rectangle(out, (x1, y1), (x2, y2), color_for_class(class_id), 2)
    return out


def draw_numbered(image: np.ndarray, preds: list[dict[str, object]]) -> np.ndarray:
    out = image.copy()
    height, width = out.shape[:2]
    for idx, pred in enumerate(preds, start=1):
        class_id = int(pred["class_id"])
        x1, y1, x2, y2 = clamp_box(pred["xyxy"], width, height)
        color = color_for_class(class_id)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = str(idx)
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        tag_x1 = x1
        tag_y1 = max(0, y1 - th - baseline - 4)
        if tag_y1 == 0:
            tag_y1 = min(height - th - baseline - 5, y2 + 3)
        cv2.rectangle(
            out,
            (tag_x1, tag_y1),
            (min(width - 1, tag_x1 + tw + 8), min(height - 1, tag_y1 + th + baseline + 6)),
            color,
            -1,
        )
        cv2.putText(
            out,
            label,
            (tag_x1 + 4, tag_y1 + th + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
    return out


def area_color(pred: dict[str, object]) -> tuple[int, int, int]:
    area = pred.get("area_kind")
    owner = pred.get("owner")
    if area == "self_hand":
        return (255, 220, 80)
    if area == "river" and owner == "self":
        return (80, 255, 120)
    if area == "river" and owner == "left_opponent":
        return (255, 120, 80)
    if area == "river" and owner == "across_opponent":
        return (120, 160, 255)
    if area == "river" and owner == "right_opponent":
        return (220, 120, 255)
    return (180, 180, 180)


def area_label(pred: dict[str, object]) -> str:
    area = pred.get("area_kind", "")
    owner = pred.get("owner", "")
    if area == "self_hand":
        return "H"
    if area == "river":
        return {
            "self": "RS",
            "left_opponent": "RL",
            "across_opponent": "RA",
            "right_opponent": "RR",
        }.get(str(owner), "R")
    return "O"


def draw_area_numbered(image: np.ndarray, preds: list[dict[str, object]]) -> np.ndarray:
    out = image.copy()
    height, width = out.shape[:2]
    for idx, pred in enumerate(preds, start=1):
        x1, y1, x2, y2 = clamp_box(pred["xyxy"], width, height)
        color = area_color(pred)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{idx}:{area_label(pred)}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 2)
        tag_y1 = max(0, y1 - th - baseline - 4)
        if tag_y1 == 0:
            tag_y1 = min(height - th - baseline - 5, y2 + 3)
        cv2.rectangle(
            out,
            (x1, tag_y1),
            (min(width - 1, x1 + tw + 8), min(height - 1, tag_y1 + th + baseline + 6)),
            color,
            -1,
        )
        cv2.putText(
            out,
            label,
            (x1 + 4, tag_y1 + th + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
    return out


def make_crop_sheet(image: np.ndarray, preds: list[dict[str, object]]) -> np.ndarray:
    height, width = image.shape[:2]
    cell_w, cell_h = 160, 190
    cols = 5
    rows = max(1, math.ceil(len(preds) / cols))
    sheet = np.full((rows * cell_h, cols * cell_w, 3), 245, dtype=np.uint8)
    for idx, pred in enumerate(preds, start=1):
        row = (idx - 1) // cols
        col = (idx - 1) % cols
        x1, y1, x2, y2 = clamp_box(pred["xyxy"], width, height)
        pad = 8
        crop = image[
            max(0, y1 - pad) : min(height, y2 + pad),
            max(0, x1 - pad) : min(width, x2 + pad),
        ]
        if crop.size == 0:
            continue
        crop_h, crop_w = crop.shape[:2]
        scale = min((cell_w - 18) / crop_w, (cell_h - 55) / crop_h)
        resized = cv2.resize(
            crop,
            (max(1, int(crop_w * scale)), max(1, int(crop_h * scale))),
            interpolation=cv2.INTER_CUBIC,
        )
        ox = col * cell_w + (cell_w - resized.shape[1]) // 2
        oy = row * cell_h + 10
        sheet[oy : oy + resized.shape[0], ox : ox + resized.shape[1]] = resized
        text = f"#{idx} {pred['class_name']} {float(pred['confidence']):.2f}"
        cv2.putText(
            sheet,
            text,
            (col * cell_w + 8, row * cell_h + cell_h - 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
    return sheet


def run_predict(
    model: object,
    image_path: Path,
    conf: float,
    iou: float,
    device: str,
    end2end: str,
) -> list[dict[str, object]]:
    predict_kwargs: dict[str, object] = {
        "conf": conf,
        "iou": iou,
        "device": device,
        "verbose": False,
    }
    if end2end != "auto":
        predict_kwargs["end2end"] = end2end == "true"
    results = model.predict(str(image_path), **predict_kwargs)
    result = results[0]
    names = result.names
    preds: list[dict[str, object]] = []
    for box in result.boxes:
        class_id = int(box.cls.item())
        preds.append(
            {
                "class_id": class_id,
                "class_name": names[class_id],
                "confidence": float(box.conf.item()),
                "xyxy": [float(v) for v in box.xyxy[0].tolist()],
            }
        )
    return sorted_predictions(preds)


def write_review_case(
    case_dir: Path,
    output_dir: Path,
    model: object,
    thresholds: list[float],
    device: str,
    iou: float,
    end2end: str,
    postprocess: bool,
    post_config: PostprocessConfig,
) -> dict[str, object]:
    image_path = case_dir / "09-warp-square-800.png"
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read warped image: {image_path}")

    case_summary: dict[str, object] = {"case": case_dir.name, "thresholds": {}}
    output_dir.mkdir(parents=True, exist_ok=True)
    for conf in thresholds:
        started = time.perf_counter()
        preds = run_predict(model, image_path, conf, iou, device, end2end)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        threshold_name = f"conf_{conf:.2f}".replace(".", "p")
        threshold_dir = output_dir / threshold_name
        threshold_dir.mkdir(parents=True, exist_ok=True)

        raw_dir = threshold_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(raw_dir / "01-box-only.png"), draw_box_only(image, preds))
        cv2.imwrite(str(raw_dir / "02-numbered-boxes.png"), draw_numbered(image, preds))
        cv2.imwrite(str(raw_dir / "03-crop-sheet.png"), make_crop_sheet(image, preds))
        (raw_dir / "predictions.json").write_text(
            json.dumps(preds, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        with (raw_dir / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow(["id", "class_id", "class_name", "confidence", "area_kind", "owner", "x1", "y1", "x2", "y2"])
            for idx, pred in enumerate(preds, start=1):
                writer.writerow(
                    [
                        idx,
                        pred["class_id"],
                        pred["class_name"],
                        f"{float(pred['confidence']):.6f}",
                        pred.get("area_kind", ""),
                        pred.get("owner", ""),
                        *[f"{float(v):.2f}" for v in pred["xyxy"]],
                    ]
                )
        post_preds: list[dict[str, object]] = preds
        rejected: list[dict[str, object]] = []
        if postprocess:
            post_preds, rejected = postprocess_predictions(preds, post_config)
            post_dir = threshold_dir / "postprocessed"
            post_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(post_dir / "01-box-only.png"), draw_box_only(image, post_preds))
            cv2.imwrite(str(post_dir / "02-numbered-boxes.png"), draw_numbered(image, post_preds))
            cv2.imwrite(str(post_dir / "03-crop-sheet.png"), make_crop_sheet(image, post_preds))
            cv2.imwrite(str(post_dir / "04-area-numbered-boxes.png"), draw_area_numbered(image, post_preds))
            (post_dir / "predictions.json").write_text(
                json.dumps(post_preds, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (post_dir / "rejected.json").write_text(
                json.dumps(rejected, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            with (post_dir / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as fp:
                writer = csv.writer(fp)
                writer.writerow(["id", "class_id", "class_name", "confidence", "area_kind", "owner", "x1", "y1", "x2", "y2"])
                for idx, pred in enumerate(post_preds, start=1):
                    writer.writerow(
                        [
                            idx,
                            pred["class_id"],
                            pred["class_name"],
                            f"{float(pred['confidence']):.6f}",
                            pred.get("area_kind", ""),
                            pred.get("owner", ""),
                            *[f"{float(v):.2f}" for v in pred["xyxy"]],
                        ]
                    )
            with (post_dir / "rejected.csv").open("w", encoding="utf-8-sig", newline="") as fp:
                writer = csv.writer(fp)
                writer.writerow(
                    [
                        "id",
                        "class_id",
                        "class_name",
                        "confidence",
                        "reject_reason",
                        "area_kind",
                        "owner",
                        "x1",
                        "y1",
                        "x2",
                        "y2",
                    ]
                )
                for idx, pred in enumerate(rejected, start=1):
                    writer.writerow(
                        [
                            idx,
                            pred["class_id"],
                            pred["class_name"],
                            f"{float(pred['confidence']):.6f}",
                            pred.get("reject_reason", ""),
                            pred.get("area_kind", ""),
                            pred.get("owner", ""),
                            *[f"{float(v):.2f}" for v in pred["xyxy"]],
                        ]
                    )
        case_summary["thresholds"][str(conf)] = {
            "raw_prediction_count": len(preds),
            "postprocessed_prediction_count": len(post_preds),
            "rejected_count": len(rejected),
            "predict_ms": round(elapsed_ms, 3),
            "confidence_min": round(min((float(p["confidence"]) for p in preds), default=0.0), 3),
            "confidence_mean": round(
                sum(float(p["confidence"]) for p in preds) / len(preds), 3
            )
            if preds
            else 0.0,
            "confidence_max": round(max((float(p["confidence"]) for p in preds), default=0.0), 3),
        }
    return case_summary


def main() -> None:
    args = parse_args()
    thresholds = [float(part.strip()) for part in args.thresholds.split(",") if part.strip()]
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    model_path = args.model.resolve()
    if not input_root.exists():
        raise FileNotFoundError(f"Missing input root: {input_root}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model: {model_path}")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Missing ultralytics in training/review environment.") from exc

    model = YOLO(str(model_path))
    post_config = PostprocessConfig(
        min_confidence=args.post_min_conf,
        duplicate_iou=args.post_duplicate_iou,
    )
    run_dir = output_root / time.strftime("%Y%m%d-%H%M%S")
    summaries: list[dict[str, object]] = []
    for case_dir in sorted(p for p in input_root.iterdir() if p.is_dir()):
        summary = write_review_case(
            case_dir=case_dir,
            output_dir=run_dir / case_dir.name,
            model=model,
            thresholds=thresholds,
            device=args.device,
            iou=args.iou,
            end2end=args.end2end,
            postprocess=args.postprocess,
            post_config=post_config,
        )
        summaries.append(summary)
        counts = ", ".join(
            f"conf {conf}: raw {summary['thresholds'][str(conf)]['raw_prediction_count']}"
            f" -> post {summary['thresholds'][str(conf)]['postprocessed_prediction_count']}"
            for conf in thresholds
        )
        print(f"ok {case_dir.name}: {counts}")
    (run_dir / "review_summary.json").write_text(
        json.dumps(
            {
                "input_root": str(input_root),
                "model": str(model_path),
                "thresholds": thresholds,
                "iou": args.iou,
                "end2end": args.end2end,
                "postprocess": args.postprocess,
                "postprocess_config": post_config.__dict__,
                "cases": summaries,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"output={run_dir}")


if __name__ == "__main__":
    main()
