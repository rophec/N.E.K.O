from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Callable, TypeVar

import cv2
import numpy as np

from postprocess_yolo_predictions import PostprocessConfig, postprocess_predictions


REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_TOOLS = REPO_ROOT / "tools" / "automajhong_reference"
sys.path.insert(0, str(REFERENCE_TOOLS))

import run_tablecloth_warp_batch as warp  # noqa: E402


DEFAULT_INPUT = (
    REPO_ROOT
    / "datasets"
    / "mahjong_yolo_active"
    / "manual_validation"
    / "images"
)
DEFAULT_MODEL = (
    REPO_ROOT
    / "runs"
    / "mahjong_yolo26_hbb"
    / "hbb_v1"
    / "weights"
    / "best.pt"
)
DEFAULT_OUTPUT_PARENT = REPO_ROOT / "runs" / "mahjong_yolo26_hbb" / "manual_validation"

T = TypeVar("T")


class Timer:
    def __init__(self) -> None:
        self.rows: list[tuple[str, float]] = []

    def measure(self, name: str, fn: Callable[[], T]) -> T:
        start = time.perf_counter()
        result = fn()
        self.rows.append((name, (time.perf_counter() - start) * 1000.0))
        return result

    def as_dict(self) -> dict[str, float]:
        data = {name: round(ms, 3) for name, ms in self.rows}
        data["total_ms"] = round(sum(ms for _, ms in self.rows), 3)
        return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run raw Mahjong screenshots through table warp and YOLO prediction."
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--device", default="0")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--postprocess", action="store_true")
    parser.add_argument("--post-min-conf", type=float, default=0.05)
    parser.add_argument("--post-duplicate-iou", type=float, default=0.28)
    return parser.parse_args()


def timestamped_output_root() -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return DEFAULT_OUTPUT_PARENT / stamp


def list_images(input_root: Path, limit: int) -> list[Path]:
    images = warp.list_images(input_root)
    if limit > 0:
        return images[:limit]
    return images


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_prediction_txt(path: Path, predictions: list[dict[str, object]]) -> None:
    lines = []
    for item in predictions:
        x1, y1, x2, y2 = item["xyxy"]
        lines.append(
            f"{item['class_id']} {item['class_name']} {item['confidence']:.6f} "
            f"{x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f}"
        )
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def draw_numbered_predictions(image: np.ndarray, predictions: list[dict[str, object]]) -> np.ndarray:
    # EN: Review-friendly drawing: use stable numbers instead of overlapping text labels.
    # ZH: 便于验收的画法：只画稳定编号，避免类别文字互相遮挡。
    output = image.copy()
    for idx, item in enumerate(predictions, start=1):
        x1, y1, x2, y2 = [int(round(float(v))) for v in item["xyxy"]]
        color = (40 + (idx * 37) % 200, 80 + (idx * 71) % 160, 80 + (idx * 53) % 160)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(output, (x1, max(0, y1 - 22)), (x1 + 30, max(22, y1)), color, -1)
        cv2.putText(
            output,
            str(idx),
            (x1 + 4, max(16, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
    return output


def predict_yolo(model, warped_path: Path, case_dir: Path, conf: float, iou: float, device: str) -> list[dict[str, object]]:
    result = model.predict(
        str(warped_path),
        conf=conf,
        iou=iou,
        device=device,
        verbose=False,
        save=False,
    )[0]

    annotated = result.plot()
    warp.write_png(case_dir / "10-yolo-prediction.png", annotated)

    predictions: list[dict[str, object]] = []
    if result.boxes is not None:
        names = result.names
        for box in result.boxes:
            class_id = int(box.cls[0].detach().cpu().item())
            confidence = float(box.conf[0].detach().cpu().item())
            xyxy = [float(v) for v in box.xyxy[0].detach().cpu().tolist()]
            predictions.append(
                {
                    "class_id": class_id,
                    "class_name": names.get(class_id, str(class_id)),
                    "confidence": confidence,
                    "xyxy": xyxy,
                }
            )
    write_json(case_dir / "10-yolo-prediction.json", predictions)
    write_prediction_txt(case_dir / "10-yolo-prediction.txt", predictions)
    return predictions


def process_image(model, image_path: Path, case_dir: Path, args: argparse.Namespace) -> dict[str, object]:
    timer = Timer()

    image = timer.measure("01_read_image", lambda: warp.read_image(image_path))
    timer.measure("02_write_source", lambda: warp.write_png(case_dir / "01-source.png", image))

    mask_before, hsv_info = timer.measure("03_hsv_tablecloth_mask", lambda: warp.hsv_tablecloth_mask(image))
    timer.measure("04_write_mask_before", lambda: warp.write_png(case_dir / "02-mask-before.png", mask_before))

    mask_after = timer.measure(
        "05_morphology_close",
        lambda: cv2.morphologyEx(
            mask_before,
            cv2.MORPH_CLOSE,
            np.ones((7, 7), np.uint8),
            iterations=3,
        ),
    )
    timer.measure("06_write_mask_after", lambda: warp.write_png(case_dir / "03-mask-after.png", mask_after))

    component, component_info = timer.measure("07_largest_component", lambda: warp.largest_component(mask_after))
    timer.measure("08_write_largest_component", lambda: warp.write_png(case_dir / "04-largest-component.png", component))

    contour = timer.measure("09_external_contour_image", lambda: warp.external_contour_image(component))
    timer.measure("10_write_external_contour", lambda: warp.write_png(case_dir / "05-external-contour.png", contour))

    lines = timer.measure("11_select_support_lines", lambda: warp.select_support_lines(contour))

    def selected_lines_image() -> np.ndarray:
        selected = np.zeros_like(image)
        selected[:] = (0, 0, 0)
        return warp.draw_lines(selected, lines)

    selected_lines = timer.measure("12_draw_selected_lines", selected_lines_image)
    timer.measure("13_write_selected_lines", lambda: warp.write_png(case_dir / "06-selected-lines.png", selected_lines))

    lines_on_source = timer.measure("14_draw_lines_on_source", lambda: warp.draw_lines(image, lines))
    timer.measure("15_write_lines_on_source", lambda: warp.write_png(case_dir / "07-lines-on-source.png", lines_on_source))

    extended_lines = timer.measure("16_draw_extended_lines", lambda: warp.draw_lines(image, lines, extended=True))
    timer.measure("17_write_extended_lines", lambda: warp.write_png(case_dir / "08-extended-lines.png", extended_lines))

    warped, intersections = timer.measure("18_warp_table", lambda: warp.warp_table(image, lines))
    warped_path = case_dir / "09-warp-square-800.png"
    timer.measure("19_write_warp", lambda: warp.write_png(warped_path, warped))

    sheet = timer.measure(
        "20_contact_sheet",
        lambda: warp.contact_sheet(
            [
                ("01 source", image),
                ("03 mask", mask_after),
                ("05 contour", contour),
                ("07 lines", lines_on_source),
                ("08 extended", extended_lines),
                ("09 warp", warped),
            ]
        ),
    )
    timer.measure("21_write_contact_sheet", lambda: warp.write_png(case_dir / "contact-sheet.png", sheet))

    predictions = timer.measure(
        "22_yolo_predict",
        lambda: predict_yolo(model, warped_path, case_dir, args.conf, args.iou, args.device),
    )
    postprocessed_predictions: list[dict[str, object]] = predictions
    rejected_predictions: list[dict[str, object]] = []
    if args.postprocess:
        def run_postprocess() -> list[dict[str, object]]:
            nonlocal rejected_predictions
            post_config = PostprocessConfig(
                min_confidence=args.post_min_conf,
                duplicate_iou=args.post_duplicate_iou,
            )
            kept, rejected = postprocess_predictions(predictions, post_config)
            rejected_predictions = rejected
            return kept

        postprocessed_predictions = timer.measure("23_yolo_postprocess", run_postprocess)
        write_json(case_dir / "11-yolo-postprocessed.json", postprocessed_predictions)
        write_json(case_dir / "11-yolo-rejected.json", rejected_predictions)
        write_prediction_txt(case_dir / "11-yolo-postprocessed.txt", postprocessed_predictions)
        numbered = draw_numbered_predictions(warped, postprocessed_predictions)
        timer.measure(
            "24_write_postprocessed_numbered",
            lambda: warp.write_png(case_dir / "11-yolo-postprocessed-numbered.png", numbered),
        )

    summary = {
        "source": str(image_path),
        "case_dir": str(case_dir),
        "image_size": [int(image.shape[1]), int(image.shape[0])],
        "model": str(args.model.resolve()),
        "confidence_threshold": args.conf,
        "iou_threshold": args.iou,
        "prediction_count": len(predictions),
        "postprocess_enabled": args.postprocess,
        "postprocessed_prediction_count": len(postprocessed_predictions),
        "rejected_prediction_count": len(rejected_predictions),
        "hsv": hsv_info,
        "largest_component": component_info,
        "selected_lines": {name: asdict(line) for name, line in lines.items()},
        "intersections": intersections,
        "warp_output": str(warped_path),
        "prediction_image": str(case_dir / "10-yolo-prediction.png"),
        "prediction_json": str(case_dir / "10-yolo-prediction.json"),
        "postprocessed_prediction_image": str(case_dir / "11-yolo-postprocessed-numbered.png"),
        "postprocessed_prediction_json": str(case_dir / "11-yolo-postprocessed.json"),
        "timings_ms": timer.as_dict(),
    }
    write_json(case_dir / "summary.json", summary)
    return summary


def write_timing_csv(output_root: Path, summaries: list[dict[str, object]]) -> None:
    steps = sorted(
        {
            key
            for summary in summaries
            for key in summary["timings_ms"].keys()
            if key != "total_ms"
        }
    )
    with (output_root / "timings.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["case", "prediction_count", *steps, "total_ms"])
        for summary in summaries:
            timings = summary["timings_ms"]
            writer.writerow(
                [
                    Path(summary["source"]).stem,
                    summary["prediction_count"],
                    *[timings.get(step, "") for step in steps],
                    timings["total_ms"],
                ]
            )


def main() -> int:
    args = parse_args()
    args.input_root = args.input_root.resolve()
    args.model = args.model.resolve()
    output_root = (args.output_root.resolve() if args.output_root else timestamped_output_root())

    if not args.input_root.exists():
        raise FileNotFoundError(f"Input folder does not exist: {args.input_root}")
    if not args.model.exists():
        raise FileNotFoundError(f"YOLO model does not exist: {args.model}")

    images = list_images(args.input_root, args.limit)
    if not images:
        raise ValueError(f"No validation images found in: {args.input_root}")

    from ultralytics import YOLO

    output_root.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.model))
    used: set[str] = set()
    summaries: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    for image_path in images:
        case_id = warp.case_id_from_path(image_path, used)
        case_dir = output_root / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        try:
            summary = process_image(model, image_path, case_dir, args)
            summaries.append(summary)
            print(
                f"ok {case_id}: predictions={summary['prediction_count']} "
                f"total_ms={summary['timings_ms']['total_ms']}"
            )
        except Exception as exc:  # noqa: BLE001 - keep batch validation running.
            failure = {"case": case_id, "source": str(image_path), "error": str(exc)}
            failures.append(failure)
            write_json(case_dir / "failure.json", failure)
            print(f"fail {case_id}: {exc}")

    write_timing_csv(output_root, summaries)
    write_json(
        output_root / "summary_all.json",
        {
            "input_root": str(args.input_root),
            "output_root": str(output_root),
            "model": str(args.model),
            "count": len(images),
            "ok": len(summaries),
            "failed": len(failures),
            "summaries": summaries,
            "failures": failures,
        },
    )
    print(f"output={output_root}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
