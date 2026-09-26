from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = (
    REPO_ROOT
    / "datasets"
    / "mahjong_yolo_active"
    / "dataset_registry"
    / "15_balanced_dragon_features_20260713"
    / "data.yaml"
)
DEFAULT_RAW_VALIDATION = (
    REPO_ROOT
    / "datasets"
    / "mahjong_yolo_active"
    / "manual_ground_truth_canonical_20260712"
    / "images"
    / "test"
)
DEFAULT_MODEL = REPO_ROOT / "yolo26n.pt"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "runs" / "mahjong_yolo26_hbb" / "epoch_compare"
DEFAULT_TEST_DATA = (
    REPO_ROOT
    / "datasets"
    / "mahjong_yolo_active"
    / "manual_ground_truth_canonical_20260712"
    / "data.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train 80/150/300 epoch YOLO26 HBB models and save pure detection evidence."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--raw-validation", type=Path, default=DEFAULT_RAW_VALIDATION)
    parser.add_argument("--test-data", type=Path, default=DEFAULT_TEST_DATA)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--epochs", nargs="+", type=int, default=[80, 150, 300])
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--cache", default=False)
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--lr0", type=float, default=0.002)
    parser.add_argument("--hsv-h", type=float, default=0.015)
    parser.add_argument("--hsv-s", type=float, default=0.7)
    parser.add_argument("--hsv-v", type=float, default=0.4)
    parser.add_argument("--scale", type=float, default=0.2)
    parser.add_argument("--translate", type=float, default=0.05)
    parser.add_argument("--erasing", type=float, default=0.4)
    parser.add_argument("--save-period", type=int, default=-1)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_data_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_dataset_path(data_yaml: Path, value: str | Path) -> Path:
    value_path = Path(value)
    if value_path.is_absolute():
        return value_path

    data = load_data_yaml(data_yaml)
    root = Path(data.get("path") or data_yaml.parent)
    if not root.is_absolute():
        root = data_yaml.parent / root
    return root / value_path


def ensure_inputs(args: argparse.Namespace) -> None:
    if not args.data.exists():
        raise FileNotFoundError(f"Missing data yaml: {args.data}")
    if not args.model.exists() and not is_downloadable_model_name(args.model):
        raise FileNotFoundError(f"Missing base model: {args.model}")
    if not args.raw_validation.exists():
        raise FileNotFoundError(f"Missing raw validation directory: {args.raw_validation}")
    if args.test_data and not args.test_data.exists():
        raise FileNotFoundError(f"Missing test data yaml: {args.test_data}")


def is_downloadable_model_name(model: Path) -> bool:
    return (
        not model.is_absolute()
        and model.parent == Path(".")
        and model.suffix == ".pt"
        and model.name.startswith("yolo26")
    )


def model_argument(model: Path) -> str:
    if model.exists():
        return str(model.resolve())
    return model.as_posix()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def result_to_boxes(result: Any) -> list[dict[str, Any]]:
    names = result.names
    boxes: list[dict[str, Any]] = []
    if result.boxes is None:
        return boxes

    xyxy = result.boxes.xyxy.detach().cpu().tolist()
    conf = result.boxes.conf.detach().cpu().tolist()
    cls = result.boxes.cls.detach().cpu().tolist()
    for index, (coords, score, class_id) in enumerate(zip(xyxy, conf, cls), start=1):
        class_index = int(class_id)
        boxes.append(
            {
                "index": index,
                "class_id": class_index,
                "class_name": names.get(class_index, str(class_index)),
                "confidence": round(float(score), 6),
                "xyxy": [round(float(v), 2) for v in coords],
            }
        )
    return boxes


def save_prediction_json(model: Any, source: Path, output_json: Path, args: argparse.Namespace) -> None:
    results = model.predict(
        source=str(source),
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        stream=False,
        verbose=False,
    )
    payload = []
    for result in results:
        payload.append(
            {
                "image": str(Path(result.path).resolve()),
                "boxes": result_to_boxes(result),
            }
        )
    write_json(output_json, payload)


def run_saved_prediction(
    model: Any,
    source: Path,
    project: Path,
    name: str,
    args: argparse.Namespace,
) -> Path:
    model.predict(
        source=str(source),
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        save=True,
        save_txt=True,
        save_conf=True,
        project=str(project),
        name=name,
        exist_ok=True,
        line_width=1,
        verbose=True,
    )
    return project / name


def train_one(args: argparse.Namespace, run_root: Path, epoch_count: int) -> dict[str, Any]:
    from ultralytics import YOLO

    run_name = f"epochs_{epoch_count:03d}"
    train_project = run_root / "training"
    val_project = run_root / "validation"
    test_project = run_root / "fixed_test_21"
    detect_project = run_root / "detections"
    json_project = run_root / "raw_yolo_boxes_json" / run_name

    base_model = YOLO(model_argument(args.model))
    train_result = base_model.train(
        data=str(args.data.resolve()),
        imgsz=args.imgsz,
        epochs=epoch_count,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        cache=args.cache,
        patience=0,
        optimizer=args.optimizer,
        lr0=args.lr0,
        mosaic=0.0,
        fliplr=0.0,
        flipud=0.0,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        scale=args.scale,
        translate=args.translate,
        erasing=args.erasing,
        save_period=args.save_period,
        project=str(train_project),
        name=run_name,
        exist_ok=True,
        task="detect",
        plots=True,
    )

    train_dir = Path(getattr(train_result, "save_dir", train_project / run_name))
    best_model = train_dir / "weights" / "best.pt"
    last_model = train_dir / "weights" / "last.pt"
    if not best_model.exists():
        raise FileNotFoundError(f"Training did not produce best.pt: {best_model}")

    trained_model = YOLO(str(best_model))
    val_result = trained_model.val(
        data=str(args.data.resolve()),
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(val_project),
        name=run_name,
        exist_ok=True,
        plots=True,
    )
    test_result = None
    if args.test_data:
        test_result = trained_model.val(
            data=str(args.test_data.resolve()),
            split="test",
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            project=str(test_project),
            name=run_name,
            exist_ok=True,
            plots=True,
        )

    data = load_data_yaml(args.data)
    val_source = resolve_dataset_path(args.data, data["val"])
    val_detect_dir = run_saved_prediction(
        trained_model, val_source, detect_project, f"{run_name}_val_boxes", args
    )
    raw_detect_dir = run_saved_prediction(
        trained_model,
        args.raw_validation,
        detect_project,
        f"{run_name}_raw6_boxes",
        args,
    )
    save_prediction_json(trained_model, val_source, json_project / "val_boxes.json", args)
    save_prediction_json(
        trained_model,
        args.raw_validation,
        json_project / "raw6_boxes.json",
        args,
    )

    summary = {
        "epochs": epoch_count,
        "train_dir": str(train_dir.resolve()),
        "best_model": str(best_model.resolve()),
        "last_model": str(last_model.resolve()),
        "validation_dir": str((val_project / run_name).resolve()),
        "test_dir": str((test_project / run_name).resolve()) if args.test_data else None,
        "val_detection_boxes_dir": str(val_detect_dir.resolve()),
        "raw6_detection_boxes_dir": str(raw_detect_dir.resolve()),
        "val_boxes_json": str((json_project / "val_boxes.json").resolve()),
        "raw6_boxes_json": str((json_project / "raw6_boxes.json").resolve()),
        "val_metrics": {
            "box_map50": float(getattr(val_result.box, "map50", 0.0)),
            "box_map50_95": float(getattr(val_result.box, "map", 0.0)),
            "box_precision": float(getattr(val_result.box, "mp", 0.0)),
            "box_recall": float(getattr(val_result.box, "mr", 0.0)),
        },
    }
    if test_result is not None:
        summary["test_metrics"] = {
            "box_map50": float(getattr(test_result.box, "map50", 0.0)),
            "box_map50_95": float(getattr(test_result.box, "map", 0.0)),
            "box_precision": float(getattr(test_result.box, "mp", 0.0)),
            "box_recall": float(getattr(test_result.box, "mr", 0.0)),
        }
    print(
        json.dumps(
            {
                "epoch_summary": run_name,
                "val_metrics": summary["val_metrics"],
                "test_metrics": summary.get("test_metrics"),
                "best_model": summary["best_model"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return summary


def write_readme(run_root: Path, args: argparse.Namespace) -> None:
    readme = f"""# Mahjong YOLO26 HBB Epoch Comparison

Created: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

This run trains three independent YOLO26 HBB models from the same base model and dataset.
The goal is to compare underfit vs overfit behavior across epoch counts.

## Inputs

- data: `{args.data.resolve()}`
- fixed test data: `{args.test_data.resolve() if args.test_data else "disabled"}`
- base model: `{model_argument(args.model)}`
- raw validation screenshots: `{args.raw_validation.resolve()}`
- epochs: `{", ".join(str(e) for e in args.epochs)}`
- confidence threshold for saved detection evidence: `{args.conf}`

## Output Layout

- `training/epochs_*/`: Ultralytics training output and weights.
- `validation/epochs_*/`: validation plots and metrics for each trained model.
- `fixed_test_21/epochs_*/`: held-out test plots and metrics for each trained model.
- `detections/epochs_*_val_boxes/`: pure YOLO detection-box images for the labeled validation split.
- `detections/epochs_*_raw6_boxes/`: pure YOLO detection-box images for the six raw screenshots.
- `raw_yolo_boxes_json/epochs_*/`: per-image raw YOLO boxes as JSON.
- `summary.json`: compact metrics and important paths.

These detection folders are before Mahjong strategy, ownership, river/hand judgment, or plugin decision logic.
"""
    (run_root / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_inputs(args)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_root = args.output_root / f"epoch_compare_{timestamp}"
    run_root.mkdir(parents=True, exist_ok=False)
    write_readme(run_root, args)

    manifest = {
        "run_root": str(run_root.resolve()),
        "data": str(args.data.resolve()),
        "test_data": str(args.test_data.resolve()) if args.test_data else None,
        "base_model": model_argument(args.model),
        "raw_validation": str(args.raw_validation.resolve()),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "confidence_for_detection_evidence": args.conf,
    }
    write_json(run_root / "manifest.json", manifest)
    shutil.copy2(args.data, run_root / "data.yaml")

    if args.dry_run:
        write_json(run_root / "dry_run.json", manifest)
        print(f"Dry run created: {run_root}")
        return

    summaries = []
    for epoch_count in args.epochs:
        print(f"\n=== Training YOLO26 HBB for {epoch_count} epochs ===", flush=True)
        summaries.append(train_one(args, run_root, epoch_count))
        write_json(run_root / "summary.json", summaries)

    print("\n=== Epoch comparison complete ===", flush=True)
    print(run_root.resolve(), flush=True)


if __name__ == "__main__":
    main()
