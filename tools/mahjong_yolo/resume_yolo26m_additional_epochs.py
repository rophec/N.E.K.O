"""Resume an interrupted YOLO26m run for an exact number of additional epochs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEST_DATA = (
    "datasets/mahjong_yolo_active/manual_ground_truth_canonical_20260712/data.yaml"
)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resume a YOLO checkpoint while preserving optimizer state and add exact extra epochs."
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--additional-epochs", type=positive_int, default=50)
    parser.add_argument("--batch", type=int, default=6)
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--device", default="0")
    parser.add_argument("--test-data", default=DEFAULT_TEST_DATA)
    parser.add_argument("--output-root", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint_path = args.checkpoint.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved_epoch = int(checkpoint.get("epoch", -1))
    completed_epochs = saved_epoch + 1
    if completed_epochs <= 0 or checkpoint.get("optimizer") is None:
        raise ValueError(f"Checkpoint cannot resume optimizer state: {checkpoint_path}")

    total_epochs = completed_epochs + args.additional_epochs
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_root = args.output_root or (
        checkpoint_path.parents[3] / f"resume_additional_{args.additional_epochs}_{timestamp}"
    )
    output_root.mkdir(parents=True, exist_ok=False)
    resume_checkpoint = output_root / "resume_checkpoint_total_epochs.pt"

    # EN: Ultralytics reads total epochs from train_args during resume.
    # ZH: Ultralytics 恢复时从 train_args 读取总轮数，因此只修改检查点副本。
    train_args = dict(checkpoint.get("train_args", {}))
    # EN: Ultralytics reads total epochs from train_args during resume.
    # ZH: Ultralytics 恢复时从 train_args 读取总轮数，因此只修改检查点副本。
    train_args = dict(checkpoint.get("train_args", {}))
    train_args = dict(checkpoint.get("train_args", {}))
    train_args["epochs"] = total_epochs
    train_args["project"] = str(output_root / "training")
    train_args["name"] = f"continued_to_{total_epochs}"
    train_args["exist_ok"] = True
    checkpoint["train_args"] = train_args
    torch.save(checkpoint, resume_checkpoint)

    metadata = {
        "source_checkpoint": str(checkpoint_path),
        "saved_epoch_zero_based": saved_epoch,
        "completed_epochs": completed_epochs,
        "additional_epochs": args.additional_epochs,
        "total_epochs": total_epochs,
        "resume_checkpoint": str(resume_checkpoint),
        "training_output": str(output_root / "training" / f"continued_to_{total_epochs}"),
        "test_data": args.test_data,
    }
    (output_root / "resume_plan.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    from ultralytics import YOLO

    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)
    model = YOLO(str(resume_checkpoint))
    model.train(
        resume=str(resume_checkpoint),
        batch=args.batch,
        device=args.device,
        workers=0,
        cache="ram",
        plots=True,
    )

    # EN: The edited checkpoint redirects resumed weights to this new output directory.
    # ZH: 检查点副本已重定向恢复产物到新目录，原训练目录不会被覆盖。
    resumed_weights = output_root / "training" / f"continued_to_{total_epochs}" / "weights"
    best_model = resumed_weights / "best.pt"
    if not best_model.exists():
        best_model = resumed_weights / "last.pt"
    evaluator = YOLO(str(best_model))
    metrics = evaluator.val(
        data=args.test_data,
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=0,
        project=str(output_root / "fixed_test"),
        name="combined_test_54",
        exist_ok=True,
        plots=True,
    )
    summary = {
        **metadata,
        "evaluated_model": str(best_model),
        "test_metrics": {
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "map50": float(metrics.box.map50),
            "map50_95": float(metrics.box.map),
        },
    }
    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
