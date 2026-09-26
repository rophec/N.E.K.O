"""Fine-tune YOLO26m for Mahjong tile-face classification details."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import yaml
from ultralytics import YOLO


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = (
    REPO_ROOT
    / "datasets"
    / "mahjong_yolo_active"
    / "dataset_registry"
    / "16_real_val_face_detail_20260714"
    / "data.yaml"
)
DEFAULT_MODEL = (
    REPO_ROOT
    / "runs"
    / "mahjong_yolo26_hbb"
    / "single_stage_improve"
    / "real_slot_synthetic_yolo26m_800_b4_test54_epochs_150_20260713-140213"
    / "epoch_compare_20260713-140213"
    / "training"
    / "epochs_150"
    / "weights"
    / "best.pt"
)
DEFAULT_PROJECT = (
    REPO_ROOT / "runs" / "mahjong_yolo26_hbb" / "single_stage_improve" / "face_detail_finetune"
)


PROFILES = {
    "baseline": {
        "imgsz": 800,
        "batch": 4,
        "cls": 0.5,
        "lr0": 0.0005,
    },
    "detail": {
        "imgsz": 960,
        "batch": 2,
        "cls": 0.8,
        "lr0": 0.0004,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="detail")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--cls", type=float)
    parser.add_argument("--lr0", type=float)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--cache", choices=["ram", "disk", "false"], default="ram")
    parser.add_argument("--name")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume optimizer, scheduler, scaler, and epoch state from --model.",
    )
    parser.add_argument(
        "--milestone-epochs",
        type=int,
        nargs="*",
        default=[],
        help="Copy exact completed-epoch checkpoints, for example: 50 100 150.",
    )
    parser.add_argument(
        "--stop-after-epoch",
        type=int,
        help="Stop cleanly after saving this completed epoch when resuming a longer run.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate paths and print the resolved config only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = dict(PROFILES[args.profile])
    for key in ("imgsz", "batch", "cls", "lr0"):
        value = getattr(args, key)
        if value is not None:
            profile[key] = value

    data = args.data.resolve()
    model_path = args.model.resolve()
    project = args.project.resolve()
    if not data.exists():
        raise FileNotFoundError(f"Dataset config not found: {data}")
    if not model_path.exists():
        raise FileNotFoundError(f"Starting model not found: {model_path}")
    dataset_config = yaml.safe_load(data.read_text(encoding="utf-8"))
    dataset_root = Path(dataset_config.get("path", data.parent)).resolve()
    split_counts: dict[str, int] = {}
    for split in ("train", "val", "test"):
        split_value = dataset_config.get(split)
        if not split_value:
            raise ValueError(f"Dataset is missing the {split} split: {data}")
        split_path = Path(split_value)
        if not split_path.is_absolute():
            split_path = dataset_root / split_path
        if not split_path.exists():
            raise FileNotFoundError(f"Dataset split not found: {split_path}")
        split_counts[split] = sum(
            path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
            for path in split_path.iterdir()
        )

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = args.name or f"yolo26m_{args.profile}_{profile['imgsz']}_e{args.epochs}_{timestamp}"
    cache: str | bool = False if args.cache == "false" else args.cache
    run_config = {
        "profile": args.profile,
        "data": str(data),
        "model": str(model_path),
        "project": str(project),
        "name": name,
        "epochs": args.epochs,
        "resume": args.resume,
        "milestone_epochs": sorted(set(args.milestone_epochs)),
        "stop_after_epoch": args.stop_after_epoch,
        "device": args.device,
        "split_counts": split_counts,
        **profile,
    }
    print(json.dumps(run_config, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        return

    # EN: Fine-detail training keeps geometry stable and lets the model learn
    # tile-face differences instead of synthetic layout variation.
    # 中文：细节微调保持几何稳定，让模型学习牌面差异，而不是合成布局变化。
    model = YOLO(str(model_path))
    milestones = {epoch for epoch in args.milestone_epochs if 0 < epoch <= args.epochs}
    stop_after_epoch = args.stop_after_epoch

    def save_milestone(trainer: object) -> None:
        # EN: Ultralytics names periodic files with a zero-based epoch index.
        # 中文：Ultralytics 的周期文件名使用从零开始的轮次，这里保存真实完成轮次。
        completed_epoch = int(trainer.epoch) + 1
        if completed_epoch in milestones:
            destination = Path(trainer.wdir) / f"epoch_{completed_epoch:04d}.pt"
            shutil.copy2(Path(trainer.last), destination)
            print(f"Milestone checkpoint saved: {destination}", flush=True)
        if stop_after_epoch is not None and completed_epoch >= stop_after_epoch:
            # EN: Stop only after last.pt and any milestone copy are durable.
            # 中文：仅在 last.pt 和里程碑副本写盘后停止，保证后续仍可续训。
            trainer.stop = True
            print(f"Requested stop reached after epoch {completed_epoch}.", flush=True)

    if milestones or stop_after_epoch is not None:
        model.add_callback("on_model_save", save_milestone)
    model.train(
        resume=args.resume,
        data=str(data),
        epochs=args.epochs,
        imgsz=profile["imgsz"],
        batch=profile["batch"],
        device=args.device,
        workers=args.workers,
        cache=cache,
        project=str(project),
        name=name,
        exist_ok=False,
        optimizer="AdamW",
        lr0=profile["lr0"],
        lrf=0.05,
        weight_decay=0.0005,
        warmup_epochs=2.0,
        patience=0,
        save=True,
        save_period=-1,
        pretrained=True,
        amp=True,
        deterministic=True,
        seed=20260714,
        box=7.5,
        cls=profile["cls"],
        dfl=1.5,
        hsv_h=0.002,
        hsv_s=0.04,
        hsv_v=0.04,
        degrees=0.0,
        translate=0.01,
        scale=0.03,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.0,
        mosaic=0.0,
        mixup=0.0,
        cutmix=0.0,
        copy_paste=0.0,
        erasing=0.0,
        close_mosaic=0,
        plots=True,
        val=True,
        verbose=True,
    )
    save_dir = Path(model.trainer.save_dir)
    best = save_dir / "weights" / "best.pt"
    print(f"Training complete. Best model: {best}", flush=True)

    # EN: Test remains read-only and is evaluated only after training.
    # 中文：测试集保持只读，仅在训练结束后执行最终评估。
    test_model = YOLO(str(best))
    metrics = test_model.val(
        data=str(data),
        split="test",
        imgsz=profile["imgsz"],
        batch=profile["batch"],
        device=args.device,
        workers=args.workers,
        project=str(save_dir / "fixed_test"),
        name="best",
        plots=True,
        verbose=True,
    )
    summary = {
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "best_model": str(best),
    }
    (save_dir / "fixed_test_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
