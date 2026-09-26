from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = (
    REPO_ROOT
    / "datasets"
    / "mahjong_yolo_active"
    / "train_ready"
    / "hbb_v1"
    / "data.yaml"
)
DEFAULT_MODEL = REPO_ROOT / "yolo26n.pt"
DEFAULT_PROJECT = REPO_ROOT / "runs" / "mahjong_yolo26_hbb"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO26 HBB on Mahjong tiles.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--name", default="hbb_v1")
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--lr0", type=float, default=0.002)
    parser.add_argument("--mosaic", type=float, default=0.0)
    parser.add_argument("--fliplr", type=float, default=0.0)
    parser.add_argument("--flipud", type=float, default=0.0)
    parser.add_argument("--scale", type=float, default=0.2)
    parser.add_argument("--translate", type=float, default=0.05)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_train_kwargs(args: argparse.Namespace) -> dict[str, object]:
    return {
        "data": str(args.data.resolve()),
        "imgsz": args.imgsz,
        "epochs": args.epochs,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "patience": args.patience,
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "mosaic": args.mosaic,
        "fliplr": args.fliplr,
        "flipud": args.flipud,
        "scale": args.scale,
        "translate": args.translate,
        "project": str(args.project.resolve()),
        "name": args.name,
        "exist_ok": True,
        "task": "detect",
    }


def validate_inputs(args: argparse.Namespace) -> None:
    if not args.data.exists():
        raise FileNotFoundError(f"Missing data yaml: {args.data}")
    if not args.model.exists():
        raise FileNotFoundError(f"Missing YOLO26 model: {args.model}")


def main() -> None:
    args = parse_args()
    validate_inputs(args)
    train_kwargs = build_train_kwargs(args)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "model": str(args.model.resolve()),
                    "train_kwargs": train_kwargs,
                    "will_import": "ultralytics.YOLO",
                    "will_run": "YOLO(model).train(**train_kwargs)",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Missing training dependency: ultralytics. "
            "Install it only in the training environment, not in the plugin runtime."
        ) from exc

    model = YOLO(str(args.model.resolve()))
    results = model.train(**train_kwargs)
    print(results)


if __name__ == "__main__":
    main()
