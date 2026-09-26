"""Train one YOLO26m experiment against the canonical fixed test set."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = "datasets/mahjong_yolo_active/dataset_registry/15_balanced_dragon_features_20260713/data.yaml"
DEFAULT_TEST_DATA = "datasets/mahjong_yolo_active/manual_ground_truth_canonical_20260712/data.yaml"


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one YOLO26m HBB run and evaluate it against the canonical fixed test set."
    )
    parser.add_argument("--epochs", required=True, type=positive_int, help="Epoch count, for example: 100")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--device", default="0")
    parser.add_argument("--model", default="yolo26m.pt")
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--test-data", default=DEFAULT_TEST_DATA)
    parser.add_argument("--save-period", type=positive_int, default=50)
    parser.add_argument("--output-root", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_root = args.output_root or (
        REPO_ROOT
        / "runs"
        / "mahjong_yolo26_hbb"
        / "single_stage_improve"
        / f"real_slot_synthetic_yolo26m_{args.imgsz}_b{args.batch}_test54_epochs_{args.epochs}_{timestamp}"
    )

    # EN: Keep training and fixed-test reports on the same comparison path.
    # ZH: Tongyi xunlian yu guding ceshi liucheng, bianyu duibi lishi zhibiao.
    command = [
        sys.executable,
        "tools/mahjong_yolo/run_epoch_comparison.py",
        "--data",
        args.data,
        "--test-data",
        args.test_data,
        "--model",
        args.model,
        "--output-root",
        str(output_root),
        "--epochs",
        str(args.epochs),
        "--imgsz",
        str(args.imgsz),
        "--batch",
        str(args.batch),
        "--device",
        str(args.device),
        "--workers",
        "0",
        "--cache",
        "ram",
        "--optimizer",
        "AdamW",
        "--lr0",
        "0.002",
        "--hsv-h",
        "0.003",
        "--hsv-s",
        "0.08",
        "--hsv-v",
        "0.08",
        "--scale",
        "0.05",
        "--translate",
        "0.02",
        "--erasing",
        "0.0",
        "--save-period",
        str(args.save_period),
    ]
    print(f"Output root: {output_root}", flush=True)
    print("Command:", " ".join(command), flush=True)
    return subprocess.run(command, cwd=REPO_ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
