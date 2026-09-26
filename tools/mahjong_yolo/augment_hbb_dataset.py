from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    REPO_ROOT
    / "datasets"
    / "mahjong_yolo_active"
    / "train_ready"
    / "hbb_v1"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "datasets"
    / "mahjong_yolo_active"
    / "augmented"
    / "hbb_v1_safe_aug"
)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class YoloBox:
    cls: int
    x: float
    y: float
    w: float
    h: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an offline, safe-augmented YOLO HBB Mahjong dataset."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--variants", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_labels(path: Path) -> list[YoloBox]:
    if not path.exists():
        return []
    boxes: list[YoloBox] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls, x, y, w, h = parts
        boxes.append(YoloBox(int(cls), float(x), float(y), float(w), float(h)))
    return boxes


def write_labels(path: Path, boxes: list[YoloBox]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{b.cls} {b.x:.6f} {b.y:.6f} {b.w:.6f} {b.h:.6f}" for b in boxes]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def image_label_pairs(dataset_root: Path, split: str) -> list[tuple[Path, Path]]:
    image_dir = dataset_root / "images" / split
    label_dir = dataset_root / "labels" / split
    pairs: list[tuple[Path, Path]] = []
    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTS:
            continue
        pairs.append((image_path, label_dir / f"{image_path.stem}.txt"))
    return pairs


def copy_pair(
    image_path: Path,
    label_path: Path,
    output_root: Path,
    split: str,
    stem: str,
) -> None:
    target_image = output_root / "images" / split / f"{stem}{image_path.suffix.lower()}"
    target_label = output_root / "labels" / split / f"{stem}.txt"
    target_image.parent.mkdir(parents=True, exist_ok=True)
    target_label.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, target_image)
    if label_path.exists():
        shutil.copy2(label_path, target_label)
    else:
        target_label.write_text("", encoding="utf-8")


def adjust_hsv_brightness(image: np.ndarray, rng: random.Random) -> np.ndarray:
    # EN: Mirrors Ultralytics-style HSV augmentation without changing geometry.
    # ZH: 类似 Ultralytics 的 HSV 增强，只改颜色，不改变标注框几何。
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-4.0, 4.0)) % 180.0
    hsv[..., 1] *= rng.uniform(0.82, 1.18)
    hsv[..., 2] *= rng.uniform(0.80, 1.22)
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    alpha = rng.uniform(0.92, 1.08)
    beta = rng.uniform(-10.0, 10.0)
    return cv2.convertScaleAbs(out, alpha=alpha, beta=beta)


def add_camera_noise(image: np.ndarray, rng: random.Random) -> np.ndarray:
    # EN: Simulates compression/capture noise seen in live screenshots.
    # ZH: 模拟实时截图里的压缩噪声和截图噪点。
    sigma = rng.uniform(1.5, 5.5)
    noise = np.random.default_rng(rng.randrange(1 << 30)).normal(0, sigma, image.shape)
    noisy = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if rng.random() < 0.45:
        noisy = cv2.GaussianBlur(noisy, (3, 3), rng.uniform(0.2, 0.7))
    return noisy


def labels_to_corners(boxes: list[YoloBox], width: int, height: int) -> list[tuple[int, np.ndarray]]:
    corners: list[tuple[int, np.ndarray]] = []
    for box in boxes:
        cx = box.x * width
        cy = box.y * height
        bw = box.w * width
        bh = box.h * height
        x1 = cx - bw / 2.0
        y1 = cy - bh / 2.0
        x2 = cx + bw / 2.0
        y2 = cy + bh / 2.0
        pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
        corners.append((box.cls, pts))
    return corners


def corners_to_labels(corners: list[tuple[int, np.ndarray]], width: int, height: int) -> list[YoloBox]:
    boxes: list[YoloBox] = []
    for cls, pts in corners:
        xs = np.clip(pts[:, 0], 0, width - 1)
        ys = np.clip(pts[:, 1], 0, height - 1)
        x1, x2 = float(xs.min()), float(xs.max())
        y1, y2 = float(ys.min()), float(ys.max())
        bw = x2 - x1
        bh = y2 - y1
        if bw < 4 or bh < 4:
            continue
        boxes.append(
            YoloBox(
                cls=cls,
                x=((x1 + x2) / 2.0) / width,
                y=((y1 + y2) / 2.0) / height,
                w=bw / width,
                h=bh / height,
            )
        )
    return boxes


def affine_matrix(width: int, height: int, rng: random.Random, variant: int) -> np.ndarray:
    # EN: Small geometry augmentation only; no flips because they corrupt tile identity.
    # ZH: 只做小幅几何增强；不做翻转，因为翻转会破坏麻将牌面语义。
    angle = rng.uniform(-3.0, 3.0)
    scale = rng.uniform(0.94, 1.06)
    tx = rng.uniform(-0.035, 0.035) * width
    ty = rng.uniform(-0.035, 0.035) * height
    if variant % 2 == 0:
        angle *= 0.5
        scale = rng.uniform(0.97, 1.04)
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, scale)
    matrix[:, 2] += (tx, ty)
    return matrix


def apply_affine(
    image: np.ndarray,
    boxes: list[YoloBox],
    rng: random.Random,
    variant: int,
) -> tuple[np.ndarray, list[YoloBox]]:
    height, width = image.shape[:2]
    matrix = affine_matrix(width, height, rng, variant)
    border = tuple(int(v) for v in image.reshape(-1, 3).mean(axis=0))
    transformed_image = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border,
    )
    corners = labels_to_corners(boxes, width, height)
    transformed_corners: list[tuple[int, np.ndarray]] = []
    for cls, pts in corners:
        ones = np.ones((pts.shape[0], 1), dtype=np.float32)
        hom = np.hstack([pts, ones])
        transformed_corners.append((cls, hom @ matrix.T))
    return transformed_image, corners_to_labels(transformed_corners, width, height)


def augment_one(
    image: np.ndarray,
    boxes: list[YoloBox],
    rng: random.Random,
    variant: int,
) -> tuple[np.ndarray, list[YoloBox], list[str]]:
    operations: list[str] = []
    out = image.copy()
    out = adjust_hsv_brightness(out, rng)
    operations.append("hsv_brightness")

    if variant % 4 in {1, 2, 3}:
        out, boxes = apply_affine(out, boxes, rng, variant)
        operations.append("small_affine_no_flip")

    if variant % 4 in {2, 3}:
        out = add_camera_noise(out, rng)
        operations.append("noise_or_blur")

    if variant % 4 == 3:
        # EN: Simulates slight JPEG/Web capture loss without changing labels.
        # ZH: 模拟网页/截图压缩损失，不改变标注框。
        quality = rng.randint(72, 92)
        ok, enc = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if ok:
            out = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            operations.append(f"jpeg_quality_{quality}")

    return out, boxes, operations


def copy_dataset_metadata(input_root: Path, output_root: Path, train_count: int, val_count: int) -> None:
    classes = (input_root / "classes.txt").read_text(encoding="utf-8")
    (output_root / "classes.txt").write_text(classes, encoding="utf-8")
    names = [line.strip() for line in classes.splitlines() if line.strip()]
    yaml_lines = [
        f"path: {output_root.as_posix()}",
        "train: images/train",
        "val: images/val",
        f"nc: {len(names)}",
        "names:",
    ]
    yaml_lines.extend(f"  {idx}: {name}" for idx, name in enumerate(names))
    (output_root / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    manifest = {
        "source": str(input_root),
        "train_images": train_count,
        "val_images": val_count,
        "notes": [
            "Offline augmentation materialized for inspection.",
            "No fliplr/flipud because Mahjong glyph identity changes under flips.",
            "Validation images are copied without augmentation.",
        ],
        "ultralytics_analogs": {
            "hsv_h": "small hue jitter",
            "hsv_s": "small saturation jitter",
            "hsv_v": "small value jitter",
            "degrees": "+/- 3 degrees",
            "translate": "+/- 3.5%",
            "scale": "0.94-1.06",
            "fliplr": 0.0,
            "flipud": 0.0,
            "mosaic": 0.0,
        },
    }
    (output_root / "augmentation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    input_root = args.input.resolve()
    output_root = args.output.resolve()
    if not input_root.exists():
        raise FileNotFoundError(f"Missing input dataset: {input_root}")
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists, pass --overwrite: {output_root}")
        shutil.rmtree(output_root)
    for split in ("train", "val"):
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    train_written = 0
    train_pairs = image_label_pairs(input_root, "train")
    for image_path, label_path in train_pairs:
        boxes = read_labels(label_path)
        copy_pair(image_path, label_path, output_root, "train", image_path.stem)
        train_written += 1

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        for idx in range(args.variants):
            aug_image, aug_boxes, _operations = augment_one(image, boxes, rng, idx)
            stem = f"{image_path.stem}_aug{idx + 1:02d}"
            out_image = output_root / "images" / "train" / f"{stem}{image_path.suffix.lower()}"
            out_label = output_root / "labels" / "train" / f"{stem}.txt"
            cv2.imwrite(str(out_image), aug_image)
            write_labels(out_label, aug_boxes)
            train_written += 1

    val_written = 0
    for image_path, label_path in image_label_pairs(input_root, "val"):
        copy_pair(image_path, label_path, output_root, "val", image_path.stem)
        val_written += 1

    copy_dataset_metadata(input_root, output_root, train_written, val_written)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    print(
        json.dumps(
            {
                "input": str(input_root),
                "output": str(output_root),
                "train_images_written": train_written,
                "val_images_written": val_written,
                "variants_per_train_image": args.variants,
                "elapsed_ms": round(elapsed_ms, 3),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
