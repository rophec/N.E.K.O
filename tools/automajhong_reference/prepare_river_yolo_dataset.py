"""Prepare Mahjong Soul river YOLO dataset drafts.

This script creates two practical artifacts:

1. A detection-only dataset from warped Mahjong Soul screenshots. The class is
   `tile`, because the current template classifier is not reliable enough to
   assign tile names automatically.
2. A review pack with per-slot crops and metadata so a human can fill real tile
   labels later.

中文说明：
本脚本只自动生成“牌的位置”训练草稿，不自动生成牌名标签。当前模板分类不够可靠，
所以不能把它的错误结果直接写进训练集，否则会污染 YOLO 模型。
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


WARP_NAME = "09-warp-square-800.png"


@dataclass(frozen=True)
class RiverGroup:
    owner: str
    orient: str
    rotation_hint: str
    bbox_xywh: tuple[int, int, int, int]
    area: float


@dataclass(frozen=True)
class TileSlot:
    case_id: str
    owner: str
    index: int
    rotation_hint: str
    bbox_xyxy: tuple[int, int, int, int]


def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"failed to read image: {path}")
    return image


def write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError(f"failed to encode png: {path}")
    encoded.tofile(str(path))


def detect_river_groups(image: np.ndarray) -> tuple[np.ndarray, list[RiverGroup]]:
    """Detect four river/discard groups on the warped table image.

    English: The mask is intentionally group-level. Mahjong discard tiles touch
    each other, so connected components usually represent a whole player's river.

    中文：这里先找“四家牌河大块”，不是直接找单张牌。牌河里的牌彼此贴得近，
    连通域通常天然会形成一整组。
    """

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = ((gray > 160) | (hsv[:, :, 2] > 180)).astype(np.uint8) * 255

    roi = np.zeros(mask.shape, np.uint8)
    roi[220:570, 230:610] = 255
    cv2.rectangle(roi, (335, 333), (465, 465), 0, -1)
    mask = cv2.bitwise_and(mask, roi)
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        iterations=2,
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
        iterations=1,
    )

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    groups: list[RiverGroup] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = float(cv2.contourArea(contour))
        if area < 1200:
            continue
        cx = x + w / 2
        cy = y + h / 2
        if cy < 330:
            owner, orient, rotation = "opposite", "horizontal", "180"
        elif cy > 470:
            owner, orient, rotation = "self", "horizontal", "0"
        elif cx < 370:
            owner, orient, rotation = "left_opponent", "vertical", "90"
        else:
            owner, orient, rotation = "right_opponent", "vertical", "270"
        groups.append(RiverGroup(owner, orient, rotation, (x, y, w, h), area))

    order = {"opposite": 0, "left_opponent": 1, "right_opponent": 2, "self": 3}
    groups.sort(key=lambda group: order.get(group.owner, 9))
    return mask, groups


def split_group_to_slots(case_id: str, group: RiverGroup) -> list[TileSlot]:
    """Split one river group into rough per-tile slots.

    English: This is a draft geometry splitter. It is good enough to create
    review crops, but L-shaped rivers still need human review or a stronger
    detector.

    中文：这是草稿拆槽逻辑，适合生成审核裁图；右家那种 L 形牌河仍需要人工审核
    或后续用更强的检测模型替代。
    """

    x, y, w, h = group.bbox_xywh
    slots: list[TileSlot] = []
    if group.orient == "horizontal":
        count = max(1, int(round(w / max(1, h * 0.58))))
        count = max(1, min(8, count))
        step = w / count
        for index in range(count):
            x0 = int(round(x + index * step))
            x1 = int(round(x + (index + 1) * step))
            slots.append(
                TileSlot(case_id, group.owner, index + 1, group.rotation_hint, (x0, y, x1, y + h))
            )
    else:
        count = max(1, int(round(h / max(1, w * 0.92))))
        count = max(1, min(8, count))
        step = h / count
        for index in range(count):
            y0 = int(round(y + index * step))
            y1 = int(round(y + (index + 1) * step))
            slots.append(
                TileSlot(case_id, group.owner, index + 1, group.rotation_hint, (x, y0, x + w, y1))
            )
    return slots


def yolo_bbox_from_xyxy(box: tuple[int, int, int, int], width: int, height: int) -> str:
    x0, y0, x1, y1 = box
    cx = ((x0 + x1) / 2) / width
    cy = ((y0 + y1) / 2) / height
    bw = (x1 - x0) / width
    bh = (y1 - y0) / height
    return f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def draw_groups(image: np.ndarray, groups: Iterable[RiverGroup]) -> np.ndarray:
    output = image.copy()
    colors = {
        "opposite": (0, 255, 255),
        "left_opponent": (0, 255, 0),
        "right_opponent": (255, 0, 255),
        "self": (255, 255, 0),
    }
    for group in groups:
        x, y, w, h = group.bbox_xywh
        color = colors.get(group.owner, (0, 255, 255))
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
        cv2.putText(output, group.owner, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return output


def draw_slots(image: np.ndarray, slots: Iterable[TileSlot]) -> np.ndarray:
    output = image.copy()
    colors = {
        "opposite": (0, 255, 255),
        "left_opponent": (0, 255, 0),
        "right_opponent": (255, 0, 255),
        "self": (255, 255, 0),
    }
    for slot in slots:
        x0, y0, x1, y1 = slot.bbox_xyxy
        color = colors.get(slot.owner, (0, 255, 255))
        cv2.rectangle(output, (x0, y0), (x1, y1), color, 2)
        cv2.putText(
            output,
            f"{slot.owner[0]}{slot.index}",
            (x0, y0 - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            color,
            1,
        )
    return output


def find_warp_images(input_root: Path) -> list[Path]:
    if input_root.is_file():
        return [input_root]
    return sorted(input_root.glob(f"*/{WARP_NAME}"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val"], default="train")
    args = parser.parse_args()

    output_root: Path = args.output_root
    images_dir = output_root / "images" / args.split
    labels_dir = output_root / "labels" / args.split
    review_dir = output_root / "review"
    crops_dir = review_dir / "crops"
    debug_dir = review_dir / "debug"
    for directory in (images_dir, labels_dir, crops_dir, debug_dir):
        directory.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    summary: list[dict[str, object]] = []
    warp_paths = find_warp_images(args.input_root)
    for warp_path in warp_paths:
        case_id = warp_path.parent.name if warp_path.name == WARP_NAME else warp_path.stem
        image = read_image(warp_path)
        height, width = image.shape[:2]
        mask, groups = detect_river_groups(image)
        slots = [slot for group in groups for slot in split_group_to_slots(case_id, group)]

        image_name = f"{case_id}_river.png"
        label_name = f"{case_id}_river.txt"
        write_png(images_dir / image_name, image)
        with open(labels_dir / label_name, "w", encoding="utf-8") as label_file:
            for slot in slots:
                label_file.write(yolo_bbox_from_xyxy(slot.bbox_xyxy, width, height) + "\n")

        write_png(debug_dir / f"{case_id}_01_group_mask.png", cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
        write_png(debug_dir / f"{case_id}_02_groups.png", draw_groups(image, groups))
        write_png(debug_dir / f"{case_id}_03_slots.png", draw_slots(image, slots))

        for slot in slots:
            x0, y0, x1, y1 = slot.bbox_xyxy
            crop = image[max(0, y0 - 2) : min(height, y1 + 2), max(0, x0 - 2) : min(width, x1 + 2)]
            crop_name = f"{case_id}_{slot.owner}_{slot.index:02d}.png"
            write_png(crops_dir / crop_name, crop)
            rows.append(
                {
                    "case_id": case_id,
                    "image": image_name,
                    "crop": crop_name,
                    "owner": slot.owner,
                    "index": slot.index,
                    "rotation_hint": slot.rotation_hint,
                    "bbox_xyxy": json.dumps(slot.bbox_xyxy, ensure_ascii=False),
                    "review_tile_label": "",
                    "review_keep": "",
                }
            )

        summary.append(
            {
                "case_id": case_id,
                "source": str(warp_path),
                "groups": [group.__dict__ for group in groups],
                "slot_count": len(slots),
                "label_file": str(labels_dir / label_name),
            }
        )

    with open(output_root / "data.yaml", "w", encoding="utf-8") as data_file:
        data_file.write(f"path: {output_root.as_posix()}\n")
        data_file.write("train: images/train\n")
        data_file.write("val: images/val\n")
        data_file.write("names:\n")
        data_file.write("  0: tile\n")

    with open(review_dir / "river_review.csv", "w", encoding="utf-8", newline="") as review_file:
        fieldnames = [
            "case_id",
            "image",
            "crop",
            "owner",
            "index",
            "rotation_hint",
            "bbox_xyxy",
            "review_tile_label",
            "review_keep",
        ]
        writer = csv.DictWriter(review_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with open(output_root / "summary.json", "w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, ensure_ascii=False, indent=2)

    print(f"processed={len(warp_paths)}")
    print(f"slots={len(rows)}")
    print(f"output={output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
