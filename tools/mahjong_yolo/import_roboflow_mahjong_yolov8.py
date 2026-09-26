from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = (
    REPO_ROOT
    / "datasets"
    / "mahjong_yolo_active"
    / "dataset_registry"
    / "06_external_roboflow_mahjong_yolov8_v6"
    / "raw_download"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "datasets"
    / "mahjong_yolo_active"
    / "dataset_registry"
    / "06_external_roboflow_mahjong_yolov8_v6"
    / "yolo_hbb_mapped_34"
)
LOCAL_NAMES = [
    "1m",
    "2m",
    "3m",
    "4m",
    "5m",
    "6m",
    "7m",
    "8m",
    "9m",
    "1p",
    "2p",
    "3p",
    "4p",
    "5p",
    "6p",
    "7p",
    "8p",
    "9p",
    "1s",
    "2s",
    "3s",
    "4s",
    "5s",
    "6s",
    "7s",
    "8s",
    "9s",
    "1z",
    "2z",
    "3z",
    "4z",
    "5z",
    "6z",
    "7z",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import Roboflow Mahjong-YOLOv8 v6 into the local 34-class HBB format."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def normalize_names(raw: Any) -> dict[int, str]:
    if isinstance(raw, dict):
        return {int(key): str(value) for key, value in raw.items()}
    if isinstance(raw, list):
        return {index: str(value) for index, value in enumerate(raw)}
    raise ValueError(f"Unsupported names format: {type(raw).__name__}")


def find_data_yaml(source: Path) -> Path:
    candidates = [source / "data.yaml", source / "data.yml"]
    candidates.extend(sorted(source.rglob("data.yaml")))
    candidates.extend(sorted(source.rglob("data.yml")))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No data.yaml found under {source}")


def resolve_split_dir(root: Path, split_value: str | None, fallback: str) -> Path:
    if split_value:
        path = Path(split_value)
        if path.is_absolute():
            return path
        return root / path
    return root / "images" / fallback


def map_label(source_label: str) -> str | None:
    label = source_label.strip()
    if label in LOCAL_NAMES:
        return label
    # EN: Red fives are valid Mahjong fives, but our current local class set has no red-five class.
    # ZH: 赤五在当前本地 34 类里没有独立类别，因此先归并到普通 5。
    red_five_map = {"0m": "5m", "0p": "5p", "0s": "5s"}
    if label in red_five_map:
        return red_five_map[label]
    # EN: Some Roboflow datasets include helper/background classes that are not Mahjong identities.
    # ZH: 部分 Roboflow 数据集会有辅助类/背景类，不属于具体麻将牌，导入时跳过。
    if label.lower() in {"0", "tile", "tiles", "unknown", "background"}:
        return None
    return None


def copy_split(
    source_root: Path,
    output_root: Path,
    split: str,
    image_dir: Path,
    label_dir: Path,
    source_names: dict[int, str],
    target_name_to_id: dict[str, int],
) -> dict[str, Any]:
    out_image_dir = output_root / "images" / split
    out_label_dir = output_root / "labels" / split
    out_image_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    stats: dict[str, Any] = {
        "split": split,
        "source_image_dir": str(image_dir),
        "source_label_dir": str(label_dir),
        "images": 0,
        "labels_written": 0,
        "labels_skipped": 0,
        "skipped_by_source_class": Counter(),
        "class_counts": Counter(),
        "missing_label_files": [],
    }

    image_paths = sorted(
        path
        for path in image_dir.rglob("*")
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    )
    for image_path in image_paths:
        rel_name = image_path.name
        label_path = label_dir / f"{image_path.stem}.txt"
        shutil.copy2(image_path, out_image_dir / rel_name)
        stats["images"] += 1

        out_lines: list[str] = []
        if not label_path.exists():
            stats["missing_label_files"].append(str(image_path.relative_to(source_root)))
        else:
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                source_class_id = int(float(parts[0]))
                source_label = source_names.get(source_class_id, str(source_class_id))
                mapped = map_label(source_label)
                if mapped is None:
                    stats["labels_skipped"] += 1
                    stats["skipped_by_source_class"][source_label] += 1
                    continue
                target_class_id = target_name_to_id[mapped]
                out_lines.append(" ".join([str(target_class_id), *parts[1:5]]))
                stats["labels_written"] += 1
                stats["class_counts"][mapped] += 1

        (out_label_dir / f"{image_path.stem}.txt").write_text(
            "\n".join(out_lines) + ("\n" if out_lines else ""),
            encoding="utf-8",
        )

    stats["class_counts"] = dict(sorted(stats["class_counts"].items()))
    stats["skipped_by_source_class"] = dict(sorted(stats["skipped_by_source_class"].items()))
    return stats


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if not source.exists():
        raise FileNotFoundError(f"Missing Roboflow source directory: {source}")
    if output.exists() and args.overwrite:
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    data_yaml = find_data_yaml(source)
    source_root = data_yaml.parent
    data = load_yaml(data_yaml)
    if data.get("path"):
        root = Path(data["path"])
        if not root.is_absolute():
            root = source_root / root
    else:
        root = source_root

    source_names = normalize_names(data["names"])
    target_name_to_id = {name: index for index, name in enumerate(LOCAL_NAMES)}

    splits = {
        "train": resolve_split_dir(root, data.get("train"), "train"),
        "val": resolve_split_dir(root, data.get("val") or data.get("valid"), "val"),
        "test": resolve_split_dir(root, data.get("test"), "test"),
    }
    summary: dict[str, Any] = {
        "source": str(source),
        "source_data_yaml": str(data_yaml),
        "output": str(output),
        "source_names": source_names,
        "local_names": {index: name for index, name in enumerate(LOCAL_NAMES)},
        "splits": {},
    }

    for split, image_dir in splits.items():
        if not image_dir.exists():
            summary["splits"][split] = {"missing": str(image_dir)}
            continue
        label_dir = Path(str(image_dir).replace(f"{Path.sep}images{Path.sep}", f"{Path.sep}labels{Path.sep}"))
        if not label_dir.exists():
            # EN: Roboflow sometimes uses train/images + train/labels layout.
            # ZH: Roboflow 有时使用 train/images + train/labels 的目录结构。
            label_dir = image_dir.parent.parent / "labels" / image_dir.name
        if not label_dir.exists():
            raise FileNotFoundError(f"Missing label directory for {split}: {label_dir}")
        summary["splits"][split] = copy_split(
            source_root=source_root,
            output_root=output,
            split=split,
            image_dir=image_dir,
            label_dir=label_dir,
            source_names=source_names,
            target_name_to_id=target_name_to_id,
        )

    output_data = {
        "path": str(output).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(LOCAL_NAMES),
        "names": {index: name for index, name in enumerate(LOCAL_NAMES)},
    }
    (output / "data.yaml").write_text(
        yaml.safe_dump(output_data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    (output / "import_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
