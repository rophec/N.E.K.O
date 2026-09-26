from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = (
    REPO_ROOT
    / "datasets"
    / "mahjong_yolo_active"
    / "dataset_registry"
    / "02_yolo_hbb_safe_aug_221"
    / "hbb_v1_safe_aug"
    / "data.yaml"
)
DEFAULT_FIXED_TEST = (
    REPO_ROOT
    / "datasets"
    / "mahjong_yolo_active"
    / "dataset_registry"
    / "05_yolo_hbb_fixed_test_21"
    / "data.yaml"
)
DEFAULT_OUTPUT = REPO_ROOT / "runs" / "mahjong_yolo26_hbb" / "single_stage_improve" / "dataset_audit"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EXPECTED_NAMES = [
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
    parser = argparse.ArgumentParser(description="Audit YOLO HBB dataset integrity for single-stage Mahjong training.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--fixed-test-data", type=Path, default=DEFAULT_FIXED_TEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def dataset_root(data_yaml: Path, data: dict[str, Any]) -> Path:
    root = Path(data.get("path") or data_yaml.parent)
    if not root.is_absolute():
        root = data_yaml.parent / root
    return root.resolve()


def split_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def names_from_yaml(data: dict[str, Any]) -> list[str]:
    raw_names = data.get("names")
    if isinstance(raw_names, dict):
        return [str(raw_names[idx]) for idx in sorted(raw_names)]
    if isinstance(raw_names, list):
        return [str(item) for item in raw_names]
    raise ValueError("data.yaml missing names")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def images_in(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def labels_in(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() == ".txt")


def audit_split(root: Path, split_name: str, image_rel: str, label_rel: str, class_count: int) -> tuple[dict[str, Any], list[dict[str, Any]], Counter[int]]:
    image_dir = split_path(root, image_rel)
    label_dir = split_path(root, label_rel)
    images = images_in(image_dir)
    labels = labels_in(label_dir)
    label_by_stem = {label.stem: label for label in labels}
    image_by_stem = {image.stem: image for image in images}
    issues: list[dict[str, Any]] = []
    class_counts: Counter[int] = Counter()
    label_counts: Counter[int] = Counter()

    for stem, image in image_by_stem.items():
        label = label_by_stem.get(stem)
        if label is None:
            issues.append({"split": split_name, "image": image.name, "issue": "missing_label"})
            continue
        lines = label.read_text(encoding="utf-8").splitlines()
        label_counts[len(lines)] += 1
        for line_no, line in enumerate(lines, start=1):
            parts = line.strip().split()
            if len(parts) != 5:
                issues.append({"split": split_name, "label": label.name, "line": line_no, "issue": "bad_column_count", "value": line})
                continue
            try:
                cls = int(float(parts[0]))
                x, y, w, h = [float(part) for part in parts[1:]]
            except ValueError:
                issues.append({"split": split_name, "label": label.name, "line": line_no, "issue": "non_numeric", "value": line})
                continue
            if cls < 0 or cls >= class_count:
                issues.append({"split": split_name, "label": label.name, "line": line_no, "issue": "class_out_of_range", "value": cls})
            else:
                class_counts[cls] += 1
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                issues.append({"split": split_name, "label": label.name, "line": line_no, "issue": "box_out_of_range", "value": line})
            if w <= 0.003 or h <= 0.003:
                issues.append({"split": split_name, "label": label.name, "line": line_no, "issue": "box_too_small", "value": line})

    for stem, label in label_by_stem.items():
        if stem not in image_by_stem:
            issues.append({"split": split_name, "label": label.name, "issue": "missing_image"})

    summary = {
        "split": split_name,
        "image_dir": str(image_dir),
        "label_dir": str(label_dir),
        "images": len(images),
        "labels": len(labels),
        "issues": len(issues),
        "empty_label_files": int(label_counts[0]),
        "min_labels_per_image": min(label_counts) if label_counts else 0,
        "max_labels_per_image": max(label_counts) if label_counts else 0,
    }
    return summary, issues, class_counts


def collect_hashes(data_yaml: Path) -> dict[str, list[dict[str, str]]]:
    data = load_yaml(data_yaml)
    root = dataset_root(data_yaml, data)
    out: dict[str, list[dict[str, str]]] = {}
    for split in ("train", "val", "test"):
        if split not in data:
            continue
        image_dir = split_path(root, data[split])
        out[split] = [{"image": str(path), "sha256": sha256_file(path)} for path in images_in(image_dir)]
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    data_yaml = args.data.resolve()
    data = load_yaml(data_yaml)
    root = dataset_root(data_yaml, data)
    names = names_from_yaml(data)
    output = args.output_root.resolve() / time.strftime("%Y%m%d-%H%M%S")
    output.mkdir(parents=True, exist_ok=True)

    global_issues: list[dict[str, Any]] = []
    if names != EXPECTED_NAMES:
        global_issues.append({"issue": "class_names_do_not_match_expected_34", "actual": names})
    if int(data.get("nc", -1)) != 34:
        global_issues.append({"issue": "nc_is_not_34", "actual": data.get("nc")})

    split_summaries: list[dict[str, Any]] = []
    all_issues: list[dict[str, Any]] = []
    class_counts_by_split: dict[str, Counter[int]] = {}
    for split in ("train", "val", "test"):
        if split not in data:
            continue
        label_rel = str(data[split]).replace("images", "labels", 1)
        summary, issues, counts = audit_split(root, split, data[split], label_rel, len(names))
        split_summaries.append(summary)
        all_issues.extend(issues)
        class_counts_by_split[split] = counts

    leakage: list[dict[str, str]] = []
    if args.fixed_test_data.exists():
        main_hashes = collect_hashes(data_yaml)
        test_hashes = collect_hashes(args.fixed_test_data.resolve())
        fixed = {row["sha256"]: row["image"] for rows in test_hashes.values() for row in rows}
        for split in ("train", "val"):
            for row in main_hashes.get(split, []):
                if row["sha256"] in fixed:
                    leakage.append({"split": split, "image": row["image"], "fixed_test_image": fixed[row["sha256"]]})

    class_rows: list[dict[str, Any]] = []
    for class_id, name in enumerate(names):
        row: dict[str, Any] = {"class_id": class_id, "class_name": name}
        for split, counts in class_counts_by_split.items():
            row[split] = counts[class_id]
        class_rows.append(row)

    payload = {
        "data_yaml": str(data_yaml),
        "dataset_root": str(root),
        "class_names_ok": names == EXPECTED_NAMES,
        "nc_ok": int(data.get("nc", -1)) == 34,
        "split_summaries": split_summaries,
        "global_issues": global_issues,
        "label_issues": all_issues,
        "fixed_test_leakage": leakage,
    }
    (output / "audit.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(output / "label_issues.csv", all_issues)
    write_csv(output / "class_counts.csv", class_rows)
    write_csv(output / "fixed_test_leakage.csv", leakage)

    lines = [
        "# Single-Stage Dataset Audit",
        "",
        f"- Data: `{data_yaml}`",
        f"- Root: `{root}`",
        f"- Class names ok: `{names == EXPECTED_NAMES}`",
        f"- nc ok: `{int(data.get('nc', -1)) == 34}`",
        f"- Label issues: {len(all_issues)}",
        f"- Fixed-test leakage: {len(leakage)}",
        "",
        "## Splits",
        "",
        "| split | images | labels | issues | min labels | max labels |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for summary in split_summaries:
        lines.append(
            "| {split} | {images} | {labels} | {issues} | {min_labels_per_image} | {max_labels_per_image} |".format(**summary)
        )
    lines.extend(["", "## Files", "", "- `audit.json`", "- `label_issues.csv`", "- `class_counts.csv`", "- `fixed_test_leakage.csv`"])
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
