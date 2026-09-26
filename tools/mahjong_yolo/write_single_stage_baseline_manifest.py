from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = REPO_ROOT / "runs" / "mahjong_yolo26_hbb" / "single_stage_improve" / "baselines"


BASELINES: list[dict[str, Any]] = [
    {
        "name": "old_epochs_150",
        "status": "candidate_baseline",
        "model": "runs/mahjong_yolo26_hbb/epoch_compare_fixed_test/epoch_compare_20260702-172530/training/epochs_150/weights/best.pt",
        "summary": "runs/mahjong_yolo26_hbb/epoch_compare_fixed_test/epoch_compare_20260702-172530/summary.json",
        "error_report": "runs/mahjong_yolo26_hbb/error_analysis/20260703-100504/report.md",
        "error_catalog": "runs/mahjong_yolo26_hbb/error_catalog/20260703-101838/catalog.md",
        "notes": "Best mAP50-95 among the pre-targeted-augmentation runs.",
    },
    {
        "name": "old_epochs_200",
        "status": "primary_baseline",
        "model": "runs/mahjong_yolo26_hbb/epoch_compare_fixed_test/epoch_compare_20260702-172530/training/epochs_200/weights/best.pt",
        "summary": "runs/mahjong_yolo26_hbb/epoch_compare_fixed_test/epoch_compare_20260702-172530/summary.json",
        "error_report": "runs/mahjong_yolo26_hbb/error_analysis/20260703-100650/report.md",
        "error_catalog": "runs/mahjong_yolo26_hbb/error_catalog/20260703-101838/catalog.md",
        "notes": "Current primary single-stage baseline because fixed-test recall is highest among accepted runs.",
    },
    {
        "name": "targeted_hard_aug_epochs_150",
        "status": "experimental_rejected",
        "model": "runs/mahjong_yolo26_hbb/targeted_hard_aug_compare/targeted_hard_aug_20260703-111833/epoch_compare_20260703-111833/training/epochs_150/weights/best.pt",
        "summary": "runs/mahjong_yolo26_hbb/targeted_hard_aug_compare/targeted_hard_aug_20260703-111833/epoch_compare_20260703-111833/summary.json",
        "error_report": "runs/mahjong_yolo26_hbb/error_analysis/20260703-115454/report.md",
        "error_catalog": "runs/mahjong_yolo26_hbb/error_catalog/20260703-120018/catalog.md",
        "notes": "Kept for evidence only. It reduced missed boxes but worsened class confusion, especially 3m->2m.",
    },
]


def read_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def extract_metrics(name: str, summary_path: Path) -> dict[str, Any] | None:
    data = read_json(summary_path)
    if not isinstance(data, list):
        return None
    if name.endswith("150"):
        epoch = 150
    elif name.endswith("200"):
        epoch = 200
    else:
        epoch = None
    for row in data:
        if epoch is None or row.get("epochs") == epoch:
            return {
                "val_metrics": row.get("val_metrics"),
                "test_metrics": row.get("test_metrics"),
            }
    return None


def main() -> None:
    output = OUTPUT_ROOT / time.strftime("%Y%m%d-%H%M%S")
    output.mkdir(parents=True, exist_ok=True)

    baselines: list[dict[str, Any]] = []
    for item in BASELINES:
        row = dict(item)
        for key in ("model", "summary", "error_report", "error_catalog"):
            path = REPO_ROOT / row[key]
            row[key] = str(path)
            row[f"{key}_exists"] = path.exists()
        row["metrics"] = extract_metrics(item["name"], Path(row["summary"]))
        baselines.append(row)

    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "policy": [
            "These paths are read-only baselines.",
            "Do not overwrite, delete, or promote rejected experiments.",
            "New single-stage experiments must write under runs/mahjong_yolo26_hbb/single_stage_improve.",
        ],
        "primary_baseline": "old_epochs_200",
        "baselines": baselines,
    }
    (output / "baseline_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Single-Stage YOLO Baselines",
        "",
        "- Primary baseline: `old_epochs_200`",
        "- Existing models and results are read-only evidence.",
        "- `targeted_hard_aug_epochs_150` is kept but rejected as a main model.",
        "",
        "| name | status | model exists | test precision | test recall | test mAP50 | test mAP50-95 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in baselines:
        test = (row.get("metrics") or {}).get("test_metrics") or {}
        lines.append(
            "| {name} | {status} | {exists} | {p:.4f} | {r:.4f} | {m50:.4f} | {m5095:.4f} |".format(
                name=row["name"],
                status=row["status"],
                exists="yes" if row["model_exists"] else "no",
                p=float(test.get("box_precision", 0.0)),
                r=float(test.get("box_recall", 0.0)),
                m50=float(test.get("box_map50", 0.0)),
                m5095=float(test.get("box_map50_95", 0.0)),
            )
        )
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
