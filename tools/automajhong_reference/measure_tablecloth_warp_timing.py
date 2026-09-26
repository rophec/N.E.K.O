"""Measure per-step timing for tablecloth warp preprocessing."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

import run_tablecloth_warp_batch as warp


def now_ms() -> float:
    return time.perf_counter() * 1000.0


class Timer:
    def __init__(self) -> None:
        self.rows: list[tuple[str, float]] = []

    def measure(self, name: str, fn):
        start = now_ms()
        result = fn()
        self.rows.append((name, now_ms() - start))
        return result


def process_image_timed(path: Path, case_dir: Path, write_outputs: bool) -> dict[str, object]:
    timer = Timer()

    image = timer.measure("01_read_image", lambda: warp.read_image(path))
    if write_outputs:
        timer.measure("02_write_source", lambda: warp.write_png(case_dir / "01-source.png", image))

    mask_before, hsv_info = timer.measure("03_hsv_tablecloth_mask", lambda: warp.hsv_tablecloth_mask(image))
    if write_outputs:
        timer.measure("04_write_mask_before", lambda: warp.write_png(case_dir / "02-mask-before.png", mask_before))

    mask_after = timer.measure(
        "05_morphology_close",
        lambda: cv2.morphologyEx(
            mask_before,
            cv2.MORPH_CLOSE,
            np.ones((7, 7), np.uint8),
            iterations=3,
        ),
    )
    if write_outputs:
        timer.measure("06_write_mask_after", lambda: warp.write_png(case_dir / "03-mask-after.png", mask_after))

    component, component_info = timer.measure("07_largest_component", lambda: warp.largest_component(mask_after))
    if write_outputs:
        timer.measure("08_write_largest_component", lambda: warp.write_png(case_dir / "04-largest-component.png", component))

    contour = timer.measure("09_external_contour_image", lambda: warp.external_contour_image(component))
    if write_outputs:
        timer.measure("10_write_external_contour", lambda: warp.write_png(case_dir / "05-external-contour.png", contour))

    lines = timer.measure("11_select_support_lines", lambda: warp.select_support_lines(contour))

    def build_selected_lines():
        selected = np.zeros_like(image)
        selected[:] = (0, 0, 0)
        return warp.draw_lines(selected, lines)

    selected_lines = timer.measure("12_draw_selected_lines", build_selected_lines)
    if write_outputs:
        timer.measure("13_write_selected_lines", lambda: warp.write_png(case_dir / "06-selected-lines.png", selected_lines))

    lines_on_source = timer.measure("14_draw_lines_on_source", lambda: warp.draw_lines(image, lines))
    if write_outputs:
        timer.measure("15_write_lines_on_source", lambda: warp.write_png(case_dir / "07-lines-on-source.png", lines_on_source))

    extended_lines = timer.measure("16_draw_extended_lines", lambda: warp.draw_lines(image, lines, extended=True))
    if write_outputs:
        timer.measure("17_write_extended_lines", lambda: warp.write_png(case_dir / "08-extended-lines.png", extended_lines))

    warped, intersections = timer.measure("18_warp_table", lambda: warp.warp_table(image, lines))
    if write_outputs:
        timer.measure("19_write_warp", lambda: warp.write_png(case_dir / "09-warp-square-800.png", warped))

    sheet = timer.measure(
        "20_contact_sheet",
        lambda: warp.contact_sheet(
            [
                ("01 source", image),
                ("03 mask", mask_after),
                ("05 contour", contour),
                ("07 lines", lines_on_source),
                ("08 extended", extended_lines),
                ("09 warp", warped),
            ]
        ),
    )
    if write_outputs:
        timer.measure("21_write_contact_sheet", lambda: warp.write_png(case_dir / "contact-sheet.png", sheet))

    summary = {
        "source": str(path),
        "image_size": [int(image.shape[1]), int(image.shape[0])],
        **hsv_info,
        "largest_component": component_info,
        "selected_lines": {name: asdict(line) for name, line in lines.items()},
        "intersections": intersections,
        "warp_output": str(case_dir / "09-warp-square-800.png"),
    }
    if write_outputs:
        timer.measure(
            "22_write_summary",
            lambda: (case_dir / "summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            ),
        )

    timings = {name: round(ms, 3) for name, ms in timer.rows}
    timings["total_ms"] = round(sum(ms for _, ms in timer.rows), 3)
    return {
        "case": path.stem,
        "path": str(path),
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "timings_ms": timings,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-images", action="store_true")
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    images = warp.list_images(args.input_root)
    if args.limit > 0:
        images = images[: args.limit]

    results: list[dict[str, object]] = []
    for path in images:
        case_dir = args.output_root / path.stem
        case_dir.mkdir(parents=True, exist_ok=True)
        result = process_image_timed(path, case_dir, write_outputs=not args.no_images)
        results.append(result)
        print(f"{path.name}: {result['timings_ms']['total_ms']}ms")

    all_steps = sorted({step for result in results for step in result["timings_ms"].keys() if step != "total_ms"})
    csv_path = args.output_root / "timings.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["case", "width", "height", *all_steps, "total_ms"])
        for result in results:
            timings = result["timings_ms"]
            writer.writerow(
                [
                    result["case"],
                    result["width"],
                    result["height"],
                    *[timings.get(step, "") for step in all_steps],
                    timings["total_ms"],
                ]
            )

    aggregates: dict[str, dict[str, float]] = {}
    for step in [*all_steps, "total_ms"]:
        values = [float(result["timings_ms"][step]) for result in results if step in result["timings_ms"]]
        if values:
            aggregates[step] = {
                "avg_ms": round(sum(values) / len(values), 3),
                "min_ms": round(min(values), 3),
                "max_ms": round(max(values), 3),
            }
    (args.output_root / "timings_summary.json").write_text(
        json.dumps({"count": len(results), "aggregates": aggregates, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

