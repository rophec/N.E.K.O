"""Command-line live backend for Mahjong Coach timing diagnostics.

This script runs the same capture + perception + coach pipeline as the plugin
live mode, then prints per-frame timing so latency regressions are visible
without opening the N.E.K.O web UI.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


REPO_ROOT = _repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from plugin.plugins.mahjong_coach.capture import DefaultCaptureProvider, prune_frames  # noqa: E402
from plugin.plugins.mahjong_coach.coach import RoundCoachEngine  # noqa: E402
from plugin.plugins.mahjong_coach.models import MahjongCoachConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Mahjong Coach live timing backend from the command line.")
    parser.add_argument("--keywords", default="雀魂,Mahjong Soul", help="Comma-separated window title keywords.")
    parser.add_argument("--interval-ms", type=int, default=400, help="Normal observation interval.")
    parser.add_argument("--fast-interval-ms", type=int, default=300, help="Interval after action-required frames.")
    parser.add_argument("--keep-frames", type=int, default=1000, help="How many captured frames to keep.")
    parser.add_argument("--save-format", default="jpg", choices=["jpg", "jpeg", "png"], help="Screenshot format.")
    parser.add_argument("--play-style", default="riichi", choices=["riichi", "fast"], help="Coach play style.")
    parser.add_argument("--round-wind", default="1z", help="Round wind tile code, e.g. 1z for east.")
    parser.add_argument("--seat-wind", default="", help="Seat wind tile code, optional.")
    parser.add_argument("--dora", default="", help="Comma-separated dora tile codes.")
    parser.add_argument("--frames-dir", default="", help="Directory for live frames. Defaults to plugin app data.")
    parser.add_argument("--limit", type=int, default=0, help="Stop after N frames. 0 means run until Ctrl+C.")
    parser.add_argument("--once", action="store_true", help="Capture and analyze one frame, then exit.")
    parser.add_argument("--jsonl", action="store_true", help="Print machine-readable JSON lines.")
    parser.add_argument("--force-checkpoint", action="store_true", help="Force strategy checkpoint every analyzed frame.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    keywords = _split_csv(args.keywords) or ["雀魂", "Mahjong Soul"]
    frames_dir = Path(args.frames_dir) if args.frames_dir else _default_frames_dir()
    frames_dir.mkdir(parents=True, exist_ok=True)

    config = MahjongCoachConfig(
        play_style=args.play_style,
        live_interval_ms=max(200, int(args.interval_ms)),
        live_fast_interval_ms=max(100, int(args.fast_interval_ms)),
        live_keep_frames=max(5, int(args.keep_frames)),
        live_save_format=args.save_format,
        round_wind=args.round_wind,
        seat_wind=args.seat_wind,
        dora_tiles=_split_csv(args.dora),
    )
    calibration_dir = REPO_ROOT / "plugin" / "plugins" / "mahjong_coach" / "data" / "calibration" / "profiles"
    engine = RoundCoachEngine(config, calibration_dir=calibration_dir)
    provider = DefaultCaptureProvider()

    print(_startup_line(args, keywords, frames_dir, calibration_dir), flush=True)
    frame_index = 0
    try:
        while True:
            loop_started = time.perf_counter()
            locate_started = time.perf_counter()
            binding = provider.locate_window(keywords)
            locate_ms = _elapsed_ms(locate_started)

            if not binding.bound:
                payload = {
                    "ts": _timestamp(),
                    "frame": frame_index,
                    "status": "waiting_for_window",
                    "error": binding.error or "window_not_found",
                    "decision": "",
                    "source": "",
                    "locate_ms": locate_ms,
                    "capture_ms": None,
                    "analyze_ms": None,
                    "engine_total_ms": None,
                    "hand_ms": None,
                    "meld_ms": None,
                    "action_ms": None,
                    "river_ms": None,
                    "strategy_ms": None,
                    "loop_ms": _elapsed_ms(loop_started),
                    "hand_count": 0,
                    "open_meld_count": 0,
                    "buttons": [],
                }
                _print_payload(payload, jsonl=args.jsonl)
                _sleep_remaining(loop_started, config.live_interval_ms)
                if args.once:
                    return 1
                continue

            capture_started = time.perf_counter()
            packet = provider.capture_frame(
                samples_dir=frames_dir,
                binding_result=binding,
                save_format=config.live_save_format,
            )
            capture_ms = _elapsed_ms(capture_started)

            analyze_started = time.perf_counter()
            decision = engine.analyze_frame(packet.image_path, force_checkpoint=bool(args.force_checkpoint))
            analyze_ms = _elapsed_ms(analyze_started)
            decision_payload = decision.to_dict()
            frame_index += 1

            payload = _timing_payload(
                frame_index=frame_index,
                packet=packet.to_dict(),
                decision=decision_payload,
                locate_ms=locate_ms,
                capture_ms=capture_ms,
                analyze_ms=analyze_ms,
                loop_ms=_elapsed_ms(loop_started),
            )
            _print_payload(payload, jsonl=args.jsonl)
            prune_frames(frames_dir, keep=config.live_keep_frames)

            if args.once or (args.limit > 0 and frame_index >= args.limit):
                return 0
            next_interval = config.live_fast_interval_ms if decision.action_required else config.live_interval_ms
            _sleep_remaining(loop_started, next_interval)
    except KeyboardInterrupt:
        print("stopped by Ctrl+C", flush=True)
        return 130


def _startup_line(args: argparse.Namespace, keywords: list[str], frames_dir: Path, calibration_dir: Path) -> str:
    return (
        "Mahjong Coach live timing backend started "
        f"keywords={keywords} interval_ms={args.interval_ms} fast_interval_ms={args.fast_interval_ms} "
        f"frames_dir={frames_dir} calibration_dir={calibration_dir}"
    )


def _timing_payload(
    *,
    frame_index: int,
    packet: dict[str, Any],
    decision: dict[str, Any],
    locate_ms: float,
    capture_ms: float,
    analyze_ms: float,
    loop_ms: float,
) -> dict[str, Any]:
    perception = decision.get("perception") if isinstance(decision.get("perception"), dict) else {}
    engine_meta = decision.get("engine_meta") if isinstance(decision.get("engine_meta"), dict) else {}
    engine_timings = engine_meta.get("timings_ms") if isinstance(engine_meta.get("timings_ms"), dict) else {}
    hand = perception.get("hand") if isinstance(perception.get("hand"), dict) else {}
    meld = perception.get("meld") if isinstance(perception.get("meld"), dict) else {}
    action = perception.get("action") if isinstance(perception.get("action"), dict) else {}
    river = perception.get("river") if isinstance(perception.get("river"), dict) else {}
    return {
        "ts": _timestamp(),
        "frame": frame_index,
        "status": "observing",
        "window": packet.get("window_title") or "",
        "image_path": packet.get("image_path") or "",
        "decision": decision.get("decision_type") or "",
        "source": engine_meta.get("source") or "",
        "locate_ms": round(float(locate_ms), 1),
        "capture_ms": round(float(capture_ms), 1),
        "analyze_ms": round(float(analyze_ms), 1),
        "engine_total_ms": round(float(engine_meta.get("elapsed_ms") or engine_timings.get("total") or 0.0), 1),
        "hand_ms": _step_ms(hand),
        "meld_ms": _step_ms(meld),
        "action_ms": _step_ms(action),
        "river_ms": _step_ms(river),
        "strategy_ms": _optional_ms(engine_timings.get("strategy")),
        "loop_ms": round(float(loop_ms), 1),
        "hand_count": len(hand.get("hand_tiles") or decision.get("hand_tiles") or []),
        "open_meld_count": int(meld.get("open_meld_count") or 0),
        "buttons": list(decision.get("buttons") or []),
        "reasons": list(decision.get("reason_codes") or []),
    }


def _print_payload(payload: dict[str, Any], *, jsonl: bool) -> None:
    if jsonl:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)
        return
    print(
        "[{ts}] frame={frame} status={status} decision={decision} source={source} "
        "locate={locate_ms}ms capture={capture_ms}ms analyze={analyze_ms}ms "
        "engine={engine_total_ms}ms hand={hand_ms}ms meld={meld_ms}ms action={action_ms}ms "
        "river={river_ms}ms strategy={strategy_ms}ms loop={loop_ms}ms "
        "hand_count={hand_count} melds={open_meld_count} buttons={buttons}".format(**payload),
        flush=True,
    )


def _step_ms(payload: dict[str, Any]) -> float | None:
    return _optional_ms(payload.get("elapsed_ms"))


def _optional_ms(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), 1)
    except (TypeError, ValueError):
        return None


def _default_frames_dir() -> Path:
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / "N.E.K.O" / "plugins" / "mahjong_coach" / "data" / "live_frames"
    return REPO_ROOT / "tmp" / "mahjong_coach_live_frames"


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def _timestamp() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000.0, 1)


def _sleep_remaining(started: float, interval_ms: int) -> None:
    remaining = max(0.0, (float(interval_ms) / 1000.0) - (time.perf_counter() - started))
    if remaining > 0:
        time.sleep(remaining)


if __name__ == "__main__":
    raise SystemExit(main())
