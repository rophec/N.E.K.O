from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from plugin.plugins.mahjong_coach.overlay import (
    _CONFIG_STYLE_HINT,
    _SWITCH_STYLE_LABEL,
    _overlay_panel_text,
    _win32_topbar_action,
    CoachOverlayController,
    _load_prefs,
    _save_prefs,
    overlay_strategy_card_from_payload,
    overlay_strategy_card_text_from_payload,
    overlay_text_from_payload,
)


def test_overlay_preferences_use_injected_plugin_data_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unrelated_local_app_data = tmp_path / "unrelated-local-app-data"
    monkeypatch.setenv("LOCALAPPDATA", str(unrelated_local_app_data))
    prefs_path = tmp_path / "custom-plugin-root" / "mahjong_coach" / "overlay_prefs.json"
    controller = CoachOverlayController(prefs_path=prefs_path)

    _save_prefs(controller.prefs_path, 520, 180, 22, "beginner")
    stored = json.loads(prefs_path.read_text(encoding="utf-8"))

    assert controller.prefs_path == prefs_path.resolve()
    assert stored["display_mode"] == "beginner"
    assert _load_prefs(controller.prefs_path)["font_size"] == 22
    assert not unrelated_local_app_data.exists()


def test_overlay_preferences_persist_compact_or_full_strategy_panel(tmp_path: Path) -> None:
    prefs_path = tmp_path / "overlay.json"

    _save_prefs(prefs_path, 620, 320, 18, "compact", "full", "standard")

    assert _load_prefs(prefs_path)["panel_mode"] == "full"
    assert _load_prefs(prefs_path)["strategy_preset"] == "standard"
    assert _overlay_panel_text("compact", "简洁主建议", "完整策略卡片") == "简洁主建议"
    assert _overlay_panel_text("full", "简洁主建议", "完整策略卡片") == "完整策略卡片"


def test_full_strategy_card_contains_same_top_three_candidates() -> None:
    payload = {
        "last_decision": {
            "decision_type": "defense_alert",
            "action_required": True,
            "suggestion": "兜牌：先打1万，保留和牌路线。",
            "perception": {
                "strategy": {
                    "posture": "mawashi",
                    "risk_budget": 62,
                    "risk_scale_note": "0–100规则型相对危险指数，不是放铳概率；数值越高越危险",
                    "risk_scale_legend": "0–9极低｜10–39较低｜40–59中等｜60–79高｜80–100很高",
                    "risk_model_legend": "基准：现物0、全见5、筋30、壁42、字牌已见3/2/0–1枚=38/55/72、无筋幺九58、无筋2/8为72、无筋中张84",
                    "risk_budget_calculation": "听牌基础62 均衡风格+0 = 上限62",
                    "win_potential": "strong",
                    "top_candidates": [
                        {"tile": "1m", "safety": "无筋幺九", "defense_risk": 58, "safety_eligible": True, "shanten": 0, "shape_loss": 0, "effective_count": 8},
                        {
                            "tile": "1z",
                            "safety": "字牌已见2枚",
                            "defense_risk": 55,
                            "safety_eligible": True,
                            "shanten": 0,
                            "shape_loss": 0,
                            "effective_count": 3,
                            "visibility": {
                                "tile": "1z",
                                "tile_name": "东",
                                "known_count": 2,
                                "unseen_count": 2,
                                "summary": "东已知2枚：手牌东×1、上家牌河东×1；尚有东×2未见",
                            },
                            "safety_evidence": [
                                "东已知2枚：手牌东×1、上家牌河东×1；尚有东×2未见；不是把中、发等其他字牌合并计算。"
                            ],
                        },
                        {"tile": "2s", "safety": "无筋2/8", "defense_risk": 72, "safety_eligible": False, "shanten": 0, "shape_loss": 0, "effective_count": 11},
                    ],
                }
            },
        },
        "round_state": {"defense_posture": "mawashi", "defense_risk_budget": 62},
    }

    card = overlay_strategy_card_text_from_payload(payload)
    structured = overlay_strategy_card_from_payload(payload)

    assert "兜牌 · 守中求和" in card
    assert "重点分析　1万" in card
    assert "方案A　候选牌1万" in card
    assert "方案B　候选牌东" in card
    assert "方案C　候选牌2索" in card
    assert "风险高 72/100（超上限10）" in card
    assert "不是放铳概率" in card
    assert "0–9极低" in card
    assert "现物0、全见5、筋30" in card
    assert "字牌已见3/2/0–1枚=38/55/72" in card
    assert "听牌基础62 均衡风格+0 = 上限62" in card
    assert structured["focus_tile"] == "1万"
    assert structured["candidates"][2]["tone"] == "danger"
    assert "相比东" in structured["candidates"][0]["comparison_reason"]
    assert structured["candidates"][2]["risk_delta_label"] == "超上限10"
    assert structured["candidates"][2]["risk_level"] == "高"
    assert "72/100" in structured["candidates"][2]["risk_calculation"]
    assert "手牌东×1" in structured["candidates"][1]["safety_reason"]
    assert "上家牌河东×1" in structured["candidates"][1]["safety_reason"]
    assert "不是把中、发等其他字牌合并计算" in structured["candidates"][1]["safety_reason"]
    assert "超预算风险" in structured["candidates"][2]["tradeoff"]
    assert "首选" not in card
    assert "备选" not in card
    assert "不推荐" not in card
    assert "本卡只解释权衡，不直接给出操作指令" in structured["decision_process"]


def test_full_strategy_card_explains_scores_rank_honba_and_deposits() -> None:
    payload = {
        "last_decision": {
            "decision_type": "defense_alert",
            "action_required": True,
            "perception": {
                "strategy": {
                    "posture": "mawashi",
                    "risk_budget": 56,
                    "risk_budget_model": {
                        "placement_adjustment": 8,
                        "table_reward_bonus": 1900,
                    },
                    "table_context": {
                        "scores": {
                            "self": 12000,
                            "left_opponent": 20000,
                            "top_opponent": 31000,
                            "right_opponent": 34000,
                        },
                        "ranks": {
                            "self": 4,
                            "left_opponent": 3,
                            "top_opponent": 2,
                            "right_opponent": 1,
                        },
                        "self_rank": 4,
                        "gap_above": 8000,
                        "honba_count": 3,
                        "riichi_stick_count": 1,
                        "win_reward_bonus": 1900,
                    },
                    "top_candidates": [
                        {"tile": "1m", "defense_risk": 40, "safety_eligible": True, "shanten": 1, "effective_count": 8},
                    ],
                },
            },
        },
        "round_state": {},
    }

    text = overlay_strategy_card_text_from_payload(payload)
    card = overlay_strategy_card_from_payload(payload)

    assert "自己 12,000（4位）" in card["match_context_summary"]
    assert "本场3｜供托1" in card["match_context_summary"]
    assert "距第3位8,000点" in card["match_context_explanation"]
    assert "风险上限+8" in card["match_context_explanation"]
    assert "合计1,900点计入和牌收益" in card["match_context_explanation"]
    assert "场况：自己 12,000（4位）" in text


def test_overlay_text_without_injected_path_has_no_filesystem_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    text = overlay_text_from_payload(
        {
            "decision_type": "waiting_for_game",
            "summary": "waiting",
            "round_state": {},
        }
    )

    assert isinstance(text, str)
    assert not (tmp_path / "overlay_prefs.json").exists()


def test_overlay_start_waits_for_real_readiness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = CoachOverlayController(prefs_path=tmp_path / "overlay.json")
    release = threading.Event()

    def fake_run() -> None:
        controller._ready_event.set()
        release.wait(timeout=1.0)

    monkeypatch.setattr(controller, "_run", fake_run)

    assert controller.start(timeout=0.5) is True
    assert controller._thread is not None and controller._thread.is_alive()

    release.set()
    controller._thread.join(timeout=1.0)


def test_overlay_start_reports_thread_initialization_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = CoachOverlayController(prefs_path=tmp_path / "overlay.json")

    def fake_run() -> None:
        controller.last_error = "tk failed"
        controller._ready_event.set()

    monkeypatch.setattr(controller, "_run", fake_run)

    assert controller.start(timeout=0.5) is False
    assert controller.last_error == "tk failed"


def test_overlay_start_uses_win32_fallback_when_frozen_host_omits_tk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = CoachOverlayController(prefs_path=tmp_path / "overlay.json")
    release = threading.Event()

    def missing_tk() -> None:
        raise SystemExit("Nuitka: Need to use '--enable-plugin=tk-inter' option during compilation")

    def fake_win32() -> None:
        controller.backend = "win32"
        controller.window_handle = 42
        controller.window_visible = True
        controller._ready_event.set()
        release.wait(timeout=1.0)

    monkeypatch.setattr(controller, "_run", missing_tk)
    monkeypatch.setattr(controller, "_run_win32", fake_win32)

    assert controller.start(timeout=0.5) is True
    assert controller.backend == "win32"
    assert controller.window_visible is True

    release.set()
    controller._thread.join(timeout=1.0)


def test_overlay_start_reports_win32_fallback_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = CoachOverlayController(prefs_path=tmp_path / "overlay.json")

    def missing_tk() -> None:
        raise SystemExit("Nuitka: Need to use '--enable-plugin=tk-inter' option during compilation")

    def broken_win32() -> None:
        raise TypeError("native startup failed")

    monkeypatch.setattr(controller, "_run", missing_tk)
    monkeypatch.setattr(controller, "_run_win32", broken_win32)

    assert controller.start(timeout=0.5) is False
    assert controller.last_error == "Win32 overlay TypeError: native startup failed"


def test_external_overlay_exposes_an_obvious_style_switch_action() -> None:
    width = 420

    assert _SWITCH_STYLE_LABEL == "打法/显示"
    assert _SWITCH_STYLE_LABEL in _CONFIG_STYLE_HINT
    assert _win32_topbar_action("strategy", width - 80, 20, width) == "switch_style"
    assert _win32_topbar_action("strategy", width - 165, 20, width) == "detail"
    assert _win32_topbar_action("strategy", width - 10, 20, width) == "close"
    assert _win32_topbar_action("config", width - 80, 20, width) == ""
