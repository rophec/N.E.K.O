from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from plugin.plugins.mahjong_coach import capture
from plugin.plugins.mahjong_coach.__init__ import MahjongCoachPlugin, _live_performance_summary
from plugin.plugins.mahjong_coach.capture import CaptureSession, DefaultCaptureProvider
from plugin.plugins.mahjong_coach.companion_transport import DirectCompanionSubmitResult
from plugin.plugins.mahjong_coach.coach import (
    RoundCoachEngine,
    _discard_ranking_text,
    build_round_plan,
    rank_discard_decisions,
)
from plugin.plugins.mahjong_coach.models import (
    CapturePreferences,
    FramePacket,
    LiveSessionState,
    MahjongCoachConfig,
    PlayerProfile,
    WindowTargetDescriptor,
)
from plugin.plugins.mahjong_coach.overlay import (
    overlay_strategy_card_from_payload,
    overlay_strategy_card_text_from_payload,
    overlay_text_from_payload,
)
from plugin.plugins.mahjong_coach.perception.river_state import RiverStateResult
from plugin.plugins.mahjong_coach.player_profile import (
    AmaeKoromoProvider,
    ProfileLookupError,
)
from plugin.plugins.mahjong_coach.preferences import PreferencesStore
from plugin.plugins.mahjong_coach.presentation import (
    build_absurd_banter_reply,
    build_neko_companion_cue,
    build_public_payload,
)
from plugin.plugins.mahjong_coach.window_binding import WindowBindingResult
from plugin.plugins.mahjong_coach.yakuman import (
    YakumanEstimateService,
    assess_yakuman_routes,
    is_route_complete,
    monte_carlo_yakuman,
    route_distance,
)


mahjong_plugin_module = importlib.import_module("plugin.plugins.mahjong_coach.__init__")
yakuman_module = importlib.import_module("plugin.plugins.mahjong_coach.yakuman")


def test_legacy_fast_style_migrates_to_versioned_aggressive_profile() -> None:
    profile = PlayerProfile.from_payload({}, legacy_play_style="fast")

    assert profile.version == 1
    assert profile.risk_tolerance == "aggressive"
    assert profile.goal_bias == "speed"
    assert profile.call_bias == "open"

    config = MahjongCoachConfig.from_payload({"decision": {"play_style": "fast"}})
    assert config.play_style == "fast"
    assert config.strategy_preset == "simple"
    assert config.player_profile.risk_tolerance == "aggressive"
    assert config.player_profile.goal_bias == "speed"

    standard = MahjongCoachConfig.from_payload({"decision": {"strategy_preset": "standard"}})
    assert standard.strategy_preset == "standard"


def test_guidance_and_inference_defaults_migrate_without_actionable_output() -> None:
    assert MahjongCoachConfig.from_payload({}).live_advice_mode == "companion"
    assert MahjongCoachConfig.from_payload({}).neko_companion_enabled is True
    assert MahjongCoachConfig.from_payload({}).absurd_banter_enabled is False
    assert MahjongCoachConfig.from_payload(
        {"decision": {"neko_companion_enabled": False}}
    ).neko_companion_enabled is False
    assert MahjongCoachConfig.from_payload(
        {"decision": {"absurd_banter_enabled": True}}
    ).absurd_banter_enabled is True
    assert MahjongCoachConfig.from_payload({}).inference_provider == "speed"
    assert MahjongCoachConfig.from_payload(
        {"decision": {"live_advice_mode": "coach"}}
    ).live_advice_mode == "strategy"
    assert MahjongCoachConfig.from_payload(
        {"perception": {"inference_provider": "directml"}}
    ).inference_provider == "speed"


def test_public_modes_keep_companion_observational_and_make_strategy_actionable() -> None:
    raw = {
        "last_decision": {
            "decision_type": "defense_alert",
            "action_required": True,
            "summary": "打4万",
            "detail": "先打4万再兜牌",
            "suggestion": "打4万",
            "buttons": ["立直"],
            "perception": {
                "hand": {"hand_tiles": ["7z", "6z", "1p", "2p", "3p"]},
                "strategy": {
                    "posture": "mawashi",
                    "risk_budget": 62,
                    "win_potential": "live",
                    "top_candidates": [
                        {
                            "tile": "4m",
                            "defense_risk": 84,
                            "safety_evidence": ["4m是无筋中张"],
                        },
                        {
                            "tile": "6p",
                            "defense_risk": 0,
                            "shanten": 0,
                            "effective_count": 3,
                            "safety_evidence": ["6p是下家现物"],
                        },
                        {
                            "tile": "2s",
                            "defense_risk": 55,
                            "shanten": 1,
                            "effective_count": 7,
                            "shape_loss": 1,
                            "safety_evidence": ["2s按无筋2/8计算"],
                        },
                    ],
                },
            },
        },
        "round_state": {
            "riichi_players": ["right_opponent"],
            "last_hand_tiles": ["7z", "6z", "1p", "2p", "3p"],
            "last_hand_confidence": 0.92,
            "last_river_confidence": 0.88,
            "current_plan": "切4万",
            "local_direction": "兜牌",
        },
    }

    companion = build_public_payload(raw, mode="companion")
    strategy = build_public_payload(raw, mode="strategy")

    companion_decision = companion["last_decision"]
    companion_state = companion["round_state"]
    assert companion_decision["action_required"] is False
    assert companion_decision["buttons"] == []
    assert companion_decision["suggestion"] == ""
    assert companion_decision["perception"]["strategy"]["top_candidates"] == []
    assert companion_state["current_plan"] == ""
    assert companion_state["local_direction"] == ""
    assert "4m" not in str(companion_decision)
    assert "6p" not in str(companion_decision)
    assert "2s" not in str(companion_decision)

    strategy_decision = strategy["last_decision"]
    strategy_state = strategy["round_state"]
    assert strategy_decision["action_required"] is False
    assert strategy_decision["buttons"] == []
    assert strategy_decision["suggestion"] == "主建议：打4万"
    assert strategy_state["current_plan"] == "主建议：打4万"
    assert strategy_state["local_plan"] == "主建议：打4万"
    assert strategy_state["local_direction"] == strategy["presentation"]["strategy_direction"]
    assert [item["tile"] for item in strategy_decision["perception"]["strategy"]["top_candidates"]] == [
        "4m",
        "6p",
        "2s",
    ]

    assert companion["presentation"]["mode_label"] == "吐槽伙伴"
    assert companion["presentation"]["strategy_direction"] == ""
    assert "完整手牌" not in companion["presentation"]["detail"]
    assert strategy["presentation"]["risk_range"] == {"min": 0.0, "max": 84.0}
    assert companion["presentation"]["risk_options"] == []
    assert [item["id"] for item in strategy["presentation"]["risk_options"]] == ["A", "B", "C"]
    assert [item["risk_basis"] for item in strategy["presentation"]["risk_options"]] == [
        "无筋中张",
        "现物",
        "无筋2/8",
    ]
    assert [item["tile_label"] for item in strategy["presentation"]["risk_options"]] == ["4万", "6筒", "2索"]
    assert strategy["presentation"]["primary_discard"] == "4m"
    assert strategy["presentation"]["primary_discard_label"] == "4万"
    assert strategy["presentation"]["primary_action"] == "主建议：打4万"
    assert strategy["presentation"]["non_prescriptive"] is False
    assert len(strategy["last_decision"]["perception"]["strategy"]["risk_options"]) == 3
    assert "不是放铳概率" in strategy["presentation"]["risk_scale_note"]
    assert any("无筋中张" in item for item in strategy["presentation"]["risk_sources"])
    assert strategy["presentation"]["posture_label"] == "保形评估"
    assert any("完整手牌" in item and "发" in item and "中" in item for item in strategy["presentation"]["evidence"])
    assert any("界面按钮" in item and "立直" in item for item in strategy["presentation"]["evidence"])
    public_card = overlay_strategy_card_from_payload(strategy)
    assert public_card["kind"] == "public_observation"
    assert [section["title"] for section in public_card["sections"]] == [
        "主建议",
        "当前策略方向",
        "看见了什么",
        "风险与原因",
        "牌型与场况",
    ]
    assert public_card["primary_action"] == "主建议：打4万"
    assert public_card["non_prescriptive"] is False
    assert len(public_card["risk_options"]) == 3
    full_text = overlay_strategy_card_text_from_payload(strategy)
    assert "【主建议】" in full_text
    assert "主建议：打4万" in full_text
    assert "【当前策略方向】" in full_text
    assert "【A/B/C 候选打法对照】" in full_text
    assert "方案A｜打4万｜相对风险很高 84/100｜高于参考线22" in full_text
    assert "方案B｜打6筒｜相对风险极低 0/100｜低于参考线62" in full_text
    assert "方案C｜打2索｜相对风险中等 55/100｜低于参考线7" in full_text
    assert "策略模式给出当前主建议与候选对照" in full_text


def test_companion_overlay_hides_persisted_shape_and_hand_analysis() -> None:
    public = build_public_payload(
        {
            "last_decision": {
                "decision_type": "coach_checkpoint",
                "perception": {"hand": {"hand_tiles": ["1m", "1m", "3m", "3m"]}},
            },
            "round_state": {
                "strategy_preset": "simple",
                "closest_shape_route": "七对子",
                "current_shanten": 1,
                "current_effective_count": 9,
            },
        },
        mode="companion",
    )

    assert public["presentation"]["shape_summary"] == ""
    compact = overlay_text_from_payload(public)
    assert "最近牌型" not in compact
    assert "向听" not in compact
    assert "一万、一万、三万、三万" not in compact


def test_neko_companion_cue_keeps_facts_without_internal_tile_candidates() -> None:
    raw = {
        "last_decision": {
            "decision_type": "coach_checkpoint",
            "summary": "打4万",
            "perception": {
                "hand": {"hand_tiles": ["7z", "6z", "1p", "2p", "3p"]},
                "strategy": {
                    "top_candidates": [
                        {
                            "tile": "4m",
                            "defense_risk": 84,
                            "safety_evidence": ["4m是无筋中张"],
                        }
                    ]
                },
            },
        },
        "round_state": {
            "round_id": "east-1",
            "riichi_players": ["right_opponent"],
            "last_hand_tiles": ["7z", "6z", "1p", "2p", "3p"],
            "last_observed_discard": "3s",
            "companion_reaction_kind": "disagree",
            "companion_reaction": "哦不对不对，这一下跟我刚才想得不太一样。",
        },
    }
    public = build_public_payload(raw, mode="companion")

    regular_cue, _regular_signature, _regular_event_kind = build_neko_companion_cue(public)
    cue, signature, event_kind = build_neko_companion_cue(
        public,
        include_rare_joke=True,
    )

    assert event_kind == "player_interaction"
    assert signature
    assert "不要评价这次操作对不对" in regular_cue
    assert "跟我刚才想得不太一样" not in regular_cue
    assert "铜须是生物吗？" not in regular_cue
    assert "不要求真的经历三阶段推理" in regular_cue
    assert "字牌共2张，具体为发、中" not in cue
    assert "不要念出完整手牌" in cue
    assert "不得告诉玩家应该打哪张牌" in cue
    assert "或问“发是字牌吗？”" in cue
    assert "铜须是生物吗？" in cue
    assert "跑题式玩笑，不代表真的识别失败" in cue
    assert "不得同时使用其他口头禅" in cue
    assert "哦对的对的" not in cue
    assert "哎呀不对不对" not in cue
    assert "4m" not in cue
    assert "打4万" not in cue
    assert "猫猫" not in cue


def test_neko_companion_does_not_speak_internal_insufficiency_fallbacks() -> None:
    public = build_public_payload(
        {
            "last_decision": {"decision_type": "opening_plan"},
            "round_state": {"round_id": "east-1"},
        },
        mode="companion",
    )

    # Companion output omits internal risk and confidence diagnostics entirely.
    assert public["presentation"]["risk_sources"] == []
    assert public["presentation"]["uncertainty"] == []

    cue, _signature, event_kind = build_neko_companion_cue(public)

    assert event_kind == "opening_observation"
    for text in (public["presentation"]["detail"], cue):
        assert "牌河证据不足" not in text
        assert "尚无稳定置信度" not in text
        assert "信息不足" not in text
        assert "不敢动" not in text
    assert "不要评价识别能力、数据完整度" in cue


def test_no_riichi_strategy_keeps_normal_efficiency_basis() -> None:
    raw = {
        "last_decision": {
            "decision_type": "risk_observation",
            "perception": {
                "strategy": {
                    "posture": "push",
                    "risk_budget": 58,
                    "top_candidates": [
                        {
                            "tile": "1z",
                            "defense_risk": 0,
                            "shanten": 1,
                            "effective_count": 8,
                            "safety": "无立直压力",
                            "risk_components": {"base_reason": "无立直压力"},
                            "risk_by_player": {},
                        },
                        {
                            "tile": "9m",
                            "defense_risk": 0,
                            "shanten": 1,
                            "effective_count": 7,
                            "safety": "无立直压力",
                            "risk_components": {"base_reason": "无立直压力"},
                            "risk_by_player": {},
                        },
                    ],
                }
            },
        },
        "round_state": {"riichi_players": []},
    }

    public = build_public_payload(raw, mode="strategy")
    presentation = public["presentation"]

    assert [item["tile"] for item in presentation["risk_options"]] == ["1z", "9m"]
    assert all(
        item["risk_basis"] == "当前无人立直，按牌效与役种路线比较"
        for item in presentation["risk_options"]
    )
    assert any("防守风险不覆盖正常牌效" in item for item in presentation["risk_sources"])
    assert "牌河证据不足" not in str(presentation)


def test_neko_companion_keeps_shape_distance_and_zero_melds_out_of_speech() -> None:
    public = build_public_payload(
        {
            "last_decision": {
                "decision_type": "opening_plan",
                "perception": {
                    "yakuman": {
                        "routes": [
                            {"route": "daisangen", "label": "大三元", "distance": 8}
                        ]
                    }
                },
            },
            "round_state": {
                "round_id": "east-1",
                "last_open_meld_count": 0,
                "closest_shape_route": "普通面子手",
                "current_shanten": 3,
                "current_effective_count": 7,
            },
        },
        mode="companion",
    )

    # Retired strategy facts are absent from both the panel and spoken context.
    assert public["presentation"]["evidence"] == ["新一局已经开始，继续观察公开事件。"]

    cue, _signature, event_kind = build_neko_companion_cue(public)

    assert event_kind == "opening_observation"
    assert "大三元" not in cue
    assert "距离约8张" not in cue
    assert "自己0组副露" not in cue
    assert "不要播报向听数、有效进张数" in cue
    assert "不要主动强调“没有副露”" in cue


def test_absurd_banter_mode_rotates_daily_cards_and_keeps_win_reaction_normal() -> None:
    opening = build_public_payload(
        {
            "last_decision": {"decision_type": "opening_plan"},
            "round_state": {"round_id": "east-1"},
        },
        mode="companion",
    )

    first, _first_signature, first_kind = build_neko_companion_cue(
        opening,
        absurd_banter_enabled=True,
        absurd_variant=0,
    )
    second, _second_signature, second_kind = build_neko_companion_cue(
        opening,
        absurd_banter_enabled=True,
        absurd_variant=1,
    )

    assert first_kind == second_kind == "opening_observation"
    assert "本次启用逆天模式" in first
    assert "先别急，我先急一下" in first
    assert "合理吗" not in first
    assert "本次启用逆天模式" in second
    assert "这合理吗？这不合理吧" in second
    assert "先别急，我先急一下" not in second
    assert "铜须是生物吗" not in first + second
    assert "一条回复只允许一个梗" in first

    coffee, _coffee_signature, coffee_kind = build_neko_companion_cue(
        opening,
        absurd_banter_enabled=True,
        absurd_variant=6,
    )
    assert coffee_kind == "opening_observation"
    assert "卡布奇诺" in coffee
    assert "不要顺带评价手牌或牌运" in coffee
    assert "不要评价自己的手牌好坏、牌型强弱" in coffee
    assert "本次输出只能围绕上面指定的话术卡与玩家互动" in coffee
    assert "即使上下文里存在这些信息也必须忽略" in coffee

    win = build_public_payload(
        {
            "last_decision": {"decision_type": "win_window", "buttons": ["tsumo"]},
            "round_state": {"round_id": "east-1"},
        },
        mode="companion",
    )
    critical, _signature, critical_kind = build_neko_companion_cue(
        win,
        absurd_banter_enabled=True,
        absurd_variant=0,
    )
    assert critical_kind == "win_opportunity"
    assert "本次启用逆天模式" not in critical
    assert "先别急，我先急一下" not in critical


def test_absurd_banter_fixed_replies_never_comment_hand_luck() -> None:
    public = build_public_payload(
        {
            "last_decision": {
                "decision_type": "coach_checkpoint",
                "perception": {"strategy": {"win_potential": "strong"}},
            },
            "round_state": {
                "round_id": "east-1",
                "last_hand_tiles": ["1m", "1m", "2m", "3m"],
                "last_observed_discard": "9p",
            },
        },
        mode="companion",
    )

    for variant in range(24):
        reply = build_absurd_banter_reply(
            public,
            event_kind="player_interaction",
            absurd_variant=variant,
        )
        assert reply
        assert all(
            marker not in reply
            for marker in ("牌运", "手气", "好牌", "烂牌", "牌型", "和牌前景")
        )

    riichi = build_public_payload(
        {
            "last_decision": {"decision_type": "defense_alert"},
            "round_state": {"round_id": "east-1", "riichi_players": ["left_opponent"]},
        },
        mode="companion",
    )
    assert "上家立直了" in build_absurd_banter_reply(
        riichi,
        event_kind="riichi_pressure",
        absurd_variant=0,
    )


def test_companion_ignores_hand_quality_for_daily_banter() -> None:
    weak = build_public_payload(
        {
            "last_decision": {
                "decision_type": "opening_plan",
                "perception": {"strategy": {"win_potential": "weak"}},
            },
            "round_state": {"round_id": "east-1"},
        },
        mode="companion",
    )
    weak_cue, _, _ = build_neko_companion_cue(
        weak,
        absurd_banter_enabled=True,
        absurd_variant=6,
    )
    assert "当前和牌路线偏弱" not in weak_cue
    assert "手气好" not in weak_cue
    assert "直接和玩家互动或说一个短梗" in weak_cue

    strong = build_public_payload(
        {
            "last_decision": {
                "decision_type": "opening_plan",
                "perception": {"strategy": {"win_potential": "strong"}},
            },
            "round_state": {"round_id": "east-1"},
        },
        mode="companion",
    )
    strong_cue, _, _ = build_neko_companion_cue(
        strong,
        absurd_banter_enabled=True,
        absurd_variant=6,
    )
    assert "当前和牌可能性较强" not in strong_cue
    assert "手气挺好" not in strong_cue
    assert "不要评价自己的手牌好坏、牌型强弱" in strong_cue


def test_absurd_banter_reacts_to_riichi_before_using_event_joke() -> None:
    riichi = build_public_payload(
        {
            "last_decision": {"decision_type": "defense_alert"},
            "round_state": {
                "round_id": "east-1",
                "riichi_players": ["top_opponent"],
            },
        },
        mode="companion",
    )

    cue, _signature, event_kind = build_neko_companion_cue(
        riichi,
        absurd_banter_enabled=True,
        absurd_variant=1,
    )

    assert event_kind == "riichi_pressure"
    assert "本次启用逆天模式" in cue
    assert "有人立直" in cue
    assert "卡布奇诺先放稳" in cue
    assert "必须先明确反应立直，再接梗" in cue


def test_absurd_banter_reports_opponent_meld_before_using_event_joke() -> None:
    observed = build_public_payload(
        {
            "last_decision": {"decision_type": "observe"},
            "round_state": {"round_id": "east-1"},
        },
        mode="companion",
    )
    event = {
        "new_melds": [
            {
                "owner": "left_opponent",
                "kind": "pon",
                "tiles": ["5z", "5z", "5z"],
            }
        ],
        "current_melds": {
            "left_opponent": [
                {"kind": "pon", "tiles": ["5z", "5z", "5z"]}
            ]
        },
    }

    cue, _signature, event_kind = build_neko_companion_cue(
        observed,
        opponent_meld_event=event,
        absurd_banter_enabled=True,
        absurd_variant=2,
    )

    assert event_kind == "opponent_meld"
    assert "上家碰了白" in cue
    assert "本次启用逆天模式" in cue
    assert "先准确播报副露" in cue
    assert "必须先准确说清玩家与吃碰杠动作，再接梗" in cue


def test_new_opponent_pon_reports_owner_tile_and_current_open_melds_once() -> None:
    plugin = MahjongCoachPlugin.__new__(MahjongCoachPlugin)
    plugin._reset_neko_companion_ambient_state()

    baseline = build_public_payload(
        {
            "last_decision": {"decision_type": "observe"},
            "round_state": {
                "round_id": "east-1",
                "opening_emitted": True,
                "settlement_phase": "playing",
                "last_opponent_melds": {
                    "right_opponent": [{"kind": "chi", "tiles": ["3m", "4m", "5m"]}],
                },
            },
        },
        mode="companion",
    )
    assert plugin._observe_neko_companion_opponent_meld(baseline) == {}

    after_pon = build_public_payload(
        {
            "last_decision": {"decision_type": "observe"},
            "round_state": {
                "round_id": "east-1",
                "opening_emitted": True,
                "settlement_phase": "playing",
                "last_opponent_melds": {
                    "right_opponent": [
                        {"kind": "chi", "tiles": ["3m", "4m", "5m"]},
                        {"kind": "pon", "tiles": ["5z", "5z", "5z"]},
                    ],
                },
            },
        },
        mode="companion",
    )
    event = plugin._observe_neko_companion_opponent_meld(after_pon)
    cue, signature, event_kind = build_neko_companion_cue(
        after_pon,
        opponent_meld_event=event,
    )

    assert event_kind == "opponent_meld"
    assert signature
    assert "下家碰了白" in cue
    assert "下家当前副露：吃（3万、4万、5万）；碰白（白、白、白）" in cue
    assert "必须说出碰的是哪张牌" in cue
    assert any(
        "对手副露：下家2组，吃（3万、4万、5万）；碰白（白、白、白）" in item
        for item in after_pon["presentation"]["evidence"]
    )
    assert plugin._observe_neko_companion_opponent_meld(after_pon) == event
    plugin._finalize_neko_companion_cadence(
        event_kind="opponent_meld",
        activity_signature="",
        rare_joke_due=False,
        events_since_rare=0,
        now=1.0,
    )
    assert plugin._observe_neko_companion_opponent_meld(after_pon) == {}

    # A temporary visual dropout must not make the same pon look new again.
    dropout = build_public_payload(
        {
            "last_decision": {"decision_type": "observe"},
            "round_state": {
                "round_id": "east-1",
                "opening_emitted": True,
                "settlement_phase": "playing",
                "last_opponent_melds": {},
            },
        },
        mode="companion",
    )
    assert plugin._observe_neko_companion_opponent_meld(dropout) == {}
    assert plugin._observe_neko_companion_opponent_meld(after_pon) == {}


@pytest.mark.asyncio
async def test_new_opponent_pon_pushes_without_riichi_or_ambient_trigger() -> None:
    plugin = MahjongCoachPlugin.__new__(MahjongCoachPlugin)
    plugin._cfg = MahjongCoachConfig(neko_companion_enabled=True)
    plugin._neko_companion_last_signature = ""
    plugin._neko_companion_last_push_at = 0.0
    plugin._neko_companion_push_count = 0
    plugin._neko_companion_events_since_rare_joke = 0
    plugin._neko_companion_last_rare_joke_at = 0.0
    plugin._reset_neko_companion_ambient_state()
    pushed: list[dict[str, object]] = []

    async def available(*, now: float) -> bool:
        _ = now
        return True

    async def unconfirmed(*_args, **_kwargs) -> None:
        return None

    plugin._probe_companion_message_plane = available
    plugin._confirm_companion_message_ingest = unconfirmed
    plugin.push_message = lambda **kwargs: pushed.append(kwargs)

    def snapshot(melds: list[dict[str, object]]) -> dict[str, object]:
        return build_public_payload(
            {
                "last_decision": {"decision_type": "observe"},
                "round_state": {
                    "round_id": "east-1",
                    "opening_emitted": True,
                    "settlement_phase": "playing",
                    "riichi_players": [],
                    "last_opponent_melds": {"right_opponent": melds},
                },
            },
            mode="companion",
        )

    assert await plugin._push_neko_companion_if_needed(snapshot([])) is False
    pon_snapshot = snapshot([{"kind": "pon", "tiles": ["5z", "5z", "5z"]}])
    assert await plugin._push_neko_companion_if_needed(pon_snapshot) is True
    assert pushed[-1]["metadata"]["event_kind"] == "opponent_meld"
    assert pushed[-1]["priority"] == 4
    assert "下家碰了白" in pushed[-1]["parts"][0]["text"]
    assert await plugin._push_neko_companion_if_needed(pon_snapshot) is False


def test_tsumo_window_has_immediate_non_prescriptive_companion_cue() -> None:
    public = build_public_payload(
        {
            "last_decision": {
                "decision_type": "win_window",
                "buttons": ["kan", "tsumo"],
                "perception": {"action": {"source": "fast_color_scan"}},
            },
            "round_state": {
                "round_id": "east-1",
                "opening_emitted": True,
                "settlement_phase": "playing",
            },
        },
        mode="companion",
    )

    cue, signature, event_kind = build_neko_companion_cue(
        public,
        include_rare_joke=True,
    )

    assert public["presentation"]["action_event"] == "tsumo"
    assert public["presentation"]["headline"] in cue
    assert public["last_decision"]["buttons"] == []
    assert event_kind == "win_opportunity"
    assert signature
    assert "界面出现自摸按钮" in cue
    assert "不要固定复读“总算没白等”" in cue
    assert "不得告诉玩家应该打哪张牌" in cue
    assert "铜须是生物吗？" not in cue


def test_win_companion_reactions_vary_across_rounds_but_stay_stable_per_event() -> None:
    reactions: dict[str, set[str]] = {"tsumo": set(), "ron": set()}
    for action in reactions:
        for index in range(18):
            raw = {
                "last_decision": {
                    "decision_type": "win_window",
                    "buttons": [action],
                },
                "round_state": {
                    "round_id": f"auto-round-{index}",
                    "update_count": 20 + index,
                    "last_hand_signature": f"hand-{index}",
                    "opening_emitted": True,
                    "settlement_phase": "playing",
                },
            }
            first = build_public_payload(raw, mode="companion")
            second = build_public_payload(raw, mode="companion")
            assert first["presentation"]["headline"] == second["presentation"]["headline"]
            cue, _signature, event_kind = build_neko_companion_cue(first)
            assert event_kind == "win_opportunity"
            assert first["presentation"]["headline"] in cue
            reactions[action].add(first["presentation"]["headline"])

    assert len(reactions["tsumo"]) >= 4
    assert len(reactions["ron"]) >= 4


def test_strategy_win_window_does_not_reuse_stale_discard_candidates() -> None:
    public = build_public_payload(
        {
            "last_decision": {
                "decision_type": "win_window",
                "buttons": ["tsumo"],
                "perception": {
                    "strategy": {
                        "top_candidates": [
                            {"tile": "4m", "defense_risk": 20, "shanten": 0, "effective_count": 8}
                        ]
                    }
                },
            },
            "round_state": {"round_id": "east-1"},
        },
        mode="strategy",
    )

    assert public["presentation"]["headline"] == "界面事件：检测到自摸按钮"
    assert public["presentation"]["primary_action"] == ""
    assert public["presentation"]["risk_options"] == []
    assert public["last_decision"]["suggestion"] == ""
    assert public["last_decision"]["perception"]["strategy"]["top_candidates"] == []


def test_strategy_call_window_prioritizes_call_decision_over_stale_discard() -> None:
    raw = {
        "last_decision": {
            "decision_type": "call_window",
            "buttons": ["pon", "skip"],
            "suggestion": "主建议：碰白。碰后进入听牌。",
            "perception": {
                "action": {
                    "source": "fast_color_scan",
                    "claimed_tile": "5z",
                    "call_recommendation": {
                        "version": 1,
                        "decision": "call",
                        "action": "pon",
                        "action_label": "碰",
                        "primary_action": "主建议：碰白",
                        "reason": "碰后可以进入听牌，预计有效牌约3枚；鸣牌后优先打9索。",
                        "claimed_tile": "5z",
                        "claimed_tile_label": "白",
                        "claimed_tile_known": True,
                        "post_discard": "9s",
                        "post_discard_label": "9索",
                        "baseline_shanten": 1,
                        "post_shanten": 0,
                        "effective_count": 3,
                        "available_actions": ["pon"],
                        "available_action_labels": ["碰"],
                    },
                },
                "strategy": {
                    "top_candidates": [
                        {"tile": "4m", "defense_risk": 20, "shanten": 1, "effective_count": 8}
                    ]
                },
            },
        },
        "round_state": {"round_id": "east-1"},
    }

    strategy = build_public_payload(raw, mode="strategy")
    companion = build_public_payload(raw, mode="companion")

    assert strategy["presentation"]["headline"] == "鸣牌窗口：主建议：碰白"
    assert strategy["presentation"]["primary_action"] == "主建议：碰白"
    assert strategy["presentation"]["primary_discard"] == ""
    assert strategy["presentation"]["risk_options"] == []
    assert "鸣牌理由：碰后可以进入听牌" in strategy["presentation"]["detail"]
    assert strategy["last_decision"]["suggestion"] == "主建议：碰白"
    assert strategy["last_decision"]["perception"]["action"]["call_recommendation"]["action"] == "pon"
    assert strategy["last_decision"]["perception"]["strategy"]["top_candidates"] == []
    assert "主建议：碰白" in overlay_text_from_payload(strategy)

    assert companion["presentation"]["primary_action"] == ""
    assert companion["presentation"]["call_recommendation"] == {}
    assert companion["last_decision"]["suggestion"] == ""
    assert "call_recommendation" not in companion["last_decision"]["perception"]["action"]
    assert "碰白" not in overlay_text_from_payload(companion)


def test_neko_companion_cue_supports_non_prescriptive_ambient_events() -> None:
    public = build_public_payload(
        {
            "last_decision": {
                "decision_type": "observe",
                "perception": {
                    "hand": {"hand_tiles": ["6z", "7z", "1m", "2m", "3m"]},
                    "strategy": {"top_candidates": [{"tile": "4m"}]},
                },
            },
            "round_state": {
                "round_id": "east-1",
                "opening_emitted": True,
                "settlement_phase": "playing",
                "last_hand_tiles": ["6z", "7z", "1m", "2m", "3m"],
                "riichi_players": ["right_opponent"],
            },
        },
        mode="companion",
    )

    ambient_cue, ambient_signature, ambient_kind = build_neko_companion_cue(
        public,
        ambient_event="ambient_observation",
        ambient_signature="activity-1",
    )
    idle_cue, idle_signature, idle_kind = build_neko_companion_cue(
        public,
        ambient_event="idle_heartbeat",
        ambient_signature="activity-1",
    )

    assert ambient_kind == "ambient_observation"
    assert idle_kind == "idle_heartbeat"
    assert ambient_signature != idle_signature
    assert "立直压力仍在持续" in ambient_cue
    assert "立直后的场面安静了一会儿" in idle_cue
    for cue in (ambient_cue, idle_cue):
        assert "不得告诉玩家应该打哪张牌" in cue
        assert "4m" not in cue
        assert "打4万" not in cue
        assert "没有检测到立直" not in cue
        assert "没看到有人立直" not in cue


def test_neko_companion_uses_casual_chat_instead_of_fake_no_riichi_observation() -> None:
    public = build_public_payload(
        {
            "last_decision": {"decision_type": "observe"},
            "round_state": {
                "round_id": "east-1",
                "opening_emitted": True,
                "settlement_phase": "playing",
                "riichi_players": [],
            },
        },
        mode="companion",
    )

    assert build_neko_companion_cue(
        public,
        ambient_event="ambient_observation",
        ambient_signature="activity-1",
    ) == ("", "", "")
    assert build_neko_companion_cue(
        public,
        ambient_event="idle_heartbeat",
        ambient_signature="activity-1",
    ) == ("", "", "")
    casual_cue, casual_signature, casual_kind = build_neko_companion_cue(
        public,
        ambient_event="casual_chat",
        ambient_signature="activity-1|casual:0",
    )
    assert casual_kind == "casual_chat"
    assert casual_signature
    assert "直接进行自然的普通聊天" in casual_cue
    assert "不必谈麻将" in casual_cue
    assert "不要复述上一条回复" in casual_cue
    assert "暂未检测到对手立直" not in casual_cue
    assert "牌河证据不足" not in casual_cue

    plugin = MahjongCoachPlugin.__new__(MahjongCoachPlugin)
    plugin._neko_companion_ambient_count = 0
    plugin._neko_companion_last_push_at = 0.0
    plugin._neko_companion_activity_changed_at = 1.0
    plugin._neko_companion_pending_activity_signature = "activity-1"
    assert plugin._select_neko_companion_ambient_event(
        public,
        activity_signature="activity-1",
        now=5.0,
    ) == ""
    assert plugin._select_neko_companion_ambient_event(
        public,
        activity_signature="activity-1",
        now=20.0,
    ) == "casual_chat"


@pytest.mark.asyncio
async def test_neko_companion_push_is_hidden_deduplicated_and_non_prescriptive() -> None:
    plugin = MahjongCoachPlugin.__new__(MahjongCoachPlugin)
    plugin._cfg = MahjongCoachConfig(
        live_advice_mode="companion",
        neko_companion_enabled=True,
    )
    plugin._neko_companion_last_signature = ""
    plugin._neko_companion_last_push_at = 0.0
    plugin._neko_companion_push_count = 0
    plugin._neko_companion_events_since_rare_joke = 0
    plugin._neko_companion_last_rare_joke_at = 0.0
    pushed: list[dict[str, object]] = []
    plugin.push_message = lambda **kwargs: pushed.append(kwargs)
    snapshot = build_public_payload(
        {
            "last_decision": {
                "decision_type": "opening_plan",
                "perception": {
                    "hand": {"hand_tiles": ["6z", "7z", "1m", "2m", "3m"]},
                    "strategy": {"top_candidates": [{"tile": "9p"}]},
                },
            },
            "round_state": {"round_id": "east-1", "last_open_meld_count": 0},
        },
        mode="companion",
    )

    assert await plugin._push_neko_companion_if_needed(snapshot) is True
    assert await plugin._push_neko_companion_if_needed(snapshot) is False
    assert len(pushed) == 1
    message = pushed[0]
    assert message["visibility"] == []
    assert message["ai_behavior"] == "respond"
    assert message["coalesce_key"] == "mahjong_companion"
    assert message["metadata"]["non_prescriptive"] is True
    cue = message["parts"][0]["text"]
    assert "只读观察" in cue
    assert "铜须是生物吗？" not in cue
    assert "9p" not in cue
    assert "暂未检测到对手立直" not in cue
    assert "没有检测到立直" not in cue

    plugin._cfg = MahjongCoachConfig(
        live_advice_mode="companion",
        neko_companion_enabled=True,
        absurd_banter_enabled=True,
    )
    plugin._neko_companion_last_signature = ""
    direct_replies: list[dict[str, object]] = []

    async def submit_fixed(cue: str, **kwargs: object) -> DirectCompanionSubmitResult:
        direct_replies.append({"cue": cue, **kwargs})
        return DirectCompanionSubmitResult(True, "tcp://127.0.0.1:48962")

    plugin._submit_companion_direct_fallback = submit_fixed
    assert await plugin._push_neko_companion_if_needed(snapshot) is True
    assert direct_replies[-1]["direct_reply"] is True
    assert "牌运" not in str(direct_replies[-1]["cue"])
    assert len(pushed) == 1

    plugin._cfg = MahjongCoachConfig(
        live_advice_mode="strategy",
        neko_companion_enabled=True,
    )
    plugin._neko_companion_last_signature = ""
    strategy_snapshot = build_public_payload(
        {
            "last_decision": {
                "decision_type": "opening_plan",
                "perception": {
                    "hand": {"hand_tiles": ["6z", "7z", "1m", "2m", "3m"]},
                    "strategy": {
                        "top_candidates": [
                            {
                                "tile": "9p",
                                "defense_risk": 58,
                                "shanten": 1,
                                "effective_count": 8,
                            }
                        ]
                    },
                },
            },
            "round_state": {"round_id": "east-1", "last_open_meld_count": 0},
        },
        mode="strategy",
    )
    assert await plugin._push_neko_companion_if_needed(strategy_snapshot) is True
    strategy_cue = pushed[-1]["parts"][0]["text"]
    assert "只读观察" in strategy_cue
    assert "9p" not in strategy_cue
    assert "打九筒" not in strategy_cue

    plugin._cfg = MahjongCoachConfig(
        live_advice_mode="strategy",
        neko_companion_enabled=False,
    )
    plugin._neko_companion_last_signature = ""
    assert await plugin._push_neko_companion_if_needed(snapshot) is False


@pytest.mark.asyncio
async def test_neko_companion_does_not_count_dead_message_plane_as_speech() -> None:
    plugin = MahjongCoachPlugin.__new__(MahjongCoachPlugin)
    plugin._cfg = MahjongCoachConfig(neko_companion_enabled=True)
    plugin._neko_companion_last_signature = ""
    plugin._neko_companion_last_push_at = 0.0
    plugin._neko_companion_push_count = 0
    plugin._neko_companion_events_since_rare_joke = 0
    plugin._neko_companion_last_rare_joke_at = 0.0
    plugin._reset_neko_companion_ambient_state()

    async def unavailable(*, now: float) -> bool:
        _ = now
        return False

    async def unavailable_direct(*_args, **_kwargs) -> DirectCompanionSubmitResult:
        return DirectCompanionSubmitResult(
            submitted=False,
            endpoint="tcp://127.0.0.1:48962",
            reason="agent_socket_unavailable",
        )

    plugin._probe_companion_message_plane = unavailable
    plugin._submit_companion_direct_fallback = unavailable_direct
    plugin.push_message = lambda **_kwargs: pytest.fail("dead transport must not receive a push")
    snapshot = build_public_payload(
        {
            "last_decision": {"decision_type": "opening_plan"},
            "round_state": {"round_id": "east-1"},
        },
        mode="companion",
    )

    assert await plugin._push_neko_companion_if_needed(snapshot) is False
    assert plugin._neko_companion_push_count == 0
    assert plugin._neko_companion_delivery_stage == "message_plane_unavailable"
    assert "兼容通道也不可用" in plugin._neko_companion_last_error


@pytest.mark.asyncio
async def test_neko_companion_uses_direct_agent_fallback_when_message_plane_is_dead() -> None:
    plugin = MahjongCoachPlugin.__new__(MahjongCoachPlugin)
    plugin._cfg = MahjongCoachConfig(neko_companion_enabled=True)
    plugin._neko_companion_last_signature = ""
    plugin._neko_companion_last_push_at = 0.0
    plugin._neko_companion_push_count = 0
    plugin._neko_companion_events_since_rare_joke = 0
    plugin._neko_companion_last_rare_joke_at = 0.0
    plugin._reset_neko_companion_ambient_state()

    async def unavailable(*, now: float) -> bool:
        _ = now
        return False

    submitted: list[dict[str, object]] = []

    async def submit_direct(
        cue: str,
        **kwargs: object,
    ) -> DirectCompanionSubmitResult:
        submitted.append({"cue": cue, **kwargs})
        return DirectCompanionSubmitResult(
            submitted=True,
            endpoint="tcp://127.0.0.1:48962",
        )

    plugin._probe_companion_message_plane = unavailable
    plugin._submit_companion_direct_fallback = submit_direct
    plugin.push_message = lambda **_kwargs: pytest.fail("dead message-plane must be bypassed")
    snapshot = build_public_payload(
        {
            "last_decision": {"decision_type": "opening_plan"},
            "round_state": {"round_id": "east-1"},
        },
        mode="companion",
    )

    assert await plugin._push_neko_companion_if_needed(snapshot) is True
    assert len(submitted) == 1
    assert submitted[0]["event_kind"] == "opening_observation"
    assert plugin._neko_companion_push_count == 1
    assert plugin._neko_companion_delivery_stage == "direct_host_submitted"
    assert "兼容通道" in plugin._neko_companion_last_error


@pytest.mark.asyncio
async def test_neko_companion_falls_back_when_sdk_rejects_local_submission() -> None:
    plugin = MahjongCoachPlugin.__new__(MahjongCoachPlugin)
    plugin._cfg = MahjongCoachConfig(neko_companion_enabled=True)
    plugin._neko_companion_last_signature = ""
    plugin._neko_companion_last_push_at = 0.0
    plugin._neko_companion_push_count = 0
    plugin._neko_companion_events_since_rare_joke = 0
    plugin._neko_companion_last_rare_joke_at = 0.0
    plugin._reset_neko_companion_ambient_state()

    async def available(*, now: float) -> bool:
        _ = now
        return True

    async def submit_direct(*_args, **_kwargs) -> DirectCompanionSubmitResult:
        return DirectCompanionSubmitResult(
            submitted=True,
            endpoint="tcp://127.0.0.1:48962",
        )

    plugin._probe_companion_message_plane = available
    plugin._submit_companion_direct_fallback = submit_direct
    plugin.push_message = lambda **_kwargs: {
        "ok": False,
        "submitted": False,
        "reason": "transport_error",
    }
    snapshot = build_public_payload(
        {
            "last_decision": {"decision_type": "opening_plan"},
            "round_state": {"round_id": "east-1"},
        },
        mode="companion",
    )

    assert await plugin._push_neko_companion_if_needed(snapshot) is True
    assert plugin._neko_companion_push_count == 1
    assert plugin._neko_companion_delivery_stage == "direct_host_submitted"
    assert "SDK 通道拒绝提交" in plugin._neko_companion_last_error


@pytest.mark.asyncio
async def test_neko_companion_ambient_cadence_fills_quiet_play_without_spam(
    monkeypatch,
    patch_module_clock,
) -> None:
    plugin = MahjongCoachPlugin.__new__(MahjongCoachPlugin)
    plugin._cfg = MahjongCoachConfig(
        live_advice_mode="companion",
        neko_companion_enabled=True,
    )
    plugin._neko_companion_last_signature = ""
    plugin._neko_companion_last_push_at = 100.0
    plugin._neko_companion_push_count = 0
    plugin._neko_companion_events_since_rare_joke = 0
    plugin._neko_companion_last_rare_joke_at = 0.0
    plugin._reset_neko_companion_ambient_state()
    pushed: list[dict[str, object]] = []
    plugin.push_message = lambda **kwargs: pushed.append(kwargs)

    def snapshot_with_discards(discards: list[str]) -> dict[str, object]:
        return build_public_payload(
            {
                "last_decision": {
                    "decision_type": "observe",
                    "perception": {"hand": {"hand_tiles": ["1m", "2m", "3m", "6z"]}},
                },
                "round_state": {
                    "round_id": "east-1",
                    "opening_emitted": True,
                    "settlement_phase": "playing",
                    "last_hand_tiles": ["1m", "2m", "3m", "6z"],
                    "riichi_players": ["right_opponent"],
                    "last_discard_piles": {
                        "left_opponent": [{"tile": tile} for tile in discards],
                    },
                },
            },
            mode="companion",
        )

    clock = {"now": 100.0}
    patch_module_clock(monkeypatch, mahjong_plugin_module, time=lambda: clock["now"])
    snapshot = snapshot_with_discards([])
    assert await plugin._push_neko_companion_if_needed(snapshot) is False

    clock["now"] = 105.0
    snapshot = snapshot_with_discards(["9m"])
    assert await plugin._push_neko_companion_if_needed(snapshot) is False

    clock["now"] = 114.0
    assert await plugin._push_neko_companion_if_needed(snapshot) is True
    assert pushed[-1]["metadata"]["event_kind"] == "ambient_observation"
    assert pushed[-1]["priority"] == 2

    clock["now"] = 126.0
    assert await plugin._push_neko_companion_if_needed(snapshot) is True
    assert pushed[-1]["metadata"]["event_kind"] == "idle_heartbeat"

    clock["now"] = 137.0
    assert await plugin._push_neko_companion_if_needed(snapshot) is False

    clock["now"] = 138.0
    assert await plugin._push_neko_companion_if_needed(snapshot) is True
    assert pushed[-1]["metadata"]["event_kind"] == "idle_heartbeat"
    assert len(pushed) == 3


@pytest.mark.asyncio
async def test_neko_companion_casual_chat_has_low_priority_cadence_and_unique_signatures(
    monkeypatch,
    patch_module_clock,
) -> None:
    plugin = MahjongCoachPlugin.__new__(MahjongCoachPlugin)
    plugin._cfg = MahjongCoachConfig(
        live_advice_mode="companion",
        neko_companion_enabled=True,
    )
    plugin._neko_companion_last_signature = ""
    plugin._neko_companion_last_push_at = 0.0
    plugin._neko_companion_push_count = 0
    plugin._neko_companion_events_since_rare_joke = 0
    plugin._neko_companion_last_rare_joke_at = 0.0
    plugin._reset_neko_companion_ambient_state()
    pushed: list[dict[str, object]] = []
    plugin.push_message = lambda **kwargs: pushed.append(kwargs)
    snapshot = build_public_payload(
        {
            "last_decision": {"decision_type": "observe"},
            "round_state": {
                "round_id": "east-1",
                "opening_emitted": True,
                "settlement_phase": "playing",
                "riichi_players": [],
            },
        },
        mode="companion",
    )

    clock = {"now": 100.0}
    patch_module_clock(monkeypatch, mahjong_plugin_module, time=lambda: clock["now"])
    assert await plugin._push_neko_companion_if_needed(snapshot) is False

    clock["now"] = 113.0
    assert await plugin._push_neko_companion_if_needed(snapshot) is True
    first_signature = plugin._neko_companion_last_signature
    assert pushed[-1]["metadata"]["event_kind"] == "casual_chat"
    assert pushed[-1]["priority"] == 2
    assert "直接进行自然的普通聊天" in pushed[-1]["parts"][0]["text"]

    clock["now"] = 114.0
    assert await plugin._push_neko_companion_if_needed(snapshot) is False

    clock["now"] = 126.0
    assert await plugin._push_neko_companion_if_needed(snapshot) is True
    assert plugin._neko_companion_last_signature != first_signature
    assert len(pushed) == 2


def test_live_loop_evaluates_quiet_frames_for_ambient_companion() -> None:
    source = inspect.getsource(MahjongCoachPlugin._run_live_loop)

    assert 'if not getattr(decision, "quiet", False):' not in source
    assert "companion_pushed = await self._push_neko_companion_if_needed(snapshot)" in source


@pytest.mark.asyncio
async def test_neko_companion_critical_riichi_bypasses_ordinary_cooldown(
    monkeypatch,
    patch_module_clock,
) -> None:
    plugin = MahjongCoachPlugin.__new__(MahjongCoachPlugin)
    plugin._cfg = MahjongCoachConfig(live_advice_mode="companion", neko_companion_enabled=True)
    plugin._neko_companion_last_signature = "previous-ambient"
    plugin._neko_companion_last_push_at = 100.0
    plugin._neko_companion_push_count = 1
    plugin._neko_companion_events_since_rare_joke = 0
    plugin._neko_companion_last_rare_joke_at = 0.0
    plugin._reset_neko_companion_ambient_state()
    pushed: list[dict[str, object]] = []
    plugin.push_message = lambda **kwargs: pushed.append(kwargs)
    patch_module_clock(monkeypatch, mahjong_plugin_module, time=lambda: 101.0)
    snapshot = build_public_payload(
        {
            "last_decision": {"decision_type": "defense_alert"},
            "round_state": {
                "round_id": "east-1",
                "opening_emitted": True,
                "settlement_phase": "playing",
                "riichi_players": ["left_opponent"],
            },
        },
        mode="companion",
    )

    assert await plugin._push_neko_companion_if_needed(snapshot) is True
    assert pushed[-1]["metadata"]["event_kind"] == "riichi_pressure"
    assert pushed[-1]["priority"] == 4
    assert plugin._neko_companion_last_event_kind == "riichi_pressure"


@pytest.mark.asyncio
async def test_neko_obvious_question_is_hard_limited_by_event_and_time_cooldowns() -> None:
    plugin = MahjongCoachPlugin.__new__(MahjongCoachPlugin)
    plugin._cfg = MahjongCoachConfig(
        live_advice_mode="companion",
        neko_companion_enabled=True,
    )
    plugin._neko_companion_last_signature = ""
    plugin._neko_companion_push_count = 0
    plugin._neko_companion_events_since_rare_joke = 11
    plugin._neko_companion_last_rare_joke_at = 0.0
    pushed: list[dict[str, object]] = []
    plugin.push_message = lambda **kwargs: pushed.append(kwargs)
    snapshot = build_public_payload(
        {
            "last_decision": {
                "decision_type": "opening_plan",
                "perception": {"hand": {"hand_tiles": ["6z", "1m", "2m", "3m"]}},
            },
            "round_state": {"round_id": "east-1", "last_hand_tiles": ["6z", "1m", "2m", "3m"]},
        },
        mode="companion",
    )

    assert await plugin._push_neko_companion_if_needed(snapshot) is True
    assert "铜须是生物吗？" in pushed[-1]["parts"][0]["text"]
    assert "哎呀不对不对" not in pushed[-1]["parts"][0]["text"]
    assert plugin._neko_companion_events_since_rare_joke == 0
    assert plugin._neko_companion_last_rare_joke_at > 0

    plugin._neko_companion_last_signature = ""
    plugin._neko_companion_events_since_rare_joke = 11
    assert await plugin._push_neko_companion_if_needed(snapshot) is True
    assert "铜须是生物吗？" not in pushed[-1]["parts"][0]["text"]
    assert plugin._neko_companion_events_since_rare_joke == 12


def test_companion_reacts_only_after_observed_player_action() -> None:
    engine = RoundCoachEngine()
    engine._companion_reference_tiles = ["1m", "9p", "7z"]
    previous = ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "1p", "2p", "3p", "1s", "2s", "3s"]

    engine._observe_player_discard(previous, ["2m", "3m", "4m", "5m", "6m", "7m", "1p", "2p", "3p", "1s", "2s", "3s", "9s"])

    assert engine.state.last_observed_discard == "1m"
    assert engine.state.companion_reaction_kind == "agree"
    assert engine.state.companion_reaction.startswith("哦对的对的")

    engine._observe_player_discard(previous, ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "1p", "2p", "3p", "1s", "2s", "9s"])

    assert engine.state.last_observed_discard == "3s"
    assert engine.state.companion_reaction_kind == "disagree"
    assert engine.state.companion_reaction.startswith(
        "哦对的对的……哎呀不对不对……对……对吗？"
    )

    mixed_previous = [
        "7z", "2m", "3m", "4m", "5m", "6m", "7m",
        "1p", "2p", "3p", "1s", "2s", "3s",
    ]
    engine._observe_player_discard(
        mixed_previous,
        ["2m", "3m", "4m", "5m", "6m", "7m", "1p", "2p", "3p", "1s", "2s", "3s", "9s"],
    )

    assert engine.state.last_observed_discard == "7z"
    assert engine.state.companion_reaction_kind == "mixed"
    assert engine.state.companion_reaction.startswith(
        "哦对的对的……哎呀不对不对……对……对吗？"
    )


def test_preferences_store_never_persists_hwnd_and_can_clear_target(tmp_path: Path) -> None:
    path = tmp_path / "coach_preferences.json"
    path.write_text(
        json.dumps(
            {
                "target": {
                    "title": "雀魂 - Mahjong Soul",
                    "app_name": "Majsoul",
                    "hwnd": 123456,
                }
            }
        ),
        encoding="utf-8",
    )
    store = PreferencesStore(path)

    loaded = store.load()
    assert loaded.target.title == "雀魂 - Mahjong Soul"
    assert "hwnd" not in loaded.target.to_dict()

    cleared = store.update(clear_target=True)
    assert cleared.target == WindowTargetDescriptor()
    assert "hwnd" not in path.read_text(encoding="utf-8")


def test_capture_session_reuses_binding_before_revalidation(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    binding = WindowBindingResult(
        bound=True,
        window_title="雀魂 - Mahjong Soul",
        hwnd=42,
        left=0,
        top=0,
        width=1280,
        height=720,
    )
    monkeypatch.setattr(capture, "bind_window_from_descriptor", lambda *_args: calls.append("bind") or binding)
    monkeypatch.setattr(capture, "refresh_cached_window", lambda current: calls.append("refresh") or current)
    session = CaptureSession(["雀魂"], revalidate_seconds=2.0)

    assert session.locate_window().bound is True
    assert session.locate_window().hwnd == 42
    assert calls == ["bind"]

    session._last_validation -= 3.0
    assert session.locate_window().bound is True
    assert calls == ["bind", "refresh"]


def test_capture_session_rebinds_after_window_size_change(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    original = WindowBindingResult(
        bound=True,
        window_title="雀魂",
        hwnd=42,
        width=1280,
        height=720,
    )
    resized = WindowBindingResult(
        bound=True,
        window_title="雀魂",
        hwnd=42,
        width=1920,
        height=1080,
    )
    monkeypatch.setattr(capture, "bind_window_from_descriptor", lambda *_args: calls.append("bind") or resized)
    monkeypatch.setattr(capture, "refresh_cached_window", lambda _current: calls.append("refresh") or resized)
    session = CaptureSession(["雀魂"], revalidate_seconds=0.25)
    session.binding = original
    session._last_validation = time.monotonic() - 1.0

    result = session.locate_window()

    assert result.width == 1920
    assert calls == ["refresh", "bind"]


def test_capture_memory_frame_does_not_write_a_frame_file(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = DefaultCaptureProvider()
    source_image = Image.new("RGB", (1280, 720), (30, 80, 120))
    monkeypatch.setattr(provider, "_capture_image", lambda _binding: (source_image.copy(), "memory-test"))

    packet = provider.capture_memory_frame(
        binding_result=WindowBindingResult(
            bound=True,
            window_title="雀魂",
            width=1280,
            height=720,
        )
    )

    assert packet.image_path == ""
    assert packet.image is not None
    assert packet.image.size == (1280, 720)
    assert packet.fingerprint
    packet.image.close()
    source_image.close()


@pytest.mark.asyncio
async def test_live_capture_queue_keeps_only_the_newest_frame(tmp_path: Path) -> None:
    plugin = MahjongCoachPlugin.__new__(MahjongCoachPlugin)
    plugin._cfg = MahjongCoachConfig()
    plugin._live_state = LiveSessionState(running=True)
    plugin._live_stop_event = asyncio.Event()
    plugin._preferences = CapturePreferences()

    binding = WindowBindingResult(
        bound=True,
        window_title="雀魂",
        hwnd=42,
        left=0,
        top=0,
        width=8,
        height=8,
    )
    session = SimpleNamespace(
        keywords=["雀魂"],
        target=WindowTargetDescriptor(),
        locate_window=lambda: binding,
        invalidate=lambda _reason: None,
    )
    counter = 0

    def capture_memory_frame(*, binding_result: WindowBindingResult) -> FramePacket:
        nonlocal counter
        counter += 1
        packet = FramePacket(
            timestamp_ms=counter,
            image=Image.new("RGB", (8, 8), (counter, 0, 0)),
            source="test",
        )
        if counter == 2:
            plugin._live_stop_event.set()
        return packet

    provider = SimpleNamespace(capture_memory_frame=capture_memory_frame)
    queue: asyncio.Queue = asyncio.Queue(maxsize=1)

    await plugin._capture_live_frames(
        session=session,
        provider=provider,
        frame_queue=queue,
        interval_ms=200,
        frames_dir=tmp_path,
    )

    packet, _locate_ms, _capture_ms = queue.get_nowait()
    assert packet.timestamp_ms == 2
    assert plugin._live_state.dropped_frames == 1
    assert not list(tmp_path.iterdir())
    packet.image.close()


def test_live_performance_summary_reports_advice_p95_and_drops() -> None:
    rows = [
        {
            "decision": "defense_alert" if index % 2 else "observe",
            "loop_ms": float(index),
        }
        for index in range(1, 21)
    ]

    summary = _live_performance_summary(rows, dropped_frames=3)

    assert summary["sample_count"] == 20
    assert summary["advice_sample_count"] == 10
    assert summary["frame_p95_ms"] == 19.0
    assert summary["advice_p95_ms"] == 19.0
    assert summary["dropped_frames"] == 3


@pytest.mark.asyncio
async def test_direct_live_entry_binds_a_unique_window_without_web_ui(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin = MahjongCoachPlugin.__new__(MahjongCoachPlugin)
    plugin._cfg = MahjongCoachConfig()
    plugin._engine = SimpleNamespace(
        config=plugin._cfg,
        state=SimpleNamespace(play_style="riichi"),
    )
    plugin._start_yolo26_warmup = lambda *_args, **_kwargs: None
    plugin._live_task = None
    plugin._live_state = LiveSessionState()
    plugin._preferences = CapturePreferences()
    plugin.logger = SimpleNamespace(info=lambda *_args, **_kwargs: None, warning=lambda *_args, **_kwargs: None)
    plugin._clear_live_hand_gap = lambda: None
    captured: dict[str, object] = {}

    async def idle_live_loop(**kwargs) -> None:
        captured.update(kwargs)
        await asyncio.sleep(60)

    plugin._run_live_loop = idle_live_loop
    monkeypatch.setattr(
        mahjong_plugin_module,
        "list_window_candidates",
        lambda _keywords: [
            {
                "title": "雀魂 - Mahjong Soul",
                "app_name": "Majsoul",
                "match_keyword": "雀魂",
                "matches_keywords": True,
            }
        ],
    )

    result = await plugin._overlay_start_live(overlay=False)
    await asyncio.sleep(0)

    assert result.unwrap()["status"] == "starting"
    assert captured["target"] == WindowTargetDescriptor(
        title="雀魂 - Mahjong Soul",
        app_name="Majsoul",
        match_keyword="雀魂",
    )
    plugin._live_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await plugin._live_task


def test_amae_koromo_profile_is_cached_and_requires_confirmation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    provider = AmaeKoromoProvider(cache_path=tmp_path / "profile-cache.json")
    calls: list[str] = []

    def request(url: str):
        calls.append(url)
        if "search_player" in url:
            return [{"id": 123, "nickname": "Neko", "level": {"id": 40101, "score": 800}}]
        if "player_extended_stats" in url:
            return {"和牌率": 0.24, "放铳率": 0.12, "立直率": 0.21, "副露率": 0.31, "count": 200}
        return {"nickname": "Neko", "level": {"id": 50101}, "count": 200}

    monkeypatch.setattr(provider, "_request_json", request)

    candidates = provider.search("Neko")
    profile = provider.fetch_profile("123", nickname="Neko")
    call_count = len(calls)
    cached_profile = provider.fetch_profile("123", nickname="Neko")

    assert candidates[0]["account_id"] == "123"
    assert candidates[0]["rank"] == "master"
    assert profile.rank == "saint"
    assert profile.sample_count == 200
    assert profile.win_rate == pytest.approx(0.24)
    assert profile.confirmed is False
    assert cached_profile == profile
    assert len(calls) == call_count


@pytest.mark.asyncio
async def test_external_profile_failure_falls_back_without_reassigning_user() -> None:
    class FailingProvider:
        def search(self, *_args, **_kwargs):
            raise ProfileLookupError("offline")

    existing = PlayerProfile(rank="master", confirmed=True)
    plugin = MahjongCoachPlugin.__new__(MahjongCoachPlugin)
    plugin._profile_provider = FailingProvider()
    plugin._preferences = CapturePreferences(profile=existing)

    result = await plugin.mahjong_coach_search_player("Neko")
    payload = result.unwrap()

    assert payload["status"] == "fallback_manual"
    assert payload["profile"]["rank"] == "master"
    assert plugin._preferences.profile == existing


def test_player_risk_profile_changes_edge_defense_result() -> None:
    hand = ["1m", "2m", "3m", "4p", "5p", "6p", "2s", "3s", "4s", "5s", "6s", "7s", "1z", "1z"]
    river = RiverStateResult(
        ok=True,
        discard_piles={"right_opponent": [{"tile": "9m", "confidence": 0.96}]},
        visible_tiles=["9m"],
        confidence=0.96,
        reason="test",
    )
    conservative = rank_discard_decisions(
        hand,
        MahjongCoachConfig(player_profile=PlayerProfile(risk_tolerance="conservative")),
        ["shimocha"],
        river,
        turn_number=10,
    )
    balanced = rank_discard_decisions(
        hand,
        MahjongCoachConfig(player_profile=PlayerProfile(risk_tolerance="balanced")),
        ["shimocha"],
        river,
        turn_number=10,
    )
    aggressive = rank_discard_decisions(
        hand,
        MahjongCoachConfig(player_profile=PlayerProfile(risk_tolerance="aggressive")),
        ["shimocha"],
        river,
        turn_number=10,
    )
    standard_balanced = rank_discard_decisions(
        hand,
        MahjongCoachConfig(
            strategy_preset="standard",
            player_profile=PlayerProfile(risk_tolerance="balanced"),
        ),
        ["shimocha"],
        river,
        turn_number=10,
    )

    assert conservative["posture"] == "fold"
    assert conservative["simple_policy_active"] is False
    assert balanced["posture"] == "push"
    assert balanced["legacy_mode"] == "attack"
    assert balanced["preserve_win_chance"] is True
    assert balanced["win_potential"] == "strong"
    assert balanced["top_candidates"][0]["within_risk_budget"] is True
    assert balanced["top_candidates"][0]["defense_risk"] <= balanced["risk_budget"]
    assert balanced["top_candidates"][0]["shanten"] == 0
    assert balanced["simple_policy_active"] is True
    assert balanced["strategy_preset"] == "simple"
    assert "简易策略轻防守+12" in balanced["risk_budget_calculation"]
    assert aggressive["posture"] == "push"
    assert aggressive["risk_budget"] > conservative["risk_budget"]
    assert standard_balanced["posture"] == "mawashi"
    assert standard_balanced["simple_policy_active"] is False
    assert balanced["risk_weight"] < standard_balanced["risk_weight"]
    assert balanced["risk_budget"] > standard_balanced["risk_budget"]
    assert len(aggressive["top_candidates"]) == 3
    assert {"safety", "shape_loss", "effective_count", "effective_count_delta"} <= aggressive["top_candidates"][0].keys()


def test_low_value_bad_shape_folds_against_multiple_riichi_players() -> None:
    hand = ["1m", "1m", "3m", "5m", "7m", "9m", "1p", "4p", "7p", "1s", "4s", "7s", "1z", "5z"]
    river = RiverStateResult(
        ok=True,
        discard_piles={
            "right_opponent": [{"tile": "9m", "confidence": 0.96}],
            "left_opponent": [{"tile": "1p", "confidence": 0.96}],
        },
        visible_tiles=["9m", "1p"],
        confidence=0.96,
        reason="test",
    )

    ranking = rank_discard_decisions(
        hand,
        MahjongCoachConfig(),
        ["shimocha", "kamicha"],
        river,
        turn_number=14,
    )

    assert ranking["posture"] == "fold"
    assert ranking["riichi_count"] == 2


def test_yakuman_routes_report_distance_keys_and_blockers() -> None:
    kokushi = ["1m", "9m", "1p", "9p", "1s", "9s", "1z", "2z", "3z", "4z", "5z", "6z", "7z"]
    completed = [*kokushi, "1m"]
    chuuren = ["1m", "1m", "1m", "2m", "3m", "4m", "5m", "5m", "6m", "7m", "8m", "9m", "9m", "9m"]

    routes = {item.route: item for item in assess_yakuman_routes(kokushi)}

    assert route_distance("kokushi", kokushi) == 1
    assert set(routes["kokushi"].key_tiles) == set(kokushi)
    assert is_route_complete("kokushi", completed)
    assert is_route_complete("chuuren", chuuren)
    assert "该路线要求门清" in {
        item.route: item for item in assess_yakuman_routes(kokushi, open_melds=1)
    }["kokushi"].blockers


def test_yakuman_preference_surfaces_close_route_even_with_cleanup_tiles() -> None:
    hand = ["1m", "9m", "1p", "9p", "1s", "9s", "1z", "2z", "3z", "4z", "5z", "6z", "5m"]
    config = MahjongCoachConfig(player_profile=PlayerProfile(goal_bias="yakuman"))

    plan = build_round_plan(hand, config)

    assert plan["direction"].startswith("役满候选：国士无双")
    assert any(item.startswith("役满路线：国士无双") for item in plan["targets"])


def test_seeded_yakuman_estimate_is_bounded_and_reproducible() -> None:
    hand = ["1m", "9m", "1p", "9p", "1s", "9s", "1z", "2z", "3z", "4z", "5z", "6z", "7z"]

    started = time.perf_counter()
    first = monte_carlo_yakuman(hand, max_trials=3, time_budget_ms=1000, seed=19)
    second = monte_carlo_yakuman(hand, max_trials=3, time_budget_ms=1000, seed=19)
    elapsed = time.perf_counter() - started

    assert [item.to_dict() for item in first] == [item.to_dict() for item in second]
    estimated = [item for item in first if item.trials]
    assert len(estimated) == 9
    assert elapsed < 1.0
    for item in estimated:
        assert set(item.tsumo_probability) == {"6", "12", "18"}
        assert all(0.0 <= value <= 1.0 for value in item.tsumo_probability.values())
        assert all(0.0 <= low <= high <= 1.0 for low, high in item.confidence_interval.values())


def test_yakuman_service_returns_immediately_and_finishes_in_background() -> None:
    hand = ["1m", "9m", "1p", "9p", "1s", "9s", "1z", "2z", "3z", "4z", "5z", "6z", "7z"]
    service = YakumanEstimateService(max_trials=20, time_budget_ms=50)
    try:
        started = time.perf_counter()
        initial = service.request(hand)
        call_elapsed = time.perf_counter() - started
        assert initial["status"] in {"running", "ready"}
        assert call_elapsed < 0.2

        deadline = time.monotonic() + 2.0
        result = initial
        while result["status"] != "ready" and time.monotonic() < deadline:
            time.sleep(0.01)
            result = service.request(hand)
        assert result["status"] == "ready"
        assert result["routes"]
    finally:
        service.close()


def test_yakuman_service_skips_worker_for_instant_route_screening() -> None:
    hand = ["1m", "9m", "1p", "9p", "1s", "9s", "1z", "2z", "3z", "4z", "5z", "6z", "7z"]
    service = YakumanEstimateService(max_trials=20, time_budget_ms=50)
    try:
        result = service.request(hand, run_background=False)

        assert result["status"] == "instant"
        assert result["routes"]
        assert service.runtime_stats() == {
            "worker_started": False,
            "active_jobs": 0,
            "pending_jobs": 0,
            "cached_results": 0,
            "result_cache_size": 4,
        }
    finally:
        service.close()


def test_yakuman_service_keeps_only_active_and_latest_pending_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hand = ["1m", "9m", "1p", "9p", "1s", "9s", "1z", "2z", "3z", "4z", "5z", "6z", "7z"]
    started = threading.Event()
    release = threading.Event()

    def slow_estimate(
        hand_tiles: list[str],
        *,
        visible_tiles: list[str] | None = None,
        open_melds: int = 0,
        **_kwargs,
    ):
        started.set()
        release.wait(timeout=2.0)
        return assess_yakuman_routes(
            hand_tiles,
            visible_tiles=visible_tiles,
            open_melds=open_melds,
        )

    monkeypatch.setattr(yakuman_module, "monte_carlo_yakuman", slow_estimate)
    service = YakumanEstimateService(max_trials=20, time_budget_ms=50, result_cache_size=2)
    try:
        service.request(hand, visible_tiles=["1m"])
        assert started.wait(timeout=1.0)

        for tile in ("2m", "3m", "4m", "5m", "6m"):
            service.request(hand, visible_tiles=[tile])

        stats = service.runtime_stats()
        assert stats["active_jobs"] == 1
        assert stats["pending_jobs"] == 1
        assert stats["cached_results"] == 0

        release.set()
        deadline = time.monotonic() + 2.0
        result = service.request(hand, visible_tiles=["6m"])
        while result["status"] != "ready" and time.monotonic() < deadline:
            time.sleep(0.01)
            result = service.request(hand, visible_tiles=["6m"])

        assert result["status"] == "ready"
        assert service.runtime_stats()["cached_results"] <= 2
    finally:
        release.set()
        service.close()
