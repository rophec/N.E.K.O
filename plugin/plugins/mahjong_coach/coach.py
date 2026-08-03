from __future__ import annotations

import time
from collections import Counter
from copy import deepcopy
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Any

from .models import CoachDecision, DefensePosture, MahjongCoachConfig, RoundCoachState, _valid_play_style
from .perception.action_detector import detect_action_buttons_fast
from .perception.fast_hand_path import FastHandResult, detect_fast_hand_path, quick_frame_fingerprint
from .perception.game_scene import GameSceneResult, detect_game_scene_path
from .perception.image_source import ImageSource, source_identity
from .perception.meld_state import MeldStateResult, detect_meld_state_path
from .perception.river_state import RiverStateResult, detect_incremental_river_state_path, detect_river_state_path
from .perception.settlement_detector import (
    SettlementFrameResult,
    SettlementTracker,
    SettlementTransition,
    detect_settlement_path,
)
from .perception.table_context import TableContextResult, detect_table_context
from .perception.yolo26_visible_tiles import Yolo26TableStateResult, detect_yolo26_table_state_path
from .tile_labels import hand_signature, is_honor, is_simple, is_terminal, normalize_tile, tile_rank, tile_suit
from .yakuman import assess_yakuman_routes


CRITICAL_BUTTONS = {"chi", "pon", "kan", "ron", "tsumo", "riichi"}
CALL_BUTTONS = {"chi", "pon", "kan"}
WIN_BUTTONS = {"ron", "tsumo"}
SUIT_NAMES = {"m": "万", "p": "筒", "s": "索", "z": "字"}
HONOR_NAMES = {"1z": "东", "2z": "南", "3z": "西", "4z": "北", "5z": "白", "6z": "发", "7z": "中"}
TEXT_TILE_ALIASES = {"东": "1z", "南": "2z", "西": "3z", "北": "4z", "白": "5z", "发": "6z", "發": "6z", "中": "7z"}
TILE_TYPES = [f"{rank}{suit}" for suit in ("m", "p", "s") for rank in range(1, 10)] + [
    f"{rank}z" for rank in range(1, 8)
]
ORPHAN_TYPES = {"1m", "9m", "1p", "9p", "1s", "9s", "1z", "2z", "3z", "4z", "5z", "6z", "7z"}

_RIICHI_SUIT_THRESHOLD = 7
_FAST_SUIT_THRESHOLD = 9
_RIICHI_PAIR_THRESHOLD = 4
_FAST_PAIR_THRESHOLD = 5
_RIICHI_SIMPLE_THRESHOLD = 10
_FAST_SIMPLE_THRESHOLD = 8
_ROUND_HISTORY_LIMIT = 2
_RIVER_FULL_RESCAN_INTERVAL = 8
_RIVER_CORRECTION_CONFIRM_FRAMES = 2
_RIVER_MATCH_MIN_IOU = 0.25
_GAME_SCENE_CONFIRM_FRAMES = 2
_SELF_CALL_CLAIM_MAX_AGE_SECONDS = 15.0
_OPPONENT_CALL_LINK_MAX_SCANS = 3
_TABLE_PLAYERS = ("self", "right_opponent", "top_opponent", "left_opponent")
_PLAYER_DISPLAY_NAMES = {
    "self": "自己",
    "left_opponent": "上家",
    "top_opponent": "对家",
    "right_opponent": "下家",
}
_RISK_SCALE_NOTE = "0–100规则型相对危险指数，不是放铳概率；数值越高越危险"
_RISK_SCALE_LEGEND = "0–9极低｜10–39较低｜40–59中等｜60–79高｜80–100很高"
_RISK_MODEL_LEGEND = (
    "基准：现物0、全见5、筋30、壁42、字牌已见3/2/0–1枚=38/55/72、"
    "无筋幺九58、无筋2/8为72、无筋中张84、座位未知90；"
    "宝牌+14、宝牌周边+8、多家立直每多一家+4（最多+10），总分封顶100"
)
_CHI_DISCARD_SOURCE = {
    "self": "left_opponent",
    "right_opponent": "self",
    "top_opponent": "right_opponent",
    "left_opponent": "top_opponent",
}


def _frame_identity(path: ImageSource) -> tuple[str, int, int, str] | None:
    """Return a cache identity that changes even when a frame path is reused."""
    return source_identity(path)


def _table_context_signature(result: TableContextResult) -> str:
    score_part = ",".join(
        f"{player}:{int(result.scores[player])}"
        for player in _TABLE_PLAYERS
        if player in result.scores
    )
    honba = "?" if result.honba_count is None else str(int(result.honba_count))
    sticks = "?" if result.riichi_stick_count is None else str(int(result.riichi_stick_count))
    return f"{score_part}|honba:{honba}|sticks:{sticks}"


def _committed_table_context_signature(state: RoundCoachState) -> str:
    score_part = ",".join(
        f"{player}:{int(state.player_scores[player])}"
        for player in _TABLE_PLAYERS
        if player in state.player_scores
    )
    honba = "?" if state.honba_count is None else str(int(state.honba_count))
    sticks = (
        "?"
        if state.table_riichi_stick_count is None
        else str(int(state.table_riichi_stick_count))
    )
    return f"{score_part}|honba:{honba}|sticks:{sticks}"


class RoundCoachEngine:
    def __init__(
        self,
        config: MahjongCoachConfig | None = None,
        *,
        calibration_dir: Path | None = None,
    ) -> None:
        self.config = config or MahjongCoachConfig()
        self.calibration_dir = calibration_dir
        self.state = RoundCoachState()
        self._last_fingerprints: dict[str, bytes] = {}
        self._last_yolo26_path: Path | None = None
        self._last_yolo26_identity: tuple[str, int, int, str] | None = None
        self._last_yolo26_result: Yolo26TableStateResult | None = None
        self._last_game_scene_path: Path | None = None
        self._last_game_scene_identity: tuple[str, int, int, str] | None = None
        self._last_game_scene_result: GameSceneResult | None = None
        self._last_table_context_result: TableContextResult | None = None
        self._game_scene_confirmation_frames = 0
        self._game_scene_confirmed = False
        self._river_frames_since_full_scan = 0
        self._river_correction_candidates: dict[str, dict[str, Any]] = {}
        self._pending_self_call_claim: dict[str, Any] = {}
        self._new_round_candidate_frames = 0
        self._new_round_candidate_tiles: list[str] = []
        self._settlement_tracker = SettlementTracker(
            confirm_frames=self.config.settlement_confirm_frames,
            confirm_max_gap_ms=self.config.settlement_confirm_max_gap_ms,
        )
        self._settlement_new_hand_frames = 0
        self._settlement_new_hand_tiles: list[str] = []
        self._pre_settlement_round_phase = ""
        self._pre_settlement_update_reason = ""
        self._auto_round_index = 0
        self._round_history: list[dict[str, Any]] = []
        self._round_archive_index = 0
        self._settlement_archived_for_round = False

    def reset_round(self, round_id: str = "default") -> RoundCoachState:
        self.state = RoundCoachState(round_id=round_id or "default", play_style=self.config.play_style)
        self._last_fingerprints = {}
        self._last_yolo26_path = None
        self._last_yolo26_identity = None
        self._last_yolo26_result = None
        self._last_game_scene_path = None
        self._last_game_scene_identity = None
        self._last_game_scene_result = None
        self._last_table_context_result = None
        self._river_frames_since_full_scan = 0
        self._river_correction_candidates = {}
        self._pending_self_call_claim = {}
        self._clear_new_round_candidate()
        self._settlement_tracker.reset()
        self._clear_settlement_new_hand_candidate()
        self._pre_settlement_round_phase = ""
        self._pre_settlement_update_reason = ""
        self._settlement_archived_for_round = False
        return self.state

    def request_full_rescan(self) -> None:
        # 中文：遮挡恢复后的候选新手牌必须再完整识别一帧，不能被指纹快路径跳过。
        # English: Force one full recognition pass after an obstructed view recovers.
        self._last_fingerprints = {}

    def has_pending_new_round_confirmation(self) -> bool:
        return self._new_round_candidate_frames > 0 or self._settlement_new_hand_frames > 0

    @property
    def round_history(self) -> list[dict[str, Any]]:
        return deepcopy(self._round_history)

    @property
    def last_round_archive(self) -> dict[str, Any]:
        if not self._round_history:
            return {}
        return deepcopy(self._round_history[-1])

    def _handle_settlement_boundary(
        self,
        path: Path | None,
        started: float,
    ) -> CoachDecision | None:
        # 中文：结算门控必须先于手牌/按钮识别，防止结果动画污染上一局状态。
        # English: Gate settlement before hand/button recognition so result animations cannot mutate round state.
        if not bool(getattr(self.config, "settlement_recognition_enabled", True)):
            if self._settlement_tracker.phase != "playing":
                self._settlement_tracker.reset()
                self._clear_settlement_new_hand_candidate()
                self.state.settlement_phase = "playing"
                self.state.settlement_kind = "none"
                self.state.settlement_confidence = 0.0
                self.state.settlement_evidence = []
                self.state.settlement_confirmation_frames = 0
                self.state.round_phase = self._pre_settlement_round_phase or "normal_tracking"
                self.state.last_update_reason = self._pre_settlement_update_reason
                self._pre_settlement_round_phase = ""
                self._pre_settlement_update_reason = ""
            return None

        self._settlement_tracker.confirm_frames = max(
            1,
            int(getattr(self.config, "settlement_confirm_frames", 2) or 2),
        )
        self._settlement_tracker.confirm_max_gap_ms = max(
            200,
            int(getattr(self.config, "settlement_confirm_max_gap_ms", 2500) or 2500),
        )
        frame_result = self._detect_settlement(path)
        previous_phase = self._settlement_tracker.phase
        transition = self._settlement_tracker.observe(
            frame_result,
            round_active=self.state.opening_emitted,
        )

        if previous_phase == "playing" and transition.phase == "settlement_candidate":
            self._pre_settlement_round_phase = self.state.round_phase
            self._pre_settlement_update_reason = self.state.last_update_reason

        if transition.phase == "playing":
            if previous_phase == "settlement_candidate":
                self.state.round_phase = self._pre_settlement_round_phase or "normal_tracking"
                self.state.last_update_reason = self._pre_settlement_update_reason
                self._pre_settlement_round_phase = ""
                self._pre_settlement_update_reason = ""
            self._sync_settlement_state(transition)
            return None

        if transition.phase == "settlement_latched":
            self._archive_current_round(transition)
        self._sync_settlement_state(transition)
        if transition.phase == "awaiting_next_round":
            return self._awaiting_next_round_decision(
                path,
                started,
                transition=transition,
                frame_result=frame_result,
            )
        return self._settlement_boundary_decision(
            started,
            transition=transition,
            frame_result=frame_result,
        )

    def _detect_settlement(self, path: Path | None) -> SettlementFrameResult:
        if path is None:
            return SettlementFrameResult(reason="image_path_missing")
        return detect_settlement_path(
            path,
            min_confidence=float(getattr(self.config, "settlement_min_confidence", 0.72) or 0.72),
        )

    def _sync_settlement_state(self, transition: SettlementTransition) -> None:
        result = transition.result
        self.state.settlement_phase = transition.phase
        self.state.settlement_confirmation_frames = int(transition.confirmation_frames)
        if transition.phase == "playing":
            self.state.settlement_kind = "none"
            self.state.settlement_confidence = 0.0
            self.state.settlement_evidence = []
            return
        self.state.settlement_kind = result.kind
        self.state.settlement_confidence = round(float(result.confidence), 4)
        self.state.settlement_evidence = list(result.evidence)
        self.state.round_phase = transition.phase
        self.state.last_update_reason = transition.phase

    def _settlement_boundary_decision(
        self,
        started: float,
        *,
        transition: SettlementTransition,
        frame_result: SettlementFrameResult,
    ) -> CoachDecision:
        candidate = transition.phase == "settlement_candidate"
        kind_text = _settlement_kind_text(transition.result.kind)
        summary = "检测到结算候选，正在复核" if candidate else f"{kind_text}已确认"
        detail = (
            f"需要连续 {self._settlement_tracker.confirm_frames} 帧；"
            f"当前 {transition.confirmation_frames} 帧，"
            f"帧间隔不超过 {transition.confirm_max_gap_ms} ms，"
            f"置信度 {transition.result.confidence:.0%}。"
            if candidate
            else "上一小局的手牌、牌河和策略已冻结；结算消失后等待稳定的新手牌。"
        )
        return CoachDecision(
            decision_type="settlement_candidate" if candidate else "round_settlement",
            priority=82 if candidate else 92,
            action_required=False,
            summary=summary,
            detail=detail,
            suggestion="暂不刷新策略，保留上一小局结果。",
            hand_tiles=list(self.state.last_hand_tiles),
            reason_codes=[
                "settlement_visual_candidate" if candidate else "settlement_visual_confirmed",
                transition.result.reason,
            ],
            coach_state=self.state.to_dict(),
            perception=self._frozen_settlement_perception(transition, frame_result),
            engine_meta={
                **self._meta(started, "settlement_gate"),
                "settlement_phase": transition.phase,
                "settlement_kind": transition.result.kind,
                "settlement_confirmation_frames": transition.confirmation_frames,
                "settlement_confirmation_elapsed_ms": transition.confirmation_elapsed_ms,
                "settlement_last_frame_gap_ms": transition.last_frame_gap_ms,
                "settlement_confirm_max_gap_ms": transition.confirm_max_gap_ms,
            },
            quiet=not transition.changed,
        )

    def _awaiting_next_round_decision(
        self,
        path: Path | None,
        started: float,
        *,
        transition: SettlementTransition,
        frame_result: SettlementFrameResult,
    ) -> CoachDecision:
        hand_result = self._detect_hand(path)
        meld_result = self._detect_melds(path, hand_result=hand_result)
        current_tiles = [normalize_tile(tile) for tile in hand_result.hand_tiles if normalize_tile(tile)]
        plausible_opening = (
            hand_result.ok
            and 12 <= len(current_tiles) <= 14
            and int(meld_result.open_meld_count or 0) == 0
        )

        if not plausible_opening:
            self._clear_settlement_new_hand_candidate()
        elif self._settlement_new_hand_frames == 0:
            self._settlement_new_hand_tiles = list(current_tiles)
            self._settlement_new_hand_frames = 1
        elif _shared_tile_count(self._settlement_new_hand_tiles, current_tiles) < 10:
            self._settlement_new_hand_tiles = list(current_tiles)
            self._settlement_new_hand_frames = 1
        else:
            self._settlement_new_hand_frames += 1

        if plausible_opening and self._settlement_new_hand_frames >= 2:
            previous_round_id = self.state.round_id
            previous_river_count = _discard_count(self.state.last_discard_piles)
            previous_settlement = transition.result.to_dict()
            previous_archive_id = str(self.last_round_archive.get("archive_id") or "")
            self._auto_round_index += 1
            self.reset_round(f"auto-settlement-round-{self._auto_round_index}")
            self._remember_hand(hand_result)
            if meld_result.ok:
                self._remember_melds(meld_result)
            river_result = RiverStateResult(
                ok=True,
                discard_piles={},
                visible_tiles=[],
                confidence=1.0,
                reason="new_round_empty_river",
            )
            decision = self._opening_decision(
                hand_result,
                river_result,
                started,
                action_meta={"source": "settlement_new_round_detection", "skipped": True},
                meld_result=meld_result,
            )
            self.state.last_update_reason = "auto_new_round_detected"
            return replace(
                decision,
                reason_codes=[
                    *decision.reason_codes,
                    "settlement_closed",
                    "auto_new_round_detected",
                ],
                coach_state=self.state.to_dict(),
                engine_meta={
                    **decision.engine_meta,
                    "round_transition": "settlement_new_hand_confirmed",
                    "previous_round_id": previous_round_id,
                    "previous_river_count": previous_river_count,
                    "current_river_count": 0,
                    "previous_settlement": previous_settlement,
                    "previous_round_archive_id": previous_archive_id,
                    "confirmation_frames": 2,
                },
            )

        settlement_payload = self._settlement_payload(transition, frame_result)
        settlement_payload["new_hand_confirmation_frames"] = self._settlement_new_hand_frames
        settlement_payload["new_hand_tile_count"] = len(current_tiles)
        return CoachDecision(
            decision_type="awaiting_next_round",
            priority=70,
            action_required=False,
            summary="上一小局已结束，等待下一局",
            detail=(
                f"已发现 {len(current_tiles)} 张候选新手牌，"
                f"正在确认第 {self._settlement_new_hand_frames}/2 帧。"
                if plausible_opening
                else "结算画面已消失；尚未看到 12 至 14 张稳定、无副露的新手牌。"
            ),
            suggestion="保留上一局数据，确认新手牌后自动清空并生成新策略。",
            hand_tiles=list(current_tiles),
            reason_codes=["settlement_closed", "awaiting_stable_new_hand"],
            coach_state=self.state.to_dict(),
            perception={
                "settlement": settlement_payload,
                "hand": hand_result.to_dict(),
                "meld": meld_result.to_dict(),
                "action": {"source": "settlement_gate", "skipped": True, "elapsed_ms": 0.0},
                "river": self._river_result_from_state("settlement_state_frozen").to_dict(),
            },
            engine_meta={
                **self._meta(started, "awaiting_next_round"),
                "settlement_phase": transition.phase,
                "settlement_kind": transition.result.kind,
                "new_hand_confirmation_frames": self._settlement_new_hand_frames,
            },
        )

    def _frozen_settlement_perception(
        self,
        transition: SettlementTransition,
        frame_result: SettlementFrameResult,
    ) -> dict[str, Any]:
        return {
            "settlement": self._settlement_payload(transition, frame_result),
            "hand": {
                "ok": bool(self.state.last_hand_tiles),
                "hand_tiles": list(self.state.last_hand_tiles),
                "confidence": self.state.last_hand_confidence,
                "reason": "settlement_state_frozen",
                "elapsed_ms": 0.0,
            },
            "meld": {
                "ok": bool(self.state.last_melds),
                "melds": list(self.state.last_melds),
                "tiles": list(self.state.last_meld_tiles),
                "open_meld_count": self.state.last_open_meld_count,
                "confidence": self.state.last_meld_confidence,
                "reason": "settlement_state_frozen",
                "elapsed_ms": 0.0,
            },
            "action": {"source": "settlement_gate", "skipped": True, "elapsed_ms": 0.0},
            "river": self._river_result_from_state("settlement_state_frozen").to_dict(),
        }

    def _settlement_payload(
        self,
        transition: SettlementTransition,
        frame_result: SettlementFrameResult,
    ) -> dict[str, Any]:
        payload = transition.to_dict()
        payload.update(
            {
                "frame_detected": frame_result.detected,
                "frame_reason": frame_result.reason,
                "frame_confidence": round(float(frame_result.confidence), 4),
                "elapsed_ms": round(float(frame_result.elapsed_ms), 1),
                "required_frames": self._settlement_tracker.confirm_frames,
            }
        )
        if self._settlement_archived_for_round:
            payload["round_archive_id"] = str(self.last_round_archive.get("archive_id") or "")
        return payload

    def _archive_current_round(
        self,
        transition: SettlementTransition,
    ) -> dict[str, Any]:
        # 中文：只在结算确认时快照一次；后续动画反复出现只更新更可靠的结算证据。
        # English: Snapshot once per confirmed round and only refresh stronger settlement evidence.
        if self._settlement_archived_for_round and self._round_history:
            latest = self._round_history[-1]
            previous = latest.get("settlement") if isinstance(latest.get("settlement"), dict) else {}
            if float(transition.result.confidence) >= float(previous.get("confidence") or 0.0):
                latest["settlement"] = transition.result.to_dict()
            return deepcopy(latest)

        self._round_archive_index += 1
        archived_at = time.time()
        state_snapshot = deepcopy(self.state.to_dict())
        state_snapshot.update(
            {
                "round_phase": self._pre_settlement_round_phase or state_snapshot["round_phase"],
                "last_update_reason": (
                    self._pre_settlement_update_reason or state_snapshot["last_update_reason"]
                ),
                "settlement_phase": "playing",
                "settlement_kind": "none",
                "settlement_confidence": 0.0,
                "settlement_evidence": [],
                "settlement_confirmation_frames": 0,
            }
        )
        archive = {
            "archive_id": f"round-archive-{self._round_archive_index}",
            "round_id": self.state.round_id,
            "archived_at": archived_at,
            "settlement": transition.result.to_dict(),
            "state": state_snapshot,
        }
        self._round_history.append(archive)
        del self._round_history[:-_ROUND_HISTORY_LIMIT]
        self._settlement_archived_for_round = True
        return deepcopy(archive)

    def _clear_settlement_new_hand_candidate(self) -> None:
        self._settlement_new_hand_frames = 0
        self._settlement_new_hand_tiles = []

    def _handle_game_scene_gate(
        self,
        path: Path | None,
        started: float,
    ) -> CoachDecision | None:
        result = self._detect_game_scene(path)
        if not result.detected:
            self._game_scene_confirmation_frames = 0
            self._game_scene_confirmed = False
            return self._waiting_for_game_decision(result, started, confirming=False)

        if not self._game_scene_confirmed:
            self._game_scene_confirmation_frames += 1
            if self._game_scene_confirmation_frames < _GAME_SCENE_CONFIRM_FRAMES:
                return self._waiting_for_game_decision(result, started, confirming=True)
            self._game_scene_confirmed = True
        return None

    def _detect_game_scene(self, path: Path | None) -> GameSceneResult:
        if path is None:
            return GameSceneResult(reason="image_path_missing")
        frame_identity = _frame_identity(path)
        if (
            frame_identity is not None
            and self._last_game_scene_identity == frame_identity
            and self._last_game_scene_result is not None
        ):
            return self._last_game_scene_result
        result = detect_game_scene_path(path)
        self._last_game_scene_path = path
        self._last_game_scene_identity = frame_identity
        self._last_game_scene_result = result
        return result

    def _observe_table_context(self, path: ImageSource | None) -> TableContextResult:
        if path is None:
            result = TableContextResult(reason="image_path_missing")
            self._last_table_context_result = result
            return result
        scene = self._detect_game_scene(path)
        if not scene.detected or scene.table_surface is None:
            result = TableContextResult(reason="active_table_not_confirmed")
            self._last_table_context_result = result
            self.state.table_context_pending_signature = ""
            self.state.table_context_pending_frames = 0
            return result
        result = detect_table_context(
            path,
            table_surface_result=scene.table_surface,
        )
        self._last_table_context_result = result
        self.state.table_context_reason = result.reason
        if not result.ok:
            self.state.table_context_pending_signature = ""
            self.state.table_context_pending_frames = 0
            return result

        signature = _table_context_signature(result)
        committed_signature = _committed_table_context_signature(self.state)
        if self.state.player_scores and signature == committed_signature:
            self.state.table_context_confidence = round(float(result.confidence), 4)
            self.state.table_context_pending_signature = ""
            self.state.table_context_pending_frames = 0
            return result
        if signature == self.state.table_context_pending_signature:
            self.state.table_context_pending_frames += 1
        else:
            self.state.table_context_pending_signature = signature
            self.state.table_context_pending_frames = 1
        if self.state.table_context_pending_frames < 2:
            return result

        self.state.player_scores = dict(result.scores)
        self.state.player_ranks = dict(result.ranks)
        if result.honba_count is not None:
            self.state.honba_count = int(result.honba_count)
        if result.riichi_stick_count is not None:
            self.state.table_riichi_stick_count = int(result.riichi_stick_count)
            self.state.last_riichi_stick_count = int(result.riichi_stick_count)
        self.state.table_context_confidence = round(float(result.confidence), 4)
        self.state.table_context_reason = "table_context_confirmed"
        self.state.table_context_pending_signature = ""
        self.state.table_context_pending_frames = 0
        return result

    def _table_context_perception(self) -> dict[str, Any]:
        if self._last_table_context_result is None:
            return {
                "ok": False,
                "reason": self.state.table_context_reason or "table_context_not_scanned",
                "confirmed": bool(self.state.player_scores),
            }
        payload = self._last_table_context_result.to_dict()
        payload["confirmed"] = bool(self.state.player_scores)
        payload["confirmation_frames"] = self.state.table_context_pending_frames
        payload["committed_scores"] = dict(self.state.player_scores)
        payload["committed_ranks"] = dict(self.state.player_ranks)
        return payload

    def _waiting_for_game_decision(
        self,
        result: GameSceneResult,
        started: float,
        *,
        confirming: bool,
    ) -> CoachDecision:
        self.state.round_phase = "waiting_for_game"
        self.state.last_update_reason = (
            "game_scene_confirmation_pending"
            if confirming
            else result.reason or "game_scene_not_detected"
        )
        return CoachDecision(
            decision_type="waiting_for_game",
            priority=2,
            action_required=False,
            summary="牌桌已出现，正在确认" if confirming else "等待进入牌局",
            detail=(
                f"需要连续 {_GAME_SCENE_CONFIRM_FRAMES} 帧同时确认牌桌边缘和中央点数盘；"
                f"当前 {_GAME_SCENE_CONFIRM_FRAMES - 1}/{_GAME_SCENE_CONFIRM_FRAMES} 帧。"
                if confirming
                else "尚未同时看到完整牌桌几何和中央点数盘，手牌、牌河、按钮与策略检测均已暂停。"
            ),
            suggestion="进入正式牌局并保持画面无遮挡后会自动开始。",
            reason_codes=[
                "game_scene_confirmation_pending"
                if confirming
                else "game_scene_not_detected",
                result.reason or "game_scene_unavailable",
            ],
            coach_state=self.state.to_dict(),
            perception={
                "game_scene": result.to_dict(),
                "hand": {"ok": False, "reason": "game_scene_gate"},
                "meld": {"ok": False, "reason": "game_scene_gate"},
                "river": {"ok": False, "reason": "game_scene_gate"},
                "action": {"source": "game_scene_gate", "skipped": True},
            },
            engine_meta={
                **self._meta(started, "game_scene_gate"),
                "game_scene_detected": result.detected,
                "game_scene_confidence": round(float(result.confidence), 4),
                "game_scene_confirmation_frames": self._game_scene_confirmation_frames,
                "game_scene_required_frames": _GAME_SCENE_CONFIRM_FRAMES,
            },
        )

    def analyze_frame(
        self,
        image_path: str | Path | None = None,
        *,
        image: Any | None = None,
        observed_buttons: list[str] | None = None,
        self_turn_index: int | None = None,
        riichi_players: list[str] | None = None,
        force_checkpoint: bool = False,
        require_game_scene: bool = False,
    ) -> CoachDecision:
        started = time.perf_counter()
        path: ImageSource | None = image if image is not None else (Path(image_path) if image_path else None)
        riichi_players = [str(item) for item in (riichi_players or []) if str(item).strip()]

        # Before the first playable hand, scene confirmation is the only
        # perception allowed to run. There is no prior round to settle, and
        # lobby frames must not reach hand/river/button/settlement detectors.
        if require_game_scene and not self.state.opening_emitted:
            scene_decision = self._handle_game_scene_gate(path, started)
            if scene_decision is not None:
                return scene_decision

        settlement_decision = self._handle_settlement_boundary(path, started)
        if settlement_decision is not None:
            return settlement_decision

        # Once a round is active, settlement keeps priority. The scene gate
        # then freezes ordinary perception under menus, lobby pages, or other
        # non-table views without erasing the current round.
        if require_game_scene and self.state.opening_emitted:
            scene_decision = self._handle_game_scene_gate(path, started)
            if scene_decision is not None:
                return scene_decision

        # Scores and counters are read from the already rectified table. They
        # only become strategy inputs after two consecutive identical reads.
        self._observe_table_context(path)

        if not self.state.opening_emitted:
            hand_result = self._detect_hand(path)
            meld_result = self._detect_melds(path, hand_result=hand_result)
            self._detect_riichi_players(path)
            if meld_result.ok:
                self._remember_melds(meld_result)
            if not hand_result.ok and meld_result.ok and _play_style(self.config) == "fast":
                hand_result = self._detect_hand(path, min_hand_tiles=2 if meld_result.open_meld_count >= 3 else 4)
            hand_result = self._accept_plausible_open_hand(hand_result, meld_result)
            if hand_result.ok:
                self._remember_hand(hand_result)
            action_meta = {"source": "opening_hand_scan", "skipped": True}
            if self._uses_live_river_tracking():
                river_result = self._track_river_for_frame(path, "opening_river_tracking", ok_if_cached=False)
            else:
                river_result = self._river_result_from_state("opening_skips_river_scan", ok_if_cached=False)
            if hand_result.ok:
                return self._opening_decision(
                    hand_result,
                    river_result,
                    started,
                    action_meta=action_meta,
                    meld_result=meld_result,
                )
            return self._observe_decision(
                hand_result,
                action_meta,
                river_result,
                started,
                phase="opening_hand_scan",
                meld_result=meld_result,
            )

        buttons, action_meta = self._resolve_buttons(path, observed_buttons)
        critical = [button for button in buttons if button in CRITICAL_BUTTONS]
        win_buttons = [button for button in critical if button in WIN_BUTTONS]
        if self.config.critical_action_interrupts and win_buttons:
            river_result = self._river_result_from_state("action_window_uses_cached_river")
            return self._critical_decision(win_buttons, action_meta, started, river_result=river_result)

        # Fingerprint gate: only skip when the original-frame action/hand
        # regions and every perspective-corrected river lane are unchanged.
        # If table geometry is unavailable, quick_frame_fingerprint fails open
        # and the complete perception pass still runs.
        warped_table = None
        if path is not None and self._cached_game_scene_matches(path):
            table_surface = self._last_game_scene_result.table_surface
            if table_surface is not None and table_surface.ok:
                warped_table = table_surface.warped_image
        fp = (
            quick_frame_fingerprint(path, self._last_fingerprints, warped_table=warped_table)
            if path
            else None
        )
        if fp is not None:
            self._last_fingerprints = fp["hashes"]
        if (
            fp
            and not fp["action_changed"]
            and not fp["hand_changed"]
            and not fp.get("river_changed", True)
            and self._new_round_candidate_frames == 0
        ):
            river_result = self._river_result_from_state("fingerprint_no_change")
            return self._observe_decision(
                FastHandResult(reason="fingerprint_match"),
                action_meta,
                river_result,
                started,
                phase="fingerprint_no_change",
            )

        previous_hand_tiles = list(self.state.last_hand_tiles)
        hand_result, meld_result = self._detect_and_remember(path)
        round_transition = self._maybe_confirm_yolo26_new_round(
            path=path,
            previous_hand_tiles=previous_hand_tiles,
            hand_result=hand_result,
            meld_result=meld_result,
            started=started,
        )
        if round_transition is not None:
            return round_transition

        call_buttons = [button for button in critical if button in CALL_BUTTONS]
        riichi_buttons = [button for button in critical if button == "riichi"]
        if self.config.critical_action_interrupts and (call_buttons or riichi_buttons):
            river_result = self._river_result_from_state("action_window_uses_cached_river")
            if call_buttons and self._uses_live_river_tracking():
                river_result, claimed_tile = self._call_window_river(path, call_buttons)
                if claimed_tile:
                    action_meta = {
                        **action_meta,
                        "claimed_tile": claimed_tile,
                        "claimed_tile_source": "river_delta",
                    }
            return self._critical_decision(
                [*call_buttons, *riichi_buttons],
                action_meta,
                started,
                hand_result=hand_result,
                meld_result=meld_result,
                river_result=river_result,
            )

        river_result = self._track_river_for_frame(path, "live_river_tracking") if self._uses_live_river_tracking() else None

        if riichi_players:
            self.state.riichi_players = list(riichi_players)
        elif self.state.opening_emitted and self.state.update_count >= 2:
            riichi_players = self._detect_riichi_players(path)
            if riichi_players:
                self.state.riichi_players = list(riichi_players)

        if riichi_players:
            if river_result is None:
                river_result = self._detect_river(path)
                if river_result.ok:
                    self._remember_river(river_result)
            return self._defense_decision(riichi_players, hand_result, river_result, started, meld_result=meld_result)

        turn_number = _coerce_turn(self_turn_index)
        if hand_result.ok and self._checkpoint_due(turn_number, force_checkpoint=force_checkpoint):
            if river_result is None:
                river_result = self._detect_river(path)
                if river_result.ok:
                    self._remember_river(river_result)
            return self._checkpoint_decision(hand_result, river_result, turn_number, force_checkpoint, started, meld_result=meld_result)

        if river_result is None:
            river_result = self._river_result_from_state("river_scan_not_due")
        return self._observe_decision(hand_result, action_meta, river_result, started, phase="normal_tracking", meld_result=meld_result)

    def _observe_decision(
        self,
        hand_result: FastHandResult,
        action_meta: dict[str, Any],
        river_result: RiverStateResult,
        started: float,
        *,
        phase: str,
        meld_result: MeldStateResult | None = None,
    ) -> CoachDecision:
        self.state.round_phase = phase
        summary, detail, reason_codes = self._observe_message(hand_result)
        if hand_result.ok and hand_result.hand_tiles:
            self.state.last_hand_tiles = [normalize_tile(t) for t in hand_result.hand_tiles if normalize_tile(t)]
        return CoachDecision(
            decision_type="observe",
            priority=5,
            action_required=False,
            summary=summary,
            detail=detail,
            suggestion=self.state.current_plan,
            hand_tiles=list(hand_result.hand_tiles),
            reason_codes=reason_codes,
            coach_state=self.state.to_dict(),
            perception={
                "table_context": self._table_context_perception(),
                "hand": hand_result.to_dict(),
                "action": action_meta,
                "river": river_result.to_dict(),
                "meld": meld_result.to_dict() if meld_result is not None else {},
            },
            engine_meta=self._meta(started, "observe"),
        )

    def _resolve_buttons(
        self,
        path: Path | None,
        observed_buttons: list[str] | None,
    ) -> tuple[list[str], dict[str, Any]]:
        normalized = _normalize_buttons(observed_buttons)
        meta: dict[str, Any] = {
            "source": "provided" if normalized else "fast_color_scan",
            "provided_buttons": list(normalized),
        }
        if normalized or path is None:
            return normalized, meta
        detected, metrics = detect_action_buttons_fast(path)
        meta.update({"detected_buttons": detected, "metrics": metrics})
        if isinstance(metrics, dict) and "elapsed_ms" in metrics:
            # 展平按钮扫描耗时，供插件 UI 的运行日志直接显示。
            # Flatten action scan timing so the plugin UI can render it directly.
            meta["elapsed_ms"] = metrics.get("elapsed_ms")
        return _normalize_buttons(detected), meta

    def _detect_hand(self, path: Path | None, *, min_hand_tiles: int | None = None) -> FastHandResult:
        if path is None:
            return FastHandResult(reason="image_path_missing")
        if self._uses_yolo26_tiles():
            yolo_result = self._detect_yolo26_table(path)
            yolo_hand = yolo_result.to_hand_result(
                min_hand_tiles=max(1, int(min_hand_tiles or self._min_hand_tiles_for_scan()))
            )
            if yolo_hand.ok:
                return yolo_hand
            fallback = detect_fast_hand_path(
                path,
                calibration_dir=self.calibration_dir,
                min_hand_tiles=max(1, int(min_hand_tiles or self._min_hand_tiles_for_scan())),
                use_onnx_hand=self.config.onnx_hand_enabled,
            )
            return replace(
                fallback,
                analysis_hints={
                    **fallback.analysis_hints,
                    "tile_recognition_mode": "yolo26",
                    "fallback_mode": "legacy",
                    "fallback_reason": yolo_hand.reason or yolo_result.reason or "yolo26_unavailable",
                    "yolo26": yolo_result.to_dict(),
                },
            )
        return detect_fast_hand_path(
            path,
            calibration_dir=self.calibration_dir,
            min_hand_tiles=max(1, int(min_hand_tiles or self._min_hand_tiles_for_scan())),
            use_onnx_hand=self.config.onnx_hand_enabled,
        )

    def _min_hand_tiles_for_scan(self) -> int:
        if not self.state.opening_emitted:
            return 12
        if _play_style(self.config) != "fast":
            return 12
        last_count = len([tile for tile in self.state.last_hand_tiles if normalize_tile(tile)])
        return 2 if 0 < last_count <= 5 else 4

    def _detect_and_remember(self, path: Path | None) -> tuple[FastHandResult, MeldStateResult]:
        hand_result = self._detect_hand(path)
        meld_result = self._detect_melds(path, hand_result=hand_result)
        if meld_result.ok:
            self._remember_melds(meld_result)
        hand_result = self._accept_plausible_open_hand(hand_result, meld_result)
        if hand_result.ok:
            self._remember_hand(hand_result)
        return hand_result, meld_result

    def _maybe_confirm_yolo26_new_round(
        self,
        *,
        path: Path | None,
        previous_hand_tiles: list[str],
        hand_result: FastHandResult,
        meld_result: MeldStateResult,
        started: float,
    ) -> CoachDecision | None:
        # 中文：新局必须同时满足“旧牌河明显存在、当前牌河归零、新手牌稳定”，并连续两帧确认。
        # English: Confirm a new round for two frames using an old populated river, a reset river, and a stable new hand.
        if not self._uses_yolo26_tiles() or path is None or not hand_result.ok:
            self._clear_new_round_candidate()
            return None

        previous_river_count = _discard_count(self.state.last_discard_piles)
        current_tiles = [normalize_tile(tile) for tile in hand_result.hand_tiles if normalize_tile(tile)]
        yolo_result = self._detect_yolo26_table(path)
        current_river_count = _discard_count(yolo_result.discard_piles)
        river_ok = yolo_result.ok if yolo_result.river_inference_ok is None else yolo_result.river_inference_ok
        plausible_opening = (
            river_ok
            and 12 <= len(current_tiles) <= 14
            and int(meld_result.open_meld_count or 0) == 0
            and previous_river_count >= 4
            and current_river_count <= 2
        )
        if not plausible_opening:
            self._clear_new_round_candidate()
            return None

        if self._new_round_candidate_frames == 0:
            shared_with_previous = _shared_tile_count(previous_hand_tiles, current_tiles)
            if previous_hand_tiles and shared_with_previous > 9:
                return None
            self._new_round_candidate_tiles = list(current_tiles)
            self._new_round_candidate_frames = 1
            return None

        shared_with_candidate = _shared_tile_count(self._new_round_candidate_tiles, current_tiles)
        if shared_with_candidate < 10:
            self._new_round_candidate_tiles = list(current_tiles)
            self._new_round_candidate_frames = 1
            return None

        self._new_round_candidate_frames += 1
        if self._new_round_candidate_frames < 2:
            return None

        raw_river = yolo_result.to_river_result()
        self._auto_round_index += 1
        self.reset_round(f"auto-round-{self._auto_round_index}")
        self._remember_hand(hand_result)
        if meld_result.ok:
            self._remember_melds(meld_result)
        if raw_river.ok:
            self._remember_river(raw_river)
        decision = self._opening_decision(
            hand_result,
            raw_river,
            started,
            action_meta={"source": "auto_new_round_detection", "skipped": True},
            meld_result=meld_result,
        )
        self.state.last_update_reason = "auto_new_round_detected"
        return replace(
            decision,
            reason_codes=[*decision.reason_codes, "auto_new_round_detected"],
            coach_state=self.state.to_dict(),
            engine_meta={
                **decision.engine_meta,
                "round_transition": "auto_new_round_detected",
                "previous_river_count": previous_river_count,
                "current_river_count": current_river_count,
                "confirmation_frames": 2,
            },
        )

    def _clear_new_round_candidate(self) -> None:
        self._new_round_candidate_frames = 0
        self._new_round_candidate_tiles = []

    def _accept_plausible_open_hand(
        self,
        hand_result: FastHandResult,
        meld_result: MeldStateResult | None = None,
    ) -> FastHandResult:
        if hand_result.ok:
            return hand_result
        if str(hand_result.reason or "") != "unstable_hand_count":
            return hand_result
        count = len([tile for tile in hand_result.hand_tiles if normalize_tile(tile)])
        inferred_open_melds = _inferred_open_melds_from_closed_count(count)
        if inferred_open_melds <= 0:
            return hand_result
        if meld_result is not None and meld_result.ok:
            inferred_open_melds = max(inferred_open_melds, int(meld_result.open_meld_count or 0))
        self.state.last_open_meld_count = max(self.state.last_open_meld_count, inferred_open_melds)
        return replace(
            hand_result,
            ok=True,
            reason=f"inferred_open_{count}_hand_tiles",
            confidence=round(float(hand_result.confidence) * 0.96, 4),
        )

    def _detect_river(self, path: Path | None) -> RiverStateResult:
        if not self.config.river_recognition_enabled:
            return RiverStateResult(reason="river_recognition_disabled")
        if path is None:
            return RiverStateResult(reason="image_path_missing")
        if self._uses_yolo26_tiles():
            yolo_result = self._detect_yolo26_table(path)
            yolo_river = yolo_result.to_river_result()
            if yolo_river.ok:
                return yolo_river
            fallback = detect_river_state_path(
                path,
                calibration_dir=self.calibration_dir,
                min_confidence=self.config.river_min_confidence,
            )
            return replace(
                fallback,
                analysis_hints={
                    **fallback.analysis_hints,
                    "tile_recognition_mode": "yolo26",
                    "fallback_mode": "legacy",
                    "fallback_reason": yolo_river.reason or yolo_result.reason or "yolo26_unavailable",
                    "yolo26": yolo_result.to_dict(),
                },
            )
        return detect_river_state_path(
            path,
            calibration_dir=self.calibration_dir,
            min_confidence=self.config.river_min_confidence,
        )

    def _track_river_for_frame(
        self,
        path: Path | None,
        reason: str,
        *,
        ok_if_cached: bool = True,
    ) -> RiverStateResult:
        # 中文：实时模式先建立全量快照；YOLO 每帧复核，legacy 增量检测并定期做全量纠错。
        # English: Bootstrap a full snapshot; reconcile YOLO every frame and audit legacy tracking periodically.
        if self._uses_live_river_tracking() and self.state.river_tracking_initialized:
            if self._uses_yolo26_tiles():
                full_result = self._detect_river(path)
                if full_result.ok:
                    reconciled = self._reconcile_full_river(
                        full_result,
                        confirm_new=reason != "call_window_river_delta",
                    )
                    self._river_frames_since_full_scan = 0
                    self._remember_river(reconciled)
                    return reconciled
                cached = self._river_result_from_state(full_result.reason or reason, ok_if_cached=ok_if_cached)
                if cached.ok and full_result.elapsed_ms:
                    return replace(cached, elapsed_ms=full_result.elapsed_ms)
                return full_result

            self._river_frames_since_full_scan += 1
            if self._river_frames_since_full_scan >= _RIVER_FULL_RESCAN_INTERVAL:
                full_result = self._detect_river(path)
                if full_result.ok:
                    reconciled = self._reconcile_full_river(full_result)
                    self._river_frames_since_full_scan = 0
                    self._remember_river(reconciled)
                    return reconciled
                cached = self._river_result_from_state(full_result.reason or reason, ok_if_cached=ok_if_cached)
                if cached.ok and full_result.elapsed_ms:
                    return replace(cached, elapsed_ms=full_result.elapsed_ms)
                return full_result

            delta_result = self._detect_incremental_river(path)
            if delta_result.ok:
                merged = self._merge_incremental_river(delta_result)
                self._remember_river(merged)
                return merged
            cached = self._river_result_from_state(delta_result.reason or reason, ok_if_cached=ok_if_cached)
            if cached.ok and delta_result.elapsed_ms:
                return replace(cached, elapsed_ms=delta_result.elapsed_ms)
            return delta_result

        river_result = self._detect_river(path)
        if river_result.ok:
            if self._uses_live_river_tracking():
                self._river_frames_since_full_scan = 0
                self._river_correction_candidates = {}
            self._remember_river(river_result)
            if river_result.reason:
                return river_result
            return replace(river_result, reason=reason)
        cached = self._river_result_from_state(river_result.reason or reason, ok_if_cached=ok_if_cached)
        if cached.ok and river_result.elapsed_ms:
            return replace(cached, elapsed_ms=river_result.elapsed_ms)
        return river_result

    def _detect_incremental_river(self, path: Path | None) -> RiverStateResult:
        if path is None:
            return RiverStateResult(reason="image_path_missing")
        if self._uses_yolo26_tiles():
            # YOLO currently emits a whole-table detection list. Keep its existing
            # fallback path until its runtime exposes slot-level incremental crops.
            return self._detect_river(path)
        return detect_incremental_river_state_path(
            path,
            self.state.last_discard_piles,
            calibration_dir=self.calibration_dir,
            min_confidence=self.config.river_min_confidence,
        )

    def _merge_incremental_river(self, delta_result: RiverStateResult) -> RiverStateResult:
        merged_piles = {
            player: [dict(item) for item in items]
            for player, items in self.state.last_discard_piles.items()
        }
        opponent_melds = _merge_opponent_meld_snapshots(
            self.state.last_opponent_melds,
            delta_result.opponent_melds,
        )
        opponent_meld_tiles = _opponent_meld_tiles(opponent_melds)
        appended: list[dict[str, Any]] = []
        for player, items in delta_result.discard_piles.items():
            target = merged_piles.setdefault(player, [])
            known_slots = {
                str(item.get("slot_id") or f"{player}:{item.get('turn_index')}")
                for item in target
                if isinstance(item, dict)
            }
            for item in items:
                if not isinstance(item, dict):
                    continue
                slot_key = str(item.get("slot_id") or f"{player}:{item.get('turn_index')}")
                if slot_key in known_slots:
                    continue
                target.append(dict(item))
                known_slots.add(slot_key)
                appended.append(dict(item))
            target.sort(key=lambda item: int(item.get("turn_index") or 0))

        visible_tiles = [
            str(item.get("tile") or "")
            for items in merged_piles.values()
            for item in items
            if isinstance(item, dict) and str(item.get("tile") or "")
        ]
        confidences = [
            float(item.get("confidence") or 0.0)
            for items in merged_piles.values()
            for item in items
            if isinstance(item, dict) and float(item.get("confidence") or 0.0) > 0.0
        ]
        hints = {
            **delta_result.analysis_hints,
            "incremental": True,
            "river_full_rescan": False,
            "river_corrected_count": 0,
            "river_pending_corrections": len(self._river_correction_candidates),
            "new_discard_count": len(appended),
            "cached_discard_count": len(visible_tiles),
        }
        return RiverStateResult(
            ok=True,
            discard_piles=merged_piles,
            visible_tiles=visible_tiles,
            opponent_melds=opponent_melds,
            opponent_meld_tiles=opponent_meld_tiles,
            confidence=round(sum(confidences) / len(confidences), 4) if confidences else 0.0,
            reason=delta_result.reason,
            elapsed_ms=delta_result.elapsed_ms,
            raw_detections=delta_result.raw_detections,
            analysis_hints=hints,
        )

    def _reconcile_full_river(
        self,
        snapshot: RiverStateResult,
        *,
        confirm_new: bool = True,
    ) -> RiverStateResult:
        """Reconcile a full snapshot with stable history instead of replacing it blindly."""
        merged_piles = {
            player: [dict(item) for item in items]
            for player, items in self.state.last_discard_piles.items()
        }
        opponent_melds = _merge_opponent_meld_snapshots(
            self.state.last_opponent_melds,
            snapshot.opponent_melds,
        )
        active_candidate_keys: set[str] = set()
        corrected: list[dict[str, Any]] = []
        appended: list[dict[str, Any]] = []

        for player, current_items in snapshot.discard_piles.items():
            target = merged_piles.setdefault(player, [])
            matched_indices: set[int] = set()
            for raw_item in current_items:
                if not isinstance(raw_item, dict) or not str(raw_item.get("tile") or ""):
                    continue
                current = dict(raw_item)
                match_index = _match_river_item_index(target, current, matched_indices)
                if match_index is None:
                    candidate_key = _river_candidate_key("append", player, current)
                    if confirm_new:
                        active_candidate_keys.add(candidate_key)
                        confirmations = self._advance_river_candidate(candidate_key, "append", player, current)
                        if confirmations < _RIVER_CORRECTION_CONFIRM_FRAMES:
                            continue
                    new_item = dict(current)
                    new_item["player"] = player
                    new_item["turn_index"] = max(
                        [int(item.get("turn_index") or 0) for item in target if isinstance(item, dict)] or [0]
                    ) + 1
                    target.append(new_item)
                    appended.append(dict(new_item))
                    self._river_correction_candidates.pop(candidate_key, None)
                    continue

                matched_indices.add(match_index)
                cached = target[match_index]
                candidate_key = _river_candidate_key("replace", player, cached)
                if str(cached.get("tile") or "") == str(current.get("tile") or ""):
                    target[match_index] = _refresh_river_item(cached, current)
                    self._river_correction_candidates.pop(candidate_key, None)
                    continue

                active_candidate_keys.add(candidate_key)
                confirmations = self._advance_river_candidate(candidate_key, "replace", player, current)
                if confirmations < _RIVER_CORRECTION_CONFIRM_FRAMES:
                    continue
                replacement = _refresh_river_item(cached, current)
                replacement["corrected_from"] = str(cached.get("tile") or "")
                replacement["correction_confirmations"] = confirmations
                replacement["correction_source"] = str(current.get("source") or snapshot.reason or "full_rescan")
                target[match_index] = replacement
                corrected.append(
                    {
                        "player": player,
                        "turn_index": replacement.get("turn_index"),
                        "from": cached.get("tile"),
                        "to": replacement.get("tile"),
                        "confidence": replacement.get("confidence"),
                    }
                )
                self._river_correction_candidates.pop(candidate_key, None)

            target.sort(key=lambda item: int(item.get("turn_index") or 0))

        # 中文：候选必须在相邻的全量复核中持续出现；消失或改口都会重新计数。
        # English: Corrections must survive consecutive full audits; missing or changed evidence resets them.
        for candidate_key in list(self._river_correction_candidates):
            if candidate_key not in active_candidate_keys:
                self._river_correction_candidates.pop(candidate_key, None)

        claimed_discard_events = _link_claimed_discards(
            merged_piles,
            snapshot.discard_piles,
            opponent_melds,
        )
        opponent_meld_tiles = _opponent_meld_tiles(opponent_melds)
        physical_discard_tiles = _physical_discard_tiles(merged_piles)
        visible_tiles = [
            str(item.get("tile") or "")
            for items in merged_piles.values()
            for item in items
            if isinstance(item, dict) and str(item.get("tile") or "")
        ]
        confidences = [
            float(item.get("confidence") or 0.0)
            for items in merged_piles.values()
            for item in items
            if isinstance(item, dict) and float(item.get("confidence") or 0.0) > 0.0
        ]
        hints = {
            **snapshot.analysis_hints,
            "incremental": False,
            "river_full_rescan": True,
            "river_corrected_count": len(corrected),
            "river_pending_corrections": len(self._river_correction_candidates),
            "river_correction_confirm_frames": _RIVER_CORRECTION_CONFIRM_FRAMES,
            "river_correction_events": corrected,
            "new_discard_count": len(appended),
            "cached_discard_count": len(visible_tiles),
            "physical_visible_discard_count": len(physical_discard_tiles),
            "claimed_discard_link_count": len(claimed_discard_events),
            "claimed_discard_events": claimed_discard_events,
        }
        return RiverStateResult(
            ok=True,
            discard_piles=merged_piles,
            visible_tiles=visible_tiles,
            opponent_melds=opponent_melds,
            opponent_meld_tiles=opponent_meld_tiles,
            confidence=round(sum(confidences) / len(confidences), 4) if confidences else 0.0,
            reason="river_corrected" if corrected else snapshot.reason,
            elapsed_ms=snapshot.elapsed_ms,
            raw_detections=snapshot.raw_detections,
            analysis_hints=hints,
        )

    def _advance_river_candidate(
        self,
        candidate_key: str,
        kind: str,
        player: str,
        item: dict[str, Any],
    ) -> int:
        previous = self._river_correction_candidates.get(candidate_key, {})
        same_evidence = (
            previous.get("kind") == kind
            and previous.get("player") == player
            and previous.get("tile") == item.get("tile")
        )
        confirmations = int(previous.get("confirmations") or 0) + 1 if same_evidence else 1
        self._river_correction_candidates[candidate_key] = {
            "kind": kind,
            "player": player,
            "tile": item.get("tile"),
            "confirmations": confirmations,
            "item": dict(item),
        }
        return confirmations

    def _call_window_river(self, path: Path | None, buttons: list[str]) -> tuple[RiverStateResult, str]:
        """Refresh a live river once and accept a call tile only when its delta is unique."""
        previous_piles = {
            player: [dict(item) for item in items]
            for player, items in self.state.last_discard_piles.items()
        }
        river_result = self._track_river_for_frame(path, "call_window_river_delta")
        if not river_result.ok:
            return river_result, ""
        claim = _claimed_discard_from_river_delta(
            previous_piles,
            river_result.discard_piles,
            buttons,
        )
        claimed_tile = str(claim.get("tile") or "")
        if claimed_tile:
            self._pending_self_call_claim = {
                **claim,
                "buttons": list(_normalize_buttons(buttons)),
                "observed_at": time.monotonic(),
                "previous_open_meld_count": int(self.state.last_open_meld_count or 0),
            }
        else:
            self._pending_self_call_claim = {}
        return river_result, claimed_tile

    def _uses_live_river_tracking(self) -> bool:
        return str(getattr(self.config, "river_tracking_mode", "checkpoint") or "checkpoint").lower() == "live"

    def _detect_riichi_players(self, path: Path | None) -> list[str]:
        if not self.config.opponent_riichi_recognition_enabled:
            self.state.riichi_pending = {}
            return []
        if path is None:
            return []

        # The top-left counter beside the dora display is not a declaration
        # marker on the current table. It can remain non-zero across hands, so
        # using it here invents a persistent ``unknown`` riichi player. Player
        # state must come from a seat-specific declaration tile in the river.
        detected: set[str] = set()
        if self._uses_yolo26_tiles():
            yolo_result = self._detect_yolo26_table(path)
            river_ok = yolo_result.ok if yolo_result.river_inference_ok is None else yolo_result.river_inference_ok
            if river_ok:
                detected.update(player for player in yolo_result.riichi_players if _player_key(player))
            # Drop stale ``unknown`` state produced by the old counter-based
            # path as soon as seat-aware recognition is active.
            self.state.riichi_players = [
                player for player in self.state.riichi_players if _player_key(player)
            ]

        confirmed: list[str] = []
        pending = dict(self.state.riichi_pending)
        for player in detected:
            pending[player] = pending.get(player, 0) + 1
            if pending[player] >= 2:
                confirmed.append(player)
        for player in list(pending):
            if player not in detected:
                pending.pop(player, None)
        self.state.riichi_pending = pending
        return sorted(set(confirmed) | set(self.state.riichi_players))

    def _detect_melds(self, path: Path | None, *, hand_result: FastHandResult | None = None) -> MeldStateResult:
        if not self.config.meld_recognition_enabled:
            return MeldStateResult(reason="meld_recognition_disabled")
        if path is None:
            return MeldStateResult(reason="image_path_missing")
        if self._uses_yolo26_tiles():
            yolo_result = self._detect_yolo26_table(path)
            original_ok = yolo_result.ok if yolo_result.original_inference_ok is None else yolo_result.original_inference_ok
            if original_ok:
                return yolo_result.to_meld_result()
            closed_hand_count = len(hand_result.hand_tiles) if hand_result is not None and hand_result.hand_tiles else None
            fallback = detect_meld_state_path(
                path,
                min_confidence=self.config.meld_min_confidence,
                closed_hand_count=closed_hand_count,
            )
            return replace(
                fallback,
                analysis_hints={
                    **fallback.analysis_hints,
                    "tile_recognition_mode": "yolo26",
                    "fallback_mode": "legacy",
                    "fallback_reason": yolo_result.reason or "yolo26_unavailable",
                    "yolo26": yolo_result.to_dict(),
                },
            )
        closed_hand_count = len(hand_result.hand_tiles) if hand_result is not None and hand_result.hand_tiles else None
        return detect_meld_state_path(
            path,
            min_confidence=self.config.meld_min_confidence,
            closed_hand_count=closed_hand_count,
        )

    def _uses_yolo26_tiles(self) -> bool:
        return str(getattr(self.config, "tile_recognition_mode", "legacy") or "legacy").lower() == "yolo26"

    def _detect_yolo26_table(self, path: Path) -> Yolo26TableStateResult:
        frame_identity = _frame_identity(path)
        if (
            self._last_yolo26_result is not None
            and (
                self._last_yolo26_identity == frame_identity
                if frame_identity is not None
                else self._last_yolo26_identity is None and self._last_yolo26_path == path
            )
        ):
            return self._last_yolo26_result
        table_surface = None
        if self._cached_game_scene_matches(path, frame_identity=frame_identity):
            table_surface = self._last_game_scene_result.table_surface
        result = detect_yolo26_table_state_path(path, table_surface_result=table_surface)
        self._last_yolo26_path = path
        self._last_yolo26_identity = frame_identity
        self._last_yolo26_result = result
        return result

    def _cached_game_scene_matches(
        self,
        path: Path,
        *,
        frame_identity: tuple[str, int, int, str] | None = None,
    ) -> bool:
        if self._last_game_scene_result is None:
            return False
        identity = frame_identity if frame_identity is not None else _frame_identity(path)
        if identity is None:
            return self._last_game_scene_identity is None and self._last_game_scene_path == path
        return self._last_game_scene_identity == identity

    def _river_result_from_state(self, reason: str, *, ok_if_cached: bool = True) -> RiverStateResult:
        has_cached = bool(
            self.state.river_tracking_initialized
            or self.state.last_discard_piles
            or self.state.last_visible_discards
            or self.state.last_opponent_melds
        )
        return RiverStateResult(
            ok=ok_if_cached and has_cached,
            discard_piles={
                player: [dict(item) for item in items]
                for player, items in self.state.last_discard_piles.items()
            },
            visible_tiles=list(self.state.last_visible_discards),
            opponent_melds={
                owner: [deepcopy(item) for item in items]
                for owner, items in self.state.last_opponent_melds.items()
            },
            opponent_meld_tiles=list(self.state.last_opponent_meld_tiles),
            confidence=float(self.state.last_river_confidence),
            reason=reason,
        )

    def _observe_message(self, hand_result: FastHandResult) -> tuple[str, str, list[str]]:
        if hand_result.ok:
            if self.state.current_plan:
                return (
                    "当前主线继续有效",
                    "继续按当前主线推进；等三巡、手牌结构明显变化、或出现立直/和牌压力时再看主线。",
                    ["coach_observe", "current_plan_active"],
                )
            return (
                "Watching round state",
                "No critical action or coach checkpoint is due.",
                ["coach_observe"],
            )
        accepted = sum(1 for item in hand_result.raw_detections if item.get("accepted"))
        occupied = sum(1 for item in hand_result.raw_detections if item.get("occupied"))
        reason = str(hand_result.reason or "hand_unavailable")
        if reason == "image_path_missing":
            detail = "No screenshot path was provided yet."
        elif reason == "image_missing":
            detail = "The screenshot file could not be read."
        elif reason == "missing_hand_tile_templates":
            detail = "No exact or scaled legacy hand-template profile fits this screenshot yet."
        elif accepted == 0:
            detail = "No stable hand tiles were detected; live capture may be grabbing the desktop, menu, or a covered game window."
        else:
            min_tiles = self._min_hand_tiles_for_scan()
            expected = f"at least {min_tiles} open-hand tiles" if min_tiles <= 4 else "12-14 stable tiles"
            detail = f"Hand scan accepted {accepted} tiles from {occupied} occupied-looking slots; waiting for {expected}."
        return "Waiting for stable hand", detail, ["coach_observe", f"hand_{reason}"]

    def _remember_hand(self, hand_result: FastHandResult) -> None:
        tiles = [normalize_tile(tile) for tile in hand_result.hand_tiles if normalize_tile(tile)]
        signature = hand_signature(tiles)
        self.state.last_hand_signature = signature
        self.state.last_hand_tiles = tiles
        self.state.last_hand_confidence = float(hand_result.confidence)

    def _remember_river(self, river_result: RiverStateResult) -> None:
        self.state.last_discard_piles = {
            player: [dict(item) for item in items]
            for player, items in river_result.discard_piles.items()
        }
        self.state.last_visible_discards = list(river_result.visible_tiles)
        self.state.last_opponent_melds = {
            owner: [deepcopy(item) for item in items]
            for owner, items in river_result.opponent_melds.items()
        }
        self.state.last_opponent_meld_tiles = list(river_result.opponent_meld_tiles)
        self.state.last_river_confidence = float(river_result.confidence)
        self.state.river_tracking_initialized = bool(river_result.ok)

    def _remember_melds(self, meld_result: MeldStateResult) -> None:
        melds = [deepcopy(item) for item in meld_result.melds]
        tile_identity_reliable = meld_result.analysis_hints.get("tile_identity_reliable") is not False
        if tile_identity_reliable:
            self._link_pending_self_call(melds)
        self.state.last_melds = melds
        self.state.last_meld_tiles = [str(tile) for tile in meld_result.tiles if str(tile).strip()] if tile_identity_reliable else []
        self.state.last_open_meld_count = meld_result.open_meld_count
        self.state.last_meld_confidence = float(meld_result.confidence)

    def _link_pending_self_call(self, melds: list[dict[str, Any]]) -> None:
        pending = dict(self._pending_self_call_claim)
        if not pending:
            return
        observed_at = float(pending.get("observed_at") or 0.0)
        if observed_at <= 0.0 or time.monotonic() - observed_at > _SELF_CALL_CLAIM_MAX_AGE_SECONDS:
            self._pending_self_call_claim = {}
            return
        previous_count = int(pending.get("previous_open_meld_count") or 0)
        if len(melds) <= previous_count:
            return
        tile = _canonical_tile(str(pending.get("tile") or ""))
        if tile not in TILE_TYPES:
            self._pending_self_call_claim = {}
            return
        new_melds = melds[previous_count:]
        meld = next(
            (
                item
                for item in new_melds
                if tile in {
                    _canonical_tile(str(value or ""))
                    for value in item.get("tiles", [])
                }
            ),
            None,
        )
        if meld is None:
            return
        candidates = _pending_self_call_discard_candidates(
            self.state.last_discard_piles,
            pending,
        )
        if not candidates:
            return
        source_player, discard = candidates[0]
        buttons = [str(button) for button in pending.get("buttons", [])]
        kind = next((button for button in ("kan", "pon", "chi") if button in buttons), "unknown")
        meld.setdefault("kind", kind)
        _mark_discard_claimed(
            discard,
            source_player=source_player,
            caller="self",
            meld=meld,
            candidate_count=len(candidates),
            source="self_call_window_river_delta",
        )
        self._pending_self_call_claim = {}

    def _critical_decision(
        self,
        buttons: list[str],
        action_meta: dict[str, Any],
        started: float,
        hand_result: FastHandResult | None = None,
        meld_result: MeldStateResult | None = None,
        river_result: RiverStateResult | None = None,
    ) -> CoachDecision:
        if any(button in WIN_BUTTONS for button in buttons):
            summary = "和牌窗口"
            suggestion = "看到荣和/自摸直接点，不需要等待策略分析。"
            decision_type = "win_window"
            priority = 100
        elif any(button in CALL_BUTTONS for button in buttons):
            summary = "吃碰杠窗口"
            suggestion = self._call_suggestion(
                hand_result,
                buttons,
                river_result,
                meld_result=meld_result,
                claimed_tile=_claimed_tile_from_action(action_meta),
            )
            decision_type = "call_window"
            priority = 95
        elif "riichi" in buttons:
            summary = "立直窗口"
            suggestion = self._riichi_suggestion(hand_result, river_result)
            decision_type = "riichi_window"
            priority = 90
        else:
            summary = "操作窗口"
            suggestion = "先处理当前按钮，再回到局面策略。"
            decision_type = "action_window"
            priority = 80
        self.state.round_phase = "action_window"
        self.state.last_update_reason = decision_type
        self.state.update_count += 1
        return CoachDecision(
            decision_type=decision_type,
            priority=priority,
            action_required=True,
            summary=summary,
            detail="动作窗口只使用当前策略与本地快判。",
            suggestion=suggestion,
            buttons=list(buttons),
            hand_tiles=list(hand_result.hand_tiles) if hand_result else [],
            reason_codes=["critical_action_interrupt"],
            coach_state=self.state.to_dict(),
            perception={
                "table_context": self._table_context_perception(),
                "action": action_meta,
                "hand": hand_result.to_dict() if hand_result else {},
                "meld": meld_result.to_dict() if meld_result else {},
                "river": river_result.to_dict() if river_result else {},
            },
            engine_meta=self._meta(started, decision_type),
        )

    def _call_suggestion(
        self,
        hand_result: FastHandResult | None,
        buttons: list[str] | None = None,
        river_result: RiverStateResult | None = None,
        meld_result: MeldStateResult | None = None,
        claimed_tile: str = "",
    ) -> str:
        plan = self.state.current_plan or self.state.opening_plan
        if not plan and hand_result is not None and hand_result.ok:
            built = build_round_plan(
                hand_result.hand_tiles,
                self.config,
                visible_tiles=_visible_tiles_for_plan(river_result),
                open_melds=self._open_meld_count_for_plan(meld_result),
                meld_tiles=self._meld_tiles_for_plan(meld_result),
            )
            plan = built["summary"]
            self.state.opening_emitted = True
            self.state.round_phase = "opening_strategy"
            self._remember_local_plan(built, opening=True)
        if hand_result is not None and hand_result.ok:
            call_analysis = analyze_call_options(
                hand_result.hand_tiles,
                self.config,
                buttons or [],
                claimed_tile=claimed_tile,
                visible_tiles=_visible_tiles_for_plan(river_result),
                open_melds=self._open_meld_count_for_plan(meld_result),
            )
            call_policy = _call_policy(hand_result.hand_tiles, self.config, buttons or [])
            call_detail = _call_analysis_text(call_analysis)
            if plan:
                return f"{call_detail} {call_policy} 当前主线：{plan}"
            return f"{call_detail} {call_policy}"
        if plan:
            return f"默认跳过，除非鸣牌能明确推进当前主线：{plan}"
        return "默认跳过；只有役牌对子、直接进听、明显加速主线或安全和牌时才吃碰杠。"

    def _riichi_suggestion(
        self,
        hand_result: FastHandResult | None,
        river_result: RiverStateResult | None = None,
    ) -> str:
        if hand_result is None or not hand_result.ok:
            return "可立直：未拿到稳定手牌，先按游戏给出的立直窗口谨慎确认。"
        advice = _riichi_advice(hand_result.hand_tiles, self.config, visible_tiles=_visible_tiles_for_plan(river_result))
        if advice:
            return advice
        return "可立直：本地暂未算出明确听牌形，先确认待牌和打点。"

    def _opening_decision(
        self,
        hand_result: FastHandResult,
        river_result: RiverStateResult,
        started: float,
        *,
        action_meta: dict[str, Any] | None = None,
        meld_result: MeldStateResult | None = None,
    ) -> CoachDecision:
        plan_started = time.perf_counter()
        plan = build_round_plan(
            hand_result.hand_tiles,
            self.config,
            visible_tiles=_visible_tiles_for_plan(river_result),
            open_melds=self._open_meld_count_for_plan(meld_result),
            meld_tiles=self._meld_tiles_for_plan(meld_result),
            player_scores=self.state.player_scores,
            player_ranks=self.state.player_ranks,
            honba_count=self.state.honba_count,
            riichi_stick_count=self.state.table_riichi_stick_count,
        )
        strategy_elapsed_ms = _elapsed_ms(plan_started)
        self.state.opening_emitted = True
        self.state.round_phase = "opening_strategy"
        self._remember_local_plan(plan, opening=True)
        self.state.last_update_reason = "opening_plan"
        self.state.update_count += 1
        return CoachDecision(
            decision_type="opening_plan",
            priority=60,
            summary="Opening plan ready",
            detail=plan["detail"],
            suggestion=plan["summary"],
            hand_tiles=list(hand_result.hand_tiles),
            reason_codes=["first_stable_hand"],
            coach_state=self.state.to_dict(),
            perception={
                "table_context": self._table_context_perception(),
                "hand": hand_result.to_dict(),
                "action": action_meta or {},
                "meld": meld_result.to_dict() if meld_result is not None else {},
                "river": river_result.to_dict(),
            },
            engine_meta=self._meta(started, "opening_plan", strategy_elapsed_ms=strategy_elapsed_ms),
        )

    def _checkpoint_due(self, turn_number: int | None, *, force_checkpoint: bool) -> bool:
        if force_checkpoint:
            return True
        if turn_number is None:
            return False
        if turn_number <= self.state.last_checkpoint_self_turn:
            return False
        return (turn_number - self.state.last_checkpoint_self_turn) >= self.config.coach_checkpoint_self_turns

    def _checkpoint_decision(
        self,
        hand_result: FastHandResult,
        river_result: RiverStateResult,
        turn_number: int | None,
        force_checkpoint: bool,
        started: float,
        meld_result: MeldStateResult | None = None,
    ) -> CoachDecision:
        plan_started = time.perf_counter()
        plan = build_round_plan(
            hand_result.hand_tiles,
            self.config,
            visible_tiles=_visible_tiles_for_plan(river_result),
            open_melds=self._open_meld_count_for_plan(meld_result),
            meld_tiles=self._meld_tiles_for_plan(meld_result),
            player_scores=self.state.player_scores,
            player_ranks=self.state.player_ranks,
            honba_count=self.state.honba_count,
            riichi_stick_count=self.state.table_riichi_stick_count,
        )
        strategy_elapsed_ms = _elapsed_ms(plan_started)
        changed = self._plan_materially_changed(plan)
        if turn_number is not None:
            self.state.last_checkpoint_self_turn = turn_number
        self.state.round_phase = "checkpoint_strategy"
        self._remember_local_plan(plan)
        reason = "forced_checkpoint" if force_checkpoint else "scheduled_checkpoint"
        self.state.last_update_reason = reason
        self.state.update_count += 1
        return CoachDecision(
            decision_type="coach_checkpoint",
            priority=50 if changed else 5,
            summary="Round checkpoint updated",
            detail=plan["detail"],
            suggestion=plan["summary"],
            hand_tiles=list(hand_result.hand_tiles),
            reason_codes=[reason],
            coach_state=self.state.to_dict(),
            perception={
                "table_context": self._table_context_perception(),
                "hand": hand_result.to_dict(),
                "meld": meld_result.to_dict() if meld_result is not None else {},
                "river": river_result.to_dict(),
            },
            engine_meta=self._meta(started, "coach_checkpoint", strategy_elapsed_ms=strategy_elapsed_ms),
            quiet=not changed,
        )

    def _plan_materially_changed(self, plan: dict[str, Any]) -> bool:
        new_direction = str(plan.get("direction") or "").strip()
        if new_direction and new_direction != self.state.local_direction:
            return True
        new_discard = [str(t) for t in plan.get("discard_priority", []) if str(t).strip()]
        if new_discard and self.state.prev_discard_priority:
            if new_discard[0] != self.state.prev_discard_priority[0]:
                return True
        new_targets = [str(t) for t in plan.get("targets", []) if str(t).strip()]
        old_targets = self.state.target_shapes
        if new_targets != old_targets:
            return True
        return False

    def _remember_local_plan(self, plan: dict[str, Any], *, opening: bool = False) -> None:
        direction = str(plan.get("direction") or "").strip()
        summary = str(plan.get("summary") or "").strip()
        detail = str(plan.get("detail") or "").strip()
        targets = [str(item) for item in plan.get("targets", []) if str(item).strip()]
        cautions = [str(item) for item in plan.get("cautions", []) if str(item).strip()]
        discard_priority = [str(item) for item in plan.get("discard_priority", []) if str(item).strip()]
        display = direction if direction else summary
        self.state.prev_direction = self.state.local_direction
        self.state.prev_discard_priority = list(self.state.prev_discard_priority)
        if opening:
            self.state.opening_plan = display
        self.state.current_plan = display
        self.state.plan_source = "heuristic"
        self.state.local_direction = direction
        self.state.local_plan = summary
        self.state.local_detail = detail
        self.state.attack_defense_bias = str(plan.get("bias") or "neutral")
        self.state.target_shapes = targets
        self.state.caution_points = cautions
        self.state.prev_discard_priority = discard_priority

    def _defense_decision(
        self,
        riichi_players: list[str],
        hand_result: FastHandResult,
        river_result: RiverStateResult,
        started: float,
        meld_result: MeldStateResult | None = None,
    ) -> CoachDecision:
        strategy_started = time.perf_counter()
        ranking = rank_discard_decisions(
            list(hand_result.hand_tiles) if hand_result.ok else [],
            self.config,
            riichi_players,
            river_result,
            open_melds=self._open_meld_count_for_plan(meld_result),
            meld_tiles=self._meld_tiles_for_plan(meld_result),
            previous_posture=self.state.defense_posture,
            player_scores=self.state.player_scores,
            player_ranks=self.state.player_ranks,
            honba_count=self.state.honba_count,
            riichi_stick_count=self.state.table_riichi_stick_count,
        )
        strategy_elapsed_ms = _elapsed_ms(strategy_started)
        self.state.round_phase = "defense_mode"
        posture = str(ranking.get("posture") or DefensePosture.FOLD.value)
        self.state.attack_defense_bias = str(ranking.get("legacy_mode") or "defense")
        self.state.defense_posture = posture
        self.state.defense_risk_budget = float(ranking.get("risk_budget") or 0.0)
        self.state.last_update_reason = "riichi_defense"
        self.state.update_count += 1
        return CoachDecision(
            decision_type="defense_alert",
            priority=85,
            action_required=True,
            summary="Defense checkpoint",
            detail=f"Riichi pressure from {', '.join(riichi_players)}.",
            suggestion=self._defense_suggestion(
                riichi_players,
                river_result,
                hand_tiles=list(hand_result.hand_tiles) if hand_result.ok else None,
                ranking=ranking,
            ),
            hand_tiles=list(hand_result.hand_tiles),
            reason_codes=["riichi_players_present", f"defense_posture:{posture}"],
            coach_state=self.state.to_dict(),
            perception={
                "table_context": self._table_context_perception(),
                "hand": hand_result.to_dict(),
                "meld": meld_result.to_dict() if meld_result is not None else {},
                "river": river_result.to_dict(),
                "strategy": ranking,
            },
            engine_meta=self._meta(started, "defense_alert", strategy_elapsed_ms=strategy_elapsed_ms),
        )

    def _meta(self, started: float, source: str, *, strategy_elapsed_ms: float | None = None) -> dict[str, Any]:
        timings_ms = {"total": round((time.perf_counter() - started) * 1000.0, 1)}
        if self._last_table_context_result is not None:
            timings_ms["table_context"] = round(
                float(self._last_table_context_result.elapsed_ms),
                1,
            )
        if strategy_elapsed_ms is not None:
            timings_ms["strategy"] = round(float(strategy_elapsed_ms), 1)
        return {
            "source": source,
            "elapsed_ms": timings_ms["total"],
            "timings_ms": timings_ms,
            "live_advice_mode": self.config.live_advice_mode,
            "per_turn_discard_prompt": self.config.per_turn_discard_prompt,
            "hand_recognition_backend": self.config.hand_recognition_backend,
            "onnx_hand_enabled": self.config.onnx_hand_enabled,
            "tile_recognition_mode": getattr(self.config, "tile_recognition_mode", "legacy"),
            "river_recognition_enabled": self.config.river_recognition_enabled,
            "river_tracking_mode": getattr(self.config, "river_tracking_mode", "checkpoint"),
            "river_recognition_backend": "yolo26_lightweight" if self._uses_yolo26_tiles() else "onnx_discard_model",
            "meld_recognition_enabled": self.config.meld_recognition_enabled,
            "meld_recognition_backend": "yolo26_lightweight" if self._uses_yolo26_tiles() else "onnx_tile_classifier",
            "opponent_riichi_recognition_enabled": self.config.opponent_riichi_recognition_enabled,
            "settlement_recognition_enabled": getattr(self.config, "settlement_recognition_enabled", True),
            "settlement_phase": self.state.settlement_phase,
        }

    def _open_meld_count_for_plan(self, meld_result: MeldStateResult | None) -> int | None:
        if meld_result is not None and meld_result.ok:
            return meld_result.open_meld_count
        return self.state.last_open_meld_count if self.state.last_open_meld_count else None

    def _meld_tiles_for_plan(self, meld_result: MeldStateResult | None) -> list[str]:
        if meld_result is not None and meld_result.ok:
            if meld_result.analysis_hints.get("tile_identity_reliable") is False:
                return []
            return [str(tile) for tile in meld_result.tiles if str(tile).strip()]
        return list(self.state.last_meld_tiles)

    def _defense_suggestion(
        self,
        riichi_players: list[str],
        river_result: RiverStateResult,
        hand_tiles: list[str] | None = None,
        ranking: dict[str, Any] | None = None,
    ) -> str:
        if ranking and ranking.get("top_candidates"):
            return _discard_ranking_text(ranking)
        piles = river_result.discard_piles if river_result.ok else self.state.last_discard_piles
        held_tiles = [normalize_tile(t) for t in (hand_tiles or []) if normalize_tile(t)] or list(self.state.last_hand_tiles)
        defense = _defense_options(
            held_tiles,
            riichi_players,
            piles,
            visible_tiles=_visible_tiles_for_plan(river_result),
        )
        exact_safe = defense["exact_safe"]
        common_suji = defense["common_suji"]
        fully_visible = defense["fully_visible"]
        wall_related = defense["wall_related"]
        known_discards = defense["known_discards"]
        if exact_safe:
            safe_text = _tile_list_in_order(exact_safe)
            return f"防守优先：对全部立直者都成立的现物 {safe_text}，优先打；随后才考虑筋/壁，宝牌周边先保守。"
        if common_suji:
            suji_text = _tile_list_in_order(common_suji)
            suffix = f"；全见 { _tile_list_in_order(fully_visible) }" if fully_visible else ""
            return f"防守优先：手里没有共同现物；共同筋候选 {suji_text}，仅作降风险参考，不是绝对安全{suffix}。"
        if fully_visible:
            return f"防守优先：手里没有共同现物；全见牌 { _tile_list_in_order(fully_visible) } 可优先考虑，再比较宝牌与手牌价值。"
        if wall_related:
            return f"防守优先：手里没有共同现物；壁相关候选 { _tile_list_in_order(wall_related) } 仅作低风险参考，不要当作现物。"
        if known_discards:
            all_safe_text = _tile_list_in_order(known_discards)
            return f"防守优先：已识别立直家弃牌 {all_safe_text}，但手里没有对所有立直者共同成立的现物；先保守，避免宝牌周边。"
        if river_result.ok and river_result.visible_tiles:
            visible_text = _tile_list_in_order(river_result.visible_tiles[-10:])
            return f"牌河已识别，可见弃牌参考 {visible_text}；未标定立直家座位时先保守找现物。"
        return "Slow down and prefer safe tiles from visible information."


def _merge_opponent_meld_snapshots(
    cached: dict[str, list[dict[str, Any]]],
    current: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    """Keep public melds stable when a later frame temporarily misses a group."""

    owner_order = ("left_opponent", "top_opponent", "right_opponent")
    extra_owners = sorted((set(cached) | set(current)) - set(owner_order))
    merged_by_owner: dict[str, list[dict[str, Any]]] = {}
    for owner in (*owner_order, *extra_owners):
        cached_items = [
            deepcopy(item)
            for item in cached.get(owner, [])
            if isinstance(item, dict)
        ]
        current_items = [
            deepcopy(item)
            for item in current.get(owner, [])
            if isinstance(item, dict)
        ]
        merged_items = cached_items
        used_indices: set[int] = set()
        for item in current_items:
            match_index = _match_opponent_meld_index(merged_items, item, used_indices)
            if match_index is not None:
                merged_items[match_index] = _refresh_opponent_meld_item(
                    merged_items[match_index],
                    item,
                )
                used_indices.add(match_index)
                continue

            next_index = max(
                [int(existing.get("meld_index") or 0) for existing in merged_items]
                or [0]
            ) + 1
            requested_index = int(item.get("meld_index") or 0)
            if requested_index <= 0 or any(
                int(existing.get("meld_index") or 0) == requested_index
                for existing in merged_items
            ):
                item["meld_index"] = next_index
            # Only a newly observed meld may consume a newly disappeared discard.
            # Keep a short window because the river and meld detectors can settle
            # one or two frames apart. Long-standing melds must never claim a later
            # same-tile detection dropout.
            item["claim_link_pending_scans"] = _OPPONENT_CALL_LINK_MAX_SCANS
            merged_items.append(item)
            used_indices.add(len(merged_items) - 1)
        if merged_items:
            merged_items.sort(key=lambda item: int(item.get("meld_index") or 0))
            merged_by_owner[owner] = merged_items
    return merged_by_owner


def _match_opponent_meld_index(
    cached_items: list[dict[str, Any]],
    current: dict[str, Any],
    used_indices: set[int],
) -> int | None:
    current_bbox = _river_bbox(current)
    has_comparable_bbox = False
    if current_bbox is not None:
        scored = [
            (_river_bbox_iou(current_bbox, cached_bbox), index)
            for index, cached in enumerate(cached_items)
            if index not in used_indices and (cached_bbox := _river_bbox(cached)) is not None
        ]
        has_comparable_bbox = bool(scored)
        if scored:
            overlap, index = max(scored)
            if overlap >= _RIVER_MATCH_MIN_IOU:
                return index

    current_tiles = tuple(str(tile) for tile in current.get("tiles", []) if str(tile).strip())
    if current_tiles:
        for index, cached in enumerate(cached_items):
            cached_tiles = tuple(str(tile) for tile in cached.get("tiles", []) if str(tile).strip())
            if index not in used_indices and cached_tiles == current_tiles:
                return index

    if has_comparable_bbox:
        return None
    meld_index = int(current.get("meld_index") or 0)
    if 1 <= meld_index <= len(cached_items) and meld_index - 1 not in used_indices:
        return meld_index - 1
    return None


def _refresh_opponent_meld_item(
    cached: dict[str, Any],
    current: dict[str, Any],
) -> dict[str, Any]:
    refreshed = deepcopy(current)
    for key in (
        "claim_discard_linked",
        "called_from_owner",
        "claimed_discard",
        "claim_link_candidate_count",
        "claim_link_source",
        "claim_link_pending_scans",
    ):
        if key in cached and (key not in refreshed or not refreshed.get(key)):
            refreshed[key] = deepcopy(cached[key])
    return refreshed


def _opponent_meld_tiles(
    opponent_melds: dict[str, list[dict[str, Any]]],
) -> list[str]:
    owner_order = ("left_opponent", "top_opponent", "right_opponent")
    extra_owners = sorted(set(opponent_melds) - set(owner_order))
    return [
        str(tile)
        for owner in (*owner_order, *extra_owners)
        for meld in opponent_melds.get(owner, [])
        if isinstance(meld, dict) and meld.get("tile_identity_reliable") is not False
        for tile in meld.get("tiles", [])
        if str(tile).strip()
    ]


def _link_claimed_discards(
    discard_piles: dict[str, list[dict[str, Any]]],
    current_piles: dict[str, list[dict[str, Any]]],
    opponent_melds: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Link a disappeared discard to the called tile of a visible meld.

    The discard remains in history for genbutsu/furiten reasoning. The marker
    only removes that physical tile from visible-copy counts because the same
    tile is already represented inside the meld.
    """

    missing = _missing_unclaimed_discards(discard_piles, current_piles)
    events: list[dict[str, Any]] = []
    for caller in ("left_opponent", "top_opponent", "right_opponent"):
        for meld in opponent_melds.get(caller, []):
            if (
                not isinstance(meld, dict)
                or meld.get("claim_discard_linked")
                or int(meld.get("claim_link_pending_scans") or 0) <= 0
            ):
                continue
            called_tile = _called_tile_from_meld(meld)
            if called_tile not in TILE_TYPES:
                meld.pop("claim_link_pending_scans", None)
                continue
            kind = str(meld.get("kind") or "")
            if kind == "chi":
                allowed_sources = {_CHI_DISCARD_SOURCE.get(caller, "")}
            else:
                allowed_sources = set(_TABLE_PLAYERS) - {caller}
            candidates = [
                candidate
                for candidate in missing
                if candidate[0] in allowed_sources
                and _canonical_tile(str(candidate[2].get("tile") or "")) == called_tile
                and not candidate[2].get("claimed_into_meld")
            ]
            if not candidates:
                _advance_meld_claim_link_window(meld)
                continue
            candidates.sort(
                key=lambda candidate: (
                    int(candidate[2].get("turn_index") or 0),
                    float(candidate[2].get("confidence") or 0.0),
                ),
                reverse=True,
            )
            source_player, _item_index, discard = candidates[0]
            event = _mark_discard_claimed(
                discard,
                source_player=source_player,
                caller=caller,
                meld=meld,
                candidate_count=len(candidates),
                source="opponent_meld_called_tile_disappearance",
            )
            events.append(event)
            meld.pop("claim_link_pending_scans", None)
            missing = [candidate for candidate in missing if candidate[2] is not discard]
    return events


def _advance_meld_claim_link_window(meld: dict[str, Any]) -> None:
    remaining = max(0, int(meld.get("claim_link_pending_scans") or 0) - 1)
    if remaining:
        meld["claim_link_pending_scans"] = remaining
    else:
        meld.pop("claim_link_pending_scans", None)


def _missing_unclaimed_discards(
    historical_piles: dict[str, list[dict[str, Any]]],
    current_piles: dict[str, list[dict[str, Any]]],
) -> list[tuple[str, int, dict[str, Any]]]:
    missing: list[tuple[str, int, dict[str, Any]]] = []
    for player, historical_items in historical_piles.items():
        current_items = [
            item
            for item in current_piles.get(player, [])
            if isinstance(item, dict)
        ]
        used_indices: set[int] = set()
        for item_index, historical in enumerate(historical_items):
            if not isinstance(historical, dict) or historical.get("claimed_into_meld"):
                continue
            match_index = _match_river_item_index(current_items, historical, used_indices)
            if match_index is None:
                missing.append((player, item_index, historical))
            else:
                used_indices.add(match_index)
    return missing


def _called_tile_from_meld(meld: dict[str, Any]) -> str:
    raw_index = meld.get("called_tile_index")
    if raw_index is None:
        return ""
    try:
        called_index = int(raw_index)
    except (TypeError, ValueError):
        return ""
    tiles = meld.get("tiles")
    if not isinstance(tiles, list) or not 0 <= called_index < len(tiles):
        return ""
    return _canonical_tile(str(tiles[called_index] or ""))


def _mark_discard_claimed(
    discard: dict[str, Any],
    *,
    source_player: str,
    caller: str,
    meld: dict[str, Any],
    candidate_count: int,
    source: str,
) -> dict[str, Any]:
    meld_index = int(meld.get("meld_index") or 0)
    kind = str(meld.get("kind") or "unknown")
    tile = _canonical_tile(str(discard.get("tile") or ""))
    discard.update(
        {
            "claimed_into_meld": True,
            "claimed_by": caller,
            "claimed_meld_index": meld_index,
            "claimed_meld_kind": kind,
            "claim_link_source": source,
        }
    )
    claimed_discard = {
        "player": source_player,
        "turn_index": int(discard.get("turn_index") or 0),
        "slot_id": str(discard.get("slot_id") or ""),
        "tile": tile,
    }
    meld.update(
        {
            "claim_discard_linked": True,
            "called_from_owner": source_player,
            "claimed_discard": claimed_discard,
            "claim_link_candidate_count": max(1, int(candidate_count)),
            "claim_link_source": source,
        }
    )
    return {
        "tile": tile,
        "source_player": source_player,
        "claimed_by": caller,
        "meld_index": meld_index,
        "meld_kind": kind,
        "candidate_count": max(1, int(candidate_count)),
        "source": source,
    }


def _pending_self_call_discard_candidates(
    discard_piles: dict[str, list[dict[str, Any]]],
    pending: dict[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    tile = _canonical_tile(str(pending.get("tile") or ""))
    source_player = _player_key(pending.get("player"))
    allowed_players = (
        [source_player]
        if source_player and source_player != "self"
        else [player for player in _TABLE_PLAYERS if player != "self"]
    )
    candidates = [
        (player, item)
        for player in allowed_players
        for item in discard_piles.get(player, [])
        if isinstance(item, dict)
        and not item.get("claimed_into_meld")
        and _canonical_tile(str(item.get("tile") or "")) == tile
    ]
    slot_id = str(pending.get("slot_id") or "")
    turn_index = int(pending.get("turn_index") or 0)
    exact = [
        candidate
        for candidate in candidates
        if (slot_id and str(candidate[1].get("slot_id") or "") == slot_id)
        or (turn_index and int(candidate[1].get("turn_index") or 0) == turn_index)
    ]
    selected = exact or candidates
    selected.sort(
        key=lambda candidate: (
            int(candidate[1].get("turn_index") or 0),
            float(candidate[1].get("confidence") or 0.0),
        ),
        reverse=True,
    )
    return selected


def _physical_discard_tiles(
    discard_piles: dict[str, list[dict[str, Any]]],
) -> list[str]:
    return [
        str(item.get("tile") or "")
        for items in discard_piles.values()
        for item in items
        if isinstance(item, dict)
        and not item.get("claimed_into_meld")
        and str(item.get("tile") or "").strip()
    ]


def _match_river_item_index(
    cached_items: list[dict[str, Any]],
    current: dict[str, Any],
    used_indices: set[int],
) -> int | None:
    current_slot = str(current.get("slot_id") or "")
    if current_slot:
        for index, cached in enumerate(cached_items):
            if index not in used_indices and str(cached.get("slot_id") or "") == current_slot:
                return index

    current_bbox = _river_bbox(current)
    if current_bbox is not None:
        scored = [
            (_river_bbox_iou(current_bbox, cached_bbox), index)
            for index, cached in enumerate(cached_items)
            if index not in used_indices and (cached_bbox := _river_bbox(cached)) is not None
        ]
        if scored:
            score, index = max(scored)
            if score >= _RIVER_MATCH_MIN_IOU:
                return index
            return None

    current_turn = int(current.get("turn_index") or 0)
    if current_turn:
        for index, cached in enumerate(cached_items):
            if index not in used_indices and int(cached.get("turn_index") or 0) == current_turn:
                return index
    return None


def _refresh_river_item(cached: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    refreshed = dict(cached)
    stable_player = cached.get("player")
    stable_turn = cached.get("turn_index")
    stable_slot = cached.get("slot_id")
    for key in ("tile", "bbox", "quad", "confidence", "source"):
        if key in current:
            refreshed[key] = deepcopy(current[key])
    if stable_player is not None:
        refreshed["player"] = stable_player
    if stable_turn is not None:
        refreshed["turn_index"] = stable_turn
    if stable_slot is not None:
        refreshed["slot_id"] = stable_slot
    return refreshed


def _river_candidate_key(kind: str, player: str, item: dict[str, Any]) -> str:
    slot_id = str(item.get("slot_id") or "")
    if slot_id:
        return f"{kind}:{player}:slot:{slot_id}"
    turn_index = int(item.get("turn_index") or 0)
    if turn_index:
        return f"{kind}:{player}:turn:{turn_index}"
    bbox = _river_bbox(item)
    if bbox is None:
        return f"{kind}:{player}:unknown"
    center_x = (bbox[0] + bbox[2]) / 2.0
    center_y = (bbox[1] + bbox[3]) / 2.0
    return f"{kind}:{player}:center:{round(center_x / 16.0)}:{round(center_y / 16.0)}"


def _river_bbox(item: dict[str, Any]) -> tuple[float, float, float, float] | None:
    raw_bbox = item.get("bbox")
    if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
        return None
    try:
        left, top, right, bottom = (float(value) for value in raw_bbox)
    except (TypeError, ValueError):
        return None
    return min(left, right), min(top, bottom), max(left, right), max(top, bottom)


def _river_bbox_iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    intersection_left = max(left[0], right[0])
    intersection_top = max(left[1], right[1])
    intersection_right = min(left[2], right[2])
    intersection_bottom = min(left[3], right[3])
    intersection = max(0.0, intersection_right - intersection_left) * max(
        0.0,
        intersection_bottom - intersection_top,
    )
    if intersection <= 0.0:
        return 0.0
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return intersection / max(1e-6, left_area + right_area - intersection)


def _visible_tiles_for_plan(river_result: RiverStateResult | None) -> list[str]:
    if river_result is None or not river_result.ok:
        return []
    has_discard_records = any(
        isinstance(item, dict)
        for items in river_result.discard_piles.values()
        for item in items
    )
    tiles = (
        _physical_discard_tiles(river_result.discard_piles)
        if has_discard_records
        else list(river_result.visible_tiles)
    )
    tiles.extend(str(tile) for tile in river_result.opponent_meld_tiles if str(tile).strip())
    return tiles


def _table_match_context(
    player_scores: dict[str, int] | None,
    player_ranks: dict[str, int] | None,
    *,
    honba_count: int | None,
    riichi_stick_count: int | None,
) -> dict[str, Any]:
    scores = {
        player: int(score)
        for player, score in dict(player_scores or {}).items()
        if player in _TABLE_PLAYERS
    }
    if len(scores) != 4 or "self" not in scores:
        return {}
    ranks = {
        player: int(rank)
        for player, rank in dict(player_ranks or {}).items()
        if player in scores
    }
    if len(ranks) != 4:
        ordered_scores = sorted(set(scores.values()), reverse=True)
        ranks = {
            player: ordered_scores.index(score) + 1
            for player, score in scores.items()
        }
    self_score = scores["self"]
    higher_scores = sorted(score for score in scores.values() if score > self_score)
    lower_scores = sorted((score for score in scores.values() if score < self_score), reverse=True)
    gap_above = higher_scores[0] - self_score if higher_scores else 0
    lead_below = self_score - lower_scores[0] if lower_scores else 0
    honba = max(0, int(honba_count or 0))
    sticks = max(0, int(riichi_stick_count or 0))
    reward_bonus = honba * 300 + sticks * 1000
    display_order = ("self", "left_opponent", "top_opponent", "right_opponent")
    summary = "｜".join(
        f"{_PLAYER_DISPLAY_NAMES[player]}{scores[player]:,}（{ranks[player]}位）"
        for player in display_order
    )
    return {
        "scores": scores,
        "ranks": ranks,
        "self_score": self_score,
        "self_rank": ranks.get("self", 0),
        "gap_above": gap_above,
        "lead_below": lead_below,
        "honba_count": honba_count,
        "riichi_stick_count": riichi_stick_count,
        "win_reward_bonus": reward_bonus,
        "summary": summary,
    }


def build_round_plan(
    hand_tiles: list[str],
    config: MahjongCoachConfig | None = None,
    *,
    visible_tiles: list[str] | None = None,
    open_melds: int | None = None,
    meld_tiles: list[str] | None = None,
    player_scores: dict[str, int] | None = None,
    player_ranks: dict[str, int] | None = None,
    honba_count: int | None = None,
    riichi_stick_count: int | None = None,
) -> dict[str, Any]:
    tiles = [normalize_tile(tile) for tile in hand_tiles if normalize_tile(tile)]
    meld_tile_list = [normalize_tile(tile) for tile in (meld_tiles or []) if normalize_tile(tile)]
    shape_tiles = [*tiles, *meld_tile_list]
    counts = Counter(tiles)
    visible_counts = _visible_counts([*(visible_tiles or []), *meld_tile_list])
    value_honors = _value_honor_tiles(config)
    dora_tiles = _dora_tiles(config)
    style = _play_style(config)
    explicit_open_melds = _coerce_open_melds(open_melds)
    inferred_open_melds = _inferred_open_melds(len(tiles)) if style == "fast" else 0
    open_meld_count = explicit_open_melds if explicit_open_melds is not None else inferred_open_melds
    suit_counts = Counter(tile_suit(tile) for tile in tiles if tile_suit(tile))
    honor_count = sum(1 for tile in tiles if is_honor(tile))
    terminal_count = sum(1 for tile in tiles if is_terminal(tile))
    simple_count = sum(1 for tile in tiles if is_simple(tile))
    pair_count = sum(1 for value in counts.values() if value >= 2)
    best_suit, best_suit_count = ("", 0)
    suited = {suit: count for suit, count in suit_counts.items() if suit in {"m", "p", "s"}}
    if suited:
        best_suit, best_suit_count = max(suited.items(), key=lambda item: item[1])
    second_suit_count = max((count for suit, count in suited.items() if suit != best_suit), default=0)
    pair_tiles = [tile for tile, value in counts.items() if value >= 2]
    value_pair_tiles = [tile for tile in pair_tiles if tile in value_honors]

    suit_threshold = _RIICHI_SUIT_THRESHOLD if style == "riichi" else _FAST_SUIT_THRESHOLD
    pair_threshold = _RIICHI_PAIR_THRESHOLD if style == "riichi" else _FAST_PAIR_THRESHOLD
    simple_threshold = _RIICHI_SIMPLE_THRESHOLD if style == "riichi" else _FAST_SIMPLE_THRESHOLD
    is_pair_route = open_meld_count == 0 and pair_count >= pair_threshold and best_suit_count < suit_threshold

    cleanup_tiles = _cleanup_candidates(tiles, counts, best_suit, value_honors, dora_tiles)
    keep_tiles = pair_tiles if is_pair_route else _keep_candidates(tiles, counts, best_suit, dora_tiles)
    discard_tiles = _discard_candidates(tiles, counts, best_suit, cleanup_tiles, keep_tiles, value_honors, dora_tiles)
    efficiency = _efficiency_analysis(tiles, discard_tiles, counts, visible_counts, best_suit, value_honors, dora_tiles, open_melds=open_meld_count)
    best_efficiency_discard = _best_efficiency_discard(efficiency)
    if best_efficiency_discard:
        discard_tiles = _merge_unique_tiles([best_efficiency_discard, *discard_tiles])
    discard_text = _tile_list_in_order(discard_tiles)
    keep_text = _tile_list(keep_tiles) or "连续搭子和中张"
    suit_text = _suit_breakdown(shape_tiles) if meld_tile_list else _suit_breakdown(tiles)
    best_shape = _suit_shape(shape_tiles, best_suit) if best_suit else ""
    meld_text = _tile_list_in_order(meld_tile_list)
    cleanup_text = _tile_list_in_order(cleanup_tiles) or discard_text or "孤张字牌和远张"
    primary_discard_text = _tile_name(best_efficiency_discard) if best_efficiency_discard else _first_tile_name(discard_tiles or cleanup_tiles)
    primary_discard_text = primary_discard_text or cleanup_text
    route_options = _route_discard_options(discard_tiles, counts, best_suit, best_suit_count, second_suit_count, pair_count, bool(value_pair_tiles), honor_count, terminal_count, simple_count)
    route_text = _route_options_text(route_options)

    direction, targets, summary = _choose_direction(
        open_meld_count, best_suit, best_suit_count, best_shape, second_suit_count,
        pair_count, pair_tiles, value_pair_tiles, is_pair_route, simple_count,
        simple_threshold, honor_count, terminal_count, keep_text, efficiency, meld_text, meld_tile_list,
        dora_tiles, shape_tiles, value_honors, style,
    )
    profile = config.player_profile if config is not None else None
    if open_meld_count == 0 and getattr(profile, "goal_bias", "balanced") == "yakuman":
        yakuman_routes = assess_yakuman_routes(tiles, visible_tiles=visible_tiles or [])
        viable = next(
            (
                item
                for item in yakuman_routes
                if not any("要求门清" in blocker for blocker in item.blockers)
                and item.distance <= 5
            ),
            None,
        )
        if viable is not None:
            direction = f"役满候选：{viable.label}"
            key_text = _tile_list_in_order(viable.key_tiles)
            targets.insert(0, f"役满路线：{viable.label}，距离约{viable.distance}张")
            if key_text:
                targets.insert(1, f"关键牌：{key_text}")
            summary = f"保留{viable.label}路线；牌效明显恶化或关键牌枯竭时回到普通和牌"
    cautions = _build_cautions(
        honor_count, terminal_count, best_suit_count, open_meld_count, route_text,
        efficiency, cleanup_tiles, discard_text, cleanup_text, primary_discard_text,
        meld_tile_list, shape_tiles, value_honors, best_suit, counts, simple_count,
        is_pair_route, style,
    )

    bias = "attack" if simple_count >= 8 or best_suit_count >= 5 or value_pair_tiles else "neutral"
    table_context = _table_match_context(
        player_scores,
        player_ranks,
        honba_count=honba_count,
        riichi_stick_count=riichi_stick_count,
    )
    if table_context:
        self_rank = int(table_context.get("self_rank") or 0)
        gap_above = int(table_context.get("gap_above") or 0)
        lead_below = int(table_context.get("lead_below") or 0)
        if self_rank == 4:
            bias = "attack"
            targets.insert(
                0,
                f"四位追分：距三位{gap_above}点，边缘选择优先保留速度与打点",
            )
            summary += "；当前四位，避免把有价值的和牌路线过早打散"
        elif self_rank == 1:
            bias = "defense" if lead_below >= 12_000 else bias
            cautions.insert(
                0,
                f"一位领先{lead_below}点：同等牌效下减少无谓放铳风险",
            )
        reward_bonus = int(table_context.get("win_reward_bonus") or 0)
        if reward_bonus:
            targets.append(
                f"桌面奖励：本场与供托合计让本手和牌额外增加{reward_bonus}点",
            )
    summary = _plan_summary(summary, keep_text, primary_discard_text, efficiency)
    detail = (
        f"结构：{suit_text}；对子 {pair_count} 组。主线：{summary}。"
        f"{(' 已副露' + str(open_meld_count) + '组，') if open_meld_count else ' '}"
        f" 当前先保留 {keep_text}，{('路线选择：' + route_text) if route_text else ('候选打牌 ' + (discard_text or cleanup_text))}。"
        f" {_efficiency_detail(efficiency)}"
    )
    return {
        "direction": direction,
        "summary": summary,
        "detail": detail,
        "bias": bias,
        "targets": targets,
        "cautions": cautions,
        "discard_priority": list(discard_tiles),
        "efficiency": efficiency,
        "table_context": table_context,
    }


def _choose_direction(
    open_meld_count: int,
    best_suit: str,
    best_suit_count: int,
    best_shape: str,
    second_suit_count: int,
    pair_count: int,
    pair_tiles: list[str],
    value_pair_tiles: list[str],
    is_pair_route: bool,
    simple_count: int,
    simple_threshold: int,
    honor_count: int,
    terminal_count: int,
    keep_text: str,
    efficiency: dict[str, Any],
    meld_text: str,
    meld_tile_list: list[str],
    dora_tiles: set[str],
    shape_tiles: list[str],
    value_honors: set[str],
    style: str,
) -> tuple[str, list[str], str]:
    suit_threshold = _RIICHI_SUIT_THRESHOLD if style == "riichi" else _FAST_SUIT_THRESHOLD
    direction = "牌效推进"
    targets: list[str] = []
    summary = ""

    if open_meld_count:
        direction = _open_hand_direction(open_meld_count, efficiency)
        targets.append(f"主线：已副露{open_meld_count}组，按最快进听/听牌收束")
        targets.append(f"保留：{keep_text}")
        summary = f"已副露{open_meld_count}组，优先打向听最低、有效牌最多的牌"
    elif best_suit_count >= suit_threshold:
        direction = f"染手({SUIT_NAMES.get(best_suit, best_suit)}子)"
        targets.append(f"主线：{SUIT_NAMES.get(best_suit, best_suit)}子清一色/混一色倾向")
        summary = f"{SUIT_NAMES.get(best_suit, best_suit)}子多，优先保留同色，杂色牌逐步清理"
    elif is_pair_route:
        direction = "七对子"
        targets.append(f"主线：七对子胚子，已有对子 {_tile_list(pair_tiles)}")
        summary = "保留现有对子，不再做面子手"
    elif value_pair_tiles:
        direction = "役牌速攻"
        targets.append(f"主线：役牌速度，保留/可碰 {_tile_list(value_pair_tiles)}")
        summary = f"围绕役牌 {_tile_list(value_pair_tiles)} 加速，同时保持听牌潜力"
    elif best_suit_count >= 5 and best_suit_count >= second_suit_count + 2:
        direction = f"围绕{SUIT_NAMES.get(best_suit, best_suit)}子"
        targets.append(f"主线：围绕{SUIT_NAMES.get(best_suit, best_suit)}子 {best_shape} 推进")
        summary = f"以{SUIT_NAMES.get(best_suit, best_suit)}子为主，{best_shape} 找顺子成型"
    elif simple_count >= simple_threshold and honor_count + terminal_count <= 3:
        direction = "断幺九/平和"
        targets.extend(["主线：断幺/平和速度", f"保留：{keep_text}"])
        summary = f"断幺九或平和速度路线，保留{keep_text}"
    elif honor_count >= 4:
        direction = "清字牌"
        targets.append("主线：先清孤字牌")
        summary = "字牌偏多，先清孤字，只留役牌对子"
    else:
        targets.append("主线：牌效推进")
        summary = f"牌效推进，保留{keep_text}"

    target_shape = _target_shape_text(direction, best_suit, best_shape, keep_text, efficiency, open_meld_count, bool(value_pair_tiles), is_pair_route)
    if target_shape:
        targets.append(f"目标形：{target_shape}")
    if keep_text and not any(item.startswith("保留：") for item in targets):
        targets.append(f"保留：{keep_text}")
    if meld_text:
        targets.append(f"副露识别：{meld_text}")
    if pair_tiles:
        targets.append(f"对子：{_tile_list(pair_tiles)}")
    if value_pair_tiles:
        targets.append(f"役牌对子：{_tile_list(value_pair_tiles)}")
    open_yaku_targets, _ = _open_yaku_notes(meld_tile_list, shape_tiles, value_honors, best_suit, open_meld_count)
    targets.extend(open_yaku_targets)
    dora_text = _tile_list([tile for tile in shape_tiles if _is_dora(tile, dora_tiles)])
    if dora_text:
        targets.append(f"宝牌/红5：{dora_text}")

    return direction, targets, summary


def _build_cautions(
    honor_count: int,
    terminal_count: int,
    best_suit_count: int,
    open_meld_count: int,
    route_text: str,
    efficiency: dict[str, Any],
    cleanup_tiles: list[str],
    discard_text: str,
    cleanup_text: str,
    primary_discard_text: str,
    meld_tile_list: list[str],
    shape_tiles: list[str],
    value_honors: set[str],
    best_suit: str,
    counts: Counter[str],
    simple_count: int,
    is_pair_route: bool,
    style: str,
) -> list[str]:
    cautions: list[str] = []
    if honor_count >= 3:
        cautions.append("孤字牌不要久留，除非成对或有役牌价值。")
    if terminal_count >= 4:
        cautions.append("孤幺九可清，已经组成边搭/对子再保留。")
    if best_suit_count >= 7:
        cautions.append("有染手分支，但摸到强中张时不要硬染。")
    if route_text:
        cautions.append(f"{'副露收束' if open_meld_count else '路线选择'}：{route_text}")
    efficiency_note = _efficiency_note(efficiency)
    if efficiency_note:
        cautions.append(efficiency_note)
    if cleanup_tiles:
        cautions.append(f"优先清理：{discard_text or cleanup_text}")
    next_step = _next_step_text(primary_discard_text, efficiency, open_meld_count)
    if next_step:
        cautions.append(f"下一步：{next_step}")
    _, open_yaku_cautions = _open_yaku_notes(meld_tile_list, shape_tiles, value_honors, best_suit, open_meld_count)
    if open_meld_count:
        cautions.extend(open_yaku_cautions)
        cautions.append(_open_hand_policy_line(open_meld_count))
    else:
        cautions.append(_open_policy_line(counts, value_honors, best_suit_count, simple_count, honor_count, terminal_count, is_pair_route, style))
    if not cautions:
        cautions.append("三巡后再看主线，不要每摸一张就推翻主线。")
    return cautions


def _tile_name(tile: str) -> str:
    normalized = normalize_tile(tile)
    if normalized in HONOR_NAMES:
        return HONOR_NAMES[normalized]
    if normalized in {"0m", "0p", "0s"}:
        return f"红5{SUIT_NAMES.get(tile_suit(normalized), '')}"
    if len(normalized) == 2:
        return f"{tile_rank(normalized)}{SUIT_NAMES.get(tile_suit(normalized), '')}"
    return normalized


def _tile_list(tiles: list[str]) -> str:
    unique = []
    for tile in sorted((normalize_tile(tile) for tile in tiles if normalize_tile(tile)), key=_tile_sort_key):
        if tile not in unique:
            unique.append(tile)
    return "、".join(_tile_name(tile) for tile in unique[:6])


def _tile_list_in_order(tiles: list[str]) -> str:
    unique = []
    for tile in (normalize_tile(tile) for tile in tiles if normalize_tile(tile)):
        if tile not in unique:
            unique.append(tile)
    return "、".join(_tile_name(tile) for tile in unique[:6])


def _tile_sort_key(tile: str) -> tuple[int, int, str]:
    suit_order = {"m": 0, "p": 1, "s": 2, "z": 3}
    normalized = normalize_tile(tile)
    rank = tile_rank(normalized)
    return (suit_order.get(tile_suit(normalized), 9), int(rank) if rank.isdigit() else 0, normalized)


_PLAYER_KEY_ALIASES = {
    "self": "self",
    "left": "left_opponent",
    "left_opponent": "left_opponent",
    "kamicha": "left_opponent",
    "top": "top_opponent",
    "top_opponent": "top_opponent",
    "toimen": "top_opponent",
    "right": "right_opponent",
    "right_opponent": "right_opponent",
    "shimocha": "right_opponent",
}


def _player_key(value: Any) -> str:
    return _PLAYER_KEY_ALIASES.get(str(value or "").strip().lower(), "")


def _claimed_discard_from_river_delta(
    previous_piles: dict[str, list[dict[str, Any]]],
    current_piles: dict[str, list[dict[str, Any]]],
    buttons: list[str],
) -> dict[str, Any]:
    """Return the unique call candidate together with its river record."""
    normalized_buttons = set(_normalize_buttons(buttons))
    if "chi" in normalized_buttons and not normalized_buttons.intersection({"pon", "kan"}):
        candidate_players = {"left_opponent"}
    else:
        candidate_players = {"left_opponent", "top_opponent", "right_opponent"}

    additions: list[dict[str, Any]] = []
    for player in candidate_players:
        previous = Counter(
            _canonical_tile(str(item.get("tile") or ""))
            for item in previous_piles.get(player, [])
            if isinstance(item, dict)
        )
        for item in current_piles.get(player, []):
            if not isinstance(item, dict):
                continue
            tile = _canonical_tile(str(item.get("tile") or ""))
            if tile not in TILE_TYPES:
                continue
            if previous[tile] > 0:
                previous[tile] -= 1
            else:
                additions.append(
                    {
                        "tile": tile,
                        "player": player,
                        "turn_index": int(item.get("turn_index") or 0),
                        "slot_id": str(item.get("slot_id") or ""),
                    }
                )
    unique_tiles = _merge_unique_tiles([str(item["tile"]) for item in additions])
    if len(unique_tiles) != 1:
        return {}
    matching = [item for item in additions if item["tile"] == unique_tiles[0]]
    result = {"tile": unique_tiles[0], "player": "", "turn_index": 0, "slot_id": ""}
    if len(matching) == 1:
        result.update(matching[0])
    return result


def _claimed_tile_from_river_delta(
    previous_piles: dict[str, list[dict[str, Any]]],
    current_piles: dict[str, list[dict[str, Any]]],
    buttons: list[str],
) -> str:
    return str(
        _claimed_discard_from_river_delta(previous_piles, current_piles, buttons).get("tile")
        or ""
    )


def _defense_options(
    held_tiles: list[str],
    riichi_players: list[str],
    discard_piles: dict[str, list[dict[str, Any]]],
    *,
    visible_tiles: list[str] | None = None,
) -> dict[str, list[str]]:
    """Rank defensive candidates without treating one player's genbutsu as universal.

    Exact safety is an intersection across every detected riichi player. Suji and
    wall information are deliberately returned separately because they reduce
    risk but are not equivalent to genbutsu.
    """
    held = _merge_unique_tiles([_canonical_tile(tile) for tile in held_tiles if _canonical_tile(tile) in TILE_TYPES])
    player_keys = _merge_unique_tiles([_player_key(player) for player in riichi_players if _player_key(player)])
    player_discards: list[list[str]] = []
    known_discards: list[str] = []
    for key in player_keys:
        tiles = [
            _canonical_tile(str(item.get("tile") or ""))
            for item in discard_piles.get(key, [])
            if isinstance(item, dict)
        ]
        tiles = [tile for tile in tiles if tile in TILE_TYPES]
        player_discards.append(tiles)
        known_discards.extend(tiles)

    if not player_discards or any(not items for items in player_discards):
        exact_safe: set[str] = set()
        common_suji: set[str] = set()
    else:
        exact_safe = set(player_discards[0])
        common_suji = _suji_tiles_from_discards(player_discards[0])
        for items in player_discards[1:]:
            exact_safe.intersection_update(items)
            common_suji.intersection_update(_suji_tiles_from_discards(items))

    visible_counts = Counter(_canonical_tile(tile) for tile in held)
    if visible_tiles is None:
        public_tiles = [
            str(item.get("tile") or "")
            for items in discard_piles.values()
            for item in items
            if isinstance(item, dict)
        ]
    else:
        public_tiles = visible_tiles
    for raw_tile in public_tiles:
        tile = _canonical_tile(str(raw_tile or ""))
        if tile in TILE_TYPES:
            visible_counts[tile] += 1
    fully_visible = [tile for tile in held if visible_counts[tile] >= 4]
    wall_related = [tile for tile in held if tile not in fully_visible and _is_wall_related(tile, visible_counts)]
    return {
        "exact_safe": [tile for tile in held if tile in exact_safe],
        "common_suji": [tile for tile in held if tile in common_suji and tile not in exact_safe],
        "fully_visible": fully_visible,
        "wall_related": wall_related,
        "known_discards": _merge_unique_tiles(known_discards),
    }


def rank_discard_decisions(
    hand_tiles: list[str],
    config: MahjongCoachConfig | None,
    riichi_players: list[str],
    river_result: RiverStateResult | None,
    *,
    open_melds: int | None = None,
    meld_tiles: list[str] | None = None,
    turn_number: int | None = None,
    previous_posture: str = "",
    player_scores: dict[str, int] | None = None,
    player_ranks: dict[str, int] | None = None,
    honba_count: int | None = None,
    riichi_stick_count: int | None = None,
) -> dict[str, Any]:
    """Combine tile efficiency and per-opponent danger into one discard ranking."""
    tiles = [normalize_tile(tile) for tile in hand_tiles if _canonical_tile(normalize_tile(tile)) in TILE_TYPES]
    piles = river_result.discard_piles if river_result is not None and river_result.ok else {}
    visible_tiles = _visible_tiles_for_plan(river_result)
    plan = build_round_plan(
        tiles,
        config,
        visible_tiles=visible_tiles,
        open_melds=open_melds,
        meld_tiles=meld_tiles,
        player_scores=player_scores,
        player_ranks=player_ranks,
        honba_count=honba_count,
        riichi_stick_count=riichi_stick_count,
    )
    efficiency = plan.get("efficiency", {})
    current_shanten = int(efficiency.get("current_shanten", 8))
    raw_options = efficiency.get("all_discard_options") or efficiency.get("discard_options") or []
    options = [dict(item) for item in raw_options if isinstance(item, dict)]
    if not options:
        options = [
            {
                "tile": tile,
                "shanten": current_shanten,
                "effective_types": 0,
                "effective_count": 0,
                "effective_tiles": [],
            }
            for tile in _merge_unique_tiles(tiles)
        ]

    hand_counts = Counter(_canonical_tile(tile) for tile in tiles)
    visible_counts = Counter(hand_counts)
    for tile in visible_tiles:
        canonical = _canonical_tile(tile)
        if canonical in TILE_TYPES:
            visible_counts[canonical] += 1
    for tile in meld_tiles or []:
        canonical = _canonical_tile(tile)
        if canonical in TILE_TYPES:
            visible_counts[canonical] += 1
    dora_tiles = _dora_tiles(config)
    value_honors = _value_honor_tiles(config)
    opponents = _riichi_targets(riichi_players, piles)
    profile = config.player_profile if config is not None else None
    inferred_turn = turn_number if turn_number is not None else max(
        1,
        sum(len(items) for items in piles.values()) // 4 + 1,
    )
    risk_weight = _defense_weight(
        min((int(item.get("shanten", 8)) for item in options), default=8),
        len(opponents),
        _play_style(config),
        risk_tolerance=getattr(profile, "risk_tolerance", "balanced"),
    )
    reference_effective_count = max(
        (int(item.get("effective_count", 0)) for item in options),
        default=0,
    )

    candidates: list[dict[str, Any]] = []
    for option in options:
        tile = normalize_tile(str(option.get("tile") or ""))
        canonical = _canonical_tile(tile)
        if canonical not in TILE_TYPES:
            continue
        shanten = int(option.get("shanten", 8))
        effective_types = int(option.get("effective_types", 0))
        effective_count = int(option.get("effective_count", 0))
        attack_score = _attack_score(
            tile,
            shanten,
            effective_types,
            effective_count,
            hand_counts,
            value_honors,
            dora_tiles,
            goal_bias=getattr(profile, "goal_bias", "balanced"),
        )
        visibility = _tile_visibility_evidence(
            canonical,
            hand_counts=hand_counts,
            river_result=river_result,
            self_meld_tiles=meld_tiles or [],
        )
        risk_by_player: dict[str, dict[str, Any]] = {}
        risks: list[float] = []
        for label, player_key in opponents:
            discards = [
                _canonical_tile(str(item.get("tile") or ""))
                for item in piles.get(player_key, [])
                if isinstance(item, dict)
            ] if player_key else []
            risk, basis, evidence, risk_components = _tile_risk_against_player(
                canonical,
                discards,
                visible_counts,
                dora_tiles,
                known_player=bool(player_key),
                player_key=player_key,
                visibility=visibility,
            )
            risk_by_player[label] = {
                "risk": risk,
                "basis": basis,
                "evidence": evidence,
                "player": player_key or label,
                "player_name": _PLAYER_DISPLAY_NAMES.get(player_key, "立直家"),
                "components": risk_components,
            }
            risks.append(risk)
        primary_risk_item = max(risk_by_player.values(), key=lambda item: float(item.get("risk", 0.0)), default={})
        defense_risk = max(risks, default=0.0)
        multi_riichi_adjustment = 0.0
        if defense_risk > 0 and len(risks) > 1:
            multi_riichi_adjustment = min(10.0, (len(risks) - 1) * 4.0)
            defense_risk = min(100.0, defense_risk + multi_riichi_adjustment)
        total_score = round(attack_score - risk_weight * defense_risk, 2)
        safety = _candidate_safety_label(risk_by_player)
        candidates.append(
            {
                "tile": tile,
                "total_score": total_score,
                "attack_score": round(attack_score, 2),
                "defense_risk": round(defense_risk, 2),
                "risk_level": _risk_level(defense_risk),
                "risk_scale_note": _RISK_SCALE_NOTE,
                "risk_components": {
                    "primary_player": str(primary_risk_item.get("player") or ""),
                    "primary_player_name": str(primary_risk_item.get("player_name") or "立直家"),
                    "base_risk": float((primary_risk_item.get("components") or {}).get("base_risk") or 0.0),
                    "base_reason": str((primary_risk_item.get("components") or {}).get("base_reason") or "无立直压力"),
                    "dora_adjustment": float((primary_risk_item.get("components") or {}).get("dora_adjustment") or 0.0),
                    "dora_reason": str((primary_risk_item.get("components") or {}).get("dora_reason") or ""),
                    "primary_risk": float(primary_risk_item.get("risk") or 0.0),
                    "multi_riichi_adjustment": multi_riichi_adjustment,
                    "final_risk": round(defense_risk, 2),
                    "capped": bool(defense_risk >= 100.0),
                },
                "risk_weight": round(risk_weight, 3),
                "shanten": shanten,
                "effective_types": effective_types,
                "effective_count": effective_count,
                "effective_count_delta": effective_count - reference_effective_count,
                "effective_tiles": list(option.get("effective_tiles") or []),
                "shape_loss": max(0, shanten - current_shanten),
                "estimated_value": _estimated_hand_value(
                    tiles,
                    dora_tiles=dora_tiles,
                    value_honors=value_honors,
                    goal_bias=getattr(profile, "goal_bias", "balanced"),
                ),
                "safety": safety,
                "visibility": visibility,
                "safety_evidence": [
                    str(item.get("evidence") or "")
                    for item in risk_by_player.values()
                    if str(item.get("evidence") or "").strip()
                ],
                "risk_by_player": risk_by_player,
            }
        )

    candidates.sort(
        key=lambda item: (
            -float(item["total_score"]),
            int(item["shanten"]),
            -int(item["effective_count"]),
            float(item["defense_risk"]),
            _tile_sort_key(str(item["tile"])),
        )
    )
    provisional = candidates[0] if candidates else {}
    table_context = _table_match_context(
        player_scores,
        player_ranks,
        honba_count=honba_count,
        riichi_stick_count=riichi_stick_count,
    )
    risk_budget_model = _risk_budget_breakdown(
        shanten=int(provisional.get("shanten", current_shanten)),
        riichi_count=len(opponents),
        turn_number=inferred_turn,
        estimated_value=int(provisional.get("estimated_value", 0)),
        risk_tolerance=getattr(profile, "risk_tolerance", "balanced"),
        room=getattr(profile, "room", "unknown"),
        rank=getattr(profile, "rank", "unknown"),
        player_scores=player_scores,
        player_ranks=player_ranks,
        honba_count=honba_count,
        riichi_stick_count=riichi_stick_count,
    )
    risk_budget = float(risk_budget_model["final"])
    safety_limit = max(risk_budget, 10.0)
    for candidate in candidates:
        risk = float(candidate.get("defense_risk", 100.0))
        candidate["within_risk_budget"] = risk <= risk_budget
        candidate["safety_eligible"] = risk <= safety_limit
        candidate["risk_budget_margin"] = round(risk_budget - risk, 2)
        components = candidate.get("risk_components") if isinstance(candidate.get("risk_components"), dict) else {}
        base_risk = float(components.get("base_risk") or 0.0)
        base_reason = str(components.get("base_reason") or "无立直压力")
        player_name = str(components.get("primary_player_name") or "立直家")
        calculation_parts = [f"{player_name}{base_reason}基础{base_risk:.0f}"]
        dora_adjustment = float(components.get("dora_adjustment") or 0.0)
        if dora_adjustment:
            calculation_parts.append(f"{str(components.get('dora_reason') or '宝牌因素')}+{dora_adjustment:.0f}")
        multi_adjustment = float(components.get("multi_riichi_adjustment") or 0.0)
        if multi_adjustment:
            calculation_parts.append(f"多家立直压力+{multi_adjustment:.0f}")
        calculation = " + ".join(calculation_parts) + f" = {risk:.0f}/100"
        if bool(components.get("capped")):
            calculation += "（100封顶）"
        candidate["risk_calculation"] = calculation
        if risk <= risk_budget:
            candidate["risk_budget_explanation"] = (
                f"{risk:.0f} ≤ 当前可接受上限{risk_budget:.0f}，低{risk_budget - risk:.0f}，可进入候选。"
            )
        else:
            candidate["risk_budget_explanation"] = (
                f"{risk:.0f} > 当前可接受上限{risk_budget:.0f}，超{risk - risk_budget:.0f}，默认不推荐。"
            )

    # A mawashi decision is two-stage: first find tiles inside the safety
    # allowance, then choose the one that preserves the most hand potential.
    # Looking only at the attack/defense total can otherwise declare a fold
    # even when a slightly slower safe discard keeps tenpai or iishanten.
    safety_candidates = [item for item in candidates if item["safety_eligible"]]
    posture_reference = min(
        safety_candidates,
        key=lambda item: (
            int(item["shape_loss"]),
            int(item["shanten"]),
            -int(item["effective_count"]),
            -float(item["attack_score"]),
            float(item["defense_risk"]),
            _tile_sort_key(str(item["tile"])),
        ),
        default=provisional,
    )
    posture = _choose_defense_posture(
        posture_reference,
        risk_budget=risk_budget,
        previous_posture=previous_posture,
        riichi_count=len(opponents),
    ) if opponents else DefensePosture.PUSH.value
    if posture == DefensePosture.FOLD.value:
        candidates.sort(
            key=lambda item: (
                float(item["defense_risk"]),
                int(item["shape_loss"]),
                int(item["shanten"]),
                -int(item["effective_count"]),
                _tile_sort_key(str(item["tile"])),
            )
        )
    elif posture == DefensePosture.MAWASHI.value:
        candidates.sort(
            key=lambda item: (
                not bool(item["safety_eligible"]),
                int(item["shape_loss"]),
                int(item["shanten"]),
                -int(item["effective_count"]),
                -float(item["attack_score"]),
                float(item["defense_risk"]),
                _tile_sort_key(str(item["tile"])),
            )
        )
    elif posture == DefensePosture.PUSH.value:
        candidates.sort(
            key=lambda item: (
                -float(item["attack_score"]),
                int(item["shanten"]),
                float(item["defense_risk"]),
                -int(item["effective_count"]),
                _tile_sort_key(str(item["tile"])),
            )
        )
    for index, candidate in enumerate(candidates, start=1):
        candidate["rank"] = index
    best = candidates[0] if candidates else {}
    legacy_mode = {
        DefensePosture.PUSH.value: "attack",
        DefensePosture.MAWASHI.value: "balanced",
        DefensePosture.FOLD.value: "defense",
    }[posture]
    win_potential = _candidate_win_potential(best)
    return {
        "mode": posture,
        "posture": posture,
        "legacy_mode": legacy_mode,
        "riichi_players": [label for label, _ in opponents],
        "riichi_count": len(opponents),
        "risk_weight": round(risk_weight, 3),
        "risk_budget": round(risk_budget, 2),
        "risk_scale_note": _RISK_SCALE_NOTE,
        "risk_scale_legend": _RISK_SCALE_LEGEND,
        "risk_model_legend": _RISK_MODEL_LEGEND,
        "risk_budget_model": risk_budget_model,
        "risk_budget_calculation": str(risk_budget_model["calculation"]),
        "table_context": table_context,
        "safe_shape_candidates": len(safety_candidates),
        "win_potential": win_potential,
        "preserve_win_chance": posture != DefensePosture.FOLD.value and win_potential != "weak",
        "turn_number": inferred_turn,
        "profile": profile.to_dict() if profile is not None else {},
        "explanation_depth": _profile_explanation_depth(getattr(profile, "rank", "unknown")),
        "candidates": candidates,
        "top_candidates": candidates[:3],
        "source": "local_efficiency_plus_river_risk",
    }


def _riichi_targets(
    riichi_players: list[str],
    discard_piles: dict[str, list[dict[str, Any]]],
) -> list[tuple[str, str]]:
    targets: list[tuple[str, str]] = []
    seen: set[str] = set()
    for index, player in enumerate(riichi_players, start=1):
        key = _player_key(player)
        label = key or f"unknown_{index}"
        if label in seen:
            continue
        seen.add(label)
        targets.append((label, key))
    if not targets and riichi_players:
        targets.append(("unknown_1", ""))
    return targets


def _risk_budget_breakdown(
    *,
    shanten: int,
    riichi_count: int,
    turn_number: int,
    estimated_value: int,
    risk_tolerance: str,
    room: str,
    rank: str,
    player_scores: dict[str, int] | None = None,
    player_ranks: dict[str, int] | None = None,
    honba_count: int | None = None,
    riichi_stick_count: int | None = None,
) -> dict[str, Any]:
    if shanten <= 0:
        base, base_label = 62.0, "听牌基础"
    elif shanten == 1:
        base, base_label = 44.0, "一向听基础"
    elif shanten == 2:
        base, base_label = 27.0, "二向听基础"
    else:
        base, base_label = 14.0, "三向听以上基础"
    style_adjustment = {"conservative": -10.0, "balanced": 0.0, "aggressive": 12.0}.get(risk_tolerance, 0.0)
    style_label = {
        "conservative": "保守风格",
        "balanced": "均衡风格",
        "aggressive": "激进风格",
    }.get(risk_tolerance, "默认风格")
    multi_riichi_adjustment = -max(0, riichi_count - 1) * 12.0
    turn_adjustment = 0.0
    turn_label = ""
    if turn_number >= 16:
        turn_adjustment, turn_label = -12.0, "16巡后"
    elif turn_number >= 12:
        turn_adjustment, turn_label = -7.0, "12巡后"
    table_context = _table_match_context(
        player_scores,
        player_ranks,
        honba_count=honba_count,
        riichi_stick_count=riichi_stick_count,
    )
    table_reward_bonus = int(table_context.get("win_reward_bonus") or 0)
    estimated_value_with_rewards = max(0, int(estimated_value)) + table_reward_bonus
    value_adjustment = 0.0
    value_label = ""
    reward_suffix = "（含本场/供托）" if table_reward_bonus else ""
    if estimated_value_with_rewards >= 7700:
        value_adjustment, value_label = 8.0, f"预计7700点以上{reward_suffix}"
    elif estimated_value_with_rewards >= 3900:
        value_adjustment, value_label = 4.0, f"预计3900点以上{reward_suffix}"
    room_adjustment = -2.0 if room in {"jade", "throne"} else 0.0
    room_label = {"jade": "玉之间", "throne": "王座之间"}.get(room, "")
    # Rank only nudges genuinely marginal thresholds. Explicit risk style has
    # a much larger effect and therefore remains authoritative.
    rank_adjustment = {
        "novice": -1.0,
        "adept": 0.0,
        "expert": 0.0,
        "master": 1.0,
        "saint": 2.0,
        "celestial": 2.0,
    }.get(rank, 0.0)
    rank_label = {
        "novice": "初心",
        "adept": "雀士",
        "expert": "雀杰",
        "master": "雀豪",
        "saint": "雀圣",
        "celestial": "魂天",
    }.get(rank, "")
    placement_adjustment = 0.0
    placement_label = ""
    self_rank = int(table_context.get("self_rank") or 0)
    gap_above = int(table_context.get("gap_above") or 0)
    lead_below = int(table_context.get("lead_below") or 0)
    if self_rank == 4 and gap_above > 0:
        placement_adjustment = 8.0 if gap_above >= 8_000 else 5.0
        placement_label = f"四位追三位差{gap_above}点"
    elif self_rank == 3 and gap_above >= 8_000:
        placement_adjustment = 4.0
        placement_label = f"三位追二位差{gap_above}点"
    elif self_rank == 1 and lead_below > 0:
        placement_adjustment = -8.0 if lead_below >= 12_000 else -4.0
        placement_label = f"一位领先{lead_below}点保护"
    elif self_rank == 2 and lead_below >= 8_000:
        placement_adjustment = -3.0
        placement_label = f"二位领先三位{lead_below}点保护"
    components = [
        {"label": base_label, "value": base},
        {"label": style_label, "value": style_adjustment},
        {"label": f"{riichi_count}家立直压力", "value": multi_riichi_adjustment},
    ]
    for label, value in (
        (turn_label, turn_adjustment),
        (value_label, value_adjustment),
        (room_label, room_adjustment),
        (rank_label, rank_adjustment),
        (placement_label, placement_adjustment),
    ):
        if label and value:
            components.append({"label": label, "value": value})
    raw = sum(float(item["value"]) for item in components)
    final = max(5.0, min(78.0, raw))
    formula_parts = [f"{components[0]['label']}{base:.0f}"]
    for component in components[1:]:
        value = float(component["value"])
        sign = "+" if value >= 0 else "−"
        formula_parts.append(f"{component['label']}{sign}{abs(value):.0f}")
    calculation = " ".join(formula_parts) + f" = 上限{final:.0f}"
    if final != raw:
        calculation += f"（原始{raw:.0f}，限制在5–78）"
    return {
        "base": base,
        "components": components,
        "raw": round(raw, 2),
        "final": round(final, 2),
        "calculation": calculation,
        "table_context": table_context,
        "estimated_value_before_table_rewards": max(0, int(estimated_value)),
        "table_reward_bonus": table_reward_bonus,
        "estimated_value_with_table_rewards": estimated_value_with_rewards,
        "placement_adjustment": placement_adjustment,
    }


def _risk_budget(
    *,
    shanten: int,
    riichi_count: int,
    turn_number: int,
    estimated_value: int,
    risk_tolerance: str,
    room: str,
    rank: str,
    player_scores: dict[str, int] | None = None,
    player_ranks: dict[str, int] | None = None,
    honba_count: int | None = None,
    riichi_stick_count: int | None = None,
) -> float:
    return float(_risk_budget_breakdown(
        shanten=shanten,
        riichi_count=riichi_count,
        turn_number=turn_number,
        estimated_value=estimated_value,
        risk_tolerance=risk_tolerance,
        room=room,
        rank=rank,
        player_scores=player_scores,
        player_ranks=player_ranks,
        honba_count=honba_count,
        riichi_stick_count=riichi_stick_count,
    )["final"])


def _profile_explanation_depth(rank: str) -> str:
    if rank in {"master", "saint", "celestial"}:
        return "advanced"
    if rank in {"expert", "adept"}:
        return "standard"
    return "guided"


def _choose_defense_posture(
    candidate: dict[str, Any],
    *,
    risk_budget: float,
    previous_posture: str,
    riichi_count: int,
) -> str:
    risk = float(candidate.get("defense_risk", 100.0))
    shanten = int(candidate.get("shanten", 8))
    effective_count = int(candidate.get("effective_count", 0))
    has_live_route = shanten <= 2 and effective_count >= 3
    if risk <= max(8.0, risk_budget * 0.62) and shanten <= 1 and effective_count >= 4:
        computed = DefensePosture.PUSH.value
    elif (risk <= risk_budget or risk <= 10.0) and has_live_route:
        computed = DefensePosture.MAWASHI.value
    else:
        computed = DefensePosture.FOLD.value
    previous = str(previous_posture or "")
    if previous not in {item.value for item in DefensePosture} or previous == computed:
        return computed
    # Hysteresis: immediately respect a large danger jump, but require a clear
    # margin before returning from fold to push on a neighbouring frame.
    if computed == DefensePosture.FOLD.value and risk >= risk_budget + 12.0:
        return computed
    if computed == DefensePosture.PUSH.value and risk <= risk_budget - 10.0 and riichi_count == 1:
        return computed
    return DefensePosture.MAWASHI.value


def _candidate_win_potential(candidate: dict[str, Any]) -> str:
    shanten = int(candidate.get("shanten", 8))
    effective_count = int(candidate.get("effective_count", 0))
    if shanten <= 1 and effective_count >= 4:
        return "strong"
    if shanten <= 2 and effective_count > 0:
        return "live"
    return "weak"


def _estimated_hand_value(
    tiles: list[str],
    *,
    dora_tiles: set[str],
    value_honors: set[str],
    goal_bias: str,
) -> int:
    counts = Counter(_canonical_tile(tile) for tile in tiles)
    dora_count = sum(
        amount
        for tile, amount in counts.items()
        if tile in dora_tiles or tile in {"5m", "5p", "5s"} and any(raw == f"0{tile[1]}" for raw in tiles)
    )
    value_sets = sum(1 for tile in value_honors if counts[tile] >= 2)
    base = 1300 + dora_count * 1700 + value_sets * 900
    if goal_bias == "value":
        base += 700
    elif goal_bias == "yakuman":
        base += 1000
    return min(32000, base)


def _defense_weight(
    shanten: int,
    riichi_count: int,
    style: str,
    *,
    risk_tolerance: str = "balanced",
) -> float:
    if riichi_count <= 0:
        return 0.0
    if shanten <= 0:
        weight = 0.35
    elif shanten == 1:
        weight = 0.65
    elif shanten == 2:
        weight = 0.9
    else:
        weight = 1.05
    if style == "fast" and shanten <= 1:
        weight -= 0.1
    if risk_tolerance == "aggressive":
        weight -= 0.12
    elif risk_tolerance == "conservative":
        weight += 0.14
    if riichi_count > 1:
        weight += min(0.25, (riichi_count - 1) * 0.15)
    return max(0.0, min(1.25, weight))


def _attack_score(
    tile: str,
    shanten: int,
    effective_types: int,
    effective_count: int,
    hand_counts: Counter[str],
    value_honors: set[str],
    dora_tiles: set[str],
    *,
    goal_bias: str = "balanced",
) -> float:
    canonical = _canonical_tile(tile)
    score = 100.0 - max(-1, shanten) * 24.0
    score += min(24.0, effective_count * 1.2)
    score += min(12.0, effective_types * 2.0)
    if tile in {"0m", "0p", "0s"} or _is_dora(canonical, dora_tiles):
        score -= 22.0
    if canonical in value_honors and hand_counts[canonical] >= 2:
        score -= 14.0
    if goal_bias == "speed":
        score += min(10.0, effective_count * 0.45)
    elif goal_bias in {"value", "yakuman"} and (tile in {"0m", "0p", "0s"} or _is_dora(canonical, dora_tiles)):
        score -= 8.0
    return score


def _tile_visibility_evidence(
    tile: str,
    *,
    hand_counts: Counter[str],
    river_result: RiverStateResult | None,
    self_meld_tiles: list[str],
) -> dict[str, Any]:
    """Describe exactly which copies of a candidate tile are currently known."""
    canonical = _canonical_tile(tile)
    tile_name = _tile_name(canonical)
    sources: list[dict[str, Any]] = []

    def add_source(label: str, count: int) -> None:
        if count > 0:
            sources.append({"label": label, "count": int(count), "tile": canonical, "tile_name": tile_name})

    add_source("手牌", int(hand_counts.get(canonical, 0)))
    if river_result is not None and river_result.ok:
        has_discard_records = any(
            isinstance(item, dict)
            for items in river_result.discard_piles.values()
            for item in items
        )
        if has_discard_records:
            for player in _TABLE_PLAYERS:
                count = sum(
                    1
                    for item in river_result.discard_piles.get(player, [])
                    if isinstance(item, dict)
                    and not item.get("claimed_into_meld")
                    and _canonical_tile(str(item.get("tile") or "")) == canonical
                )
                add_source(f"{_PLAYER_DISPLAY_NAMES.get(player, player)}牌河", count)
        else:
            add_source(
                "牌河识别",
                sum(1 for item in river_result.visible_tiles if _canonical_tile(str(item)) == canonical),
            )

        mapped_meld_count = 0
        for player, melds in river_result.opponent_melds.items():
            count = sum(
                1
                for meld in melds
                if isinstance(meld, dict) and meld.get("tile_identity_reliable") is not False
                for item in meld.get("tiles", [])
                if _canonical_tile(str(item)) == canonical
            )
            mapped_meld_count += count
            add_source(f"{_PLAYER_DISPLAY_NAMES.get(player, player)}副露", count)
        if mapped_meld_count == 0:
            add_source(
                "对手副露",
                sum(1 for item in river_result.opponent_meld_tiles if _canonical_tile(str(item)) == canonical),
            )

    add_source(
        "自己的副露",
        sum(1 for item in self_meld_tiles if _canonical_tile(str(item)) == canonical),
    )
    known_count = min(4, sum(int(item["count"]) for item in sources))
    unseen_count = max(0, 4 - known_count)
    if sources:
        source_text = "、".join(
            f"{item['label']}{item['tile_name']}×{item['count']}"
            for item in sources
        )
        summary = f"{tile_name}已知{known_count}枚：{source_text}；尚有{tile_name}×{unseen_count}未见"
    else:
        summary = f"尚未在手牌、牌河或副露区看见{tile_name}；{tile_name}×4均未见"
    return {
        "tile": canonical,
        "tile_name": tile_name,
        "known_count": known_count,
        "unseen_count": unseen_count,
        "sources": sources,
        "summary": summary,
    }


def _suji_anchor_tiles(tile: str) -> list[str]:
    canonical = _canonical_tile(tile)
    suit = tile_suit(canonical)
    if suit not in {"m", "p", "s"}:
        return []
    return [
        f"{rank}{suit}"
        for rank in range(1, 10)
        if canonical in _suji_tiles_from_discards([f"{rank}{suit}"])
    ]


def _risk_level(value: float) -> str:
    risk = max(0.0, min(100.0, float(value)))
    if risk <= 9:
        return "极低"
    if risk <= 39:
        return "较低"
    if risk <= 59:
        return "中等"
    if risk <= 79:
        return "高"
    return "很高"


def _tile_risk_against_player(
    tile: str,
    player_discards: list[str],
    visible_counts: Counter[str],
    dora_tiles: set[str],
    *,
    known_player: bool,
    player_key: str = "",
    visibility: dict[str, Any] | None = None,
) -> tuple[float, str, str, dict[str, Any]]:
    tile_name = _tile_name(tile)
    player_name = _PLAYER_DISPLAY_NAMES.get(player_key, "立直家")
    visible_summary = str((visibility or {}).get("summary") or "")
    if not known_player:
        return 90.0, "立直家座位未知", f"没有识别出立直家的座位，无法核对{tile_name}是否为该家的现物或筋。", {
            "base_risk": 90.0,
            "base_reason": "立直家座位未知",
            "dora_adjustment": 0.0,
            "dora_reason": "",
            "final_risk": 90.0,
        }
    discards = {_canonical_tile(item) for item in player_discards if _canonical_tile(item) in TILE_TYPES}
    if tile in discards:
        return 0.0, "现物", f"{player_name}牌河已经出现{tile_name}，所以打{tile_name}对该家是现物。", {
            "base_risk": 0.0,
            "base_reason": "现物",
            "dora_adjustment": 0.0,
            "dora_reason": "",
            "final_risk": 0.0,
        }
    if visible_counts[tile] >= 4:
        detail = visible_summary or f"{tile_name}四枚均已识别"
        return 5.0, "全见", f"{detail}；四枚位置都已知，不再有同牌未知张。", {
            "base_risk": 5.0,
            "base_reason": "全见",
            "dora_adjustment": 0.0,
            "dora_reason": "",
            "final_risk": 5.0,
        }
    suji = _suji_tiles_from_discards(list(discards))
    if tile in suji:
        risk, basis = 30.0, "筋"
        supports = [item for item in discards if tile in _suji_tiles_from_discards([item])]
        support_text = _tile_list_in_order(supports) or "对应筋牌"
        evidence = f"{player_name}牌河见{support_text}，因此{tile_name}按筋降低风险；筋仍不是绝对安全。"
    elif _is_wall_related(tile, visible_counts):
        risk, basis = 42.0, "壁相关"
        value = int(tile_rank(tile))
        support_tiles = [
            f"{neighbor}{tile_suit(tile)}"
            for neighbor in (value - 1, value + 1)
            if 1 <= neighbor <= 9 and visible_counts[f"{neighbor}{tile_suit(tile)}"] >= 4
        ]
        evidence = (
            f"{_tile_list_in_order(support_tiles)}各已知4枚，形成{tile_name}相邻的壁；"
            "壁只降低顺子等待风险，不排除双碰或单骑。"
        )
    elif is_honor(tile):
        shown = visible_counts[tile]
        risk, basis = (38.0 if shown >= 3 else 55.0 if shown == 2 else 72.0), f"字牌已见{shown}枚"
        evidence = (
            f"{visible_summary}；据此按{tile_name}自身的已知张数判断，"
            f"不是把其他字牌合并成{shown}枚。"
        )
    else:
        rank = int(tile_rank(tile))
        anchors = _tile_list_in_order(_suji_anchor_tiles(tile))
        evidence = (
            f"{player_name}牌河未见{tile_name}，它不是现物；"
            f"也未见可支持{tile_name}筋判断的{anchors or '对应牌'}。"
        )
        if rank in {1, 9}:
            risk, basis = 58.0, "无筋幺九"
        elif rank in {2, 8}:
            risk, basis = 72.0, "无筋2/8"
        else:
            risk, basis = 84.0, "无筋中张"
    base_risk = risk
    base_reason = basis
    dora_adjustment = 0.0
    dora_reason = ""
    if _is_dora(tile, dora_tiles):
        risk += 14.0
        basis += "+宝牌"
        dora_adjustment = 14.0
        dora_reason = "宝牌"
        evidence += f"{tile_name}本身是宝牌，危险度再上调14。"
    elif _is_dora_adjacent(tile, dora_tiles):
        risk += 8.0
        basis += "+宝牌周边"
        dora_adjustment = 8.0
        dora_reason = "宝牌周边"
        adjacent_dora = [
            item
            for item in dora_tiles
            if tile_suit(item) == tile_suit(tile)
            and tile_rank(item).isdigit()
            and abs(int(tile_rank(item)) - int(tile_rank(tile))) == 1
        ]
        evidence += f"{tile_name}邻近宝牌{_tile_list_in_order(adjacent_dora)}，危险度再上调8。"
    final_risk = min(100.0, risk)
    return final_risk, basis, evidence, {
        "base_risk": base_risk,
        "base_reason": base_reason,
        "dora_adjustment": dora_adjustment,
        "dora_reason": dora_reason,
        "uncapped_risk": risk,
        "final_risk": final_risk,
    }


def _is_dora_adjacent(tile: str, dora_tiles: set[str]) -> bool:
    suit = tile_suit(tile)
    rank = tile_rank(tile)
    if suit not in {"m", "p", "s"} or not rank.isdigit():
        return False
    value = int(rank)
    return any(
        tile_suit(dora) == suit
        and tile_rank(dora).isdigit()
        and abs(int(tile_rank(dora)) - value) == 1
        for dora in dora_tiles
    )


def _candidate_safety_label(risk_by_player: dict[str, dict[str, Any]]) -> str:
    if not risk_by_player:
        return "无立直压力"
    bases = [str(item.get("basis") or "") for item in risk_by_player.values()]
    if bases and all(basis == "现物" for basis in bases):
        return "共同现物"
    if any(basis == "现物" for basis in bases):
        return "单家现物，另家有风险"
    if bases and all(basis == "筋" for basis in bases):
        return "共同筋（非绝对安全）"
    return " / ".join(dict.fromkeys(bases))


def _discard_ranking_text(ranking: dict[str, Any]) -> str:
    top = [item for item in ranking.get("top_candidates", []) if isinstance(item, dict)]
    if not top:
        return "未得到稳定的攻守候选，先按现物、筋、壁顺序保守判断。"
    labels = ("首选", "次选", "第三")
    parts: list[str] = []
    for label, item in zip(labels, top, strict=False):
        tile = _tile_name(str(item.get("tile") or ""))
        shanten = _shanten_text(int(item.get("shanten", 8)))
        effective = int(item.get("effective_count", 0))
        safety = str(item.get("safety") or "风险未知")
        parts.append(f"{label}打{tile}（{shanten}，有效{effective}枚，{safety}）")
    mode_names = {
        "push": "推进",
        "mawashi": "兜牌",
        "fold": "全退",
        "attack": "推进",
        "balanced": "兜牌",
        "defense": "全退",
    }
    has_genbutsu = any("现物" in str(item.get("safety") or "") for item in top)
    warning = "手里没有对全部立直者成立的现物；" if ranking.get("riichi_count") and not has_genbutsu else ""
    budget = float(ranking.get("risk_budget") or 0.0)
    mode = str(ranking.get("mode") or "")
    best = top[0]
    best_tile = _tile_name(str(best.get("tile") or ""))
    best_shanten = _shanten_text(int(best.get("shanten", 8)))
    best_effective = int(best.get("effective_count", 0))
    best_risk = float(best.get("defense_risk") or 0.0)
    if mode == DefensePosture.MAWASHI.value:
        lead = (
            f"兜牌（风险预算{budget:.0f}）：先打{best_tile}（风险{best_risk:.0f}），"
            f"保持{best_shanten}、有效{best_effective}枚，继续保留和牌路线。候选："
        )
    elif mode == DefensePosture.PUSH.value:
        lead = f"推进（风险预算{budget:.0f}）：手牌仍有较强和牌潜力，在预算内继续做牌。候选："
    elif mode == DefensePosture.FOLD.value:
        lead = f"全退（风险预算{budget:.0f}）：当前和牌潜力不足以覆盖风险，安全优先。候选："
    else:
        lead = f"{mode_names.get(mode, '攻守排序')}（风险预算{budget:.0f}）："
    return lead + warning + "；".join(parts) + "。"


def _suji_tiles_from_discards(discards: list[str]) -> set[str]:
    result: set[str] = set()
    targets_by_rank = {
        1: (4,), 2: (5,), 3: (6,), 4: (1, 7), 5: (2, 8),
        6: (3, 9), 7: (4,), 8: (5,), 9: (6,),
    }
    for tile in discards:
        canonical = _canonical_tile(tile)
        suit = tile_suit(canonical)
        rank = tile_rank(canonical)
        if suit not in {"m", "p", "s"} or not rank.isdigit():
            continue
        for target in targets_by_rank[int(rank)]:
            result.add(f"{target}{suit}")
    return result


def _is_wall_related(tile: str, visible_counts: Counter[str]) -> bool:
    canonical = _canonical_tile(tile)
    suit = tile_suit(canonical)
    rank = tile_rank(canonical)
    if suit not in {"m", "p", "s"} or not rank.isdigit():
        return False
    value = int(rank)
    adjacent = []
    if value > 1:
        adjacent.append(f"{value - 1}{suit}")
    if value < 9:
        adjacent.append(f"{value + 1}{suit}")
    return any(visible_counts[neighbor] >= 4 for neighbor in adjacent)


def _suit_shape(tiles: list[str], suit: str) -> str:
    ranks = [tile_rank(tile) for tile in tiles if tile_suit(tile) == suit]
    return "".join(ranks) or "-"


def _suit_breakdown(tiles: list[str]) -> str:
    parts = []
    for suit in ("m", "p", "s", "z"):
        shape = _suit_shape(tiles, suit)
        if shape != "-":
            parts.append(f"{SUIT_NAMES[suit]}{shape}")
    return " / ".join(parts) if parts else "暂无稳定手牌"


def _keep_candidates(tiles: list[str], counts: Counter[str], best_suit: str, dora_tiles: set[str]) -> list[str]:
    keep: list[str] = []
    for tile in tiles:
        normalized = normalize_tile(tile)
        if counts[normalized] >= 2 or normalized in {"0m", "0p", "0s"} or _is_dora(normalized, dora_tiles):
            keep.append(normalized)
            continue
        if tile_suit(normalized) == best_suit and _has_neighbor(normalized, counts, distance=2):
            keep.append(normalized)
    return keep


def _cleanup_candidates(
    tiles: list[str],
    counts: Counter[str],
    best_suit: str,
    value_honors: set[str],
    dora_tiles: set[str],
) -> list[str]:
    cleanup: list[str] = []
    for tile in sorted((normalize_tile(tile) for tile in tiles if normalize_tile(tile)), key=_tile_sort_key):
        if counts[tile] >= 2 or tile in {"0m", "0p", "0s"} or _is_dora(tile, dora_tiles):
            continue
        suit = tile_suit(tile)
        if suit == "z":
            cleanup.append(tile)
        elif suit != best_suit and not _has_neighbor(tile, counts, distance=1):
            cleanup.append(tile)
        elif suit == best_suit and is_terminal(tile) and not _has_neighbor(tile, counts, distance=1):
            cleanup.append(tile)
    return sorted(cleanup, key=lambda tile: _cleanup_score(tile, counts, best_suit, value_honors))[:5]


def _cleanup_score(tile: str, counts: Counter[str], best_suit: str, value_honors: set[str]) -> tuple[int, int, str]:
    suit = tile_suit(tile)
    rank = tile_rank(tile)
    rank_value = int(rank) if rank.isdigit() else 0
    if suit == "z":
        if tile in value_honors:
            return (3, rank_value, tile)
        return (0, rank_value, tile)
    if best_suit and suit != best_suit and not _has_neighbor(tile, counts, distance=1):
        return (1 if is_terminal(tile) else 2, rank_value, tile)
    if suit == best_suit and is_terminal(tile) and not _has_neighbor(tile, counts, distance=1):
        return (4, rank_value, tile)
    return (6, rank_value, tile)


def _discard_candidates(
    tiles: list[str],
    counts: Counter[str],
    best_suit: str,
    cleanup_tiles: list[str],
    keep_tiles: list[str],
    value_honors: set[str],
    dora_tiles: set[str],
) -> list[str]:
    selected: list[str] = []
    keep_set = set(keep_tiles)
    for tile in sorted(cleanup_tiles, key=lambda tile: _discard_score(tile, counts, best_suit, keep_set, value_honors, dora_tiles)):
        if tile not in selected:
            selected.append(tile)

    pool: list[str] = []
    for tile in tiles:
        normalized = normalize_tile(tile)
        if not normalized or normalized in selected or normalized in {"0m", "0p", "0s"} or _is_dora(normalized, dora_tiles):
            continue
        if counts[normalized] >= 2:
            continue
        pool.append(normalized)

    pool = sorted(set(pool), key=lambda tile: _discard_score(tile, counts, best_suit, keep_set, value_honors, dora_tiles))
    for tile in pool:
        if len(selected) >= 5:
            break
        selected.append(tile)
    return selected[:4]


def _efficiency_analysis(
    tiles: list[str],
    initial_discards: list[str],
    counts: Counter[str],
    visible_counts: Counter[str],
    best_suit: str,
    value_honors: set[str],
    dora_tiles: set[str],
    *,
    open_melds: int = 0,
) -> dict[str, Any]:
    canonical_tiles = [_canonical_tile(tile) for tile in tiles if _canonical_tile(tile) in TILE_TYPES]
    if not canonical_tiles:
        return {
            "current_shanten": 8,
            "best_path": "standard",
            "discard_options": [],
            "all_discard_options": [],
            "open_melds": _clamp_melds(open_melds),
            "closed_tile_count": 0,
        }

    open_meld_count = _clamp_melds(open_melds)
    current = _best_hand_shanten(canonical_tiles, open_melds=open_meld_count)
    candidates = _efficiency_discard_candidates(tiles, initial_discards, counts, best_suit, value_honors, dora_tiles)
    options: list[dict[str, Any]] = []
    visible_total = sum(visible_counts.values())

    if len(canonical_tiles) % 3 == 2:
        hand_counts = Counter(canonical_tiles)
        for tile in candidates:
            canonical = _canonical_tile(tile)
            if canonical not in hand_counts:
                continue
            remaining = list(canonical_tiles)
            remaining.remove(canonical)
            remaining_counts = Counter(remaining)
            post_discard = _best_hand_shanten(remaining, open_melds=open_meld_count)["shanten"]
            effective_tiles: list[str] = []
            effective_count = 0
            for draw in TILE_TYPES:
                # Count from the post-discard hand. A discarded tile can itself be
                # an effective draw again (for example a tanki or shanpon wait).
                available = max(0, 4 - remaining_counts.get(draw, 0) - visible_counts.get(draw, 0))
                if available <= 0:
                    continue
                if _best_hand_shanten([*remaining, draw], open_melds=open_meld_count)["shanten"] < post_discard:
                    effective_tiles.append(draw)
                    effective_count += available
            options.append(
                {
                    "tile": tile,
                    "shanten": post_discard,
                    "effective_types": len(effective_tiles),
                    "effective_count": effective_count,
                    "effective_tiles": sorted(effective_tiles, key=_tile_sort_key),
                    "heuristic_score": _discard_score(tile, counts, best_suit, set(), value_honors, dora_tiles),
                }
            )

    options.sort(
        key=lambda item: (
            item["shanten"],
            -item["effective_count"],
            -item["effective_types"],
            item["heuristic_score"],
        )
    )
    serialized_options = [
        {
            "tile": str(item["tile"]),
            "shanten": item["shanten"],
            "effective_types": item["effective_types"],
            "effective_count": item["effective_count"],
            "effective_tiles": list(item["effective_tiles"]),
        }
        for item in options
    ]
    return {
        "current_shanten": current["shanten"],
        "best_path": current["path"],
        "open_melds": open_meld_count,
        "closed_tile_count": len(canonical_tiles),
        "discard_options": serialized_options[:5],
        "all_discard_options": serialized_options,
        "visible_tile_count": visible_total,
    }


def _efficiency_discard_candidates(
    tiles: list[str],
    initial_discards: list[str],
    counts: Counter[str],
    best_suit: str,
    value_honors: set[str],
    dora_tiles: set[str],
) -> list[str]:
    candidates = _merge_unique_tiles(initial_discards)
    keep_set: set[str] = set()
    for tile in sorted(
        {normalize_tile(tile) for tile in tiles if normalize_tile(tile)},
        key=lambda item: _discard_score(item, counts, best_suit, keep_set, value_honors, dora_tiles),
    ):
        if tile not in candidates:
            candidates.append(tile)
    return candidates


def _best_hand_shanten(tiles: list[str], *, open_melds: int = 0) -> dict[str, Any]:
    counts = _shanten_counts(tiles)
    open_meld_count = _clamp_melds(open_melds)
    standard = _standard_shanten(tuple(counts), open_meld_count)
    options = [("standard", standard)]
    if open_meld_count == 0:
        options.extend(
            [
                ("seven_pairs", _seven_pairs_shanten(counts)),
                ("thirteen_orphans", _thirteen_orphans_shanten(counts)),
            ]
        )
    path, shanten = min(options, key=lambda item: item[1])
    return {"path": path, "shanten": int(shanten)}


def _shanten_counts(tiles: list[str]) -> list[int]:
    counts = [0] * len(TILE_TYPES)
    for tile in tiles:
        index = _tile_index(_canonical_tile(tile))
        if index is not None:
            counts[index] += 1
    return counts


def _visible_counts(tiles: list[str]) -> Counter[str]:
    return Counter(_canonical_tile(tile) for tile in tiles if _canonical_tile(tile) in TILE_TYPES)


@lru_cache(maxsize=4096)
def _standard_shanten(counts_tuple: tuple[int, ...], open_melds: int = 0) -> int:
    open_meld_count = _clamp_melds(open_melds)

    @lru_cache(maxsize=262144)
    def walk(counts_state: tuple[int, ...], melds: int, taatsu: int, pair: int) -> int:
        counts = list(counts_state)
        capped_taatsu = min(taatsu, max(0, 4 - melds))
        best = 8 - (2 * melds) - capped_taatsu - pair
        index = next((offset for offset, value in enumerate(counts) if value > 0), -1)
        if index < 0:
            return best

        counts[index] -= 1
        best = min(best, walk(tuple(counts), melds, taatsu, pair))
        counts[index] += 1

        if melds < 4 and counts[index] >= 3:
            counts[index] -= 3
            best = min(best, walk(tuple(counts), melds + 1, taatsu, pair))
            counts[index] += 3

        if melds < 4 and _can_form_sequence(index) and counts[index + 1] > 0 and counts[index + 2] > 0:
            counts[index] -= 1
            counts[index + 1] -= 1
            counts[index + 2] -= 1
            best = min(best, walk(tuple(counts), melds + 1, taatsu, pair))
            counts[index] += 1
            counts[index + 1] += 1
            counts[index + 2] += 1

        if pair == 0 and counts[index] >= 2:
            counts[index] -= 2
            best = min(best, walk(tuple(counts), melds, taatsu, 1))
            counts[index] += 2

        if taatsu < 4 and counts[index] >= 2:
            counts[index] -= 2
            best = min(best, walk(tuple(counts), melds, taatsu + 1, pair))
            counts[index] += 2

        if taatsu < 4 and _can_form_sequence(index) and counts[index + 1] > 0:
            counts[index] -= 1
            counts[index + 1] -= 1
            best = min(best, walk(tuple(counts), melds, taatsu + 1, pair))
            counts[index] += 1
            counts[index + 1] += 1

        if taatsu < 4 and _can_form_sequence(index) and counts[index + 2] > 0:
            counts[index] -= 1
            counts[index + 2] -= 1
            best = min(best, walk(tuple(counts), melds, taatsu + 1, pair))
            counts[index] += 1
            counts[index + 2] += 1

        return best

    return walk(counts_tuple, open_meld_count, 0, 0)


def _seven_pairs_shanten(counts: list[int]) -> int:
    pairs = sum(1 for count in counts if count >= 2)
    unique = sum(1 for count in counts if count > 0)
    return 6 - pairs + max(0, 7 - unique)


def _thirteen_orphans_shanten(counts: list[int]) -> int:
    orphan_indexes = [_tile_index(tile) for tile in ORPHAN_TYPES]
    present = sum(1 for index in orphan_indexes if index is not None and counts[index] > 0)
    has_pair = any(index is not None and counts[index] >= 2 for index in orphan_indexes)
    return 13 - present - int(has_pair)


def _can_form_sequence(index: int) -> bool:
    return 0 <= index < 27 and index % 9 <= 6


def _tile_index(tile: str) -> int | None:
    canonical = _canonical_tile(tile)
    try:
        return TILE_TYPES.index(canonical)
    except ValueError:
        return None


def _canonical_tile(tile: str) -> str:
    normalized = normalize_tile(tile)
    if normalized in {"0m", "0p", "0s"}:
        return f"5{normalized[1]}"
    return normalized


def _best_efficiency_discard(efficiency: dict[str, Any]) -> str:
    options = efficiency.get("discard_options")
    if isinstance(options, list) and options:
        first = options[0]
        if isinstance(first, dict):
            return str(first.get("tile") or "")
    return ""


def _efficiency_note(efficiency: dict[str, Any]) -> str:
    current = _shanten_text(int(efficiency.get("current_shanten", 8)))
    open_melds = int(efficiency.get("open_melds", 0))
    prefix = f"副露牌效：估算{open_melds}组，" if open_melds else "牌效："
    options = efficiency.get("discard_options")
    if isinstance(options, list) and options:
        best = options[0]
        if isinstance(best, dict):
            tile = _tile_name(str(best.get("tile") or ""))
            shanten = _shanten_text(int(best.get("shanten", 8)))
            effective_types = int(best.get("effective_types", 0))
            effective_count = int(best.get("effective_count", 0))
            waits = _tile_list_in_order([str(tile) for tile in best.get("effective_tiles", [])])
            wait_text = f"（{waits}）" if waits else ""
            visible_count = int(efficiency.get("visible_tile_count", 0))
            visible_text = "，已扣可见牌" if visible_count else ""
            return f"{prefix}当前{current}；打{tile}后{shanten}，有效{effective_types}种{effective_count}枚{visible_text}{wait_text}。"
    return f"{prefix}当前{current}；未到出牌手数，先按主线保留结构。"


def _efficiency_detail(efficiency: dict[str, Any]) -> str:
    path_names = {"standard": "面子手", "seven_pairs": "七对子", "thirteen_orphans": "国士"}
    path = path_names.get(str(efficiency.get("best_path") or "standard"), "面子手")
    return f"牌效按{path}估算，{_efficiency_note(efficiency)}"


def _shanten_text(value: int) -> str:
    if value <= -1:
        return "和牌形"
    if value == 0:
        return "听牌"
    return f"{value}向听"


def _plan_summary(summary: str, keep_text: str, primary_discard_text: str, efficiency: dict[str, Any]) -> str:
    shanten = _shanten_text(int(efficiency.get("current_shanten", 8)))
    discard = str(primary_discard_text or "").strip()
    keep = str(keep_text or "").strip()
    if discard:
        return f"{summary}；留{keep}，先看打{discard}，当前{shanten}"
    return f"{summary}；留{keep}，当前{shanten}"


def _target_shape_text(
    direction: str,
    best_suit: str,
    best_shape: str,
    keep_text: str,
    efficiency: dict[str, Any],
    open_melds: int,
    has_value_pair: bool,
    pair_route: bool,
) -> str:
    shanten = _shanten_text(int(efficiency.get("current_shanten", 8)))
    if open_melds:
        return f"{shanten}，剩余手牌只保留进听块，避免再转慢速大牌"
    if pair_route:
        return "七对子优先，继续收对子；孤张只保留安全、宝牌或强中张"
    if has_value_pair:
        return f"役牌对子可碰，碰成刻子后按{shanten}收束"
    if best_suit and "染手" in direction:
        return f"{SUIT_NAMES.get(best_suit, best_suit)}子{best_shape}，保留同色和役牌，杂色逐步切"
    if "断幺" in direction:
        return "断幺/平和速度，保留中张两面，孤幺九和孤字先处理"
    return f"{shanten}，保留{keep_text}，每次优先比较向听和有效牌"


def _next_step_text(primary_discard_text: str, efficiency: dict[str, Any], open_melds: int) -> str:
    discard = str(primary_discard_text or "").strip()
    options = efficiency.get("discard_options")
    if isinstance(options, list) and options:
        best = options[0]
        if isinstance(best, dict):
            effective = int(best.get("effective_count", 0) or 0)
            types = int(best.get("effective_types", 0) or 0)
            shanten = _shanten_text(int(best.get("shanten", 8)))
            tile = _tile_name(str(best.get("tile") or "")) or discard
            open_note = "；副露后只开能进听/听牌的牌" if open_melds else ""
            return f"优先看打{tile}后{shanten}，有效{types}种{effective}枚{open_note}"
    if discard:
        return f"先处理{discard}，三巡后再复核主线"
    return ""


def _merge_unique_tiles(tiles: list[str]) -> list[str]:
    result: list[str] = []
    for tile in tiles:
        normalized = normalize_tile(tile)
        if normalized and normalized not in result:
            result.append(normalized)
    return result


def _first_tile_name(tiles: list[str]) -> str:
    first = _first_tile(tiles)
    return _tile_name(first) if first else ""


def _value_honor_tiles(config: MahjongCoachConfig | None) -> set[str]:
    values = {"5z", "6z", "7z"}
    if config is None:
        return values
    for tile in (config.round_wind, config.seat_wind):
        normalized = _context_tile(tile)
        if tile_suit(normalized) == "z":
            values.add(normalized)
    return values


def _dora_tiles(config: MahjongCoachConfig | None) -> set[str]:
    if config is None:
        return set()
    return {
        _canonical_tile(_context_tile(tile))
        for tile in config.dora_tiles
        if _canonical_tile(_context_tile(tile)) in TILE_TYPES
    }


def _is_dora(tile: str, dora_tiles: set[str]) -> bool:
    return _canonical_tile(tile) in dora_tiles


def _context_tile(tile: str) -> str:
    raw = str(tile or "").strip()
    return normalize_tile(TEXT_TILE_ALIASES.get(raw, raw))


def _play_style(config: MahjongCoachConfig | None) -> str:
    if config is None:
        return "riichi"
    profile = getattr(config, "player_profile", None)
    if profile is not None and (
        getattr(profile, "risk_tolerance", "balanced") == "aggressive"
        or getattr(profile, "goal_bias", "balanced") == "speed"
        or getattr(profile, "call_bias", "balanced") == "open"
    ):
        return "fast"
    return _valid_play_style(config.play_style)


def _settlement_kind_text(kind: str) -> str:
    return {
        "win": "和牌结算",
        "exhaustive_draw": "荒牌流局",
        "abortive_draw": "途中流局",
        "unknown": "小局结算",
    }.get(str(kind or "").strip(), "小局结算")


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000.0, 1)


def _discard_count(discard_piles: dict[str, list[dict[str, Any]]]) -> int:
    return sum(len(items) for items in discard_piles.values() if isinstance(items, list))


def _shared_tile_count(left: list[str], right: list[str]) -> int:
    left_counts = Counter(normalize_tile(tile) for tile in left if normalize_tile(tile))
    right_counts = Counter(normalize_tile(tile) for tile in right if normalize_tile(tile))
    return sum((left_counts & right_counts).values())


def _inferred_open_melds(closed_tile_count: int) -> int:
    count = max(0, int(closed_tile_count or 0))
    if count >= 12:
        return 0
    if count >= 9:
        return 1
    if count >= 6:
        return 2
    if count >= 3:
        return 3
    return 4 if count > 0 else 0


def _inferred_open_melds_from_closed_count(closed_tile_count: int) -> int:
    count = max(0, int(closed_tile_count or 0))
    if count in (10, 11):
        return 1
    if count in (7, 8):
        return 2
    if count in (4, 5):
        return 3
    return 0


def _coerce_open_melds(value: int | None) -> int | None:
    if value is None:
        return None
    try:
        return max(0, min(4, int(value)))
    except (TypeError, ValueError):
        return None


def _clamp_melds(value: int) -> int:
    return max(0, min(4, int(value or 0)))


def _open_yaku_notes(
    meld_tiles: list[str],
    shape_tiles: list[str],
    value_honors: set[str],
    best_suit: str,
    open_melds: int,
) -> tuple[list[str], list[str]]:
    if open_melds <= 0:
        return [], []
    canonical_melds = [_canonical_tile(tile) for tile in meld_tiles if _canonical_tile(tile) in TILE_TYPES]
    if not canonical_melds:
        return [], ["确认役种：副露牌没识别到，别只凭剩余手牌判断有役。"]

    targets: list[str] = []
    cautions: list[str] = []
    meld_counts = Counter(canonical_melds)
    value_triplets = [tile for tile, count in meld_counts.items() if count >= 3 and tile in value_honors]
    if value_triplets:
        targets.append(f"已成役：役牌副露 {_tile_list(value_triplets)}")

    canonical_shape = [_canonical_tile(tile) for tile in shape_tiles if _canonical_tile(tile) in TILE_TYPES]
    if canonical_shape and all(is_simple(tile) for tile in canonical_shape):
        targets.append("可成役：断幺九")
    else:
        non_honor_suits = {tile_suit(tile) for tile in canonical_shape if tile_suit(tile) in {"m", "p", "s"}}
        if best_suit and len(non_honor_suits) == 1:
            suit_name = SUIT_NAMES.get(best_suit, best_suit)
            if any(is_honor(tile) for tile in canonical_shape):
                targets.append(f"可成役：混一色({suit_name}子)")
            else:
                targets.append(f"可成役：清一色({suit_name}子)")

    if not targets:
        cautions.append("确认役种：没有役牌、断幺或染手时，副露手可能没役。")
    return targets, cautions


def _open_hand_direction(open_melds: int, efficiency: dict[str, Any]) -> str:
    shanten = int(efficiency.get("current_shanten", 8))
    if shanten <= 0:
        return "副露听牌/收束"
    if shanten == 1:
        return "副露一向听"
    if open_melds >= 2:
        return "副露进听优先"
    return "副露加速"


def _open_hand_policy_line(open_melds: int) -> str:
    if open_melds >= 3:
        return "鸣牌：只开能听牌/和牌的牌，其他先别破坏待牌。"
    if open_melds >= 2:
        return "鸣牌：继续快攻，但只开能进听或明显增加有效牌的牌。"
    return "鸣牌：能推进主线就开，开过后优先进听而不是改做大牌。"


def _meld_policy(
    value_pairs: str,
    best_suit_count: int,
    simple_count: int,
    honor_count: int,
    terminal_count: int,
    pair_route: bool,
    style: str,
    *,
    is_action_window: bool = False,
) -> str:
    prefix = "" if is_action_window else "鸣牌："
    if style == "fast":
        if value_pairs:
            return f"役牌对子{value_pairs}，碰到可以开；中张吃碰能加速也开。" if is_action_window else f"鸣牌：{value_pairs}对子可以碰；中张吃碰能加速也开。"
        if pair_route:
            return "七对子胚子一般不鸣；但役牌刻子或明显加速可开。"
        if best_suit_count >= 6:
            return f"{prefix}染手主线可开同色/役牌，能推进就吃碰。"
        if simple_count >= 7 and honor_count + terminal_count <= 4:
            return f"{prefix}断幺速度手积极开中张，能进听或加速就吃碰。"
        return f"{prefix}快攻风格，能推进主线或进听就吃碰；安全牌也考虑和牌。" if is_action_window else "鸣牌：快攻风格，能推进主线或进听就吃碰；安全牌也考虑和牌。"
    if value_pairs:
        return f"役牌对子{value_pairs}，碰到可以开；没碰到役牌就默认跳过。" if is_action_window else f"鸣牌：{value_pairs}对子可以碰；其余仍要能听牌或明显加速。"
    if pair_route:
        return "七对子胚子一般不鸣；只有役牌碰成刻子，或直接听牌才开。" if is_action_window else "鸣牌：七对子胚子一般不鸣，除非役牌刻子直接成型。"
    if best_suit_count >= 7:
        return f"{prefix}染手主线可开同色/役牌，非主线牌默认跳过。"
    if simple_count >= 9 and honor_count + terminal_count <= 3:
        return f"{prefix}断幺速度手可开中张；不能进听或破坏好形就跳过。"
    return "默认跳过；只有役牌、直接进听、明显加速主线或安全和牌时才吃碰杠。" if is_action_window else "鸣牌：默认跳过，只有役牌、直接听牌、或明显加速主线才开。"


def _open_policy_line(
    counts: Counter[str],
    value_honors: set[str],
    best_suit_count: int,
    simple_count: int,
    honor_count: int,
    terminal_count: int,
    pair_route: bool,
    style: str = "riichi",
) -> str:
    value_pairs = _tile_list([tile for tile, value in counts.items() if value >= 2 and tile in value_honors])
    return _meld_policy(value_pairs, best_suit_count, simple_count, honor_count, terminal_count, pair_route, style)


def _call_policy(hand_tiles: list[str], config: MahjongCoachConfig, buttons: list[str]) -> str:
    tiles = [normalize_tile(tile) for tile in hand_tiles if normalize_tile(tile)]
    counts = Counter(tiles)
    value_honors = _value_honor_tiles(config)
    suit_counts = Counter(tile_suit(tile) for tile in tiles if tile_suit(tile))
    suited = {suit: count for suit, count in suit_counts.items() if suit in {"m", "p", "s"}}
    best_suit_count = max(suited.values(), default=0)
    honor_count = sum(1 for tile in tiles if is_honor(tile))
    terminal_count = sum(1 for tile in tiles if is_terminal(tile))
    simple_count = sum(1 for tile in tiles if is_simple(tile))
    pair_count = sum(1 for value in counts.values() if value >= 2)
    all_value_pairs = _tile_list([tile for tile, value in counts.items() if value >= 2 and tile in value_honors])
    call_buttons = set(_normalize_buttons(buttons))
    style = _play_style(config)

    active_value_pairs = all_value_pairs if (not call_buttons or call_buttons & {"pon", "kan"}) else ""
    return _meld_policy(active_value_pairs, best_suit_count, simple_count, honor_count, terminal_count, pair_count >= 4 and best_suit_count < 8, style, is_action_window=True)


def _claimed_tile_from_action(action_meta: dict[str, Any] | None) -> str:
    """Read an optional call-tile field from a future action/river detector.

    The current button template detector only knows that a chi/pon/kan window is
    visible. Keeping this adapter small lets the strategy become exact as soon
    as perception can provide the just-discarded tile, without guessing it.
    """
    for key in ("claimed_tile", "discard_tile", "called_tile"):
        value = normalize_tile(str((action_meta or {}).get(key) or ""))
        if _canonical_tile(value) in TILE_TYPES:
            return value
    return ""


def analyze_call_options(
    hand_tiles: list[str],
    config: MahjongCoachConfig | None,
    buttons: list[str],
    *,
    claimed_tile: str = "",
    visible_tiles: list[str] | None = None,
    open_melds: int | None = None,
) -> dict[str, Any]:
    """Evaluate every legal chi/pon/kan branch from the currently known hand.

    This is deliberately pure and does not invent the discarded tile. When
    ``claimed_tile`` is unknown, callers receive conditional branches such as
    "if the discard is 5p, pon then discard ..." instead of a false direct
    instruction. A detector can later supply that tile through ``action_meta``.
    """
    tiles = [normalize_tile(tile) for tile in hand_tiles if _canonical_tile(normalize_tile(tile)) in TILE_TYPES]
    canonical_tiles = [_canonical_tile(tile) for tile in tiles]
    if not canonical_tiles:
        return {"claimed_tile": "", "claimed_tile_known": False, "baseline_shanten": 8, "options": []}

    current_open_melds = _coerce_open_melds(open_melds)
    if current_open_melds is None:
        current_open_melds = _inferred_open_melds(len(canonical_tiles)) if _play_style(config) == "fast" else 0
    current_open_melds = _clamp_melds(current_open_melds)
    baseline = _best_hand_shanten(canonical_tiles, open_melds=current_open_melds)["shanten"]
    normalized_buttons = set(_normalize_buttons(buttons))
    requested_tile = _canonical_tile(claimed_tile)
    if requested_tile not in TILE_TYPES:
        requested_tile = ""

    raw_options: list[dict[str, Any]] = []
    counts = Counter(canonical_tiles)
    candidate_tiles = [requested_tile] if requested_tile else list(TILE_TYPES)
    if "pon" in normalized_buttons:
        for tile in candidate_tiles:
            if counts[tile] >= 2:
                raw_options.append({"action": "pon", "claimed_tile": tile, "consume": [tile, tile]})
    if "kan" in normalized_buttons:
        for tile in candidate_tiles:
            if counts[tile] >= 3:
                raw_options.append({"action": "kan", "claimed_tile": tile, "consume": [tile, tile, tile]})
    if "chi" in normalized_buttons:
        for tile in candidate_tiles:
            if tile_suit(tile) not in {"m", "p", "s"}:
                continue
            rank = int(tile_rank(tile))
            suit = tile_suit(tile)
            for first in range(max(1, rank - 2), min(rank, 7) + 1):
                sequence = [f"{value}{suit}" for value in range(first, first + 3)]
                if tile not in sequence:
                    continue
                consume = [value for value in sequence if value != tile]
                required = Counter(consume)
                if all(counts[value] >= amount for value, amount in required.items()):
                    raw_options.append({"action": "chi", "claimed_tile": tile, "consume": consume, "sequence": sequence})

    options: list[dict[str, Any]] = []
    seen: set[tuple[str, str, tuple[str, ...]]] = set()
    for raw in raw_options:
        key = (str(raw["action"]), str(raw["claimed_tile"]), tuple(sorted(raw["consume"])))
        if key in seen:
            continue
        seen.add(key)
        action = str(raw["action"])
        remaining = _remove_canonical_tiles(canonical_tiles, list(raw["consume"]))
        option: dict[str, Any] = {
            "action": action,
            "claimed_tile": str(raw["claimed_tile"]),
            "consume": list(raw["consume"]),
            "known_claimed_tile": bool(requested_tile),
            "baseline_shanten": baseline,
        }
        if action == "kan":
            # A kan requires a rinshan draw before the next discard, so scoring it
            # as a normal chi/pon branch would be mathematically misleading.
            option.update({"status": "needs_rinshan", "recommendation": "cautious"})
            options.append(option)
            continue

        after_call = build_round_plan(
            remaining,
            config,
            visible_tiles=visible_tiles,
            open_melds=current_open_melds + 1,
        )
        efficiency = after_call["efficiency"]
        best = _best_efficiency_discard(efficiency)
        if best:
            best_option = next(
                (item for item in efficiency["discard_options"] if item.get("tile") == best),
                {},
            )
            post_shanten = int(best_option.get("shanten", efficiency["current_shanten"]))
            effective_count = int(best_option.get("effective_count", 0))
            effective_types = int(best_option.get("effective_types", 0))
        else:
            post_shanten = int(efficiency["current_shanten"])
            effective_count = 0
            effective_types = 0
        improvement = baseline - post_shanten
        value_honor = str(raw["claimed_tile"]) in _value_honor_tiles(config)
        recommendation = "skip"
        if improvement > 0 or post_shanten == 0:
            recommendation = "call"
        elif action == "pon" and value_honor and post_shanten <= baseline and effective_count > 0:
            recommendation = "consider"
        option.update(
            {
                "status": "evaluated",
                "discard": best,
                "post_shanten": post_shanten,
                "effective_types": effective_types,
                "effective_count": effective_count,
                "shanten_change": improvement,
                "value_honor": value_honor,
                "recommendation": recommendation,
            }
        )
        options.append(option)

    recommendation_rank = {"call": 0, "consider": 1, "skip": 2, "cautious": 3}
    options.sort(
        key=lambda item: (
            recommendation_rank.get(str(item.get("recommendation")), 9),
            int(item.get("post_shanten", 9)),
            -int(item.get("effective_count", 0)),
            str(item.get("action")),
            _tile_sort_key(str(item.get("claimed_tile") or "")),
        )
    )
    return {
        "claimed_tile": requested_tile,
        "claimed_tile_known": bool(requested_tile),
        "baseline_shanten": baseline,
        "open_melds": current_open_melds,
        "options": options,
    }


def _remove_canonical_tiles(tiles: list[str], to_remove: list[str]) -> list[str]:
    remaining = list(tiles)
    for tile in to_remove:
        canonical = _canonical_tile(tile)
        if canonical in remaining:
            remaining.remove(canonical)
    return remaining


def _call_analysis_text(analysis: dict[str, Any]) -> str:
    options = [item for item in analysis.get("options", []) if isinstance(item, dict)]
    if not options:
        return "本次没有算出合法的吃/碰/杠分支，默认跳过。"
    evaluated = [item for item in options if item.get("status") == "evaluated"]
    best = next((item for item in evaluated if item.get("recommendation") == "call"), None)
    if best is None:
        best = next((item for item in evaluated if item.get("recommendation") == "consider"), None)
    if best is None and evaluated:
        best = evaluated[0]
    if best is None:
        return "存在可杠牌，但杠后必须考虑岭上摸牌和手牌价值；当前不做自动推荐。"

    action_names = {"chi": "吃", "pon": "碰", "kan": "杠"}
    action = action_names.get(str(best.get("action")), str(best.get("action")))
    claimed = _tile_name(str(best.get("claimed_tile") or ""))
    discard = _tile_name(str(best.get("discard") or "")) or "再选弃牌"
    result = (
        f"分支牌效：若来牌是{claimed}，{action}后建议打{discard}，"
        f"向听{best.get('post_shanten', '?')}，有效牌{best.get('effective_count', 0)}枚。"
    )
    if not analysis.get("claimed_tile_known"):
        return "尚未识别本次被弃牌，不能直接替你鸣牌。" + result
    if best.get("recommendation") == "call":
        return "建议鸣牌。" + result
    if best.get("recommendation") == "consider":
        return "可以考虑鸣牌。" + result
    return "默认跳过。" + result


def _riichi_advice(
    hand_tiles: list[str],
    config: MahjongCoachConfig | None = None,
    *,
    visible_tiles: list[str] | None = None,
    efficiency: dict[str, Any] | None = None,
) -> str:
    if efficiency is None:
        plan = build_round_plan(hand_tiles, config, visible_tiles=visible_tiles)
        efficiency = plan.get("efficiency", {})
    options = efficiency.get("discard_options")
    if not isinstance(options, list):
        return ""
    tenpai_options = [item for item in options if isinstance(item, dict) and int(item.get("shanten", 8)) == 0]
    if not tenpai_options:
        return ""
    best = tenpai_options[0]
    discard_tile = str(best.get("tile") or "")
    discard = _tile_name(discard_tile)
    wait_tiles, wait_count = _riichi_waits_after_discard(
        hand_tiles,
        [str(tile) for tile in best.get("effective_tiles", [])],
        discard_tile,
        visible_tiles or [],
    )
    wait_text = _tile_list_in_order(wait_tiles) or "待牌未明"
    effective_types = len(wait_tiles) or int(best.get("effective_types", 0))
    effective_count = wait_count if wait_tiles else int(best.get("effective_count", 0))
    quality = _wait_quality(effective_types, effective_count)
    style = _play_style(config)
    value_text = _riichi_value_text(hand_tiles, config)
    visible_text = "，已扣可见牌" if int(efficiency.get("visible_tile_count", 0)) else ""
    if style == "fast":
        if quality == "good" and effective_count >= 8:
            action = "推荐立直"
            reason = "好形/枚数够"
        elif quality == "good":
            action = "可以立直"
            reason = "待牌尚可"
        else:
            action = "谨慎立直"
            reason = "愚形或枚数少"
    else:
        if quality == "good":
            action = "推荐立直"
            reason = "好形/枚数够"
        elif quality == "medium":
            action = "可以立直"
            reason = "待牌尚可"
        else:
            action = "谨慎立直"
            reason = "愚形或枚数少"
    return f"{action}：打{discard}听{wait_text}，有效{effective_types}种{effective_count}枚{visible_text}；{reason}{value_text}；本地快判。"


def _riichi_waits_after_discard(
    hand_tiles: list[str],
    effective_tiles: list[str],
    discard_tile: str,
    visible_tiles: list[str],
) -> tuple[list[str], int]:
    hand_counts = Counter(_canonical_tile(tile) for tile in hand_tiles if _canonical_tile(tile) in TILE_TYPES)
    visible_counts = _visible_counts(visible_tiles)
    discarded = _canonical_tile(discard_tile)
    if discarded in hand_counts:
        hand_counts[discarded] -= 1
        if hand_counts[discarded] <= 0:
            del hand_counts[discarded]
    waits = _merge_unique_tiles(
        [_canonical_tile(tile) for tile in effective_tiles if _canonical_tile(tile) in TILE_TYPES]
    )
    wait_count = sum(max(0, 4 - hand_counts.get(tile, 0) - visible_counts.get(tile, 0)) for tile in waits)
    return waits, wait_count


def _wait_quality(effective_types: int, effective_count: int) -> str:
    if effective_types >= 2 and effective_count >= 6:
        return "good"
    if effective_count >= 5:
        return "medium"
    return "poor"


def _riichi_value_text(hand_tiles: list[str], config: MahjongCoachConfig | None) -> str:
    tiles = [normalize_tile(tile) for tile in hand_tiles if normalize_tile(tile)]
    counts = Counter(tiles)
    value_pairs = _tile_list([tile for tile, value in counts.items() if value >= 2 and tile in _value_honor_tiles(config)])
    dora_tiles = _dora_tiles(config)
    dora_count = sum(1 for tile in tiles if tile in {"0m", "0p", "0s"} or _is_dora(tile, dora_tiles))
    parts: list[str] = []
    if dora_count:
        parts.append(f"有宝牌/红5 {dora_count}张")
    if value_pairs:
        parts.append(f"有役牌对子{value_pairs}")
    return f"，{'，'.join(parts)}" if parts else ""


def _route_discard_options(
    discard_tiles: list[str],
    counts: Counter[str],
    best_suit: str,
    best_suit_count: int,
    second_suit_count: int,
    pair_count: int,
    has_value_pair: bool,
    honor_count: int,
    terminal_count: int,
    simple_count: int,
) -> list[tuple[str, str]]:
    options: list[tuple[str, str]] = []
    primary = _first_tile(discard_tiles)
    if not primary:
        return options

    if best_suit_count >= 8:
        primary_label = f"{SUIT_NAMES.get(best_suit, best_suit)}染"
    elif pair_count >= 4:
        primary_label = "七对"
    elif has_value_pair:
        primary_label = "役牌"
    elif best_suit_count >= 5 and best_suit_count >= second_suit_count + 2:
        primary_label = "主线"
    elif simple_count >= 9 and honor_count + terminal_count <= 3:
        primary_label = "断幺"
    elif honor_count >= 4:
        primary_label = "孤字先"
    else:
        primary_label = "牌效"
    options.append((primary_label, primary))

    alternate = _alternate_route_tile(discard_tiles, primary, counts, best_suit)
    if alternate:
        if best_suit_count >= 7:
            alternate_label = "不硬染"
        elif pair_count >= 4:
            alternate_label = "面子手"
        else:
            alternate_label = "保守"
        options.append((alternate_label, alternate))
    return options


def _route_options_text(options: list[tuple[str, str]]) -> str:
    parts: list[str] = []
    for label, tile in options[:2]:
        name = _tile_name(tile)
        if label and name:
            parts.append(f"{label}打{name}")
    return "；".join(parts)


def _first_tile(tiles: list[str]) -> str:
    for tile in tiles:
        normalized = normalize_tile(tile)
        if normalized:
            return normalized
    return ""


def _alternate_route_tile(discard_tiles: list[str], primary: str, counts: Counter[str], best_suit: str) -> str:
    normalized = [normalize_tile(tile) for tile in discard_tiles if normalize_tile(tile)]
    for tile in normalized:
        if tile != primary:
            return tile

    for tile, value in sorted(counts.items(), key=lambda item: _tile_sort_key(item[0])):
        if value >= 2 or tile == primary or tile in {"0m", "0p", "0s"}:
            continue
        if best_suit and tile_suit(tile) == best_suit and _has_neighbor(tile, counts, distance=2):
            continue
        return tile
    return ""


def _discard_score(
    tile: str,
    counts: Counter[str],
    best_suit: str,
    keep_set: set[str],
    value_honors: set[str],
    dora_tiles: set[str],
) -> tuple[int, int, str]:
    suit = tile_suit(tile)
    rank = tile_rank(tile)
    rank_value = int(rank) if rank.isdigit() else 0
    keep_penalty = 20 if tile in keep_set else 0
    pair_penalty = 50 if counts[tile] >= 2 else 0
    dora_penalty = 35 if _is_dora(tile, dora_tiles) else 0
    value_penalty = 8 if tile in value_honors else 0
    if suit == "z":
        return (pair_penalty + keep_penalty + dora_penalty + value_penalty, rank_value, tile)

    connected = _has_neighbor(tile, counts, distance=1)
    near = _has_neighbor(tile, counts, distance=2)
    base = 8
    if not connected and is_terminal(tile):
        base = 1
    elif not connected:
        base = 3
    elif not near:
        base = 5

    if best_suit and suit != best_suit:
        base -= 1
    if best_suit and suit == best_suit:
        base += 2
    return (pair_penalty + keep_penalty + dora_penalty + base, rank_value, tile)


def _has_neighbor(tile: str, counts: Counter[str], *, distance: int) -> bool:
    suit = tile_suit(tile)
    if suit not in {"m", "p", "s"}:
        return False
    rank = tile_rank(tile)
    if not rank.isdigit():
        return False
    value = int(rank)
    for offset in range(1, distance + 1):
        if counts.get(f"{value - offset}{suit}", 0) > 0 or counts.get(f"{value + offset}{suit}", 0) > 0:
            return True
    return False


def _normalize_buttons(buttons: list[str] | None) -> list[str]:
    result: list[str] = []
    for button in buttons or []:
        normalized = str(button or "").strip().lower()
        if normalized and normalized not in result:
            result.append(normalized)
    return result


def _coerce_turn(value: int | None) -> int | None:
    if value is None:
        return None
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return None
