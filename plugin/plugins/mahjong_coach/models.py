from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


PROFILE_VERSION = 1
PREFERENCES_VERSION = 1


class DefensePosture(str, Enum):
    PUSH = "push"
    MAWASHI = "mawashi"
    FOLD = "fold"


@dataclass(frozen=True)
class WindowTargetDescriptor:
    """Stable window identity. A HWND is deliberately never persisted."""

    title: str = ""
    app_name: str = ""
    match_keyword: str = ""

    @classmethod
    def from_payload(cls, payload: dict[str, Any] | None) -> "WindowTargetDescriptor":
        payload = payload if isinstance(payload, dict) else {}
        return cls(
            title=str(payload.get("title") or "").strip(),
            app_name=str(payload.get("app_name") or "").strip(),
            match_keyword=str(payload.get("match_keyword") or "").strip(),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PlayerProfile:
    version: int = PROFILE_VERSION
    rank: str = "unknown"
    room: str = "unknown"
    risk_tolerance: str = "balanced"
    goal_bias: str = "balanced"
    call_bias: str = "balanced"
    source: str = "manual"
    account_id: str = ""
    nickname: str = ""
    confirmed: bool = False
    fetched_at: float = 0.0
    sample_count: int = 0
    win_rate: float | None = None
    deal_in_rate: float | None = None
    riichi_rate: float | None = None
    call_rate: float | None = None

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, Any] | None,
        *,
        legacy_play_style: Any = "",
    ) -> "PlayerProfile":
        payload = payload if isinstance(payload, dict) else {}
        legacy = _valid_play_style(legacy_play_style)
        risk_default = "aggressive" if legacy == "fast" else "balanced"
        goal_default = "speed" if legacy == "fast" else "balanced"
        call_default = "open" if legacy == "fast" else "closed"
        return cls(
            version=PROFILE_VERSION,
            rank=_choice(payload.get("rank"), _VALID_RANKS, "unknown"),
            room=_choice(payload.get("room"), _VALID_ROOMS, "unknown"),
            risk_tolerance=_choice(payload.get("risk_tolerance"), _VALID_RISK, risk_default),
            goal_bias=_choice(payload.get("goal_bias"), _VALID_GOALS, goal_default),
            call_bias=_choice(payload.get("call_bias"), _VALID_CALL_BIAS, call_default),
            source=_choice(payload.get("source"), {"manual", "amae_koromo"}, "manual"),
            account_id=str(payload.get("account_id") or "").strip(),
            nickname=str(payload.get("nickname") or "").strip(),
            confirmed=bool(payload.get("confirmed", False)),
            fetched_at=_float_value(payload.get("fetched_at"), 0.0),
            sample_count=max(0, _int_value(payload.get("sample_count"), 0)),
            win_rate=_optional_rate(payload.get("win_rate")),
            deal_in_rate=_optional_rate(payload.get("deal_in_rate")),
            riichi_rate=_optional_rate(payload.get("riichi_rate")),
            call_rate=_optional_rate(payload.get("call_rate")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CapturePreferences:
    version: int = PREFERENCES_VERSION
    auto_start_live: bool = False
    target: WindowTargetDescriptor = field(default_factory=WindowTargetDescriptor)
    profile: PlayerProfile = field(default_factory=PlayerProfile)

    @classmethod
    def from_payload(cls, payload: dict[str, Any] | None) -> "CapturePreferences":
        payload = payload if isinstance(payload, dict) else {}
        return cls(
            version=PREFERENCES_VERSION,
            auto_start_live=bool(payload.get("auto_start_live", False)),
            target=WindowTargetDescriptor.from_payload(payload.get("target")),
            profile=PlayerProfile.from_payload(payload.get("profile")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class YakumanEstimate:
    route: str
    label: str
    distance: int
    key_tiles: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    tenpai_probability: dict[str, float] = field(default_factory=dict)
    tsumo_probability: dict[str, float] = field(default_factory=dict)
    confidence_interval: dict[str, list[float]] = field(default_factory=dict)
    trials: int = 0
    estimated: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MahjongCoachConfig:
    live_advice_mode: str = "coach"
    coach_checkpoint_self_turns: int = 3
    critical_action_interrupts: bool = True
    per_turn_discard_prompt: bool = False
    play_style: str = "riichi"
    strategy_preset: str = "simple"
    hand_recognition_backend: str = "legacy_templates"
    onnx_hand_enabled: bool = False
    meld_recognition_enabled: bool = True
    meld_min_confidence: float = 0.72
    river_recognition_enabled: bool = True
    river_tracking_mode: str = "checkpoint"
    river_min_confidence: float = 0.90
    tile_recognition_mode: str = "legacy"
    opponent_riichi_recognition_enabled: bool = True
    settlement_recognition_enabled: bool = True
    settlement_min_confidence: float = 0.72
    settlement_confirm_frames: int = 2
    settlement_confirm_max_gap_ms: int = 2500
    live_window_keywords: list[str] = field(default_factory=lambda: ["雀魂", "Mahjong Soul"])
    live_interval_ms: int = 400
    live_fast_interval_ms: int = 300
    live_keep_frames: int = 20
    live_checkpoint_interval_seconds: int = 4
    live_overlay_enabled: bool = True
    live_save_format: str = "jpg"
    round_wind: str = "1z"
    seat_wind: str = ""
    dora_tiles: list[str] = field(default_factory=list)
    player_profile: PlayerProfile = field(default_factory=PlayerProfile)

    @classmethod
    def from_payload(cls, payload: dict[str, Any] | None) -> "MahjongCoachConfig":
        payload = payload or {}
        decision = payload.get("decision") if isinstance(payload.get("decision"), dict) else {}
        perception = payload.get("perception") if isinstance(payload.get("perception"), dict) else {}
        live = payload.get("live") if isinstance(payload.get("live"), dict) else {}
        round_context = payload.get("round") if isinstance(payload.get("round"), dict) else {}
        legacy_style = _valid_play_style(decision.get("play_style"))
        profile_payload = payload.get("profile") if isinstance(payload.get("profile"), dict) else {}
        return cls(
            live_advice_mode=str(decision.get("live_advice_mode") or "coach"),
            coach_checkpoint_self_turns=max(1, int(decision.get("coach_checkpoint_self_turns") or 1)),
            critical_action_interrupts=bool(decision.get("critical_action_interrupts", True)),
            per_turn_discard_prompt=bool(decision.get("per_turn_discard_prompt", False)),
            play_style=legacy_style,
            strategy_preset=_valid_strategy_preset(decision.get("strategy_preset")),
            hand_recognition_backend=str(perception.get("hand_recognition_backend") or "legacy_templates"),
            onnx_hand_enabled=bool(perception.get("onnx_hand_enabled", False)),
            meld_recognition_enabled=bool(perception.get("meld_recognition_enabled", True)),
            meld_min_confidence=max(0.0, min(1.0, float(perception.get("meld_min_confidence") or 0.72))),
            river_recognition_enabled=bool(perception.get("river_recognition_enabled", True)),
            river_tracking_mode=_valid_river_tracking_mode(perception.get("river_tracking_mode")),
            river_min_confidence=max(0.0, min(1.0, float(perception.get("river_min_confidence") or 0.90))),
            tile_recognition_mode=_valid_tile_recognition_mode(perception.get("tile_recognition_mode")),
            opponent_riichi_recognition_enabled=bool(perception.get("opponent_riichi_recognition_enabled", True)),
            settlement_recognition_enabled=bool(perception.get("settlement_recognition_enabled", True)),
            settlement_min_confidence=max(
                0.0,
                min(1.0, float(perception.get("settlement_min_confidence") or 0.72)),
            ),
            settlement_confirm_frames=max(
                1,
                min(8, int(perception.get("settlement_confirm_frames") or 2)),
            ),
            settlement_confirm_max_gap_ms=max(
                200,
                min(10_000, int(perception.get("settlement_confirm_max_gap_ms") or 2500)),
            ),
            live_window_keywords=_string_list(live.get("window_keywords"), ["雀魂", "Mahjong Soul"]),
            live_interval_ms=max(200, int(live.get("interval_ms") or 400)),
            live_fast_interval_ms=max(100, int(live.get("fast_interval_ms") or 300)),
            live_keep_frames=max(
                0,
                int(live.get("keep_frames") if live.get("keep_frames") is not None else 20),
            ),
            live_checkpoint_interval_seconds=max(4, int(live.get("checkpoint_interval_seconds") or 4)),
            live_overlay_enabled=bool(live.get("overlay_enabled", True)),
            live_save_format=str(live.get("save_format") or "jpg"),
            round_wind=str(round_context.get("round_wind") or "1z"),
            seat_wind=str(round_context.get("seat_wind") or ""),
            dora_tiles=_string_list(round_context.get("dora_tiles"), []),
            player_profile=PlayerProfile.from_payload(
                profile_payload,
                legacy_play_style=legacy_style,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RoundCoachState:
    round_id: str = "default"
    round_phase: str = "round_idle"
    play_style: str = "riichi"
    strategy_preset: str = "simple"
    opening_emitted: bool = False
    opening_plan: str = ""
    current_plan: str = ""
    plan_source: str = "heuristic"
    local_direction: str = ""
    local_plan: str = ""
    local_detail: str = ""
    attack_defense_bias: str = "neutral"
    defense_posture: str = ""
    defense_risk_budget: float = 0.0
    target_shapes: list[str] = field(default_factory=list)
    caution_points: list[str] = field(default_factory=list)
    last_hand_signature: str = ""
    last_hand_tiles: list[str] = field(default_factory=list)
    last_hand_confidence: float = 0.0
    last_melds: list[dict[str, Any]] = field(default_factory=list)
    last_meld_tiles: list[str] = field(default_factory=list)
    last_open_meld_count: int = 0
    last_meld_confidence: float = 0.0
    last_opponent_melds: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    last_opponent_meld_tiles: list[str] = field(default_factory=list)
    last_discard_piles: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    last_visible_discards: list[str] = field(default_factory=list)
    last_river_confidence: float = 0.0
    river_tracking_initialized: bool = False
    last_checkpoint_self_turn: int = 0
    prev_direction: str = ""
    prev_discard_priority: list[str] = field(default_factory=list)
    riichi_players: list[str] = field(default_factory=list)
    riichi_pending: dict[str, int] = field(default_factory=dict)
    riichi_stick_baseline: int | None = None
    last_riichi_stick_count: int | None = None
    player_scores: dict[str, int] = field(default_factory=dict)
    player_ranks: dict[str, int] = field(default_factory=dict)
    honba_count: int | None = None
    table_riichi_stick_count: int | None = None
    table_context_confidence: float = 0.0
    table_context_reason: str = ""
    table_context_pending_signature: str = ""
    table_context_pending_frames: int = 0
    settlement_phase: str = "playing"
    settlement_kind: str = "none"
    settlement_confidence: float = 0.0
    settlement_evidence: list[str] = field(default_factory=list)
    settlement_confirmation_frames: int = 0
    last_update_reason: str = ""
    update_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CoachDecision:
    decision_type: str = "observe"
    priority: int = 0
    action_required: bool = False
    summary: str = ""
    detail: str = ""
    suggestion: str = ""
    buttons: list[str] = field(default_factory=list)
    hand_tiles: list[str] = field(default_factory=list)
    reason_codes: list[str] = field(default_factory=list)
    coach_state: dict[str, Any] = field(default_factory=dict)
    perception: dict[str, Any] = field(default_factory=dict)
    engine_meta: dict[str, Any] = field(default_factory=dict)
    analysis_source: str = "heuristic"
    quiet: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FramePacket:
    timestamp_ms: int
    image_path: str = ""
    window_title: str = ""
    width: int = 0
    height: int = 0
    source: str = "unknown"
    image: Any = field(default=None, repr=False, compare=False)
    fingerprint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp_ms": self.timestamp_ms,
            "image_path": self.image_path,
            "window_title": self.window_title,
            "width": self.width,
            "height": self.height,
            "source": self.source,
            "fingerprint": self.fingerprint,
            "has_image": self.image is not None,
        }


@dataclass
class LiveSessionState:
    running: bool = False
    status: str = "stopped"
    frame_index: int = 0
    started_at: float = 0.0
    updated_at: float = 0.0
    last_error: str = ""
    last_frame_path: str = ""
    last_window_title: str = ""
    last_capture_source: str = ""
    last_binding: dict[str, Any] = field(default_factory=dict)
    observed_hand_changes: int = 0
    missing_hand_frames: int = 0
    dropped_frames: int = 0
    overlay_enabled: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _string_list(value: Any, fallback: list[str]) -> list[str]:
    if isinstance(value, list):
        result = [str(item).strip() for item in value if str(item).strip()]
        return result or list(fallback)
    if isinstance(value, str):
        result = [item.strip() for item in value.split(",") if item.strip()]
        return result or list(fallback)
    return list(fallback)


def _clean_string_list(items: list[Any] | None) -> list[str]:
    return [str(item).strip() for item in (items or []) if str(item).strip()]


def _valid_play_style(value: Any) -> str:
    style = str(value or "").strip().lower()
    if style in ("fast", "快攻", "aggressive"):
        return "fast"
    return "riichi"


def _valid_strategy_preset(value: Any) -> str:
    preset = str(value or "").strip().lower()
    if preset in ("standard", "full", "complete", "完整", "完整攻守"):
        return "standard"
    return "simple"


def _valid_river_tracking_mode(value: Any) -> str:
    mode = str(value or "").strip().lower()
    if mode in ("live", "realtime", "real_time", "continuous"):
        return "live"
    return "checkpoint"


def _valid_tile_recognition_mode(value: Any) -> str:
    mode = str(value or "").strip().lower()
    if mode in ("yolo", "yolo26", "ultralytics_yolo26"):
        return "yolo26"
    return "legacy"


_VALID_RANKS = {
    "unknown", "novice", "adept", "expert", "master", "saint", "celestial",
}
_VALID_ROOMS = {"unknown", "bronze", "silver", "gold", "jade", "throne", "friendly"}
_VALID_RISK = {"conservative", "balanced", "aggressive"}
_VALID_GOALS = {"speed", "balanced", "value", "yakuman"}
_VALID_CALL_BIAS = {"closed", "balanced", "open"}


def _choice(value: Any, valid: set[str], fallback: str) -> str:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in valid else fallback


def _int_value(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _float_value(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _optional_rate(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return None
