from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping

from ..tile_labels import normalize_tile


@dataclass(frozen=True)
class HeldTileSafety:
    """Genbutsu coverage for one physical tile currently held by the player.

    玩家手里一张实体牌的现物覆盖范围。not_proven_safe 只表示不是现物，
    不表示这张牌一定会放铳。
    """

    tile: str
    tile_type: str
    genbutsu_against: tuple[str, ...] = ()
    not_proven_safe_against: tuple[str, ...] = ()
    unknown_against: tuple[str, ...] = ()
    safe_against_all: bool = False


@dataclass(frozen=True)
class GenbutsuAssessment:
    """Exact genbutsu safety derived from each riichi player's own river.

    根据每名立直者自己的牌河计算精确现物。这里只表达现物，不估计筋、壁或放铳率。
    """

    safe_by_player: dict[str, tuple[str, ...]] = field(default_factory=dict)
    safe_against_all: tuple[str, ...] = ()
    held_safe_against_all: tuple[str, ...] = ()
    missing_players: tuple[str, ...] = ()
    held_tile_safety: tuple[HeldTileSafety, ...] = ()
    rule: str = "own_discard_furiten"
    rule_scope: str = "ron_only"
    can_still_tsumo: bool = True

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def assess_genbutsu(
    hand_tiles: Iterable[str],
    discard_piles: Mapping[str, Iterable[Mapping[str, Any] | str]],
    riichi_players: Iterable[str],
) -> GenbutsuAssessment:
    """Return per-player genbutsu and their intersection for multiple riichi.

    返回逐家现物；多人立直时，safe_against_all 只保留对所有立直家均为现物的牌。
    """

    players = tuple(dict.fromkeys(str(player).strip() for player in riichi_players if str(player).strip()))
    hand = tuple(tile for tile in (_physical_tile(item) for item in hand_tiles) if tile)
    safe_by_player: dict[str, tuple[str, ...]] = {}
    missing: list[str] = []
    safe_sets: list[set[str]] = []

    for player in players:
        entries = discard_piles.get(player)
        if entries is None:
            missing.append(player)
            continue
        safe = {_canonical_tile(_entry_tile(entry)) for entry in entries}
        safe.discard("")
        safe_sets.append(safe)
        safe_by_player[player] = tuple(sorted(safe, key=_tile_sort_key))

    # 缺少任一立直家的牌河时，不能证明一张牌对所有人安全。
    # Universal safety cannot be proven while any riichi river is missing.
    safe_against_all: set[str] = set()
    if players and not missing and len(safe_sets) == len(players):
        safe_against_all = set.intersection(*safe_sets) if safe_sets else set()
    held_safe = tuple(tile for tile in dict.fromkeys(hand) if _canonical_tile(tile) in safe_against_all)
    held_tile_safety = tuple(
        _held_tile_safety(tile, players, safe_by_player, set(missing))
        for tile in dict.fromkeys(hand)
    )
    return GenbutsuAssessment(
        safe_by_player=safe_by_player,
        safe_against_all=tuple(sorted(safe_against_all, key=_tile_sort_key)),
        held_safe_against_all=held_safe,
        missing_players=tuple(missing),
        held_tile_safety=held_tile_safety,
    )


def _held_tile_safety(
    tile: str,
    players: tuple[str, ...],
    safe_by_player: Mapping[str, tuple[str, ...]],
    missing_players: set[str],
) -> HeldTileSafety:
    canonical = _canonical_tile(tile)
    safe = tuple(player for player in players if canonical in safe_by_player.get(player, ()))
    unknown = tuple(player for player in players if player in missing_players)
    not_proven = tuple(player for player in players if player not in safe and player not in unknown)
    return HeldTileSafety(
        tile=tile,
        tile_type=canonical,
        genbutsu_against=safe,
        not_proven_safe_against=not_proven,
        unknown_against=unknown,
        safe_against_all=bool(players) and not unknown and not not_proven,
    )


def _entry_tile(entry: Mapping[str, Any] | str) -> str:
    if isinstance(entry, Mapping):
        return str(entry.get("tile") or "")
    return str(entry or "")


def _physical_tile(tile: str) -> str:
    normalized = normalize_tile(tile)
    return normalized if _canonical_tile(normalized) else ""


def _canonical_tile(tile: str) -> str:
    normalized = normalize_tile(tile)
    if normalized in {"0m", "0p", "0s"}:
        normalized = f"5{normalized[1]}"
    if len(normalized) != 2 or normalized[0] not in "123456789" or normalized[1] not in "mpsz":
        return ""
    if normalized[1] == "z" and normalized[0] not in "1234567":
        return ""
    return normalized


def _tile_sort_key(tile: str) -> tuple[int, int]:
    canonical = _canonical_tile(tile)
    suit_order = {"m": 0, "p": 1, "s": 2, "z": 3}
    if not canonical:
        return (9, 0)
    return (suit_order.get(canonical[1], 9), int(canonical[0]))
