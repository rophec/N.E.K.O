from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Iterable

from ..tile_labels import normalize_tile


TILE_TYPES = tuple(
    [f"{rank}{suit}" for suit in ("m", "p", "s") for rank in range(1, 10)]
    + [f"{rank}z" for rank in range(1, 8)]
)
ORPHAN_TYPES = frozenset(
    {"1m", "9m", "1p", "9p", "1s", "9s", "1z", "2z", "3z", "4z", "5z", "6z", "7z"}
)


@dataclass(frozen=True)
class ShantenResult:
    """Exact shanten result for the supplied closed tiles and open meld count.

    针对给定暗手和副露组数的精确向听结果。
    """

    shanten: int
    path: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class EffectiveTile:
    """One tile type that improves shanten and its unseen-copy count.

    能降低向听的一种牌，以及当前尚未看见的枚数。unseen_count 不等于牌山真实枚数。
    """

    tile: str
    unseen_count: int
    visible_count: int
    hand_count: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class DiscardEfficiency:
    """Efficiency after discarding one physical tile from a 3n+2 hand.

    从 3n+2 手牌打出一张实体牌后的牌效。
    """

    tile: str
    shanten: int
    effective_tiles: tuple[EffectiveTile, ...] = ()

    @property
    def effective_types(self) -> int:
        return len(self.effective_tiles)

    @property
    def effective_count(self) -> int:
        return sum(item.unseen_count for item in self.effective_tiles)

    def to_dict(self) -> dict[str, object]:
        return {
            "tile": self.tile,
            "shanten": self.shanten,
            "effective_types": self.effective_types,
            "effective_count": self.effective_count,
            "effective_tiles": [item.tile for item in self.effective_tiles],
            "effective_tile_details": [item.to_dict() for item in self.effective_tiles],
        }


@dataclass(frozen=True)
class HandEfficiencyAnalysis:
    """Deterministic hand-efficiency analysis built from visible information.

    根据可见信息计算的确定性牌效结果；不推断对手暗手或牌山顺序。
    """

    shanten: int
    best_path: str
    open_melds: int
    closed_tile_count: int
    current_effective_tiles: tuple[EffectiveTile, ...] = ()
    discard_options: tuple[DiscardEfficiency, ...] = ()
    visible_counts: dict[str, int] = field(default_factory=dict)
    source_counts: dict[str, int] = field(default_factory=dict)
    anomalies: tuple[str, ...] = ()

    def to_dict(self, *, discard_limit: int | None = None) -> dict[str, object]:
        options = self.discard_options if discard_limit is None else self.discard_options[:discard_limit]
        return {
            "current_shanten": self.shanten,
            "best_path": self.best_path,
            "open_melds": self.open_melds,
            "closed_tile_count": self.closed_tile_count,
            "current_effective_types": len(self.current_effective_tiles),
            "current_effective_count": sum(item.unseen_count for item in self.current_effective_tiles),
            "current_effective_tiles": [item.tile for item in self.current_effective_tiles],
            "current_effective_tile_details": [item.to_dict() for item in self.current_effective_tiles],
            "discard_options": [item.to_dict() for item in options],
            "all_discard_options": [item.to_dict() for item in self.discard_options],
            "visible_counts": dict(self.visible_counts),
            "visible_tile_count": sum(self.visible_counts.values()),
            "visible_source_counts": dict(self.source_counts),
            "remaining_count_kind": "unseen_not_wall",
            "anomalies": list(self.anomalies),
        }


def analyze_hand_efficiency(
    hand_tiles: Iterable[str],
    *,
    river_tiles: Iterable[str] = (),
    meld_tiles: Iterable[str] = (),
    dora_indicators: Iterable[str] = (),
    other_visible_tiles: Iterable[str] = (),
    open_melds: int = 0,
) -> HandEfficiencyAnalysis:
    """Calculate exact shanten and ukeire from all currently visible tiles.

    使用当前所有可见牌计算精确向听与有效牌。返回的剩余枚数仅表示四枚中尚未看见的
    数量，不能区分这些牌位于牌山还是对手暗手。
    """

    physical_hand = _physical_tiles(hand_tiles)
    canonical_hand = [_canonical_tile(tile) for tile in physical_hand]
    river = _canonical_tiles(river_tiles)
    melds = _canonical_tiles(meld_tiles)
    indicators = _canonical_tiles(dora_indicators)
    other = _canonical_tiles(other_visible_tiles)
    external_visible = [*river, *melds, *indicators, *other]
    visible_counts = Counter(external_visible)
    hand_counts = Counter(canonical_hand)
    open_meld_count = _clamp_melds(open_melds)
    shanten = calculate_shanten(canonical_hand, open_melds=open_meld_count)
    anomalies = _copy_count_anomalies(hand_counts, visible_counts)

    current_effective: tuple[EffectiveTile, ...] = ()
    discard_options: tuple[DiscardEfficiency, ...] = ()
    tile_modulo = len(canonical_hand) % 3
    if canonical_hand and tile_modulo == 1:
        current_effective = _effective_tiles(canonical_hand, shanten.shanten, hand_counts, visible_counts, open_meld_count)
    elif canonical_hand and tile_modulo == 2:
        discard_options = _discard_efficiencies(
            physical_hand,
            canonical_hand,
            hand_counts,
            visible_counts,
            open_meld_count,
        )
    elif canonical_hand:
        anomalies.append(f"unexpected_closed_tile_count:{len(canonical_hand)}")

    return HandEfficiencyAnalysis(
        shanten=shanten.shanten,
        best_path=shanten.path,
        open_melds=open_meld_count,
        closed_tile_count=len(canonical_hand),
        current_effective_tiles=current_effective,
        discard_options=discard_options,
        visible_counts=dict(sorted(visible_counts.items(), key=lambda item: _tile_sort_key(item[0]))),
        source_counts={
            "river": len(river),
            "meld": len(melds),
            "dora_indicator": len(indicators),
            "other": len(other),
        },
        anomalies=tuple(anomalies),
    )


def calculate_shanten(tiles: Iterable[str], *, open_melds: int = 0) -> ShantenResult:
    """Return the minimum of standard, seven-pairs, and thirteen-orphans shanten.

    返回面子手、七对子和国士无双中的最小向听；存在副露时只计算面子手。
    """

    counts = _shanten_counts(_canonical_tiles(tiles))
    open_meld_count = _clamp_melds(open_melds)
    options = [("standard", _standard_shanten(tuple(counts), open_meld_count))]
    if open_meld_count == 0:
        options.extend(
            (
                ("seven_pairs", _seven_pairs_shanten(counts)),
                ("thirteen_orphans", _thirteen_orphans_shanten(counts)),
            )
        )
    path, value = min(options, key=lambda item: item[1])
    return ShantenResult(shanten=int(value), path=path)


def _discard_efficiencies(
    physical_hand: list[str],
    canonical_hand: list[str],
    hand_counts: Counter[str],
    visible_counts: Counter[str],
    open_melds: int,
) -> tuple[DiscardEfficiency, ...]:
    options: list[DiscardEfficiency] = []
    for physical_tile in _unique_physical_discards(physical_hand):
        canonical = _canonical_tile(physical_tile)
        remaining = list(canonical_hand)
        remaining.remove(canonical)
        post_discard = calculate_shanten(remaining, open_melds=open_melds).shanten
        effective = _effective_tiles(remaining, post_discard, hand_counts, visible_counts, open_melds)
        options.append(DiscardEfficiency(tile=physical_tile, shanten=post_discard, effective_tiles=effective))
    options.sort(
        key=lambda item: (
            item.shanten,
            -item.effective_count,
            -item.effective_types,
            _discard_tie_key(item.tile),
        )
    )
    return tuple(options)


def _effective_tiles(
    base_tiles: list[str],
    base_shanten: int,
    hand_counts: Counter[str],
    visible_counts: Counter[str],
    open_melds: int,
) -> tuple[EffectiveTile, ...]:
    result: list[EffectiveTile] = []
    for draw in TILE_TYPES:
        hand_count = hand_counts.get(draw, 0)
        visible_count = visible_counts.get(draw, 0)
        unseen_count = max(0, 4 - hand_count - visible_count)
        if unseen_count <= 0:
            continue
        if calculate_shanten([*base_tiles, draw], open_melds=open_melds).shanten < base_shanten:
            result.append(
                EffectiveTile(
                    tile=draw,
                    unseen_count=unseen_count,
                    visible_count=visible_count,
                    hand_count=hand_count,
                )
            )
    return tuple(result)


def _copy_count_anomalies(hand_counts: Counter[str], visible_counts: Counter[str]) -> list[str]:
    anomalies: list[str] = []
    for tile in TILE_TYPES:
        total = hand_counts.get(tile, 0) + visible_counts.get(tile, 0)
        if total > 4:
            anomalies.append(f"visible_copy_overflow:{tile}:{total}")
    return anomalies


def _physical_tiles(tiles: Iterable[str]) -> list[str]:
    return [tile for tile in (normalize_tile(item) for item in tiles) if _canonical_tile(tile) in TILE_TYPES]


def _canonical_tiles(tiles: Iterable[str]) -> list[str]:
    return [tile for tile in (_canonical_tile(item) for item in tiles) if tile in TILE_TYPES]


def _canonical_tile(tile: str) -> str:
    normalized = normalize_tile(tile)
    if normalized in {"0m", "0p", "0s"}:
        return f"5{normalized[1]}"
    return normalized


def _unique_physical_discards(tiles: Iterable[str]) -> list[str]:
    unique = set(_physical_tiles(tiles))
    return sorted(unique, key=_discard_tie_key)


def _discard_tie_key(tile: str) -> tuple[int, int, int]:
    canonical = _canonical_tile(tile)
    suit_order = {"m": 0, "p": 1, "s": 2, "z": 3}
    # Keep red fives when structural efficiency is identical.
    red_penalty = 1 if tile in {"0m", "0p", "0s"} else 0
    return (suit_order.get(canonical[1:] if len(canonical) > 1 else "", 9), int(canonical[0]), red_penalty)


def _tile_sort_key(tile: str) -> tuple[int, int]:
    suit_order = {"m": 0, "p": 1, "s": 2, "z": 3}
    canonical = _canonical_tile(tile)
    return (suit_order.get(canonical[1:] if len(canonical) > 1 else "", 9), int(canonical[0]))


def _shanten_counts(tiles: Iterable[str]) -> list[int]:
    counts = [0] * len(TILE_TYPES)
    for tile in tiles:
        try:
            index = TILE_TYPES.index(_canonical_tile(tile))
        except ValueError:
            continue
        counts[index] += 1
    return counts


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
    indexes = [TILE_TYPES.index(tile) for tile in ORPHAN_TYPES]
    present = sum(1 for index in indexes if counts[index] > 0)
    has_pair = any(counts[index] >= 2 for index in indexes)
    return 13 - present - int(has_pair)


def _can_form_sequence(index: int) -> bool:
    return 0 <= index < 27 and index % 9 <= 6


def _clamp_melds(value: int) -> int:
    return max(0, min(4, int(value or 0)))
