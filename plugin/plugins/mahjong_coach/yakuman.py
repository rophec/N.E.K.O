from __future__ import annotations

import itertools
import math
import random
import threading
import time
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from functools import lru_cache
from typing import Any, Callable

from .models import YakumanEstimate
from .tile_labels import normalize_tile


TILE_TYPES = [f"{rank}{suit}" for suit in ("m", "p", "s") for rank in range(1, 10)] + [
    f"{rank}z" for rank in range(1, 8)
]
ORPHANS = {"1m", "9m", "1p", "9p", "1s", "9s", *{f"{rank}z" for rank in range(1, 8)}}
WINDS = {"1z", "2z", "3z", "4z"}
DRAGONS = {"5z", "6z", "7z"}
HONORS = WINDS | DRAGONS
TERMINALS = {"1m", "9m", "1p", "9p", "1s", "9s"}
GREENS = {"2s", "3s", "4s", "6s", "8s", "6z"}
MILESTONES = (6, 12, 18)


_ROUTES: tuple[tuple[str, str], ...] = (
    ("kokushi", "国士无双"),
    ("suuankou", "四暗刻"),
    ("daisangen", "大三元"),
    ("shousuushii", "小四喜"),
    ("daisuushii", "大四喜"),
    ("tsuuiisou", "字一色"),
    ("chinroutou", "清老头"),
    ("ryuuiisou", "绿一色"),
    ("chuuren", "九莲宝灯"),
)


def assess_yakuman_routes(
    hand_tiles: list[str],
    *,
    visible_tiles: list[str] | None = None,
    open_melds: int = 0,
) -> list[YakumanEstimate]:
    hand = _canonical_tiles(hand_tiles)
    visible = _canonical_tiles(visible_tiles or [])
    wall_counts = Counter({tile: 4 for tile in TILE_TYPES})
    wall_counts.subtract(hand)
    wall_counts.subtract(visible)
    estimates: list[YakumanEstimate] = []
    for route, label in _ROUTES:
        distance = route_distance(route, hand, open_melds=open_melds)
        blockers = _route_blockers(route, hand, open_melds)
        key_tiles = _route_key_tiles(
            route,
            hand,
            wall_counts,
            distance=distance,
            open_melds=open_melds,
        )
        estimates.append(
            YakumanEstimate(
                route=route,
                label=label,
                distance=distance,
                key_tiles=key_tiles,
                blockers=blockers,
                estimated=False,
            )
        )
    return sorted(
        estimates,
        key=lambda item: (_has_hard_blocker(item), item.distance, len(item.blockers), item.route),
    )


def monte_carlo_yakuman(
    hand_tiles: list[str],
    *,
    visible_tiles: list[str] | None = None,
    open_melds: int = 0,
    max_trials: int = 10_000,
    time_budget_ms: int = 250,
    seed: int | None = None,
) -> list[YakumanEstimate]:
    started = time.perf_counter()
    deterministic = assess_yakuman_routes(
        hand_tiles,
        visible_tiles=visible_tiles,
        open_melds=open_melds,
    )
    active = [item for item in deterministic if not _has_hard_blocker(item)]
    if not active:
        return deterministic
    hand = _canonical_tiles(hand_tiles)
    visible = _canonical_tiles(visible_tiles or [])
    wall = _remaining_wall(hand, visible)
    rng = random.Random(seed)
    deadline = started + max(1, int(time_budget_ms)) / 1000.0
    limit = max(len(active), min(10_000, int(max_trials)))
    stats = {
        item.route: {
            "trials": 0,
            "tenpai": {turn: 0 for turn in MILESTONES},
            "tsumo": {turn: 0 for turn in MILESTONES},
        }
        for item in active
    }
    trial_index = 0
    while trial_index < limit and (trial_index < len(active) or time.perf_counter() < deadline):
        route = active[trial_index % len(active)].route
        _simulate_route_trial(
            route,
            hand,
            wall,
            stats[route],
            rng,
            open_melds=open_melds,
            deadline=deadline,
        )
        trial_index += 1

    enriched: list[YakumanEstimate] = []
    for item in deterministic:
        route_stats = stats.get(item.route)
        if not route_stats:
            enriched.append(item)
            continue
        trials = int(route_stats["trials"])
        tenpai = {
            str(turn): _probability(route_stats["tenpai"][turn], trials)
            for turn in MILESTONES
        }
        tsumo = {
            str(turn): _probability(route_stats["tsumo"][turn], trials)
            for turn in MILESTONES
        }
        intervals = {
            f"tenpai_{turn}": list(_wilson_interval(route_stats["tenpai"][turn], trials))
            for turn in MILESTONES
        }
        intervals.update(
            {
                f"tsumo_{turn}": list(_wilson_interval(route_stats["tsumo"][turn], trials))
                for turn in MILESTONES
            }
        )
        enriched.append(
            replace(
                item,
                tenpai_probability=tenpai,
                tsumo_probability=tsumo,
                confidence_interval=intervals,
                trials=trials,
                estimated=True,
            )
        )
    return enriched


class YakumanEstimateService:
    """One-worker latest-result service so live advice never waits for simulation."""

    def __init__(self, *, max_trials: int = 10_000, time_budget_ms: int = 250) -> None:
        self.max_trials = max_trials
        self.time_budget_ms = time_budget_ms
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mahjong-yakuman")
        self._lock = threading.Lock()
        self._futures: dict[str, Future[list[YakumanEstimate]]] = {}
        self._results: dict[str, list[YakumanEstimate]] = {}

    def request(
        self,
        hand_tiles: list[str],
        *,
        visible_tiles: list[str] | None = None,
        open_melds: int = 0,
    ) -> dict[str, Any]:
        hand = _canonical_tiles(hand_tiles)
        visible = _canonical_tiles(visible_tiles or [])
        key = "|".join([",".join(sorted(hand)), ",".join(sorted(visible)), str(open_melds)])
        immediate = assess_yakuman_routes(hand, visible_tiles=visible, open_melds=open_melds)
        with self._lock:
            future = self._futures.get(key)
            if future is not None and future.done():
                try:
                    self._results[key] = future.result()
                finally:
                    self._futures.pop(key, None)
            ready = self._results.get(key)
            if ready is not None:
                return self._payload("ready", key, ready)
            if future is None:
                self._futures[key] = self._executor.submit(
                    monte_carlo_yakuman,
                    hand,
                    visible_tiles=visible,
                    open_melds=open_melds,
                    max_trials=self.max_trials,
                    time_budget_ms=self.time_budget_ms,
                    seed=None,
                )
        return self._payload("running", key, immediate)

    def _payload(self, status: str, key: str, routes: list[YakumanEstimate]) -> dict[str, Any]:
        return {
            "status": status,
            "key": key,
            "routes": [item.to_dict() for item in routes],
            "max_trials": self.max_trials,
            "time_budget_ms": self.time_budget_ms,
            "assumptions": [
                "unknown_tiles_uniform",
                "opponent_ron_not_simulated",
                "four_player_only",
            ],
        }

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


def route_distance(route: str, hand_tiles: list[str], *, open_melds: int = 0) -> int:
    hand = _canonical_tiles(hand_tiles)
    counts = Counter(hand)
    if route in {"kokushi", "suuankou", "chuuren"} and open_melds:
        return 14
    if route == "kokushi":
        return _min_replacements(counts, _route_targets(route))
    if route == "suuankou":
        return _triplet_hand_distance(counts, set(TILE_TYPES))
    if route in {"daisangen", "shousuushii", "daisuushii"}:
        return _min_replacements(counts, _route_targets(route))
    if route == "tsuuiisou":
        return _triplet_hand_distance(counts, HONORS)
    if route == "chinroutou":
        return _triplet_hand_distance(counts, TERMINALS)
    if route in {"ryuuiisou", "chuuren"}:
        return _min_replacements(counts, _route_targets(route))
    return 14


def is_route_complete(route: str, hand_tiles: list[str], *, open_melds: int = 0) -> bool:
    hand = _canonical_tiles(hand_tiles)
    if len(hand) % 3 != 2:
        return False
    if route in {"kokushi", "suuankou", "chuuren"} and open_melds:
        return False
    counts = Counter(hand)
    if route == "kokushi":
        return all(counts[tile] >= 1 for tile in ORPHANS) and any(counts[tile] >= 2 for tile in ORPHANS)
    if not _is_standard_complete(counts, open_melds=open_melds):
        return False
    if route == "suuankou":
        return sum(1 for value in counts.values() if value >= 3) >= 4
    if route == "daisangen":
        return all(counts[tile] >= 3 for tile in DRAGONS)
    if route == "shousuushii":
        return sum(counts[tile] >= 3 for tile in WINDS) == 3 and sum(counts[tile] >= 2 for tile in WINDS) == 4
    if route == "daisuushii":
        return all(counts[tile] >= 3 for tile in WINDS)
    if route == "tsuuiisou":
        return all(tile in HONORS for tile in hand)
    if route == "chinroutou":
        return all(tile in TERMINALS for tile in hand)
    if route == "ryuuiisou":
        return all(tile in GREENS for tile in hand)
    if route == "chuuren":
        suits = {tile[1] for tile in hand if tile[1] in {"m", "p", "s"}}
        if len(suits) != 1 or any(tile.endswith("z") for tile in hand):
            return False
        suit = next(iter(suits))
        return counts[f"1{suit}"] >= 3 and counts[f"9{suit}"] >= 3 and all(counts[f"{rank}{suit}"] >= 1 for rank in range(2, 9))
    return False


def _simulate_route_trial(
    route: str,
    starting_hand: list[str],
    wall: list[str],
    stats: dict[str, Any],
    rng: random.Random,
    *,
    open_melds: int,
    deadline: float,
) -> None:
    stats["trials"] += 1
    hand = list(starting_hand)
    while len(hand) > 13:
        _discard_for_route(route, hand, open_melds=open_melds)
    draws = rng.sample(wall, k=min(max(MILESTONES), len(wall)))
    tenpai_at: int | None = None
    complete_at: int | None = None
    for turn, tile in enumerate(draws, start=1):
        if time.perf_counter() >= deadline:
            break
        hand.append(tile)
        if is_route_complete(route, hand, open_melds=open_melds):
            complete_at = turn
            tenpai_at = tenpai_at or turn
            break
        _discard_for_route(route, hand, open_melds=open_melds)
        if tenpai_at is None and route_distance(route, hand, open_melds=open_melds) <= 1:
            tenpai_at = turn
    for milestone in MILESTONES:
        if tenpai_at is not None and tenpai_at <= milestone:
            stats["tenpai"][milestone] += 1
        if complete_at is not None and complete_at <= milestone:
            stats["tsumo"][milestone] += 1


def _discard_for_route(route: str, hand: list[str], *, open_melds: int) -> None:
    counts = Counter(hand)
    preferred = {
        "kokushi": ORPHANS,
        "daisangen": DRAGONS,
        "shousuushii": WINDS,
        "daisuushii": WINDS,
        "tsuuiisou": HONORS,
        "chinroutou": TERMINALS,
        "ryuuiisou": GREENS,
    }.get(route)

    def keep_score(tile: str) -> tuple[float, str]:
        if preferred is not None and tile not in preferred:
            return (-100.0, tile)
        copies = counts[tile]
        if route == "kokushi":
            return (3.0 if copies == 1 else 2.0 if copies == 2 else 0.0, tile)
        if route == "chuuren":
            suit_counts = Counter(item[1] for item in hand if not item.endswith("z"))
            best_suit = max(("m", "p", "s"), key=lambda suit: suit_counts[suit])
            if tile.endswith("z") or tile[1] != best_suit:
                return (-100.0, tile)
            needed = 3 if tile[0] in {"1", "9"} else 1
            return (4.0 if copies <= needed else 1.0, tile)
        required_bonus = 4.0 if (
            (route == "daisangen" and tile in DRAGONS)
            or (route in {"shousuushii", "daisuushii"} and tile in WINDS)
        ) else 0.0
        shape_bonus = 3.0 if copies >= 3 else 2.0 if copies == 2 else 0.5
        return (required_bonus + shape_bonus, tile)

    best_index = min(range(len(hand)), key=lambda index: keep_score(hand[index]))
    hand.pop(best_index)


@lru_cache(maxsize=None)
def _route_targets(route: str) -> tuple[Counter[str], ...]:
    targets: list[Counter[str]] = []
    if route == "kokushi":
        for pair in ORPHANS:
            target = Counter({tile: 1 for tile in ORPHANS})
            target[pair] += 1
            targets.append(target)
    elif route == "daisangen":
        targets = _complete_targets([Counter({tile: 3}) for tile in DRAGONS])
    elif route == "shousuushii":
        for pair in WINDS:
            fixed = [Counter({wind: 3}) for wind in WINDS if wind != pair]
            targets.extend(_complete_targets(fixed, fixed_pair=pair))
    elif route == "daisuushii":
        fixed = [Counter({wind: 3}) for wind in WINDS]
        for pair in TILE_TYPES:
            if pair in WINDS:
                continue
            target = sum(fixed, Counter())
            target[pair] += 2
            if max(target.values(), default=0) <= 4:
                targets.append(target)
    elif route == "ryuuiisou":
        melds = [Counter({tile: 3}) for tile in GREENS]
        melds.append(Counter(("2s", "3s", "4s")))
        for selected in itertools.combinations_with_replacement(melds, 4):
            for pair in GREENS:
                target = sum(selected, Counter())
                target[pair] += 2
                if max(target.values(), default=0) <= 4:
                    targets.append(target)
    elif route == "chuuren":
        for suit in ("m", "p", "s"):
            base = Counter({f"1{suit}": 3, f"9{suit}": 3})
            for rank in range(2, 9):
                base[f"{rank}{suit}"] = 1
            for extra in range(1, 10):
                target = Counter(base)
                target[f"{extra}{suit}"] += 1
                targets.append(target)
    return tuple(targets)


def _route_blockers(route: str, hand: list[str], open_melds: int) -> list[str]:
    blockers: list[str] = []
    if open_melds and route in {"kokushi", "suuankou", "chuuren"}:
        blockers.append("该路线要求门清")
    allowed = {
        "kokushi": ORPHANS,
        "tsuuiisou": HONORS,
        "chinroutou": TERMINALS,
        "ryuuiisou": GREENS,
    }.get(route)
    if allowed is not None:
        off_route = sum(tile not in allowed for tile in hand)
        if off_route:
            blockers.append(f"需先处理{off_route}张路线外牌")
    return blockers


def _has_hard_blocker(item: YakumanEstimate) -> bool:
    return any("要求门清" in blocker for blocker in item.blockers)


def _route_key_tiles(
    route: str,
    hand: list[str],
    wall_counts: Counter[str],
    *,
    distance: int,
    open_melds: int,
) -> list[str]:
    targets = _route_targets(route)
    if targets:
        counts = Counter(hand)
        scored = [
            (sum(min(counts[tile], amount) for tile, amount in target.items()), target)
            for target in targets
        ]
        best_score = max(score for score, _target in scored)
        missing = {
            tile
            for score, target in scored
            if score == best_score
            for tile, amount in target.items()
            if wall_counts[tile] > 0 and counts[tile] < amount
        }
        return [
            tile
            for tile in TILE_TYPES
            if tile in missing
        ]
    return [
        tile
        for tile in TILE_TYPES
        if wall_counts[tile] > 0
        and route_distance(route, [*hand, tile], open_melds=open_melds) < distance
    ][:10]


def _remaining_wall(hand: list[str], visible: list[str]) -> list[str]:
    counts = Counter({tile: 4 for tile in TILE_TYPES})
    counts.subtract(hand)
    counts.subtract(visible)
    return [tile for tile in TILE_TYPES for _ in range(max(0, counts[tile]))]


def _fixed_meld_route_distance(counts: Counter[str], fixed_melds: list[Counter[str]]) -> int:
    return _min_replacements(counts, _complete_targets(fixed_melds))


def _complete_targets(
    fixed_melds: list[Counter[str]],
    *,
    fixed_pair: str | None = None,
) -> list[Counter[str]]:
    missing_melds = 4 - len(fixed_melds)
    melds = _all_meld_patterns(TILE_TYPES)
    targets: list[Counter[str]] = []
    for selected in itertools.combinations_with_replacement(melds, missing_melds):
        pair_candidates = [fixed_pair] if fixed_pair else TILE_TYPES
        for pair in pair_candidates:
            target = sum(fixed_melds, Counter())
            for meld in selected:
                target.update(meld)
            target[str(pair)] += 2
            if sum(target.values()) == 14 and max(target.values(), default=0) <= 4:
                targets.append(target)
    return targets


def _triplet_hand_distance(counts: Counter[str], allowed: set[str]) -> int:
    best = 14
    for pair in allowed:
        matched_pair = min(2, counts[pair])
        triplet_matches = sorted(
            (min(3, counts[tile]) for tile in allowed if tile != pair),
            reverse=True,
        )[:4]
        best = min(best, 14 - matched_pair - sum(triplet_matches))
    return best


def _restricted_standard_distance(
    counts: Counter[str],
    allowed: set[str],
    *,
    extra_sequences: list[tuple[str, str, str]] | None = None,
) -> int:
    melds = [Counter({tile: 3}) for tile in allowed]
    melds.extend(Counter(sequence) for sequence in (extra_sequences or []))
    targets: list[Counter[str]] = []
    for selected in itertools.combinations_with_replacement(melds, 4):
        for pair in allowed:
            target = sum(selected, Counter())
            target[pair] += 2
            if max(target.values(), default=0) <= 4:
                targets.append(target)
    return _min_replacements(counts, targets)


def _all_meld_patterns(allowed: list[str]) -> list[Counter[str]]:
    allowed_set = set(allowed)
    melds = [Counter({tile: 3}) for tile in allowed]
    for suit in ("m", "p", "s"):
        for start in range(1, 8):
            sequence = [f"{rank}{suit}" for rank in range(start, start + 3)]
            if all(tile in allowed_set for tile in sequence):
                melds.append(Counter(sequence))
    return melds


def _min_replacements(counts: Counter[str], targets: Any) -> int:
    return min(
        (14 - sum(min(counts[tile], amount) for tile, amount in target.items()) for target in targets),
        default=14,
    )


def _is_standard_complete(counts: Counter[str], *, open_melds: int = 0) -> bool:
    required_melds = 4 - max(0, min(4, int(open_melds)))
    concealed_size = required_melds * 3 + 2
    if sum(counts.values()) != concealed_size:
        return False
    array = [counts[tile] for tile in TILE_TYPES]
    for pair_index, value in enumerate(array):
        if value < 2:
            continue
        array[pair_index] -= 2
        if _consume_melds(array, required_melds):
            array[pair_index] += 2
            return True
        array[pair_index] += 2
    return False


def _consume_melds(counts: list[int], remaining: int) -> bool:
    if remaining == 0:
        return not any(counts)
    try:
        index = next(i for i, value in enumerate(counts) if value)
    except StopIteration:
        return False
    if counts[index] >= 3:
        counts[index] -= 3
        if _consume_melds(counts, remaining - 1):
            counts[index] += 3
            return True
        counts[index] += 3
    if index < 27 and index % 9 <= 6 and counts[index + 1] and counts[index + 2]:
        counts[index] -= 1
        counts[index + 1] -= 1
        counts[index + 2] -= 1
        if _consume_melds(counts, remaining - 1):
            counts[index] += 1
            counts[index + 1] += 1
            counts[index + 2] += 1
            return True
        counts[index] += 1
        counts[index + 1] += 1
        counts[index + 2] += 1
    return False


def _canonical_tiles(tiles: list[str]) -> list[str]:
    result: list[str] = []
    for raw in tiles:
        tile = normalize_tile(raw)
        if tile in {"0m", "0p", "0s"}:
            tile = f"5{tile[1]}"
        if tile in TILE_TYPES:
            result.append(tile)
    return result


def _probability(successes: int, trials: int) -> float:
    return round(successes / trials, 6) if trials else 0.0


def _wilson_interval(successes: int, trials: int) -> tuple[float, float]:
    if trials <= 0:
        return 0.0, 0.0
    z = 1.96
    p = successes / trials
    denominator = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denominator
    margin = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * trials)) / trials) / denominator
    return round(max(0.0, center - margin), 6), round(min(1.0, center + margin), 6)
