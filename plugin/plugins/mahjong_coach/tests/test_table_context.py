from __future__ import annotations

from PIL import Image

from plugin.plugins.mahjong_coach.perception.table_context import (
    _parse_counter,
    _parse_score,
    _score_ranks,
    detect_table_context,
    reconcile_riichi_stick_count,
)
from plugin.plugins.mahjong_coach.perception.table_surface import TableSurfaceResult


def _surface() -> TableSurfaceResult:
    return TableSurfaceResult(
        ok=True,
        reason="table_surface_detected",
        warped_image=Image.new("RGB", (800, 800), "navy"),
    )


def test_table_context_requires_all_four_scores_and_reads_both_counters() -> None:
    reads = {
        "score:self": ("32100", 0.98),
        "score:right_opponent": ("26700", 0.94),
        "score:top_opponent": ("24200", 0.91),
        "score:left_opponent": ("17000", 0.89),
        "counter:riichi_stick_count": ("x2", 0.82),
        "counter:honba_count": ("x3", 0.78),
    }

    result = detect_table_context(
        Image.new("RGB", (1920, 1080), "black"),
        table_surface_result=_surface(),
        recognizer=lambda _crop, field: reads[field],
    )

    assert result.ok is True
    assert result.scores == {
        "self": 32100,
        "top_opponent": 24200,
        "left_opponent": 17000,
        "right_opponent": 26700,
    }
    assert result.ranks == {
        "self": 1,
        "top_opponent": 3,
        "left_opponent": 4,
        "right_opponent": 2,
    }
    assert result.riichi_stick_count == 2
    assert result.honba_count == 3
    assert result.confidence == 0.89
    assert result.to_dict()["seat_mapping"] == {
        "self": "bottom",
        "right_opponent": "right",
        "top_opponent": "top",
        "left_opponent": "left",
    }
    assert result.score_reads["self"]["rotation"] == 0
    assert result.score_reads["right_opponent"]["rotation"] == 270
    assert result.score_reads["top_opponent"]["rotation"] == 180
    assert result.score_reads["left_opponent"]["rotation"] == 90


def test_table_context_rejects_partial_or_low_confidence_score_set() -> None:
    reads = {
        "score:self": ("25000", 0.95),
        "score:right_opponent": ("25000", 0.95),
        "score:top_opponent": ("25000", 0.95),
        "score:left_opponent": ("25000", 0.40),
        "counter:riichi_stick_count": ("x0", 0.80),
        "counter:honba_count": ("x0", 0.80),
    }

    result = detect_table_context(
        Image.new("RGB", (1920, 1080), "black"),
        table_surface_result=_surface(),
        recognizer=lambda _crop, field: reads[field],
    )

    assert result.ok is False
    assert result.reason == "four_scores_not_confirmed"
    assert "left_opponent" not in result.scores


def test_table_context_rejects_stable_but_impossible_score_total() -> None:
    reads = {
        "score:self": ("25000", 0.95),
        "score:right_opponent": ("25000", 0.95),
        "score:top_opponent": ("25000", 0.95),
        "score:left_opponent": ("24500", 0.95),
        "counter:riichi_stick_count": ("x0", 0.80),
        "counter:honba_count": ("x0", 0.80),
    }

    result = detect_table_context(
        Image.new("RGB", (1920, 1080), "black"),
        table_surface_result=_surface(),
        recognizer=lambda _crop, field: reads[field],
    )

    assert result.ok is False
    assert result.reason == "score_total_implausible"


def test_numeric_parsing_is_strict_about_points_but_tolerates_counter_icons() -> None:
    assert _parse_score("２５０００", 0.9) == 25000
    assert _parse_score("-1200", 0.9) == -1200
    assert _parse_score("25123", 0.9) is None
    assert _parse_score("25000", 0.4) is None
    assert _parse_counter("杂字x12", 0.7) == 12
    assert _parse_counter("x0", 0.3) is None


def test_score_ranks_use_shared_rank_for_ties() -> None:
    assert _score_ranks({"self": 25000, "left_opponent": 25000, "top_opponent": 30000}) == {
        "self": 2,
        "left_opponent": 2,
        "top_opponent": 1,
    }


def test_riichi_stick_count_prefers_point_conservation_over_counter_ocr() -> None:
    count, pool, source = reconcile_riichi_stick_count(
        {
            "self": 24000,
            "right_opponent": 25000,
            "top_opponent": 25000,
            "left_opponent": 25000,
        },
        0,
    )
    assert (count, pool, source) == (1, 100000, "common_point_pool")

    count, pool, source = reconcile_riichi_stick_count(
        {
            "self": 23000,
            "right_opponent": 25000,
            "top_opponent": 25000,
            "left_opponent": 25000,
        },
        7,
        point_pool_total=100000,
    )
    assert (count, pool, source) == (2, 100000, "score_pool_conservation")


def test_riichi_stick_count_keeps_ocr_as_custom_room_fallback() -> None:
    count, pool, source = reconcile_riichi_stick_count(
        {
            "self": 30000,
            "right_opponent": 25000,
            "top_opponent": 25000,
            "left_opponent": 25000,
        },
        3,
    )
    assert (count, pool, source) == (3, 108000, "counter_ocr_baseline")
