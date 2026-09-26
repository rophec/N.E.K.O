from __future__ import annotations

from plugin.plugins.mahjong_coach.coach import RoundCoachEngine, build_round_plan
from plugin.plugins.mahjong_coach.perception.river_state import RiverStateResult
from plugin.plugins.mahjong_coach.strategy.efficiency import analyze_hand_efficiency, calculate_shanten
from plugin.plugins.mahjong_coach.strategy.risk import assess_genbutsu


def test_calculate_shanten_supports_standard_seven_pairs_and_orphans() -> None:
    complete = calculate_shanten(
        ["1m", "2m", "3m", "1p", "2p", "3p", "1s", "2s", "3s", "1z", "1z", "1z", "2z", "2z"]
    )
    seven_pairs = calculate_shanten(
        ["1m", "1m", "2m", "2m", "3p", "3p", "4p", "4p", "5s", "5s", "6s", "6s", "7z"]
    )
    orphans = calculate_shanten(
        ["1m", "9m", "1p", "9p", "1s", "9s", "1z", "2z", "3z", "4z", "5z", "6z", "7z"]
    )

    assert (complete.shanten, complete.path) == (-1, "standard")
    assert (seven_pairs.shanten, seven_pairs.path) == (0, "seven_pairs")
    assert (orphans.shanten, orphans.path) == (0, "thirteen_orphans")


def test_thirteen_tile_hand_reports_current_effective_tiles() -> None:
    analysis = analyze_hand_efficiency(
        ["1m", "2m", "3m", "1p", "2p", "3p", "1s", "2s", "3s", "1z", "1z", "1z", "2z"]
    )

    assert analysis.shanten == 0
    assert [item.tile for item in analysis.current_effective_tiles] == ["2z"]
    assert analysis.current_effective_tiles[0].unseen_count == 3


def test_visible_sources_reduce_unseen_count_without_claiming_wall_count() -> None:
    hand = ["1m", "2m", "3m", "1p", "2p", "3p", "1s", "2s", "3s", "1z", "1z", "1z", "2z", "9m"]
    analysis = analyze_hand_efficiency(
        hand,
        river_tiles=["2z"],
        meld_tiles=["2z"],
        dora_indicators=["3z"],
    )
    best = next(item for item in analysis.discard_options if item.tile == "9m")

    assert best.shanten == 0
    assert best.effective_count == 1
    assert best.effective_tiles[0].tile == "2z"
    assert best.effective_tiles[0].visible_count == 2
    assert analysis.source_counts == {"river": 1, "meld": 1, "dora_indicator": 1, "other": 0}
    assert analysis.to_dict()["remaining_count_kind"] == "unseen_not_wall"


def test_copy_overflow_is_reported_as_recognition_anomaly() -> None:
    analysis = analyze_hand_efficiency(
        ["5p", "5p", "1m", "2m", "3m", "1p", "2p", "3p", "1s", "2s", "3s", "1z", "1z"],
        river_tiles=["5p", "5p", "5p"],
    )

    assert "visible_copy_overflow:5p:5" in analysis.anomalies


def test_round_plan_exposes_exact_engine_and_visible_source_breakdown() -> None:
    plan = build_round_plan(
        ["1m", "2m", "3m", "1p", "2p", "3p", "1s", "2s", "3s", "1z", "1z", "1z", "2z", "9m"],
        visible_tiles=["2z"],
        meld_tiles=["2z", "3m", "3m"],
        dora_indicators=["3z"],
        open_melds=0,
    )
    efficiency = plan["efficiency"]

    assert efficiency["calculation_source"] == "strategy.efficiency/analyze_hand_efficiency"
    assert efficiency["visible_source_counts"] == {
        "river": 1,
        "meld": 3,
        "dora_indicator": 1,
        "other": 0,
    }
    assert efficiency["remaining_count_kind"] == "unseen_not_wall"
    assert len(efficiency["all_discard_options"]) >= len(efficiency["discard_options"])


def test_engine_state_keeps_exact_efficiency_for_dashboard() -> None:
    engine = RoundCoachEngine()
    efficiency = {"current_shanten": 1, "calculation_source": "strategy.efficiency/analyze_hand_efficiency"}

    engine._remember_local_plan({"direction": "牌效推进", "efficiency": efficiency})

    assert engine.state.plan_source == "deterministic_efficiency+heuristic_route"
    assert engine.state.last_efficiency == efficiency


def test_genbutsu_uses_intersection_for_multiple_riichi_players() -> None:
    assessment = assess_genbutsu(
        ["1m", "2m", "3m"],
        {
            "left_opponent": [{"tile": "1m"}, {"tile": "2m"}],
            "right_opponent": [{"tile": "2m"}, {"tile": "3m"}],
        },
        ["left_opponent", "right_opponent"],
    )

    assert assessment.safe_by_player["left_opponent"] == ("1m", "2m")
    assert assessment.safe_by_player["right_opponent"] == ("2m", "3m")
    assert assessment.safe_against_all == ("2m",)
    assert assessment.held_safe_against_all == ("2m",)


def test_genbutsu_does_not_claim_universal_safety_when_a_river_is_missing() -> None:
    assessment = assess_genbutsu(
        ["1m"],
        {"left_opponent": [{"tile": "1m"}]},
        ["left_opponent", "right_opponent"],
    )

    assert assessment.safe_against_all == ()
    assert assessment.held_safe_against_all == ()
    assert assessment.missing_players == ("right_opponent",)


def test_genbutsu_reports_per_tile_coverage_and_normalizes_red_five() -> None:
    assessment = assess_genbutsu(
        ["0m", "6p"],
        {
            "left_opponent": [{"tile": "5m"}],
            "right_opponent": [{"tile": "5m"}, {"tile": "6p"}],
        },
        ["left_opponent", "right_opponent"],
    )

    red_five, six_pin = assessment.held_tile_safety
    assert red_five.tile_type == "5m"
    assert red_five.genbutsu_against == ("left_opponent", "right_opponent")
    assert red_five.safe_against_all is True
    assert six_pin.genbutsu_against == ("right_opponent",)
    assert six_pin.not_proven_safe_against == ("left_opponent",)
    assert six_pin.safe_against_all is False
    assert assessment.can_still_tsumo is True


def test_defense_suggestion_never_unions_two_riichi_rivers() -> None:
    engine = RoundCoachEngine()
    river = RiverStateResult(
        ok=True,
        discard_piles={
            "left_opponent": [{"tile": "1m"}],
            "right_opponent": [{"tile": "3m"}],
        },
    )

    suggestion = engine._defense_suggestion(["left", "right"], river, hand_tiles=["1m", "3m"])

    assert "没有共同交集" in suggestion
