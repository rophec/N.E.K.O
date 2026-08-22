from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

from PIL import Image, ImageDraw

from plugin.plugins.mahjong_coach.perception import yolo26_visible_tiles as visible_tiles
from plugin.plugins.mahjong_coach.perception.table_surface import (
    _SupportLine,
    TableSurfaceResult,
    _order_quad_points,
    _quad_bbox,
    _select_supported_line,
    detect_table_surface,
)
from plugin.plugins.mahjong_coach.perception.yolo26_visible_tiles import (
    DEFAULT_MODEL_DIR,
    Yolo26TableStateResult,
    YoloTileDetection,
    _decode_end2end_output,
    detect_yolo26_table_state_path,
    load_yolo26_backend,
    postprocess_yolo26_detections,
)


def detection(tile: str, confidence: float, bbox: list[float]) -> YoloTileDetection:
    return YoloTileDetection(tile=tile, confidence=confidence, bbox=bbox, source="test_yolo26")


def test_default_yolo26_model_bundle_is_installed_and_matches_metadata() -> None:
    metadata_path = DEFAULT_MODEL_DIR / "metadata.json"
    labels_path = DEFAULT_MODEL_DIR / "labels.json"
    model_path = DEFAULT_MODEL_DIR / "model.onnx"

    assert metadata_path.is_file()
    assert labels_path.is_file()
    assert model_path.is_file()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with model_path.open("rb") as model_file:
        digest = hashlib.file_digest(model_file, "sha256").hexdigest()
    assert digest == metadata["sha256"]

    backend = load_yolo26_backend(DEFAULT_MODEL_DIR)
    assert backend.available is True
    assert backend.runtime == "onnxruntime"
    assert len(backend.labels) == 34

    hints = visible_tiles._base_hints(DEFAULT_MODEL_DIR, "", backend.runtime)
    assert hints["model_source"] == "bundled"
    assert hints["model_id"] == metadata["training_run"]
    assert hints["model_hash"] == metadata["sha256"][:12]
    assert "yolo26_model_dir" not in hints


def test_yolo26_postprocess_splits_hand_meld_and_river() -> None:
    result = postprocess_yolo26_detections(
        [
            detection("3m", 0.91, [220, 820, 260, 900]),
            detection("1m", 0.93, [120, 820, 160, 900]),
            detection("5z", 0.88, [1480, 690, 1520, 760]),
            detection("5z", 0.87, [1530, 690, 1570, 760]),
            detection("5z", 0.86, [1580, 690, 1620, 760]),
        ],
        image_size=(1920, 1080),
        river_detections=[
            detection("7p", 0.89, [420, 170, 460, 230]),
            detection("2s", 0.90, [250, 325, 290, 375]),
            detection("9m", 0.92, [610, 325, 650, 375]),
            detection("4s", 0.94, [430, 550, 470, 610]),
        ],
        river_image_size=(800, 800),
        min_confidence=0.25,
    )

    assert result["hand_tiles"] == ["1m", "3m"]
    assert result["meld_tiles"] == ["5z", "5z", "5z"]
    assert len(result["melds"]) == 1
    assert result["discard_piles"]["top_opponent"][0]["tile"] == "7p"
    assert result["discard_piles"]["left_opponent"][0]["tile"] == "2s"
    assert result["discard_piles"]["right_opponent"][0]["tile"] == "9m"
    assert result["discard_piles"]["self"][0]["tile"] == "4s"
    assert {item.coordinate_space for item in result["original_detections"]} == {"original_frame"}
    assert {item.coordinate_space for item in result["river_detections"]} == {"warped_table"}


def test_yolo26_postprocess_excludes_outer_visible_tiles_from_river() -> None:
    result = postprocess_yolo26_detections(
        [],
        image_size=(1920, 1080),
        river_detections=[
            detection("5z", 0.95, [60, 60, 100, 140]),
            detection("6p", 0.94, [690, 180, 730, 250]),
            detection("3m", 0.93, [355, 265, 395, 330]),
        ],
        river_image_size=(800, 800),
    )

    assert result["visible_tiles"] == ["3m"]
    assert result["opponent_meld_count"] == 0
    assert result["opponent_melds"] == {}
    assert result["opponent_meld_tiles"] == []
    assert result["excluded_visible_count"] == 2
    assert sorted(item.area_kind for item in result["detections"]) == [
        "excluded_table_tile",
        "excluded_table_tile",
        "river",
    ]


def test_yolo26_groups_top_and_right_opponent_meld_shelves() -> None:
    result = postprocess_yolo26_detections(
        [],
        image_size=(1920, 1080),
        river_detections=[
            # Top opponent: one chi followed by one pon.
            detection("3m", 0.95, [68, 58, 96, 100]),
            detection("4m", 0.95, [96, 58, 124, 100]),
            detection("5m", 0.95, [124, 58, 162, 88]),
            detection("7p", 0.95, [162, 58, 190, 100]),
            detection("7p", 0.95, [190, 58, 218, 100]),
            detection("7p", 0.95, [218, 58, 256, 88]),
            # Right opponent: two pons, with each called tile turned.
            detection("3z", 0.95, [710, 72, 752, 100]),
            detection("3z", 0.95, [710, 100, 752, 128]),
            detection("3z", 0.95, [722, 128, 752, 166]),
            detection("9s", 0.95, [710, 166, 752, 194]),
            detection("9s", 0.95, [710, 194, 752, 222]),
            detection("9s", 0.95, [722, 222, 752, 260]),
        ],
        river_image_size=(800, 800),
    )

    assert result["opponent_meld_count"] == 4
    assert [item["kind"] for item in result["opponent_melds"]["top_opponent"]] == ["chi", "pon"]
    assert [item["tiles"] for item in result["opponent_melds"]["top_opponent"]] == [
        ["3m", "4m", "5m"],
        ["7p", "7p", "7p"],
    ]
    assert [item["kind"] for item in result["opponent_melds"]["right_opponent"]] == ["pon", "pon"]
    assert sum(item.area_kind == "opponent_meld" for item in result["river_detections"]) == 12


def test_yolo26_groups_left_opponent_chi_and_preserves_owner() -> None:
    result = postprocess_yolo26_detections(
        [],
        image_size=(1920, 1080),
        river_detections=[
            detection("7s", 0.95, [42, 632, 70, 674]),
            detection("8s", 0.95, [42, 674, 82, 704]),
            detection("9s", 0.95, [42, 704, 82, 734]),
        ],
        river_image_size=(800, 800),
    )

    meld = result["opponent_melds"]["left_opponent"][0]
    assert meld["owner"] == "left_opponent"
    assert meld["kind"] == "chi"
    assert sorted(meld["tiles"]) == ["7s", "8s", "9s"]
    assert result["opponent_meld_tiles"] == meld["tiles"]


def test_yolo26_repairs_only_the_unique_turned_called_tile_for_legal_pon() -> None:
    result = postprocess_yolo26_detections(
        [],
        image_size=(1920, 1080),
        river_detections=[
            detection("7s", 0.40, [160, 70, 198, 100]),
            detection("5s", 0.97, [198, 70, 228, 110]),
            detection("5s", 0.96, [228, 70, 258, 110]),
        ],
        river_image_size=(800, 800),
    )

    meld = result["opponent_melds"]["top_opponent"][0]
    assert meld["kind"] == "pon"
    assert meld["observed_tiles"] == ["7s", "5s", "5s"]
    assert meld["tiles"] == ["5s", "5s", "5s"]
    assert meld["called_tile_index"] == 0
    assert meld["corrections"] == [
        {
            "tile_index": 0,
            "from": "7s",
            "to": "5s",
            "reason": "called_tile_legal_pon_recovery",
        }
    ]


def test_yolo26_groups_four_identical_opponent_tiles_as_kan() -> None:
    result = postprocess_yolo26_detections(
        [],
        image_size=(1920, 1080),
        river_detections=[
            detection("1z", 0.95, [68, 58, 96, 100]),
            detection("1z", 0.95, [96, 58, 124, 100]),
            detection("1z", 0.95, [124, 58, 152, 100]),
            detection("1z", 0.95, [152, 58, 190, 88]),
        ],
        river_image_size=(800, 800),
    )

    meld = result["opponent_melds"]["top_opponent"][0]
    assert meld["kind"] == "kan"
    assert meld["tiles"] == ["1z"] * 4
    assert result["opponent_meld_count"] == 1


def test_yolo26_does_not_promote_aligned_dora_indicators_without_called_rotation() -> None:
    result = postprocess_yolo26_detections(
        [],
        image_size=(1920, 1080),
        river_detections=[
            detection("3m", 0.95, [68, 58, 96, 100]),
            detection("4m", 0.95, [96, 58, 124, 100]),
            detection("5m", 0.95, [124, 58, 152, 100]),
        ],
        river_image_size=(800, 800),
    )

    assert result["opponent_melds"] == {}
    assert result["opponent_meld_count"] == 0
    assert {item.area_kind for item in result["river_detections"]} == {"excluded_table_tile"}


def test_yolo26_river_result_carries_structured_opponent_melds() -> None:
    opponent_melds = {
        "top_opponent": [
            {
                "owner": "top_opponent",
                "meld_index": 1,
                "kind": "pon",
                "tiles": ["5z", "5z", "5z"],
            }
        ]
    }
    state = Yolo26TableStateResult(
        ok=True,
        opponent_melds=opponent_melds,
        opponent_meld_tiles=["5z", "5z", "5z"],
        river_inference_ok=True,
    )

    river = state.to_river_result()

    assert river.opponent_melds == opponent_melds
    assert river.opponent_melds is not opponent_melds
    assert river.opponent_meld_tiles == ["5z", "5z", "5z"]


def test_yolo26_postprocess_filters_low_confidence_and_dedupes() -> None:
    result = postprocess_yolo26_detections(
        [
            detection("1m", 0.95, [100, 850, 150, 930]),
            detection("2m", 0.80, [104, 852, 154, 932]),
            detection("3m", 0.10, [220, 850, 270, 930]),
            detection("empty", 0.99, [320, 850, 370, 930]),
        ],
        image_size=(1920, 1080),
        min_confidence=0.25,
    )

    assert result["hand_tiles"] == ["1m"]
    assert len(result["detections"]) == 1


def test_yolo26_recovers_one_interior_low_confidence_hand_tile() -> None:
    detections = [
        detection(
            f"{(index % 9) + 1}m",
            0.22 if index == 12 else 0.95,
            [100 + index * 38, 720, 134 + index * 38, 790],
        )
        for index in range(14)
    ]

    result = postprocess_yolo26_detections(detections, image_size=(800, 800), min_confidence=0.25)

    assert len(result["hand_tiles"]) == 14
    assert result["hand_recovered_count"] == 1
    recovered = [item for item in result["original_detections"] if item.area_kind == "hand" and item.confidence < 0.25]
    assert len(recovered) == 1
    assert recovered[0].source.endswith(":geometric_recovery")


def test_yolo26_recovers_complete_low_confidence_self_meld_geometry() -> None:
    detections = [
        detection(f"{(index % 9) + 1}m", 0.95, [100 + index * 38, 720, 134 + index * 38, 790])
        for index in range(10)
    ]
    detections.extend(
        [
            detection("6z", 0.025, [600, 730, 676, 790]),
            detection("6s", 0.026, [674, 716, 740, 790]),
            detection("9s", 0.011, [738, 718, 802, 790]),
        ]
    )

    result = postprocess_yolo26_detections(detections, image_size=(900, 800), min_confidence=0.25)

    assert len(result["hand_tiles"]) == 10
    assert result["meld_tiles"] == ["6z", "6s", "9s"]
    assert len(result["melds"]) == 1
    assert result["meld_recovered_count"] == 3
    assert result["meld_identity_reliable"] is False
    state = Yolo26TableStateResult(
        ok=True,
        hand_tiles=result["hand_tiles"],
        melds=result["melds"],
        meld_tiles=result["meld_tiles"],
        analysis_hints={"yolo26_meld_identity_reliable": result["meld_identity_reliable"]},
        original_inference_ok=True,
    )
    assert state.to_meld_result().open_meld_count == 1
    assert state.to_meld_result().analysis_hints["tile_identity_reliable"] is False


def test_yolo26_does_not_recover_incomplete_low_confidence_side_pair() -> None:
    detections = [
        detection(f"{(index % 9) + 1}m", 0.95, [100 + index * 38, 720, 134 + index * 38, 790])
        for index in range(10)
    ]
    detections.extend(
        [
            detection("6z", 0.025, [600, 730, 676, 790]),
            detection("6s", 0.026, [674, 716, 740, 790]),
        ]
    )

    result = postprocess_yolo26_detections(detections, image_size=(900, 800), min_confidence=0.25)

    assert len(result["hand_tiles"]) == 10
    assert result["melds"] == []
    assert result["original_recovered_count"] == 0


def test_yolo26_backend_reports_missing_model_dir(tmp_path: Path) -> None:
    backend = load_yolo26_backend(tmp_path / "missing")

    assert backend.available is False
    assert backend.reason == "yolo26_model_dir_missing"


def test_yolo26_backend_accepts_export_after_model_exists(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "labels.json").write_text('["1m"]', encoding="utf-8")
    (model_dir / "metadata.json").write_text('{"runtime":"onnxruntime","model_file":"model.onnx"}', encoding="utf-8")
    (model_dir / "model.onnx").write_bytes(b"placeholder")

    backend = load_yolo26_backend(model_dir)

    assert backend.available is True
    assert backend.reason == ""
    assert backend.runtime == "onnxruntime"


def test_yolo26_backend_rejects_model_path_escape(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "labels.json").write_text('["1m"]', encoding="utf-8")
    (model_dir / "metadata.json").write_text(
        '{"runtime":"onnxruntime","model_file":"../outside.onnx"}',
        encoding="utf-8",
    )
    (tmp_path / "outside.onnx").write_bytes(b"placeholder")

    backend = load_yolo26_backend(model_dir)

    assert backend.available is False
    assert backend.reason == "yolo26_model_path_invalid"


def test_yolo26_backend_rejects_lfs_pointer(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "labels.json").write_text('["1m"]', encoding="utf-8")
    (model_dir / "metadata.json").write_text(
        '{"runtime":"onnxruntime","model_file":"model.onnx"}',
        encoding="utf-8",
    )
    (model_dir / "model.onnx").write_bytes(
        b"version https://git-lfs.github.com/spec/v1\n"
        b"oid sha256:0000000000000000000000000000000000000000000000000000000000000000\n"
        b"size 81915072\n"
    )

    backend = load_yolo26_backend(model_dir)

    assert backend.available is False
    assert backend.reason == "yolo26_model_lfs_pointer"


def test_yolo26_backend_rejects_checksum_mismatch(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "labels.json").write_text('["1m"]', encoding="utf-8")
    (model_dir / "metadata.json").write_text(
        json.dumps(
            {
                "runtime": "onnxruntime",
                "model_file": "model.onnx",
                "sha256": "0" * 64,
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "model.onnx").write_bytes(b"placeholder")

    backend = load_yolo26_backend(model_dir)

    assert backend.available is False
    assert backend.reason == "yolo26_model_checksum_mismatch"


def test_yolo26_backend_rejects_class_count_mismatch(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "labels.json").write_text('["1m"]', encoding="utf-8")
    (model_dir / "metadata.json").write_text(
        '{"runtime":"onnxruntime","model_file":"model.onnx","class_count":34}',
        encoding="utf-8",
    )
    (model_dir / "model.onnx").write_bytes(b"placeholder")

    backend = load_yolo26_backend(model_dir)

    assert backend.available is False
    assert backend.reason == "yolo26_class_count_mismatch"


def test_yolo26_backend_retries_after_model_replacement(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    valid_model = b"valid-model"
    expected_sha = hashlib.sha256(valid_model).hexdigest()
    (model_dir / "labels.json").write_text('["1m"]', encoding="utf-8")
    (model_dir / "metadata.json").write_text(
        json.dumps(
            {
                "runtime": "onnxruntime",
                "model_file": "model.onnx",
                "class_count": 1,
                "sha256": expected_sha,
            }
        ),
        encoding="utf-8",
    )
    model_path = model_dir / "model.onnx"
    model_path.write_bytes(b"broken-file")
    stale_stat = model_path.stat()

    broken = load_yolo26_backend(model_dir)
    model_path.write_bytes(valid_model)
    os.utime(model_path, ns=(stale_stat.st_atime_ns, stale_stat.st_mtime_ns))
    repaired = load_yolo26_backend(model_dir)

    assert broken.available is False
    assert broken.reason == "yolo26_model_checksum_mismatch"
    assert repaired.available is True


def test_yolo26_session_cache_uses_model_content_identity(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"model-a")
    original_stat = model_path.stat()
    sessions: list[object] = []

    class FakeOrt:
        @staticmethod
        def get_available_providers() -> list[str]:
            return ["CPUExecutionProvider"]

        @staticmethod
        def InferenceSession(_path: str, *, providers: list[str]) -> object:
            session = SimpleNamespace(providers=providers)
            sessions.append(session)
            return session

    monkeypatch.setitem(sys.modules, "onnxruntime", FakeOrt)
    visible_tiles._load_onnx_session.cache_clear()
    first_identity = visible_tiles._model_file_identity(model_path)
    first = visible_tiles._load_onnx_session(*first_identity)

    model_path.write_bytes(b"model-b")
    os.utime(model_path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    second_identity = visible_tiles._model_file_identity(model_path)
    second = visible_tiles._load_onnx_session(*second_identity)

    assert first_identity[:3] == second_identity[:3]
    assert first_identity[3] != second_identity[3]
    assert first is not second
    assert len(sessions) == 2
    visible_tiles._load_onnx_session.cache_clear()


def test_yolo26_postprocess_preserves_model_five_classes() -> None:
    result = postprocess_yolo26_detections(
        [
            detection("5m", 0.9, [220, 820, 260, 900]),
            detection("5p", 0.9, [270, 820, 310, 900]),
            detection("5s", 0.9, [320, 820, 360, 900]),
        ],
        image_size=(1920, 1080),
        river_detections=[
            detection("5m", 0.9, [355, 265, 395, 330]),
            detection("5p", 0.9, [405, 265, 445, 330]),
            detection("5s", 0.9, [455, 265, 495, 330]),
        ],
        river_image_size=(800, 800),
    )

    assert result["hand_tiles"] == ["5m", "5p", "5s"]
    assert sorted(result["visible_tiles"]) == ["5m", "5p", "5s"]


def test_yolo26_external_hints_do_not_expose_override_path(tmp_path: Path) -> None:
    frame_path = tmp_path / "frame.png"
    private_model_dir = tmp_path / "private-user-model"
    Image.new("RGB", (80, 80), "white").save(frame_path)

    result = detect_yolo26_table_state_path(
        frame_path,
        model_dir=private_model_dir,
        table_surface_result=TableSurfaceResult(reason="table_surface_unavailable"),
    )
    serialized = json.dumps(result.to_dict(), ensure_ascii=False)

    assert result.analysis_hints["model_source"] == "override"
    assert result.analysis_hints["model_id"] == "yolo26"
    assert "yolo26_model_dir" not in result.analysis_hints
    assert str(private_model_dir) not in serialized


def test_yolo26_decodes_end_to_end_rows() -> None:
    import numpy as np

    detections = _decode_end2end_output(
        np.array([[[10.0, 20.0, 30.0, 40.0, 0.9, 1.0], [1.0, 2.0, 3.0, 4.0, 0.1, 0.0]]]),
        labels=["1m", "2m"],
        image_size=(100, 100),
        scale=1.0,
        pad_x=0.0,
        pad_y=0.0,
        min_confidence=0.25,
    )

    assert len(detections) == 1
    assert detections[0].tile == "2m"
    assert detections[0].bbox == [10.0, 20.0, 30.0, 40.0]


def test_yolo26_bottom_gap_splits_closed_tiles_from_self_meld() -> None:
    detections = [
        detection(f"{(index % 9) + 1}m", 0.95, [100 + index * 38, 720, 134 + index * 38, 790])
        for index in range(11)
    ]
    detections.extend(
        [
            detection("1s", 0.95, [640, 720, 670, 790]),
            detection("2s", 0.95, [672, 720, 702, 790]),
            detection("3s", 0.95, [704, 720, 734, 790]),
        ]
    )

    result = postprocess_yolo26_detections(detections, image_size=(800, 800))

    assert len(result["hand_tiles"]) == 11
    assert result["meld_tiles"] == ["1s", "2s", "3s"]
    assert len(result["melds"]) == 1


def test_yolo26_groups_four_detected_self_meld_tiles_as_one_kan() -> None:
    detections = [
        detection(f"{(index % 9) + 1}m", 0.95, [100 + index * 38, 720, 134 + index * 38, 790])
        for index in range(11)
    ]
    detections.extend(
        detection(f"{index + 1}s", 0.95, [640 + index * 32, 720, 670 + index * 32, 790])
        for index in range(4)
    )

    result = postprocess_yolo26_detections(detections, image_size=(800, 800))

    assert len(result["hand_tiles"]) == 11
    assert result["meld_tiles"] == ["1s", "2s", "3s", "4s"]
    assert len(result["melds"]) == 1
    assert result["melds"][0]["tiles"] == ["1s", "2s", "3s", "4s"]


def test_yolo26_hand_threshold_accounts_for_recognized_self_melds() -> None:
    result = Yolo26TableStateResult(
        ok=True,
        hand_tiles=["1m"] * 11,
        melds=[{"tiles": ["1s", "2s", "3s"]}],
        confidence=0.95,
        reason="recognized_yolo26_visible_tiles",
    )

    hand = result.to_hand_result(min_hand_tiles=12)

    assert hand.ok is True
    assert hand.reason == "recognized_yolo26_hand"


def test_yolo26_low_right_river_tile_stays_with_right_player() -> None:
    result = postprocess_yolo26_detections(
        [],
        image_size=(1920, 1080),
        river_detections=[detection("2m", 0.95, [530, 448, 561, 488])],
        river_image_size=(800, 800),
    )

    assert result["discard_piles"]["right_opponent"][0]["tile"] == "2m"


def test_yolo26_uses_tile_orientation_to_break_diagonal_owner_tie() -> None:
    result = postprocess_yolo26_detections(
        [],
        image_size=(1920, 1080),
        river_detections=[
            detection("1m", 0.95, [385, 190, 415, 250]),
            detection("2m", 0.95, [385, 550, 415, 610]),
            detection("3m", 0.95, [220, 385, 280, 415]),
            detection("4m", 0.95, [520, 385, 580, 415]),
            # This right-player tile is just across the angular diagonal and
            # would otherwise be assigned to the top player. Its horizontal
            # orientation resolves only this narrow boundary ambiguity.
            detection("5p", 0.95, [465, 275, 515, 305]),
        ],
        river_image_size=(800, 800),
    )

    assert {item["tile"] for item in result["discard_piles"]["right_opponent"]} == {"4m", "5p"}
    assert [item["tile"] for item in result["discard_piles"]["top_opponent"]] == ["1m"]
    assert result["riichi_players"] == []


def test_yolo26_detects_right_player_riichi_declaration_by_orientation() -> None:
    detections = [
        detection(f"{index + 1}p", 0.95, [500, 300 + index * 32, 540, 330 + index * 32])
        for index in range(5)
    ]
    detections.append(detection("2m", 0.95, [546, 430, 576, 472]))

    result = postprocess_yolo26_detections(
        [],
        image_size=(1920, 1080),
        river_detections=detections,
        river_image_size=(800, 800),
    )

    assert result["riichi_players"] == ["right_opponent"]


def test_yolo26_estimates_shifted_river_center_instead_of_fixed_polygons() -> None:
    result = postprocess_yolo26_detections(
        [],
        image_size=(1920, 1080),
        river_detections=[
            detection("1m", 0.95, [430, 175, 470, 225]),
            detection("2m", 0.95, [280, 320, 320, 370]),
            detection("3m", 0.95, [590, 320, 630, 370]),
            detection("4m", 0.95, [430, 535, 470, 585]),
        ],
        river_image_size=(800, 800),
    )

    center_x, center_y = result["river_center"]
    assert center_x == 0.5687
    assert center_y == 0.475
    assert {owner: pile[0]["tile"] for owner, pile in result["discard_piles"].items()} == {
        "self": "4m",
        "right_opponent": "3m",
        "top_opponent": "1m",
        "left_opponent": "2m",
    }


def test_table_surface_orders_quad_and_bbox() -> None:
    quad = _order_quad_points([[500, 400], [100, 100], [520, 120], [80, 420]])

    assert quad == [[100.0, 100.0], [520.0, 120.0], [80.0, 420.0], [500.0, 400.0]]
    assert _quad_bbox(quad) == [80.0, 100.0, 520.0, 420.0]


def test_table_surface_detects_synthetic_table() -> None:
    image = Image.new("RGB", (640, 360), "black")
    draw = ImageDraw.Draw(image)
    draw.polygon([(90, 50), (550, 40), (610, 315), (30, 320)], fill=(38, 95, 155))

    result = detect_table_surface(image)

    assert result.ok is True
    assert result.reason == "table_surface_detected"
    assert result.method == "tablecloth_support_lines"
    assert result.quad
    assert result.warped_size == (800, 800)
    assert result.quality_score > 0
    assert result.quad_area_ratio >= 0.5


def test_table_surface_support_lines_ignore_large_inner_quadrilateral() -> None:
    image = Image.new("RGB", (640, 360), "black")
    draw = ImageDraw.Draw(image)
    draw.polygon([(90, 35), (550, 35), (630, 350), (10, 350)], fill=(38, 95, 155))
    # This decoy covers the legacy fixed sample and more than 22% of the image,
    # just like a complete four-player river/score-panel contour can in-game.
    draw.polygon([(55, 80), (305, 70), (320, 320), (45, 320)], fill=(45, 145, 70))

    support_result = detect_table_surface(image, legacy_fallback=False)
    legacy_result = detect_table_surface(image, mode="legacy", legacy_fallback=False)

    assert support_result.ok is True
    assert support_result.method == "tablecloth_support_lines"
    assert support_result.quad_area_ratio >= 0.5
    assert legacy_result.ok is True
    assert legacy_result.method == "automajsoul_opencv_color"
    assert _test_quad_area_ratio(legacy_result.quad, image.size) < 0.4


def test_table_surface_prefers_distributed_edge_support_over_outermost_ui_slash() -> None:
    candidates = [
        # Avatar decoration: it looks farther right only after a very short
        # segment is extrapolated to the middle of the screenshot.
        _SupportLine("", 1757, 265, 1816, 386, 134.6, 64.0),
        _SupportLine("", 1756, 265, 1815, 387, 135.5, 64.2),
        # Real table boundary: several near-collinear segments support it over
        # most of the right side even though its midpoint is slightly inward.
        _SupportLine("", 1654, 109, 1750, 406, 312.1, 72.1),
        _SupportLine("", 1746, 405, 1919, 939, 561.3, 72.1),
        _SupportLine("", 1793, 558, 1917, 942, 403.5, 72.1),
    ]

    selected = _select_supported_line(candidates, side="right", width=1920, height=1200)

    assert selected is not None
    line, diagnostics = selected
    assert 71.0 <= line.angle <= 73.0
    assert diagnostics["coverage_ratio"] >= 0.65
    assert diagnostics["longest_span_ratio"] >= 0.44


def test_table_surface_uses_weighted_median_to_keep_parallel_top_edge_level() -> None:
    candidates = [
        _SupportLine("", 290, 13, 1630, 13, 1340.0, 0.0),
        _SupportLine("", 300, 15, 1620, 15, 1320.0, 0.0),
        _SupportLine("", 350, 31, 1000, 24, 650.0, -0.6),
    ]

    selected = _select_supported_line(candidates, side="top", width=1920, height=1200)

    assert selected is not None
    line, diagnostics = selected
    assert line.angle == 0.0
    assert line.y1 == line.y2
    assert diagnostics["coverage_ratio"] >= 0.69


def _test_quad_area_ratio(quad: list[list[float]], image_size: tuple[int, int]) -> float:
    ordered = _order_quad_points(quad)
    polygon = [ordered[0], ordered[1], ordered[3], ordered[2]]
    doubled_area = 0.0
    for index, (x1, y1) in enumerate(polygon):
        x2, y2 = polygon[(index + 1) % len(polygon)]
        doubled_area += x1 * y2 - x2 * y1
    width, height = image_size
    return abs(doubled_area) / 2 / (width * height)


def test_yolo26_missing_backend_still_reports_table_surface_hints(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.jpg"
    Image.new("RGB", (320, 180), "black").save(image_path)

    result = detect_yolo26_table_state_path(image_path, model_dir=tmp_path / "missing-model")

    assert result.ok is False
    assert result.reason == "yolo26_model_dir_missing"
    assert "table_surface_ok" in result.analysis_hints


def test_yolo26_runtime_runs_original_hand_and_warped_river_as_separate_passes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    image_path = tmp_path / "frame.jpg"
    Image.new("RGB", (1920, 1080), "black").save(image_path)
    calls: list[tuple[tuple[int, int], float]] = []

    class FakeBackend:
        available = True
        reason = ""
        runtime = "test"

        def detect(self, image: Image.Image, *, min_confidence: float) -> list[YoloTileDetection]:
            calls.append((image.size, min_confidence))
            if image.size == (1920, 1080):
                return [
                    detection(f"{(index % 9) + 1}m", 0.95, [100 + index * 70, 850, 150 + index * 70, 1010])
                    for index in range(14)
                ]
            return [
                detection("1m", 0.95, [430, 175, 470, 225]),
                detection("2m", 0.95, [280, 320, 320, 370]),
                detection("3m", 0.95, [590, 320, 630, 370]),
                detection("4m", 0.95, [430, 535, 470, 585]),
            ]

    surface = SimpleNamespace(
        ok=True,
        reason="table_surface_detected",
        warped_image=Image.new("RGB", (800, 800), "black"),
        diagnostics={},
        to_hints=lambda: {"table_surface_ok": True},
    )
    monkeypatch.setattr(visible_tiles, "load_yolo26_backend", lambda _model_dir: FakeBackend())
    monkeypatch.setattr(
        visible_tiles,
        "detect_table_surface",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("scene-gate table surface should be reused")
        ),
    )

    result = detect_yolo26_table_state_path(
        image_path,
        model_dir=tmp_path / "model",
        table_surface_result=surface,
    )

    assert calls == [((1920, 1080), 0.01), ((800, 800), 0.25)]
    assert result.original_inference_ok is True
    assert result.river_inference_ok is True
    assert len(result.hand_tiles) == 14
    assert set(result.discard_piles) == {"self", "right_opponent", "top_opponent", "left_opponent"}
    assert {item["coordinate_space"] for item in result.raw_detections} == {"original_frame", "warped_table"}


def test_yolo26_table_warp_failure_keeps_original_hand_and_marks_river_unavailable(
    tmp_path: Path,
    monkeypatch,
) -> None:
    image_path = tmp_path / "frame.jpg"
    Image.new("RGB", (1280, 720), "black").save(image_path)

    class FakeBackend:
        available = True
        reason = ""
        runtime = "test"

        def detect(self, _image: Image.Image, *, min_confidence: float) -> list[YoloTileDetection]:
            return [
                detection(f"{(index % 9) + 1}m", 0.95, [80 + index * 60, 570, 125 + index * 60, 690])
                for index in range(14)
            ]

    surface = SimpleNamespace(
        ok=False,
        reason="table_surface_not_found",
        warped_image=None,
        diagnostics={},
        to_hints=lambda: {"table_surface_ok": False},
    )
    monkeypatch.setattr(visible_tiles, "load_yolo26_backend", lambda _model_dir: FakeBackend())
    monkeypatch.setattr(visible_tiles, "detect_table_surface", lambda *_args, **_kwargs: surface)

    result = detect_yolo26_table_state_path(image_path, model_dir=tmp_path / "model")

    assert result.ok is True
    assert result.original_inference_ok is True
    assert result.river_inference_ok is False
    assert result.to_hand_result(min_hand_tiles=12).ok is True
    assert result.to_river_result().ok is False
    assert result.to_river_result().reason == "table_surface_not_found"
