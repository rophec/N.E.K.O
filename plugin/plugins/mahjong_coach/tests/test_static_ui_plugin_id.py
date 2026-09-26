from pathlib import Path


MAIN_JS = Path(__file__).resolve().parents[1] / "static" / "main.js"


def test_static_ui_resolves_runtime_plugin_id_from_host_path() -> None:
    source = MAIN_JS.read_text(encoding="utf-8")

    assert "const PLUGIN_ID = resolvePluginId();" in source
    assert "const PLUGIN_ID = 'mahjong_coach';" not in source
    assert r"/^\/plugin\/([^/]+)\/ui(?:\/|$)/" in source
    assert "decodeURIComponent(pathMatch[1])" in source


def test_static_ui_keeps_canonical_id_only_as_fallback() -> None:
    source = MAIN_JS.read_text(encoding="utf-8")

    assert "const DEFAULT_PLUGIN_ID = 'mahjong_coach';" in source
    assert "new URLSearchParams" in source
    assert ".get('plugin_id')" in source


def test_static_ui_pins_all_previews_to_one_requested_frame() -> None:
    source = MAIN_JS.read_text(encoding="utf-8")

    assert "const previewArgs = { image_path: requestedPath };" in source
    assert "callPlugin('mahjong_coach_frame_preview', previewArgs" in source
    assert "callPlugin('mahjong_coach_table_region_preview', previewArgs" in source
    assert "callPlugin('mahjong_coach_settlement_preview', previewArgs" in source
    assert "queuedPreviewPath = requestedPath;" in source
