from pathlib import Path
import tomllib


MAIN_JS = Path(__file__).resolve().parents[1] / "static" / "main.js"
PLUGIN_TOML = Path(__file__).resolve().parents[1] / "plugin.toml"


def test_manifest_uses_current_plugin_package_instead_of_legacy_entry_shim() -> None:
    manifest = tomllib.loads(PLUGIN_TOML.read_text(encoding="utf-8"))

    assert manifest["plugin"]["entry"] == "plugin.plugins.mahjong_coach:MahjongCoachPlugin"


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


def test_static_ui_defaults_to_yolo26_recognition() -> None:
    source = MAIN_JS.read_text(encoding="utf-8")
    html = (MAIN_JS.parent / "index.html").read_text(encoding="utf-8")

    assert 'id="activeRecognitionMode" class="mode-badge is-yolo">YOLO26<' in html
    assert 'id="yoloModeBtn" class="mode-option is-active"' in html
    assert 'id="tileRecognitionModeInput" type="hidden" value="yolo26"' in html
    assert "const normalized = mode === 'legacy' ? 'legacy' : 'yolo26';" in source


def test_static_ui_pins_all_previews_to_one_requested_frame() -> None:
    source = MAIN_JS.read_text(encoding="utf-8")

    assert "const previewArgs = { image_path: requestedPath };" in source
    assert "callPlugin('mahjong_coach_frame_preview', previewArgs" in source
    assert "callPlugin('mahjong_coach_table_region_preview', previewArgs" in source
    assert "callPlugin('mahjong_coach_settlement_preview', previewArgs" in source
    assert "refreshFramePreview(live.last_frame_path, live.last_frame_revision)" in source
    assert "const requestedKey = `${requestedPath}#${requestedRevision}`;" in source
    assert "queuedPreviewRequest = { path: requestedPath, revision: requestedRevision, key: requestedKey };" in source
    assert "{ dashboard_visible: true }" in source
    assert "const settlementPromise = Promise.allSettled" in source
    assert "const settlementResult = await settlementPromise;" in source
    assert "}, 700);" in source


def test_static_ui_prioritizes_warped_table_evidence() -> None:
    html = (MAIN_JS.parent / "index.html").read_text(encoding="utf-8")
    styles = (MAIN_JS.parent / "style.css").read_text(encoding="utf-8")

    assert "变换后牌桌证据" in html
    assert "牌河 / 对手副露主视图" in html
    assert "原始帧识别证据" in html
    assert "手牌 / 自家副露 / 按钮" in html
    assert 'class="evidence-preview table-region-evidence-preview"' in html
    assert 'class="evidence-preview settlement-evidence-preview"' in html
    assert ".table-region-evidence-preview" in styles
    assert "grid-row: 1 / span 2;" in styles
    assert ".settlement-evidence-preview" in styles


def test_static_ui_collapses_infrequent_runtime_controls() -> None:
    html = (MAIN_JS.parent / "index.html").read_text(encoding="utf-8")
    core_controls = html.split('<section class="control-band"', 1)[1].split("</section>", 1)[0]
    runtime_controls = html.split('<details class="advanced-control-band runtime-settings-band">', 1)[1].split(
        "</details>", 1
    )[0]

    assert 'id="keywordsInput"' in core_controls
    assert 'id="guidanceModeInput"' in core_controls
    assert 'id="overlayInput"' in core_controls
    assert 'id="nekoCompanionEnabledInput"' in core_controls
    assert 'id="intervalInput"' not in core_controls
    assert 'id="intervalInput"' in runtime_controls
    assert 'id="strategyPresetInput"' in runtime_controls
    assert 'id="inferenceProviderInput"' in runtime_controls
    assert 'id="autoStartLiveInput"' in runtime_controls


def test_static_ui_prewarms_gpu_before_yolo_live_analysis() -> None:
    source = MAIN_JS.read_text(encoding="utf-8")

    assert "function requestYoloWarmup()" in source
    assert "callPlugin('mahjong_coach_warmup_yolo26'" in source
    assert "requestYoloWarmup().catch(() => {});" in source
    assert "GPU 预热中" in source
    assert "syncInferenceRuntime(data.inference_runtime || {})" in source
    assert "实际 ${backend}" in source
    assert "降级 ${compact(runtime.fallback_reason)}" in source
    assert "if (inferenceProviderInput) inferenceProviderInput.value = 'speed';" in source
    assert "await requestYoloWarmup().catch(() => {});" in source
    assert "state.status === 'fallback' || state.accelerated === false" in source


def test_running_dashboard_rehydrates_runtime_choices_from_backend() -> None:
    source = MAIN_JS.read_text(encoding="utf-8")

    assert "const syncRuntimeChoices = !preferencesHydrated || force || Boolean(data.live?.running);" in source
    assert "if (syncRuntimeChoices)" in source
    assert "setSelectValue(strategyPresetInput, data.config?.strategy_preset" in source
    assert "setGuidanceMode(data.config?.live_advice_mode" in source
    assert "nekoCompanionEnabledInput.checked = Boolean(data.config.neko_companion_enabled);" in source
    assert "const absurdBanterEnabledInput = document.getElementById('absurdBanterEnabledInput');" in source
    assert "syncAbsurdBanterAvailability" in source
    assert "absurd_banter_enabled: Boolean(" in source
