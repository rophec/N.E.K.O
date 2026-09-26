import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WS_STREAM = (
    ROOT
    / "local_server"
    / "qwen3_tts_server"
    / "server"
    / "routes"
    / "ws_stream.py"
)


def _source() -> str:
    return WS_STREAM.read_text(encoding="utf-8")


def test_clone_buffers_the_whole_utterance_before_continuation_synthesis():
    source = _source()
    tree = ast.parse(source)

    input_text_branch = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and any(
            isinstance(value, ast.Constant) and value.value == "input.text"
            for value in node.test.comparators
        )
    )
    continuation_branch = next(
        node
        for node in input_text_branch.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "clone_continuation"
    )

    clone_calls = {
        node.func.id
        for statement in continuation_branch.body
        for node in ast.walk(statement)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    preset_calls = {
        node.func.id
        for statement in continuation_branch.orelse
        for node in ast.walk(statement)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert "_extract_complete_sentences" not in clone_calls
    assert "_extract_complete_sentences" in preset_calls
    assert "[Qwen3-TTS WS][clone.buffer]" in source


def test_first_clone_synthesis_reuses_the_anchor_prepared_by_session_config():
    source = _source()

    synthesize = source.split("async def _synthesize_and_send", 1)[1].split(
        "# ── 主消息循环", 1
    )[0]
    assert "if sentence_index > 0 and wrapper:" in synthesize
    assert "stream.set_voice" not in synthesize
    assert "[Qwen3-TTS WS][clone.continuation]" in synthesize


def test_clone_synthesis_uses_stable_two_stage_seeds_without_temperature_override():
    source = _source()
    tree = ast.parse(source)

    tts_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "TTSConfig"
    ]
    assert len(tts_calls) == 1

    keyword_names = {keyword.arg for keyword in tts_calls[0].keywords}
    assert {"seed", "sub_seed"} <= keyword_names
    assert "temperature" not in keyword_names
    assert "sub_temperature" not in keyword_names
    assert 'getattr(config, "seed", 42)' in source
    assert 'getattr(config, "sub_seed", 45)' in source
    assert "seed={seed} sub_seed={sub_seed}" in source
