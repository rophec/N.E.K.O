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


def test_fixed_speaker_mode_can_be_selected_by_url_or_session_config():
    source = _source()

    assert 'config_msg.get("clone_mode")' in source
    assert 'websocket.query_params.get("clone_mode")' in source
    assert '"fixed_speaker", "fixed_speaker_codes", "icl_codes"' in source
    assert 'x_vector_only_mode = clone_mode_raw in {"x_vector", "x_vector_only"}' in source


def test_fixed_speaker_mode_keeps_codes_and_uses_complete_clone_synthesis():
    source = _source()
    tree = ast.parse(source)

    assert "fixed_speaker_embedding = np.array(" in source
    assert 'session_mode = "fixed_speaker_codes"' in source
    assert 'mode = "clone"' in source
    assert "reference_code_frames={reference_code_frames}" in source
    assert "prompt_icl=True decoder_continuation={decoder_continuation_ready}" in source
    assert "参考 codec codes 未成功编译 decoder continuation state" in source

    clone_continuation_assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "clone_continuation"
            for target in node.targets
        )
    ]
    assert len(clone_continuation_assignments) == 1
    assert ast.unparse(clone_continuation_assignments[0].value) == "mode == 'clone'"


def test_x_vector_only_mode_retains_the_failed_embedding_only_ab_path():
    source = _source()

    assert "wrapper.shutdown()" in source
    assert "stream = pool.engine.create_stream()" in source
    assert 'session_mode = "x_vector_only"' in source
    assert 'mode = "custom"' in source
    assert "reference_codes=False decoder_continuation=False" in source
    assert 'f"external_speaker_embedding(dim={speaker.size})"' in source
