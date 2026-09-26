import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
INFERENCE = ROOT / "local_server" / "qwen3_tts_server" / "inference"


def _function(path: Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    )


def test_reference_prefix_is_prepared_without_finalizing_decoder():
    handler = _function(INFERENCE / "workers" / "decoder.py", "handle_prepare_state")
    decode_call = next(
        node
        for node in ast.walk(handler)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "decode"
    )
    is_final = next(keyword.value for keyword in decode_call.keywords if keyword.arg == "is_final")

    assert isinstance(is_final, ast.Constant)
    assert is_final.value is False


def test_reference_state_protocol_is_wired_from_stream_to_worker():
    proxy_source = (INFERENCE / "proxy.py").read_text(encoding="utf-8")
    stream_source = (INFERENCE / "stream.py").read_text(encoding="utf-8")
    worker_source = (INFERENCE / "workers" / "decoder.py").read_text(encoding="utf-8")

    assert 'msg_type="PREPARE_STATE"' in proxy_source
    assert '_ref_init' not in proxy_source
    assert 'is_final=True' not in proxy_source.split(
        'if isinstance(input, TTSResult):', 1
    )[1].split('t_start = time.time()', 1)[0]
    assert 'req.msg_type == "PREPARE_STATE"' in worker_source
    assert 'prepare_continuation_state' in stream_source
    assert 'is_final=False' in stream_source
