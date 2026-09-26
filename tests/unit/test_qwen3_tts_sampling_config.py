import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SERVER_ROOT = ROOT / "local_server" / "qwen3_tts_server"


def _parse(relative_path: str) -> ast.AST:
    source = (SERVER_ROOT / relative_path).read_text(encoding="utf-8")
    return ast.parse(source)


def test_server_defaults_match_upstream_deterministic_clone_seeds():
    tree = _parse("server/config.py")
    defaults = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name):
            continue
        if node.target.id in {"seed", "sub_seed"}:
            defaults[node.target.id] = ast.literal_eval(node.value)

    assert defaults == {"seed": 42, "sub_seed": 45}


def test_http_config_passes_both_sampling_seeds_without_temperature_override():
    tree = _parse("server/routes/http_speech.py")
    build_config = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_build_config"
    )
    tts_call = next(
        node
        for node in ast.walk(build_config)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "TTSConfig"
    )
    keyword_names = {keyword.arg for keyword in tts_call.keywords}

    assert {"seed", "sub_seed"} <= keyword_names
    assert "temperature" not in keyword_names
    assert "sub_temperature" not in keyword_names
