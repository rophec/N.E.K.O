from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def create_inference_session(
    ort: Any,
    model_path: str,
    providers: Sequence[str],
    *,
    low_memory: bool = False,
) -> Any:
    """Create one session with the plugin's shared bounded-memory policy."""
    provider_list = list(providers)
    options = None
    if (low_memory or "DmlExecutionProvider" in provider_list) and hasattr(ort, "SessionOptions"):
        options = ort.SessionOptions()
        options.enable_mem_pattern = False
        options.enable_cpu_mem_arena = False
        if hasattr(ort, "ExecutionMode"):
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        if low_memory:
            options.intra_op_num_threads = 2
            options.inter_op_num_threads = 1
    if options is None:
        return ort.InferenceSession(model_path, providers=provider_list)
    return ort.InferenceSession(model_path, sess_options=options, providers=provider_list)
