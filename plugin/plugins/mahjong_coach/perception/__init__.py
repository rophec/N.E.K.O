from __future__ import annotations

import gc

from .fast_hand_path import FastHandResult, detect_fast_hand_path


def release_perception_runtime_caches() -> None:
    """Release optional model and template caches while live mode is idle."""
    from .tile_templates import release_template_matrix_cache
    from .table_context import release_table_context_runtime_cache
    from .vit_tile_classifier_onnx import release_onnx_tile_classifier_cache
    from .yolo26_visible_tiles import release_yolo26_runtime_cache

    release_yolo26_runtime_cache()
    release_onnx_tile_classifier_cache()
    release_template_matrix_cache()
    release_table_context_runtime_cache()
    gc.collect()


def perception_runtime_stats() -> dict[str, object]:
    from .table_context import table_context_runtime_stats
    from .tile_templates import template_matrix_runtime_stats
    from .vit_tile_classifier_onnx import onnx_tile_classifier_runtime_stats
    from .yolo26_visible_tiles import yolo26_runtime_stats

    return {
        **yolo26_runtime_stats(),
        **onnx_tile_classifier_runtime_stats(),
        **template_matrix_runtime_stats(),
        **table_context_runtime_stats(),
    }


__all__ = [
    "FastHandResult",
    "detect_fast_hand_path",
    "perception_runtime_stats",
    "release_perception_runtime_caches",
]
