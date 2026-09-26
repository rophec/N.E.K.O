"""Shared preset voice selection for HTTP and WebSocket routes."""


def resolve_requested_voice(
    requested_voice: str | None,
    default_voice: str = "",
    engine_default_voice: str = "",
) -> str:
    """Resolve API ``default``/blank sentinels to the loaded model's default."""
    requested = str(requested_voice or "").strip()
    if not requested or requested.casefold() == "default":
        return str(default_voice or engine_default_voice or "").strip()
    return requested
