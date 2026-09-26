"""Compatibility transport for Mahjong companion dialogue.

The normal route is the public ``push_message`` SDK.  Older N.E.K.O builds can
leave the embedded message-plane process stopped after a plugin-server reload,
while the main agent event socket remains healthy.  In that specific failure
mode the SDK cannot reach the proactive bridge, so the Mahjong plugin uses the
same ``proactive_message`` event shape as that bridge and submits it directly
to the already-running main agent socket.

This module is deliberately small and lazy-imports pyzmq.  It is a fault-only
compatibility path, not a replacement for the SDK message plane.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
import socket
from typing import Any


_DEFAULT_AGENT_PUSH_PORT = 48962


@dataclass(frozen=True, slots=True)
class DirectCompanionSubmitResult:
    submitted: bool
    endpoint: str
    reason: str = ""


def resolve_agent_push_endpoint() -> str:
    """Resolve the same loopback endpoint used by the host proactive bridge."""

    raw_port = os.getenv("NEKO_ZMQ_AGENT_PUSH_PORT", "").strip()
    try:
        port = int(raw_port) if raw_port else _DEFAULT_AGENT_PUSH_PORT
    except (TypeError, ValueError):
        port = _DEFAULT_AGENT_PUSH_PORT
    if not 1 <= port <= 65535:
        port = _DEFAULT_AGENT_PUSH_PORT
    return f"tcp://127.0.0.1:{port}"


def _parse_loopback_endpoint(endpoint: str) -> tuple[str, int] | None:
    prefix = "tcp://"
    if not str(endpoint).startswith(prefix):
        return None
    address = str(endpoint)[len(prefix) :]
    if ":" not in address:
        return None
    host, raw_port = address.rsplit(":", 1)
    try:
        port = int(raw_port)
    except (TypeError, ValueError):
        return None
    if host not in {"127.0.0.1", "localhost"} or not 1 <= port <= 65535:
        return None
    return host, port


def _agent_socket_is_listening(endpoint: str, *, timeout: float) -> bool:
    parsed = _parse_loopback_endpoint(endpoint)
    if parsed is None:
        return False
    try:
        with socket.create_connection(parsed, timeout=max(0.01, float(timeout))):
            return True
    except OSError:
        return False


def submit_companion_direct(
    cue: str,
    *,
    plugin_id: str = "mahjong_coach",
    delivery_id: str,
    event_kind: str,
    priority: int,
    metadata: dict[str, Any] | None = None,
    direct_reply: bool = False,
    endpoint: str | None = None,
    timeout: float = 0.35,
) -> DirectCompanionSubmitResult:
    """Submit one cue to the host agent socket when message-plane is down.

    ``submitted=True`` means the host's agent PULL endpoint was listening and
    accepted the local ZeroMQ send.  It is not a guarantee that the model has
    generated or spoken a reply.
    """

    target = str(endpoint or resolve_agent_push_endpoint())
    text = str(cue or "").strip()
    if not text:
        return DirectCompanionSubmitResult(False, target, "empty_cue")
    if not _agent_socket_is_listening(target, timeout=min(float(timeout), 0.15)):
        return DirectCompanionSubmitResult(False, target, "agent_socket_unavailable")

    try:
        import zmq
    except Exception:
        return DirectCompanionSubmitResult(False, target, "pyzmq_unavailable")

    resolved_plugin_id = str(plugin_id or "mahjong_coach").strip() or "mahjong_coach"
    event_metadata = dict(metadata or {})
    event_metadata.setdefault("delivery_id", str(delivery_id))
    event_metadata.setdefault("context_type", "mahjong_companion")
    event_metadata.setdefault("event_kind", str(event_kind))
    event_metadata.setdefault("non_prescriptive", True)
    event_metadata["transport_fallback"] = "direct_agent_event"

    event = {
        "event_type": "proactive_message",
        "lanlan_name": None,
        "text": text,
        "summary": text,
        "detail": text,
        "channel": f"plugin:{resolved_plugin_id}",
        "task_id": "",
        "success": True,
        "status": "completed",
        "direct_reply": bool(direct_reply),
        "delivery_mode": "proactive",
        "source_kind": "plugin",
        "source_name": resolved_plugin_id,
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "metadata": event_metadata,
        "media_parts": [],
        "visibility": [],
        "ai_behavior": "respond",
        "priority": int(priority),
        "coalesce_key": f"{resolved_plugin_id}:mahjong_companion",
    }

    context = zmq.Context.instance()
    push_socket = context.socket(zmq.PUSH)
    linger_ms = max(1, int(float(timeout) * 1000))
    try:
        push_socket.setsockopt(zmq.LINGER, linger_ms)
        push_socket.setsockopt(zmq.IMMEDIATE, 1)
        push_socket.setsockopt(zmq.SNDTIMEO, linger_ms)
        push_socket.connect(target)
        push_socket.send_json(event)
    except Exception as exc:
        return DirectCompanionSubmitResult(False, target, type(exc).__name__)
    finally:
        push_socket.close(linger=linger_ms)
    return DirectCompanionSubmitResult(True, target)


__all__ = [
    "DirectCompanionSubmitResult",
    "resolve_agent_push_endpoint",
    "submit_companion_direct",
]
