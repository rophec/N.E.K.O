from __future__ import annotations

import zmq

from plugin.plugins.mahjong_coach.companion_transport import (
    resolve_agent_push_endpoint,
    submit_companion_direct,
)


def test_resolve_agent_push_endpoint_uses_host_port_env(monkeypatch) -> None:
    monkeypatch.setenv("NEKO_ZMQ_AGENT_PUSH_PORT", "49001")

    assert resolve_agent_push_endpoint() == "tcp://127.0.0.1:49001"


def test_submit_companion_direct_matches_proactive_bridge_event_contract() -> None:
    context = zmq.Context.instance()
    receiver = context.socket(zmq.PULL)
    receiver.setsockopt(zmq.LINGER, 0)
    port = receiver.bind_to_random_port("tcp://127.0.0.1")
    endpoint = f"tcp://127.0.0.1:{port}"
    try:
        result = submit_companion_direct(
            "只读牌局观察",
            plugin_id="mahjong_coach_beta_20260813_0_3_27b1",
            delivery_id="delivery-1",
            event_kind="riichi_pressure",
            priority=4,
            metadata={"reply_max_sentences": 2},
            endpoint=endpoint,
            timeout=1.0,
        )

        assert result.submitted is True
        assert receiver.poll(timeout=1000, flags=zmq.POLLIN)
        event = receiver.recv_json()
    finally:
        receiver.close(linger=0)

    assert event["event_type"] == "proactive_message"
    assert event["delivery_mode"] == "proactive"
    assert event["ai_behavior"] == "respond"
    assert event["direct_reply"] is False
    assert event["visibility"] == []
    assert event["text"] == "只读牌局观察"
    assert event["metadata"]["delivery_id"] == "delivery-1"
    assert event["metadata"]["transport_fallback"] == "direct_agent_event"
    assert event["channel"] == "plugin:mahjong_coach_beta_20260813_0_3_27b1"
    assert event["source_name"] == "mahjong_coach_beta_20260813_0_3_27b1"
    assert event["coalesce_key"] == (
        "mahjong_coach_beta_20260813_0_3_27b1:mahjong_companion"
    )


def test_submit_companion_direct_can_bypass_llm_for_fixed_banter() -> None:
    context = zmq.Context.instance()
    receiver = context.socket(zmq.PULL)
    receiver.setsockopt(zmq.LINGER, 0)
    port = receiver.bind_to_random_port("tcp://127.0.0.1")
    endpoint = f"tcp://127.0.0.1:{port}"
    try:
        result = submit_companion_direct(
            "先别急，我先急一下。",
            delivery_id="delivery-fixed",
            event_kind="casual_chat",
            priority=2,
            direct_reply=True,
            endpoint=endpoint,
            timeout=1.0,
        )
        assert result.submitted is True
        assert receiver.poll(timeout=1000, flags=zmq.POLLIN)
        event = receiver.recv_json()
    finally:
        receiver.close(linger=0)

    assert event["direct_reply"] is True
    assert event["text"] == "先别急，我先急一下。"


def test_submit_companion_direct_rejects_missing_agent_socket() -> None:
    result = submit_companion_direct(
        "test",
        delivery_id="delivery-2",
        event_kind="opening_plan",
        priority=4,
        endpoint="tcp://127.0.0.1:9",
        timeout=0.05,
    )

    assert result.submitted is False
    assert result.reason == "agent_socket_unavailable"
