from __future__ import annotations

from urllib.parse import urlparse

import pytest

from plugin.core.zmq_transport import HostTransport


pytestmark = pytest.mark.plugin_unit


def _port(endpoint: str) -> int:
    return int(urlparse(endpoint).port or 0)


def test_host_transport_never_reuses_fixed_message_plane_ports() -> None:
    transport = HostTransport()
    try:
        ports = {_port(transport.downlink_endpoint), _port(transport.uplink_endpoint)}
        assert ports.isdisjoint({38865, 38866, 38867})
        assert all(49152 <= port <= 65535 for port in ports)
    finally:
        transport.close()
