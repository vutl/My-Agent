"""Network-boundary helpers for local-only diagnostic capabilities."""

from __future__ import annotations

from ipaddress import ip_address

from starlette.requests import Request


def request_client_is_loopback(request: Request) -> bool:
    """Return true only when the ASGI peer itself is a loopback client.

    Do not trust the configured bind host or forwarded headers here: either can
    disagree with the socket peer, and forwarded headers are caller-controlled
    unless a separate trusted-proxy policy is configured.
    """

    client = request.client
    if client is None:
        return False
    host = str(client.host or "").strip().strip("[]")
    if host.casefold() == "localhost":
        return True
    try:
        address = ip_address(host)
    except ValueError:
        return False
    if address.is_loopback:
        return True
    mapped = getattr(address, "ipv4_mapped", None)
    return bool(mapped and mapped.is_loopback)
