"""Keep other websites away from the local backend.

Any page open in the browser can try to talk to http://127.0.0.1:8765: open the /ws socket
(WebSockets are not covered by CORS) or send requests, including after DNS rebinding (a site's
own domain made to resolve to 127.0.0.1). Two checks, for HTTP and WebSocket alike:

- Host must be localhost, an IP address or a name in LIVETRANS_ALLOWED_HOSTS. A rebinding
  attack always arrives with the attacker's domain name as Host.
- Origin, when the browser sends one, must be this server itself (same scheme-less host:port
  as Host). Scripts and command-line tools send no Origin and are not affected.
"""

from __future__ import annotations

import ipaddress
import logging
from urllib.parse import urlsplit

from starlette.responses import PlainTextResponse

log = logging.getLogger("livetrans.guard")


def _hostname(host: str) -> str:
    """'127.0.0.1:8765' -> '127.0.0.1', '[::1]:8765' -> '::1', 'Localhost' -> 'localhost'."""
    host = host.strip().lower()
    if host.startswith("["):
        return host[1 : host.find("]")] if "]" in host else host[1:]
    return host.rsplit(":", 1)[0] if host.count(":") == 1 else host


def host_allowed(host: str, extra: frozenset[str] = frozenset()) -> bool:
    name = _hostname(host)
    if not name:
        return False
    if name == "localhost" or name.endswith(".localhost") or name in extra:
        return True
    try:
        ipaddress.ip_address(name)
    except ValueError:
        return False
    return True


def origin_allowed(origin: str | None, host: str) -> bool:
    if origin is None:
        return True
    parts = urlsplit(origin.strip())
    return parts.scheme in ("http", "https") and parts.netloc.lower() == host.strip().lower()


def rejection(host: str, origin: str | None, extra: frozenset[str] = frozenset()) -> str | None:
    """Why a request must be refused, or None if it may proceed."""
    if not host_allowed(host, extra):
        return f"host {host!r} not allowed (set LIVETRANS_ALLOWED_HOSTS to permit it)"
    if not origin_allowed(origin, host):
        return f"cross-site request from origin {origin!r} refused"
    return None


class LocalOnlyMiddleware:
    """ASGI middleware applying rejection() to every HTTP request and WebSocket handshake."""

    def __init__(self, app, allowed_hosts: frozenset[str] = frozenset()):
        self.app = app
        self.extra = frozenset(h.strip().lower() for h in allowed_hosts if h.strip())

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            return await self.app(scope, receive, send)
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}
        reason = rejection(headers.get("host", ""), headers.get("origin"), self.extra)
        if reason is None:
            return await self.app(scope, receive, send)
        log.warning("Refused %s %s: %s", scope["type"], scope.get("path", ""), reason)
        if scope["type"] == "http":
            return await PlainTextResponse(reason, status_code=403)(scope, receive, send)
        # Closing before accepting makes the server answer the handshake with HTTP 403.
        await receive()  # websocket.connect
        await send({"type": "websocket.close", "code": 1008})
