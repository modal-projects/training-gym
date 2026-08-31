"""A tiny, dependency-free HTTP Basic Auth reverse proxy.

Modal exposes container ports with ``modal.forward(port)``, which creates a
*public* ``*.w.modal.host`` tunnel with no authentication (proxy-auth only
applies to decorated web endpoints, not raw tunnels). Dashboards we forward this
way -- the Ray dashboard and the self-hosted Trackio server -- would therefore be
world-readable.

To gate them we run this proxy inside the head container: it listens on an
ephemeral local port, requires HTTP Basic Auth, and forwards authorized traffic
to the real dashboard on ``127.0.0.1``. The launcher then forwards the *proxy's*
port instead of the dashboard's, so the tunnel URL prompts for credentials.

The proxy is deliberately stdlib-only (imports cleanly on the driver, the head
container, and locally) and supports the two traffic shapes these dashboards
need beyond plain requests:

- streamed responses (Server-Sent Events -- Gradio's event queue), by relaying
  the body until the upstream closes rather than buffering it;
- ``Upgrade: websocket`` (Ray's live views), by splicing the two TCP sockets
  bidirectionally after replaying the client's handshake upstream.
"""

from __future__ import annotations

import base64
import binascii
import hmac
import http.client
import socket
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Headers that are connection-specific and must not be copied verbatim across a
# proxy hop (RFC 7230 §6.1). ``Upgrade``/``Connection`` are handled explicitly
# for the websocket path.
_HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
    }
)

_CHUNK = 1 << 16


def _pipe(src: socket.socket, dst: socket.socket) -> None:
    """Relay bytes ``src`` -> ``dst`` until either end closes, then tear both down."""
    try:
        while True:
            data = src.recv(_CHUNK)
            if not data:
                break
            dst.sendall(data)
    except OSError:
        pass
    finally:
        for sock in (src, dst):
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


class _AuthProxyHandler(BaseHTTPRequestHandler):
    # Set on the concrete subclass built per-server in ``start_auth_proxy``.
    target_host: str = "127.0.0.1"
    target_port: int = 0
    expected_auth: str = ""  # base64("user:pass")
    expected_password: str = ""
    expected_write_token: str = ""
    realm: str = "Restricted"

    protocol_version = "HTTP/1.1"

    def log_message(self, *_args: object) -> None:  # noqa: D401 — silence default logging
        """Suppress the default stderr access log (noisy under a training run)."""

    def _authorized(self) -> bool:
        write_token = self.headers.get("X-Trackio-Write-Token", "")
        if self.expected_write_token and hmac.compare_digest(
            write_token, self.expected_write_token
        ):
            return True
        if not self.expected_auth:
            return True
        header = self.headers.get("Authorization", "")
        if not header.startswith("Basic "):
            return False
        if self.expected_password:
            try:
                decoded = base64.b64decode(
                    header[len("Basic ") :].strip(), validate=True
                ).decode("utf-8")
            except (binascii.Error, ValueError, UnicodeDecodeError):
                return False
            _username, separator, supplied = decoded.partition(":")
            return bool(separator) and hmac.compare_digest(
                supplied, self.expected_password
            )
        # Constant-time compare so the tunnel can't be probed via timing.
        return hmac.compare_digest(header[len("Basic ") :].strip(), self.expected_auth)

    def _send_unauthorized(self) -> None:
        self.send_response(401)
        self.send_header("WWW-Authenticate", f'Basic realm="{self.realm}"')
        self.send_header("Content-Length", "0")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

    def _proxy(self) -> None:
        if not self._authorized():
            self._send_unauthorized()
            return
        if self.headers.get("Upgrade", "").lower() == "websocket":
            self._proxy_websocket()
        else:
            self._proxy_http()

    def _proxy_http(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else None

        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in _HOP_BY_HOP
        }
        try:
            upstream = http.client.HTTPConnection(
                self.target_host, self.target_port, timeout=300
            )
            upstream.request(self.command, self.path, body=body, headers=headers)
            resp = upstream.getresponse()
        except OSError as exc:
            self.send_error(502, f"upstream unavailable: {exc}")
            self.close_connection = True
            return

        # Delimit the body by connection close and stream it through, so both
        # fixed-length and open-ended (SSE) responses relay correctly without us
        # having to reconcile Content-Length / chunked framing.
        self.send_response_only(resp.status, resp.reason)
        for key, value in resp.getheaders():
            if key.lower() in _HOP_BY_HOP or key.lower() == "content-length":
                continue
            self.send_header(key, value)
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        try:
            while True:
                chunk = resp.read(_CHUNK)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        except OSError:
            pass
        finally:
            upstream.close()

    def _proxy_websocket(self) -> None:
        try:
            upstream = socket.create_connection(
                (self.target_host, self.target_port), timeout=30
            )
        except OSError as exc:
            self.send_error(502, f"upstream unavailable: {exc}")
            self.close_connection = True
            return

        # Replay the client's handshake (request line + headers) upstream, then
        # hand off to a raw bidirectional splice -- once upgraded the framing is
        # opaque to us.
        request = [f"{self.command} {self.path} {self.request_version}"]
        request += [f"{key}: {value}" for key, value in self.headers.items()]
        upstream.sendall(("\r\n".join(request) + "\r\n\r\n").encode("latin-1"))

        client = self.connection
        self.close_connection = True
        upstream_to_client = threading.Thread(
            target=_pipe, args=(upstream, client), daemon=True
        )
        upstream_to_client.start()
        _pipe(client, upstream)
        upstream_to_client.join()

    # All methods route through the same auth + proxy path.
    do_GET = _proxy
    do_POST = _proxy
    do_PUT = _proxy
    do_DELETE = _proxy
    do_PATCH = _proxy
    do_HEAD = _proxy
    do_OPTIONS = _proxy


def start_auth_proxy(
    *,
    target_host: str,
    target_port: int,
    username: str = "",
    password: str = "",
    ignore_username: bool = False,
    write_token: str = "",
    realm: str = "Training Gym",
    bind_host: str = "0.0.0.0",
    bind_port: int = 0,
) -> tuple[int, Callable[[], None]]:
    """Start a Basic Auth reverse proxy in a background thread.

    Returns ``(listen_port, stop)`` where ``listen_port`` is the selected port
    (ephemeral by default) and ``stop()`` shuts it down. Trackio metric clients
    may authenticate with ``X-Trackio-Write-Token`` instead of browser Basic
    Auth when ``write_token`` is configured. ``ignore_username`` matches the
    main gym dashboard's password-only Basic Auth behavior.
    """
    expected_auth = (
        base64.b64encode(f"{username}:{password}".encode()).decode("ascii")
        if password
        else ""
    )

    handler = type(
        "_BoundAuthProxyHandler",
        (_AuthProxyHandler,),
        {
            "target_host": target_host,
            "target_port": int(target_port),
            "expected_auth": expected_auth,
            "expected_password": password if ignore_username else "",
            "expected_write_token": write_token,
            "realm": realm,
        },
    )

    server = ThreadingHTTPServer((bind_host, bind_port), handler)
    server.daemon_threads = True
    listen_port = server.server_address[1]

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def stop() -> None:
        server.shutdown()
        server.server_close()

    return listen_port, stop
