from __future__ import annotations

import base64
import http.client
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from modal_training_gym.common.auth_proxy import start_auth_proxy


class _Backend(BaseHTTPRequestHandler):
    def log_message(self, *_args: object) -> None:
        pass

    def do_GET(self) -> None:
        body = b"ok"
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _request(port: int, headers: dict[str, str] | None = None) -> int:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    connection.request("GET", "/", headers=headers or {})
    response = connection.getresponse()
    response.read()
    connection.close()
    return response.status


def test_auth_proxy_accepts_browser_or_trackio_credentials() -> None:
    backend = ThreadingHTTPServer(("127.0.0.1", 0), _Backend)
    threading.Thread(target=backend.serve_forever, daemon=True).start()
    port, stop = start_auth_proxy(
        target_host="127.0.0.1",
        target_port=backend.server_address[1],
        username="training-gym",
        password="password",
        ignore_username=True,
        write_token="write-token",
    )
    basic = base64.b64encode(b"training-gym:password").decode()
    blank_username = base64.b64encode(b":password").decode()

    try:
        assert _request(port) == 401
        assert _request(port, {"Authorization": f"Basic {basic}"}) == 200
        assert (
            _request(port, {"Authorization": f"Basic {blank_username}"}) == 200
        )
        assert _request(port, {"X-Trackio-Write-Token": "write-token"}) == 200
        assert _request(port, {"X-Trackio-Write-Token": "wrong"}) == 401
    finally:
        stop()
        backend.shutdown()
        backend.server_close()
