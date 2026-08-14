"""Standalone dev server for observatory/static — fixture-backed API on :8899.

Serves static/ at / and maps the DESIGN.md API routes onto the fixture files,
so the frontend can be developed and integration-tested without Modal:

    python3 observatory/static_dev_server.py [port]
"""

from __future__ import annotations

import json
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"
FIXTURES = HERE / "fixtures"
DEFAULT_PORT = 8899


def _fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def _raw_dir(run_id: str) -> Path:
    # demo run dir keeps raw artifacts where the real run wrote them
    return (FIXTURES / "demo" / f"ws_{run_id}" / "workspace" / "agents" / "_runs" / run_id)


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC), **kwargs)

    def end_headers(self):
        self.send_header("Cache-Control", "no-cache")  # dev: always revalidate
        super().end_headers()

    def _json(self, obj, status: int = 200) -> None:
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _text(self, body: bytes, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802 (http.server API)
        path = self.path.split("?", 1)[0].rstrip("/") or "/"

        if path == "/healthz":
            return self._json({"ok": True})

        if path.startswith("/api/"):
            return self._api(path)

        if path == "/":
            self.path = "/index.html"
        elif path == "/run":
            self.path = "/run.html"
        elif path in ("/how", "/docs"):
            self.path = "/docs.html"
        elif path.startswith("/docs/"):
            self.path = f"/docs-{path.split('/', 2)[2]}.html"
        elif path == "/tools":
            self.path = "/tools.html"
        return super().do_GET()

    def _api(self, path: str) -> None:
        record = _fixture("sample_record.json")
        run_id = record.get("index_row", {}).get("run_id", "")

        if path == "/api/runs":
            return self._json([record.get("index_row", {})])

        parts = path.split("/")  # ['', 'api', 'runs', <id>, ...]
        if len(parts) < 4 or parts[2] != "runs" or parts[3] != run_id:
            return self._json({"error": "unknown run"}, 404)

        rest = parts[4:]
        if not rest:
            return self._json(record)
        if rest == ["status"]:
            return self._json(_fixture("sample_status.json"))
        if rest == ["workspace"]:
            return self._json(_fixture("sample_workspace.json"))
        if len(rest) == 2 and rest[0] == "raw":
            name = rest[1]
            f = _raw_dir(run_id) / name
            if "/" in name or name.startswith(".") or not f.is_file():
                return self._json({"error": "unknown raw artifact"}, 404)
            return self._text(f.read_bytes())
        return self._json({"error": "not found"}, 404)


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PORT
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"observatory dev server: http://127.0.0.1:{port}/  (static={STATIC})")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
