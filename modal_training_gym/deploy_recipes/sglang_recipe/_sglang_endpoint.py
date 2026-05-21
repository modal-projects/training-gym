"""SGLang server subprocess management.

Vendored from autoinference's ``runtime/endpoint.py`` — contains only the
``SGLangEndpoint`` class and the helpers needed by ``serve_sglang.py``.
"""

from __future__ import annotations

import json
import shlex
import subprocess
import time
import urllib.error
import urllib.request
from typing import Any, Mapping, Optional


class SGLangEndpoint:
    """Manages an SGLang server subprocess."""

    DEFAULT_OPERATIONAL_ARGS: dict[str, str] = {
        "--enable-metrics": "",
        "--decode-log-interval": "1",
        "--enable-cache-report": "",
        "--model-loader-extra-config": '{"enable_multithread_load":true,"num_threads":64}',
    }

    def __init__(
        self,
        *,
        model_path: str,
        worker_port: int = 8000,
        tp: int | None = None,
        dp: int | None = None,
        extra_server_args: dict[str, str] | None = None,
        health_timeout: float = 20 * 60,
        health_poll_interval: float = 5.0,
    ):
        self.model_path = model_path
        self.worker_port = worker_port
        self.tp = tp
        self.dp = dp
        self.extra_server_args = dict(extra_server_args) if extra_server_args else {}
        self.health_timeout = health_timeout
        self.health_poll_interval = health_poll_interval
        self._proc: Optional[subprocess.Popen] = None

    def _build_cmd(self) -> list[str]:
        cmd = [
            "python",
            "-m",
            "sglang.launch_server",
            "--host",
            "0.0.0.0",
            "--port",
            str(self.worker_port),
            "--model-path",
            self.model_path,
        ]
        if self.tp is not None:
            cmd.extend(["--tp", str(self.tp)])
        if self.dp is not None:
            cmd.extend(["--dp", str(self.dp), "--enable-dp-attention"])

        merged = {**self.DEFAULT_OPERATIONAL_ARGS, **self.extra_server_args}
        for key, value in merged.items():
            if value == "":
                cmd.append(key)
            else:
                cmd.extend([key, value])
        return cmd

    def _rewrite_chat_template_kwargs(self) -> None:
        """Convert ``--chat-template-kwargs`` to ``--chat-template``.

        SGLang doesn't expose ``--chat-template-kwargs`` as a CLI flag
        (it's a per-request API parameter only).  Work around this by
        downloading the model's chat template from its tokenizer config,
        prepending Jinja ``{% set %}`` statements for the requested
        kwargs, and pointing sglang at the modified template file.
        """
        raw = self.extra_server_args.pop("--chat-template-kwargs", None)
        if raw is None:
            return

        kwargs = json.loads(raw) if isinstance(raw, str) else raw

        template = self._load_chat_template()
        if not template:
            print(
                "[sglang] warning: could not load chat template from model; "
                "ignoring --chat-template-kwargs"
            )
            return

        prefix = "\n".join(
            f"{{% set {k} = {json.dumps(v)} %}}" for k, v in kwargs.items()
        )
        path = "/tmp/_sglang_chat_template.jinja"
        with open(path, "w") as f:
            f.write(prefix + "\n" + template)

        self.extra_server_args["--chat-template"] = path
        print(f"[sglang] rewrote --chat-template-kwargs as --chat-template {path}")

    def _load_chat_template(self) -> str:
        """Return the model's Jinja chat template string, or ``""``."""
        import os

        config = None
        # Local checkpoint path
        local = os.path.join(self.model_path, "tokenizer_config.json")
        if os.path.isfile(local):
            with open(local) as f:
                config = json.load(f)
        else:
            try:
                from huggingface_hub import hf_hub_download

                path = hf_hub_download(self.model_path, "tokenizer_config.json")
                with open(path) as f:
                    config = json.load(f)
            except Exception as exc:
                print(f"[sglang] failed to download tokenizer_config.json: {exc}")
                return ""

        template = config.get("chat_template", "")
        if isinstance(template, list):
            template = next(
                (t["template"] for t in template if t.get("name") == "default"),
                template[0]["template"] if template else "",
            )
        return template

    def start(self) -> None:
        self._rewrite_chat_template_kwargs()
        cmd = self._build_cmd()
        print(f"[sglang] starting: {shlex.join(cmd)}")
        self._proc = subprocess.Popen(cmd)
        _wait_ready(
            self._proc,
            port=self.worker_port,
            timeout=self.health_timeout,
            poll_interval=self.health_poll_interval,
        )

    def stop(self) -> None:
        if self._proc is None or self._proc.poll() is not None:
            return
        self._proc.terminate()
        try:
            self._proc.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()
        self._proc = None


def warmup_chat_completions(
    *,
    port: int,
    payload: Mapping[str, Any],
    successful_requests: int = 3,
    request_timeout: float = 30.0,
) -> None:
    """Send a few chat-completion requests to warm caches and JIT."""
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    body = json.dumps(dict(payload)).encode()

    for i in range(successful_requests):
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=request_timeout):
                pass
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            print(f"[sglang] warmup {i + 1}/{successful_requests} failed: {exc}")


# ── Internal ──────────────────────────────────────────────────────────────────


def _wait_ready(
    proc: subprocess.Popen,
    *,
    port: int,
    timeout: float,
    poll_interval: float = 5.0,
) -> None:
    deadline = time.time() + timeout
    url = f"http://127.0.0.1:{port}/health"

    while time.time() < deadline:
        if (rc := proc.poll()) is not None:
            raise subprocess.CalledProcessError(rc, cmd=proc.args)
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                if 200 <= resp.getcode() < 300:
                    return
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
        time.sleep(poll_interval)

    raise TimeoutError(f"SGLang health check timed out after {timeout}s on port {port}")
