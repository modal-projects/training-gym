"""Launch-side helpers for the disaggregated miles flow.

Vendored from the stitch cookbook (``cookbook/common/launch.py`` +
``cookbook/common/process.py`` + ``cookbook/common/smoke.py``): resolve HF repo
ids and materialize inline YAML configs, patch the runtime Megatron checkout,
build the ``train.py`` command, reach the rollout pool's Flash gateway, and smoke
a deployed Flash pool.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
import urllib.request
from collections.abc import Iterable
from typing import Any

import modal
import modal.experimental

# ── Config preparation ────────────────────────────────────────────────────────


def prepare_config(cfg: Any, tmpdir: str, yaml_config_fields: Iterable[str]) -> None:
    """Resolve HF repo IDs to local paths and materialize inline YAML configs."""
    import yaml
    from huggingface_hub import snapshot_download

    for attr in ("hf_checkpoint", "load", "ref_load", "critic_load"):
        if (val := getattr(cfg, attr, None)) and not str(val).startswith("/"):
            setattr(cfg, attr, snapshot_download(val, local_files_only=True))

    for field in yaml_config_fields:
        if isinstance(val := getattr(cfg, field, None), dict):
            path = os.path.join(tmpdir, f"{field}.yaml")
            with open(path, "w") as f:
                yaml.dump(val, f)
            setattr(cfg, field, path)


def model_is_cached(model: Any) -> bool:
    """Whether the HF cache already holds the model's weights, so ``train`` can
    self-heal a launch that skipped the client-side download step."""
    from huggingface_hub import snapshot_download

    if path := getattr(model, "model_path", ""):
        return os.path.isdir(path) and bool(os.listdir(path))
    try:
        snapshot_download(model.model_name, local_files_only=True)
    except Exception:  # noqa: BLE001 — any cache miss means "download it"
        return False
    return True


def materialize_node_local_yaml(
    cfg: Any, field: str, dest_dir: str = "/root/.node_yaml"
) -> None:
    """Write an inline-dict config field to a deterministic node-local path.

    Unlike :func:`prepare_config`'s per-launch tmpdir, every node writes the same
    content at the same path, so Ray actors on other nodes can read a field the
    trainer only passes as a filename (miles' ``te_precision_config_file``). Call
    on every node, before the rank gate.
    """
    import yaml

    if isinstance(val := getattr(cfg, field, None), dict):
        os.makedirs(dest_dir, exist_ok=True)
        path = os.path.join(dest_dir, f"{field}.yaml")
        with open(path, "w") as f:
            yaml.dump(val, f)
        setattr(cfg, field, path)


def apply_git_patches(patch_paths: Iterable[str], repo_dir: str, label: str) -> None:
    """Apply git patches to a runtime checkout, tolerating an already-applied
    patch so a container restart (or a second run in the same container) is a
    no-op rather than a failure."""
    for patch_path in patch_paths:
        if not os.path.exists(patch_path):
            raise FileNotFoundError(f"{label} not found: {patch_path}")
        check = subprocess.run(
            ["git", "-C", repo_dir, "apply", "--check", patch_path],
            capture_output=True,
            text=True,
        )
        if check.returncode == 0:
            subprocess.run(["git", "-C", repo_dir, "apply", patch_path], check=True)
            print(f"[{label}] applied {patch_path}", flush=True)
            continue
        reverse = subprocess.run(
            ["git", "-C", repo_dir, "apply", "--reverse", "--check", patch_path],
            capture_output=True,
            text=True,
        )
        if reverse.returncode == 0:
            print(f"[{label}] already applied {patch_path}", flush=True)
            continue
        raise RuntimeError(
            f"cannot apply {label} {patch_path}\n"
            f"check: {check.stderr}\nreverse: {reverse.stderr}"
        )


def build_train_cmd(cfg: Any, trainer_root: str, *, model_script_attr: str) -> str:
    """Build the training command, sourcing model arch args if needed."""
    train_script = (
        f"{trainer_root}/{'train_async.py' if cfg.async_mode else 'train.py'}"
    )
    model_script = getattr(cfg, model_script_attr, "")
    if model_script:
        inner = (
            f"source {trainer_root}/{model_script} && "
            f"python3 {train_script} ${{MODEL_ARGS[@]}} {shlex.join(cfg.cli_args())}"
        )
        return f"bash -c {shlex.quote(inner)}"
    return f"python3 {train_script} {shlex.join(cfg.cli_args())}"


# ── Flash pool smoke check ────────────────────────────────────────────────────


class VersionAheadError(RuntimeError):
    """A monotonic rollout pool has already advanced past the smoke's expected version."""


def smoke_flash_pool(
    *,
    app_name: str,
    cls_name: str,
    model_name: str,
    weight_version: int,
    expect_min_containers: int,
    timeout_seconds: int,
) -> None:
    """Poll until the pool serves completions at ``weight_version`` — through the
    gateway (Flash holds the request through a scaled-down pool's cold start) and then
    each live replica's ``/server_info``.

    This runs on the launching client, which has no ``stitch`` install, so the pool
    is addressed through Modal's Flash APIs directly rather than ``ModalFlashPool``.
    """
    pool = _FlashPool(app_name, cls_name)
    deadline = time.time() + timeout_seconds
    last_error: str | None = None
    while True:
        try:
            _smoke_once(pool, model_name, weight_version, expect_min_containers)
            return
        except VersionAheadError:
            raise
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
        if time.time() >= deadline:
            raise TimeoutError(
                f"Flash pool smoke did not pass before timeout: {last_error}"
            )
        print(f"Waiting for Flash pool readiness: {last_error}")
        time.sleep(10)


def flash_gateway_url(server_cls: Any) -> str:
    """The rollout pool's Flash gateway URL, from the app's own ``Server`` handle.

    Called from the trainer container with the class object the app was built
    with: that handle is hydrated for the container's own app, so it resolves in
    an ephemeral run — unlike ``Cls.from_name``/``ModalFlashPool``, which can only
    look a Flash service up once the app is deployed.
    """
    urls = server_cls._experimental_get_flash_urls()
    if not urls:
        raise RuntimeError(f"no Flash gateway URL for {server_cls}")
    return str(urls[0]).rstrip("/")


def deployed_gateway_url(app_name: str, cls_name: str = "Server") -> str:
    """The pool's Flash gateway URL looked up by app name — deployed pools only."""
    return _FlashPool(app_name, cls_name).gateway_url()


def await_gateway_ready(
    gateway: str,
    *,
    timeout_seconds: float = 20 * 60,
    interval_seconds: float = 15.0,
) -> bool:
    """Block until the pool's gateway answers ``/health`` 200.

    Flash holds requests through a cold-starting pool, so this only matters for
    the trainer's first rollout meeting engines that are still loading. On
    timeout it warns and returns ``False``; the caller proceeds because the
    trainer's rollout requests retry.
    """
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            _get_json(f"{gateway}/health", timeout=10)
            return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(interval_seconds)
    print(f"WARNING: pool at {gateway} not ready after {timeout_seconds:.0f}s")
    return False


class _FlashPool:
    """The two Flash lookups the smoke needs: gateway URL and live replica URLs."""

    def __init__(self, app_name: str, cls_name: str) -> None:
        self.app_name = app_name
        self.cls_name = cls_name

    def _cls(self) -> modal.Cls:
        try:
            return modal.Cls.from_name(self.app_name, self.cls_name)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"cannot resolve {self.app_name}.{self.cls_name} — is the pool deployed?"
            ) from exc

    def gateway_url(self) -> str:
        urls = self._cls()._experimental_get_flash_urls()
        if not urls:
            raise RuntimeError(
                f"no Flash gateway URL for {self.app_name}.{self.cls_name}"
            )
        return str(urls[0]).rstrip("/")

    def discover_replicas(self) -> list[str]:
        containers = modal.experimental.flash_get_containers(
            self.app_name, self.cls_name
        )
        hosts = [
            c.get("host") if isinstance(c, dict) else getattr(c, "host", None)
            for c in containers
        ]
        return [_url(str(h)) for h in hosts if h]


def _url(host: str) -> str:
    host = host.rstrip("/")
    return host if host.startswith(("http://", "https://")) else f"https://{host}"


def _smoke_once(
    pool: _FlashPool,
    model_name: str,
    expected: int,
    expect_min_containers: int,
) -> None:
    gateway = pool.gateway_url()
    print(f"Gateway URL: {gateway}")
    # A fresh pool has no claimed run, so version 0 is unpinnable — an exact-version
    # request would 409. The claimed run shows up as the run component of the applied
    # pointer (``server_info["run_id"]`` is only the sidecar's static label).
    info = _get_json(f"{gateway}/server_info", timeout=60)
    print(f"Gateway server_info: {info}")
    if not _applied_run(info):
        if expected != 0:
            raise RuntimeError(
                f"pool is unclaimed; cannot serve expected weight version {expected}"
            )
        data = _post_json(
            f"{gateway}/v1/chat/completions", _completion(model_name), timeout=900
        )
        _check_serves(data)
        print(f"Pool serves base (unclaimed): {data.get('choices')}")
        return
    data = _post_json(
        f"{gateway}/v1/chat/completions",
        _completion(model_name, expected),
        timeout=900,
    )
    print(f"Gateway completion: {data}")
    _check_completion(data, expected)
    replicas = pool.discover_replicas()
    if len(replicas) < expect_min_containers:
        raise RuntimeError(
            f"expected at least {expect_min_containers} containers, "
            f"found {len(replicas)}: {replicas}"
        )
    for target in replicas:
        info = _get_json(f"{target}/server_info", timeout=30)
        print(f"{target} server_info={info}")
        _check_version(_applied_version(info), expected, target)


def _applied_run(info: dict) -> str:
    """Run id the replica's applied pointer belongs to, empty if it has none."""
    return str(info.get("applied") or "").rpartition("/")[0]


def _applied_version(info: dict) -> int:
    """Version of the replica's applied pointer (``[<run_id>/]weight_vNNNNNN``)."""
    applied = str(info.get("applied") or "").strip()
    if not applied:
        return -1
    tail = applied.rpartition("/")[2].removeprefix("weight_v")
    if not tail.isdigit():
        raise ValueError(f"unparseable applied pointer: {applied!r}")
    return int(tail)


def _check_version(current: int, expected: int, target: str) -> None:
    if current > expected:
        raise VersionAheadError(
            f"{target} applied={current} already past expected {expected}"
        )
    if current != expected:
        raise RuntimeError(f"{target} applied={current}, expected {expected}")


def _check_completion(data: dict, expected: int) -> None:
    start = int(data.get("weight_version_start", -1))
    end = int(data.get("weight_version_end", -1))
    if start > expected or end > expected:
        raise VersionAheadError(
            f"gateway served {start}->{end}, already past expected {expected}"
        )
    if start != expected or end != expected:
        raise RuntimeError(f"unexpected gateway weight metadata: {data}")


def _check_serves(data: dict) -> None:
    choices = data.get("choices") or []
    if not choices or not ((choices[0].get("message") or {}).get("content")):
        raise RuntimeError(f"pool did not return a completion: {data}")


def _completion(model_name: str, expected: int | None = None) -> dict:
    payload: dict[str, Any] = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Reply with exactly OK."}],
        "max_tokens": 8,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if expected is not None:  # pin the version only against a claimed pool
        payload["weight_version"] = {"exact_version": expected}
    return payload


def _get_json(url: str, *, timeout: float) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        body = resp.read().decode()
    return json.loads(body) if body.strip().startswith("{") else {}


def _post_json(url: str, payload: dict, *, timeout: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as resp:
        return json.load(resp)
