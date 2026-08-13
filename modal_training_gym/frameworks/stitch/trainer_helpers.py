"""Trainer-side helpers for the disaggregated miles flow.

Vendored from the stitch cookbook (``cookbook/common/launch.py`` +
``cookbook/common/process.py``): resolve HF repo ids and materialize inline YAML
configs, patch the runtime Megatron checkout, build the ``train.py`` command, and
reach the rollout pool's Flash gateway.
"""

from __future__ import annotations

import os
import shlex
import socket
import subprocess
import time
import urllib.request
from collections.abc import Iterable
from typing import Any

import modal

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


# ── Rollout pool endpoints ────────────────────────────────────────────────────


def modal_cluster_context(n_nodes: int) -> tuple[int, str, str]:
    """Return ``(rank, head_addr, node_ip)`` for this container's Modal cluster.

    Unlike :class:`~modal_training_gym.common.ray_cluster.ModalRayCluster`, the
    addresses are needed before Ray starts: miles reads them off the environment
    (``MASTER_ADDR``/``MILES_HOST_IP``) on every rank.
    """
    import modal.experimental

    try:
        info = modal.experimental.get_cluster_info()
        ips = list(info.container_ipv4_ips or [])
    except Exception:  # noqa: BLE001 — not running as a clustered function
        info, ips = None, []
    if not ips and n_nodes == 1:
        # Modal may omit container IPv4s for a size-1 cluster.
        ip = _local_ip()
        return 0, ip, ip
    if info is None or len(ips) != n_nodes:
        raise RuntimeError(
            f"Modal cluster size mismatch: expected {n_nodes} nodes, got {len(ips)}"
        )
    return info.rank, ips[0], ips[info.rank]


def _local_ip() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        try:
            sock.connect(("8.8.8.8", 80))
            return str(sock.getsockname()[0])
        except OSError:
            return socket.gethostbyname(socket.gethostname())


def flash_gateway_url(server_cls: Any) -> str:
    """The rollout pool's Flash gateway URL, from the app's own ``Server`` handle.

    Called with the class object the app was built with: that handle is hydrated
    for the container's own app, so it resolves in an ephemeral run — unlike
    ``Cls.from_name``/``ModalFlashPool``, which need a deployed app.
    """
    urls = server_cls._experimental_get_flash_urls()
    if not urls:
        raise RuntimeError(f"no Flash gateway URL for {server_cls}")
    return str(urls[0]).rstrip("/")


def deployed_gateway_url(app_name: str, cls_name: str = "Server") -> str:
    """The pool's Flash gateway URL looked up by app name — deployed pools only."""
    try:
        cls = modal.Cls.from_name(app_name, cls_name)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"cannot resolve {app_name}.{cls_name} — is the pool deployed?"
        ) from exc
    return flash_gateway_url(cls)


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
            urllib.request.urlopen(f"{gateway}/health", timeout=10).close()
            return True
        except Exception:  # noqa: BLE001
            time.sleep(interval_seconds)
    print(f"WARNING: pool at {gateway} not ready after {timeout_seconds:.0f}s")
    return False
