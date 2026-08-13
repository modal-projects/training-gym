"""Subprocess + runtime helpers for the disaggregated flow.

Vendored from the stitch cookbook (``cookbook/common/process.py``): launch the
sidecar beside SGLang, wait on HTTP liveness, terminate cleanly, and probe
torch-distributed rank/barrier for the rank-gated publish hooks.
"""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

SIDECAR_MODULE = "modal_training_gym.frameworks.stitch.sidecar"


def start_sidecar(
    flags: list[str],
    *,
    sidecar_port: int,
    sglang_port: int,
    log_path: str | None = None,
) -> subprocess.Popen:
    """Launch the versioned rollout proxy (the stitch sidecar) beside SGLang.

    ``flags`` are the recipe-derived sidecar CLI flags; only the ports are wired
    here, since they belong to this container's process layout.
    """
    cmd = [
        "python3",
        "-m",
        SIDECAR_MODULE,
        "--host",
        "0.0.0.0",
        "--port",
        str(sidecar_port),
        "--upstream",
        f"http://127.0.0.1:{sglang_port}",
        *flags,
    ]
    print("Starting sidecar:", " ".join(cmd))
    if log_path is None:
        return subprocess.Popen(cmd, start_new_session=True)
    # A replica's container log window is overrun by SGLang's per-batch output
    # within seconds, so keep a durable copy of the sync decisions. ``log_path``
    # must not live on the bulletin volume: an open file there makes the
    # reconciler's ``Volume.reload()`` fail with "there are open files preventing
    # the operation", and the replica then never sees a new version.
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(
        [
            "bash",
            "-lc",
            f"set -o pipefail; {shlex.join(cmd)} 2>&1 | tee -a {shlex.quote(log_path)}",
        ],
        start_new_session=True,
    )


def wait_http(url: str, process: subprocess.Popen | None, timeout: int) -> None:
    deadline = time.time() + timeout
    last_error: str | None = None
    while time.time() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(
                f"process exited while waiting for {url}: code={process.returncode}"
            )
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if 200 <= resp.status < 500:
                    return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(2)
    raise TimeoutError(f"Timed out waiting for {url}; last error: {last_error}")


def terminate_process(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=20)
    except Exception:  # noqa: BLE001
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except Exception:  # noqa: BLE001
            pass


def dist_rank() -> int | None:
    """This process's torch-distributed rank, or ``None`` off the distributed path.

    Gates rank-0-only side effects (pointer writes, pool wakes) so only one writer
    acts per publish.
    """
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank())
    except Exception:  # noqa: BLE001
        return None
    return None


def dist_barrier() -> None:
    """Wait for all ranks; a no-op off the distributed path."""
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
