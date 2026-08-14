"""Interactive GPU sandbox — an H200 box with SSH access (workflow in
sandbox.md).

    _SANDBOX_GPU=H200 python -m modal run toolbox/gpu_tools/gpu_sandbox.py \\
        --key-path ~/.ssh/id_ed25519.pub --sandbox-id my-box

_SANDBOX_GPU ("H200", "H200:N", or "cpu") must be set on the launch command:
Modal reads the GPU spec when this file is imported. SSH connection info is
written to the gpu-sandbox-workspace volume at /ssh-info/<sandbox-id>.json.
On exit the launcher appends the session's GPU accounting line to
runs/GPU_LOG.jsonl (gpu_tools/README.md rule 5).
"""

import json
import os
import pathlib
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

import modal

APP_NAME = "gpu-sandbox"
CUDA_TAG = "12.6.0-devel-ubuntu22.04"
LOCAL_SSHD_CONFIG_PATH = pathlib.Path(__file__).resolve().parent / "sshd_config"
REMOTE_WORKSPACE_DIR = "/root/workspace"
VOLUME_ROOT = "/vol"
VOLUME_WORKSPACE_DIR = f"{VOLUME_ROOT}/workspace"
SSH_INFO_DIR = f"{VOLUME_ROOT}/ssh-info"
SYNC_INTERVAL_SECONDS = 30

# GPU spec — read at import time (before Modal registers the function).
# H200 only (gpu_tools/README.md rule 2); "cpu" for CPU-only debugging.
GPU_SPEC = os.environ.get("_SANDBOX_GPU", "H200")
_GPU_PARAM = None if GPU_SPEC.lower() == "cpu" else GPU_SPEC

app = modal.App(APP_NAME)

workspace_volume = modal.Volume.from_name(
    "gpu-sandbox-workspace", create_if_missing=True
)
out_volume = modal.Volume.from_name("lab-out")
hf_cache_volume = modal.Volume.from_name("lab-hf-cache")

image = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python="3.12")
    .apt_install(
        "openssh-server",
        "rsync",
        "git",
        "curl",
        "wget",
        "vim",
        "htop",
        "tmux",
        "jq",
    )
    .add_local_file(str(LOCAL_SSHD_CONFIG_PATH), "/etc/ssh/sshd_config", copy=True)
    .run_commands(
        "mkdir -p /root/workspace /var/run/sshd /root/.ssh",
        "chmod 700 /root/.ssh",
    )
    .pip_install(
        "sglang[all]",
        "torch",
        "transformers",
        "datasets",
        "numpy",
        "pandas",
        "nvitop",
    )
    .env({"HF_HOME": "/hf-cache"})
)


def _rsync(src: str, dst: str) -> None:
    subprocess.run(
        ["rsync", "-a", "--delete", "--exclude", ".git/",
         f"{src.rstrip('/')}/", f"{dst.rstrip('/')}/"],
        check=True,
    )


def _initial_sync() -> None:
    os.makedirs(VOLUME_WORKSPACE_DIR, exist_ok=True)
    os.makedirs(REMOTE_WORKSPACE_DIR, exist_ok=True)
    workspace_volume.reload()
    _rsync(VOLUME_WORKSPACE_DIR, REMOTE_WORKSPACE_DIR)


def _background_sync(stop: threading.Event) -> None:
    while not stop.is_set():
        _rsync(REMOTE_WORKSPACE_DIR, VOLUME_WORKSPACE_DIR)
        workspace_volume.commit()
        stop.wait(SYNC_INTERVAL_SECONDS)


def _write_authorized_key(pubkey: str) -> None:
    path = "/root/.ssh/authorized_keys"
    with open(path, "w") as f:
        f.write(f"{pubkey.strip()}\n")
    os.chmod(path, 0o600)


def _write_ssh_info_to_volume(sandbox_id: str, host: str, port: int) -> None:
    """Write SSH connection info to the volume so the local client can read it."""
    os.makedirs(SSH_INFO_DIR, exist_ok=True)
    info = {"host": host, "port": port, "sandbox_id": sandbox_id, "time": time.time()}
    info_path = os.path.join(SSH_INFO_DIR, f"{sandbox_id}.json")
    with open(info_path, "w") as f:
        json.dump(info, f)
    workspace_volume.commit()


@app.function(
    image=image,
    gpu=_GPU_PARAM,
    timeout=4 * 60 * 60,
    volumes={VOLUME_ROOT: workspace_volume,
             "/out": out_volume,
             "/hf-cache": hf_cache_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def sandbox(ssh_public_key: str, sandbox_id: str = "default") -> None:
    _write_authorized_key(ssh_public_key)

    # Start sshd FIRST so SSH is available immediately (don't block on volume sync)
    sshd = subprocess.Popen(["/usr/sbin/sshd", "-D", "-e"])

    # Run initial volume sync in background — SSH works while this catches up
    init_thread = threading.Thread(target=_initial_sync, daemon=True)
    init_thread.start()

    stop = threading.Event()
    sync_thread = threading.Thread(target=_background_sync, args=(stop,), daemon=True)
    sync_thread.start()

    try:
        with modal.forward(22, unencrypted=True) as tunnel:
            host, port = tunnel.tcp_socket
            # Print for logs (best-effort)
            print(f"SANDBOX_SSH={host}:{port}")
            print(f"ssh root@{host} -p {port}")
            # Write to volume for reliable handshake with local client
            _write_ssh_info_to_volume(sandbox_id, host, port)
            sshd.wait()
    finally:
        stop.set()
        sync_thread.join(timeout=5)
        _rsync(REMOTE_WORKSPACE_DIR, VOLUME_WORKSPACE_DIR)
        workspace_volume.commit()
        # Clean up SSH info file
        info_path = os.path.join(SSH_INFO_DIR, f"{sandbox_id}.json")
        if os.path.exists(info_path):
            os.remove(info_path)
            workspace_volume.commit()
        if sshd.poll() is None:
            sshd.terminate()


def _log_gpu_use(started: float, rc: int, sandbox_id: str) -> None:
    """Same ledger line gpu_launcher.py writes (gpu_tools/README.md rule 5).
    Accounting must never fail the session."""
    n = 0 if _GPU_PARAM is None else (
        int(GPU_SPEC.split(":")[1]) if ":" in GPU_SPEC else 1)
    row = {"ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "seconds": round(time.time() - started, 1),
           "gpu": "none" if _GPU_PARAM is None else GPU_SPEC, "n_gpus": n,
           "command": f"gpu_sandbox.py --sandbox-id {sandbox_id}"[:120],
           "exit": rc}
    try:
        path = pathlib.Path(__file__).resolve().parents[2] / "runs" / "GPU_LOG.jsonl"
        path.parent.mkdir(exist_ok=True)
        with path.open("a") as f:
            f.write(json.dumps(row) + "\n")
    except OSError as e:
        print(f"[gpu_sandbox] GPU_LOG write failed: {e}", flush=True)


@app.local_entrypoint()
def main(key_path: str = "", sandbox_id: str = "default") -> None:
    default_key = pathlib.Path.home() / ".ssh" / "id_ed25519.pub"
    path = pathlib.Path(key_path.strip() or str(default_key)).expanduser()
    if not path.exists():
        raise ValueError(f"SSH public key not found: {path}")
    pubkey = path.read_text().strip()

    # `kill $SANDBOX_PID` sends SIGTERM — turn it into SystemExit so the
    # finally block still writes the accounting line.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    started = time.time()
    rc = 0
    try:
        sandbox.remote(ssh_public_key=pubkey, sandbox_id=sandbox_id)
    except SystemExit as e:
        rc = int(e.code or 0)
        raise
    except BaseException:
        rc = 1
        raise
    finally:
        _log_gpu_use(started, rc, sandbox_id)
