#!/usr/bin/env python3
"""gpu_launcher — run ONE portable command on a GPU host.

Compose a plain shell command that runs on any machine with the right
environment; this launcher is how that command reaches your GPUs, which
live on Modal: it builds the requested image, mounts the standard volumes,
and executes the command in a Modal Sandbox — no per-tool Modal apps, no
function registration, nothing tool-specific.

    python3 toolbox/gpu_tools/gpu_launcher.py \
        --pip-e toolbox/training_tool/automodel \
        --gpu H200 --timeout-min 240 \
        --upload runs/job1:/root/job \
        -- python /root/job/train.py

Image sources (pick one base, then layer):
  --image <ref>            docker registry image from any registry
  --python <ver>           debian_slim base with this python (default 3.12)
Layers, applied in order given:
  --requirements <file>    pip install -r <file>
  --pip <spec>             pip install <spec>            (repeatable)
  --pip-e <dir>            copy <dir> into the image and pip install -e it
  --upload <src>:<dst>     copy a local dir/file into the image (repeatable)
Runtime:
  --gpu H200|H200:4|...    GPU spec (default H200)
  --timeout-min <n>        wall-clock cap (default 120)
  --volume <name>:<path>   Modal volume mount (repeatable; default lab-out:/out
                           and lab-hf-cache:/hf-cache — pass --no-default-volumes
                           to drop them)
  --secret <name>          Modal secret (repeatable; default huggingface-secret)
  --env K=V                environment variable in the sandbox (repeatable)
  --workdir <path>         working directory for the command
  --dry-run                print the resolved sandbox spec and exit (no modal
                           import, no network)

Exit code = the command's exit code. Logs stream to stdout/stderr. Volume
writes flush when the sandbox terminates, so /out artifacts persist.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_VOLUMES = [("lab-out", "/out"), ("lab-hf-cache", "/hf-cache")]
DEFAULT_SECRETS = ["huggingface-secret"]
# The HF discipline every trainer image in this repo uses:
# cache on the shared volume so weights download once, then stay local.
DEFAULT_ENV = {"HF_HOME": "/hf-cache", "HF_HUB_ENABLE_HF_TRANSFER": "1"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run one portable command on a Modal GPU sandbox.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--image", default=None, help="docker registry image base")
    ap.add_argument("--python", default="3.12", help="debian_slim python (when no --image)")
    ap.add_argument("--requirements", action="append", default=[],
                    help="pip install -r FILE (repeatable)")
    ap.add_argument("--pip", action="append", default=[], help="pip install SPEC (repeatable)")
    ap.add_argument("--pip-e", action="append", default=[],
                    help="copy DIR into the image and pip install -e it (repeatable)")
    ap.add_argument("--upload", action="append", default=[],
                    help="SRC:DST — copy local dir/file into the image (repeatable)")
    ap.add_argument("--gpu", default="H200",
                    help="GPU spec, or 'none' for a CPU-only sandbox")
    ap.add_argument("--timeout-min", type=int, default=120)
    ap.add_argument("--volume", action="append", default=[],
                    help="NAME:PATH Modal volume mount (repeatable)")
    ap.add_argument("--no-default-volumes", action="store_true",
                    help="do not mount lab-out:/out and lab-hf-cache:/hf-cache")
    ap.add_argument("--secret", action="append", default=[],
                    help="Modal secret name (repeatable; default huggingface-secret)")
    ap.add_argument("--env", action="append", default=[], help="K=V (repeatable)")
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the resolved spec, run nothing")
    ap.add_argument("command", nargs=argparse.REMAINDER,
                    help="-- the command to run (everything after --)")
    args = ap.parse_args(argv)
    cmd = args.command
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        ap.error("no command given (put it after `--`)")
    args.command = cmd
    return args


def resolve_spec(args: argparse.Namespace) -> dict:
    """The launcher's full intent as plain data — what --dry-run prints and
    what the live path executes. Kept separate so tests can assert on it
    without modal installed."""
    volumes = [] if args.no_default_volumes else list(DEFAULT_VOLUMES)
    for v in args.volume:
        name, _, path = v.partition(":")
        if not name or not path:
            raise SystemExit(f"--volume must be NAME:PATH, got {v!r}")
        volumes.append((name, path))
    uploads = []
    for u in args.upload:
        src, _, dst = u.partition(":")
        if not src or not dst:
            raise SystemExit(f"--upload must be SRC:DST, got {u!r}")
        if not Path(src).exists():
            raise SystemExit(f"--upload source does not exist: {src}")
        uploads.append((str(Path(src).resolve()), dst))
    for d in args.pip_e:
        if not Path(d).is_dir():
            raise SystemExit(f"--pip-e directory does not exist: {d}")
    for r in args.requirements:
        if not Path(r).is_file():
            raise SystemExit(f"--requirements file does not exist: {r}")
    env = dict(DEFAULT_ENV)
    # The run's /out/models namespace + metering attribution: forward the
    # workspace's LEARNING_AGENT_RUN_ID into every job (explicit --env overrides).
    if os.environ.get("LEARNING_AGENT_RUN_ID"):
        env["LEARNING_AGENT_RUN_ID"] = os.environ["LEARNING_AGENT_RUN_ID"]
    for kv in args.env:
        k, _, v = kv.partition("=")
        if not k:
            raise SystemExit(f"--env must be K=V, got {kv!r}")
        env[k] = v
    return {
        "image": args.image or f"debian_slim(python_version={args.python})",
        "requirements": [str(Path(r).resolve()) for r in args.requirements],
        "pip": list(args.pip),
        "pip_e": [str(Path(d).resolve()) for d in args.pip_e],
        "uploads": uploads,
        "gpu": args.gpu,
        "timeout_min": args.timeout_min,
        "volumes": volumes,
        "secrets": list(args.secret) or list(DEFAULT_SECRETS),
        "env": env,
        "workdir": args.workdir,
        "command": list(args.command),
    }


def run(spec: dict) -> int:
    import modal  # lazy: --dry-run and tests never need it

    if spec["image"].startswith("debian_slim"):
        image = modal.Image.debian_slim(python_version=spec["image"].split("=")[-1].rstrip(")"))
    else:
        # entrypoint([]) — a registry image may ship a blocking ENTRYPOINT;
        # the sandbox supplies the command, so always run it directly.
        image = modal.Image.from_registry(spec["image"]).entrypoint([])
    for r in spec["requirements"]:
        image = image.pip_install_from_requirements(r)
    if spec["pip"]:
        image = image.pip_install(*spec["pip"])
    for d in spec["pip_e"]:
        dst = f"/root/_pip_e/{Path(d).name}"
        image = image.add_local_dir(d, dst, copy=True).run_commands(f"pip install -e {dst}")
    for src, dst in spec["uploads"]:
        if Path(src).is_dir():
            image = image.add_local_dir(src, dst)
        else:
            image = image.add_local_file(src, dst)

    app = modal.App.lookup("lab-gpu-launcher", create_if_missing=True)
    volumes = {path: modal.Volume.from_name(name, create_if_missing=False)
               for name, path in spec["volumes"]}
    run_id = spec["env"].get("LEARNING_AGENT_RUN_ID") or os.environ.get("LEARNING_AGENT_RUN_ID", "")
    sb = modal.Sandbox.create(
        *spec["command"],
        app=app,
        image=image,
        gpu=None if spec["gpu"] == "none" else spec["gpu"],
        timeout=spec["timeout_min"] * 60,
        volumes=volumes,
        secrets=[modal.Secret.from_name(s) for s in spec["secrets"]],
        env=spec["env"],
        workdir=spec["workdir"],
        tags={"learning_agent_run_id": run_id} if run_id else None,
    )
    # Clock starts AFTER create returns: image build + scheduling are not
    # GPU time and must not inflate the ledger.
    started = time.time()
    rc = -1  # unknown until wait() reports; -1 in the log = died unaccounted
    print(f"[gpu_launcher] sandbox {sb.object_id} gpu={spec['gpu']} "
          f"timeout={spec['timeout_min']}min", flush=True)
    try:
        for line in sb.stdout:
            print(line, end="", flush=True)
        for line in sb.stderr:
            print(line, end="", file=sys.stderr, flush=True)
        sb.wait()
        rc = sb.returncode if sb.returncode is not None else -1
    except BaseException:
        # Ctrl-C / timeout / network drop: don't leave the GPU burning.
        try:
            sb.terminate()
            print("[gpu_launcher] interrupted — sandbox terminated", flush=True)
        except Exception:
            print(f"[gpu_launcher] interrupted — could not terminate "
                  f"{sb.object_id}; check `modal sandbox list`", flush=True)
        raise
    finally:
        # No client-side volume.commit(): sandbox-attached volumes flush their
        # writes when the sandbox terminates (commit() is container-only API).
        _log_gpu_use(spec, started, rc)
    print(f"[gpu_launcher] exit {rc}", flush=True)
    return rc


def _log_gpu_use(spec: dict, started: float, rc: int) -> None:
    """One line per job in runs/GPU_LOG.jsonl — the run's GPU-hours ledger
    (the dashboard sums it). Agent-written Modal apps append their own line
    in the same shape (see toolbox/gpu_tools/README.md)."""
    gpu = spec["gpu"]
    try:
        n = 0 if gpu == "none" else (int(gpu.split(":")[1]) if ":" in gpu else 1)
    except ValueError:  # malformed spec must not lose the log line
        n = 1
    row = {"ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "seconds": round(time.time() - started, 1),
           "gpu": gpu, "n_gpus": n,
           "command": " ".join(spec["command"])[:120], "exit": rc}
    try:
        path = Path(__file__).resolve().parents[2] / "runs" / "GPU_LOG.jsonl"
        path.parent.mkdir(exist_ok=True)
        with path.open("a") as f:
            f.write(json.dumps(row) + "\n")
    except OSError as e:  # accounting must never fail the job
        print(f"[gpu_launcher] GPU_LOG write failed: {e}", flush=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    spec = resolve_spec(args)
    if args.dry_run:
        print(json.dumps(spec, indent=2))
        return 0
    return run(spec)


if __name__ == "__main__":
    raise SystemExit(main())
