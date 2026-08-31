"""Add native Trackio tracking to slime.

Applied only for recipes that configure Trackio. The patch is idempotent and
keeps the existing W&B and TensorBoard paths intact.
"""

from __future__ import annotations

from pathlib import Path


SLIME_ROOT = Path("/root/slime")

TRACKIO_UTILS = '''"""Trackio helpers for distributed slime processes."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import urllib.request


_PROJECT = ""
_RUN_ID = ""
_RUN_NAME = ""
_SERVER_URL = ""
_WRITE_TOKEN = ""


def _enabled(args) -> bool:
    return bool(getattr(args, "use_trackio", False))


def _init(args) -> None:
    global _PROJECT, _RUN_ID, _RUN_NAME, _SERVER_URL, _WRITE_TOKEN

    _PROJECT = getattr(args, "trackio_project", "") or ""
    _RUN_ID = os.environ.get("TRAINING_GYM_TRAINING_RUN_ID", "")
    _RUN_NAME = os.environ.get("TRACKIO_RUN_NAME", "") or _RUN_ID
    _SERVER_URL = os.environ.get("TRACKIO_SERVER_URL", "").rstrip("/")
    _WRITE_TOKEN = os.environ.get("TRACKIO_WRITE_TOKEN", "")
    missing = [
        name
        for name, value in (
            ("trackio_project", _PROJECT),
            ("TRAINING_GYM_TRAINING_RUN_ID", _RUN_ID),
            ("TRACKIO_SERVER_URL", _SERVER_URL),
            ("TRACKIO_WRITE_TOKEN", _WRITE_TOKEN),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(f"Trackio configuration is missing: {', '.join(missing)}")
    args.trackio_run_name = _RUN_NAME


def init_trackio_primary(args) -> None:
    if not _enabled(args):
        args.trackio_run_name = None
        return
    _init(args)


def init_trackio_secondary(args) -> None:
    if not _enabled(args):
        return
    _init(args)


def _json_default(value):
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _log_id(step: int | None, key: str) -> str:
    raw = f"{_RUN_ID}\\0{step}\\0{key}".encode()
    return hashlib.sha256(raw).hexdigest()


def _bulk_log_payload(metrics, step: int | None = None) -> dict:
    return {
        "logs": [
            {
                "project": _PROJECT,
                "run": _RUN_NAME,
                "run_id": _RUN_ID,
                "metrics": {key: value},
                "step": step,
                "config": None,
                "log_id": _log_id(step, key),
            }
            for key, value in metrics.items()
        ],
        "hf_token": None,
    }


def _post(payload: dict) -> None:
    payload = json.dumps(
        payload,
        default=_json_default,
    ).encode()
    request = urllib.request.Request(
        f"{_SERVER_URL}/api/bulk_log",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "X-Trackio-Write-Token": _WRITE_TOKEN,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            response.read()
    except Exception:
        logging.getLogger(__name__).exception("Trackio bulk_log")


def log(metrics, step: int | None = None) -> None:
    if not metrics:
        return
    _post(_bulk_log_payload(metrics, step))


def finish() -> None:
    pass
'''


def _replace_once(path: Path, old: str, new: str) -> None:
    source = path.read_text()
    if new in source:
        return
    if old not in source:
        raise RuntimeError(f"Trackio patch marker not found in {path}: {old[:80]!r}")
    path.write_text(source.replace(old, new, 1))


def _patch_arguments() -> None:
    path = SLIME_ROOT / "slime/utils/arguments.py"
    source = path.read_text()
    if "def add_trackio_arguments(parser):" not in source:
        marker = "def add_wandb_arguments(parser):"
        marker_index = source.find(marker)
        if marker_index < 0:
            raise RuntimeError(f"Trackio patch marker not found in {path}: {marker!r}")
        line_start = source.rfind("\n", 0, marker_index) + 1
        indent = source[line_start:marker_index]
        block = (
            f"{indent}# trackio\n"
            f"{indent}def add_trackio_arguments(parser):\n"
            f'{indent}    parser.add_argument("--use-trackio", action="store_true", default=False)\n'
            f'{indent}    parser.add_argument("--trackio-project", type=str, default=None)\n'
            f"{indent}    return parser\n\n"
        )
        source = source[:line_start] + block + source[line_start:]

    call = "parser = add_wandb_arguments(parser)"
    trackio_call = "parser = add_trackio_arguments(parser)"
    if trackio_call not in source:
        call_index = source.find(call)
        if call_index < 0:
            raise RuntimeError(f"Trackio patch marker not found in {path}: {call!r}")
        line_end = source.find("\n", call_index)
        if line_end < 0:
            line_end = len(source)
        line_start = source.rfind("\n", 0, call_index) + 1
        indent = source[line_start:call_index]
        source = (
            source[: line_end + 1]
            + f"{indent}{trackio_call}\n"
            + source[line_end + 1 :]
        )
    path.write_text(source)


def _patch_logging_utils() -> None:
    path = SLIME_ROOT / "slime/utils/logging_utils.py"
    _replace_once(
        path,
        "from . import wandb_utils\n",
        "from . import trackio_utils, wandb_utils\n",
    )
    _replace_once(
        path,
        """def init_tracking(args, primary: bool = True, **kwargs):
    if primary:
        wandb_utils.init_wandb_primary(args, **kwargs)
    else:
        wandb_utils.init_wandb_secondary(args, **kwargs)
""",
        """def init_tracking(args, primary: bool = True, **kwargs):
    if primary:
        wandb_utils.init_wandb_primary(args, **kwargs)
        trackio_utils.init_trackio_primary(args)
    else:
        wandb_utils.init_wandb_secondary(args, **kwargs)
        trackio_utils.init_trackio_secondary(args)
""",
    )
    _replace_once(
        path,
        """def finish_tracking(args):
    if not args.use_wandb:
        return
    try:
        if wandb.run is not None:
            wandb.finish()
    except Exception:
        logging.getLogger(__name__).exception("Failed to finish wandb run")
""",
        """def finish_tracking(args):
    if args.use_wandb:
        try:
            if wandb.run is not None:
                wandb.finish()
        except Exception:
            logging.getLogger(__name__).exception("Failed to finish wandb run")
    if getattr(args, "use_trackio", False):
        trackio_utils.finish()
""",
    )
    _replace_once(
        path,
        """    if args.use_wandb:
        wandb.log(metrics)

    if args.use_tensorboard:
""",
        """    if args.use_wandb:
        wandb.log(metrics)
    if getattr(args, "use_trackio", False):
        trackio_utils.log(metrics, step=metrics.get(step_key))

    if args.use_tensorboard:
""",
    )


def main() -> None:
    utils_path = SLIME_ROOT / "slime/utils/trackio_utils.py"
    utils_path.write_text(TRACKIO_UTILS)
    _patch_arguments()
    _patch_logging_utils()
    print("Patched slime with native Trackio tracking")


if __name__ == "__main__":
    main()
