#!/usr/bin/env python3
"""hf_tasks — task assets on the HuggingFace Hub (private dataset repo).

The git repo carries only the small tracked task files; the heavy and the
held-out assets (corpus/, dev.json, test.json) live in ONE private HF
dataset repo, mirrored per task:

    <task>/corpus/...   <task>/dev.json   <task>/test.json   <task>/task.md ...

Creating a task gains one step: author it under workspace_setup/tasks/<T>,
then `upload --task <T>`. A fresh checkout restores assets with `fetch`.

    python3 workspace_setup/hf_tasks.py upload                # every task
    python3 workspace_setup/hf_tasks.py upload --task fav2
    python3 workspace_setup/hf_tasks.py fetch --task fav2     # corpus + dev
    python3 workspace_setup/hf_tasks.py fetch --task fav2 --with-test

fetch never downloads test.json unless --with-test is passed: seeding
machines don't need it, only the scoring operator does. The repo id comes
from LEARNING_AGENT_TASKS_DATASET (default leonmodal/learning-agent-tasks);
the token from HUGGINGFACEHUB_API_TOKEN (env, then .env). The repo is
created private and STAYS private — it holds the held-out gold.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKS_DIR = ROOT / "workspace_setup" / "tasks"
DEFAULT_REPO = "universal-learning-agent/tasks"


def token() -> str:
    tok = os.environ.get("HUGGINGFACEHUB_API_TOKEN", "")
    if not tok and (ROOT / ".env").exists():
        for line in (ROOT / ".env").read_text().splitlines():
            if line.startswith("HUGGINGFACEHUB_API_TOKEN="):
                tok = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not tok:
        raise SystemExit("no HUGGINGFACEHUB_API_TOKEN in env or .env")
    return tok


def repo_id() -> str:
    return os.environ.get("LEARNING_AGENT_TASKS_DATASET", DEFAULT_REPO)


def api():
    from huggingface_hub import HfApi
    return HfApi(token=token())


def ensure_private_repo(a) -> None:
    from huggingface_hub.utils import RepositoryNotFoundError
    rid = repo_id()
    try:
        info = a.repo_info(rid, repo_type="dataset")
        if not info.private:
            raise SystemExit(f"{rid} exists but is PUBLIC — refusing to upload "
                             "held-out gold; make it private first")
    except RepositoryNotFoundError:
        a.create_repo(rid, repo_type="dataset", private=True)
        print(f"created private dataset {rid}")


def upload(tasks: list[str]) -> None:
    a = api()
    ensure_private_repo(a)
    for t in tasks:
        src = TASKS_DIR / t
        if not src.is_dir():
            raise SystemExit(f"no task assets at {src}")
        print(f"uploading {t} ...")
        a.upload_folder(repo_id=repo_id(), repo_type="dataset",
                        folder_path=str(src), path_in_repo=t,
                        commit_message=f"tasks: upload {t}",
                        ignore_patterns=["__pycache__/*", ".DS_Store"])
    print(f"done -> https://huggingface.co/datasets/{repo_id()}")


def fetch(task: str, with_test: bool) -> None:
    from huggingface_hub import snapshot_download
    patterns = [f"{task}/*"]
    ignore = [] if with_test else [f"{task}/test.json"]
    dest = TASKS_DIR
    snapshot_download(repo_id=repo_id(), repo_type="dataset", token=token(),
                      allow_patterns=patterns, ignore_patterns=ignore,
                      local_dir=str(dest))
    print(f"fetched {task} -> {dest / task}" + ("" if with_test else "  (test.json excluded)"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    up = sub.add_parser("upload", help="push task assets to the private dataset")
    up.add_argument("--task", default="", help="one task (default: every dir under workspace_setup/tasks)")
    fe = sub.add_parser("fetch", help="restore a task's assets from the dataset")
    fe.add_argument("--task", required=True)
    fe.add_argument("--with-test", action="store_true",
                    help="also download the held-out test.json (scoring operator only)")
    args = ap.parse_args()
    if args.cmd == "upload":
        tasks = [args.task] if args.task else sorted(
            p.name for p in TASKS_DIR.iterdir() if p.is_dir())
        upload(tasks)
    else:
        fetch(args.task, args.with_test)


if __name__ == "__main__":
    sys.exit(main())
