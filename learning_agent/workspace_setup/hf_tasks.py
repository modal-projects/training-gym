#!/usr/bin/env python3
"""Task-asset distribution via the HF dataset universal-learning-agent/tasks.

The git repo carries the code, configs, and small tracked task files; the
private HF dataset carries the rest — corpora and held-out gold — one
TOP-LEVEL <task>/ folder per task (the dataset's own README documents this
contract). fetch maps <task>/ -> workspace_setup/tasks/<task>/ so files land where the
codebase expects, and finishes with `bench.py verify`: the integrity pins
stay the arbiter of consistency.

    python3 workspace_setup/hf_tasks.py fetch --task fav2
    python3 workspace_setup/hf_tasks.py fetch --task fav2 --with-test
    python3 workspace_setup/hf_tasks.py upload --task tau2_banking
    python3 workspace_setup/hf_tasks.py upload            # every registered task
    python3 workspace_setup/hf_tasks.py list

HELD-OUT PROTECTION: fetch SKIPS <task>/test.json unless --with-test is
passed — the held-out sets are for the scoring operator only and must never
land where an agent workspace could be seeded from a tree that has them
casually lying around. upload DOES include test.json (the dataset is the
one complete home; it is private).

Auth (first match wins): HUGGINGFACEHUB_API_TOKEN (the .env is loaded
first, per the dataset README), then HF_TOKEN_JUNLIN, then HF_TOKEN. The
token must be scoped to the universal-learning-agent org (fine-grained
token -> Repositories permissions -> add the org with repo.content.read +
repo.write); a personal-namespace token 404s/403s on every call.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_ID = "universal-learning-agent/tasks"

sys.path.insert(0, str(ROOT / "harness"))


def _token() -> str:
    try:  # the repo's own .env loader (harness/envfile.py) — optional
        import envfile
        envfile.load_env(ROOT)
    except Exception:  # noqa: BLE001
        pass
    for var in ("HUGGINGFACEHUB_API_TOKEN", "HF_TOKEN_JUNLIN", "HF_TOKEN"):
        tok = os.environ.get(var, "")
        if tok:
            return tok
    raise SystemExit("no HF token: set HUGGINGFACEHUB_API_TOKEN in .env "
                     "(or export HF_TOKEN_JUNLIN / HF_TOKEN)")


def _api():
    try:
        from huggingface_hub import HfApi
    except ImportError:
        raise SystemExit("pip install huggingface_hub (or: uv run --with "
                         "huggingface_hub,pyyaml python3 workspace_setup/hf_tasks.py ...)")
    return HfApi(token=_token())


def registered_tasks() -> list[str]:
    """Tasks that exist (config present) AND have an asset dir to distribute."""
    from config import load_task, toolbox_config
    out = []
    for cfg in sorted((ROOT / "task_configs").glob("*.yaml")):
        tcfg = load_task(ROOT, cfg.stem)
        assets = toolbox_config(tcfg)["assets_task"]
        if (ROOT / "workspace_setup" / "tasks" / assets).is_dir() and assets not in out:
            out.append(assets)
    return out


def cmd_upload(tasks: list[str]) -> None:
    api = _api()
    api.create_repo(REPO_ID, repo_type="dataset", private=True, exist_ok=True)
    for t in tasks:
        src = ROOT / "workspace_setup" / "tasks" / t
        if not src.is_dir():
            raise SystemExit(f"no asset dir tasks/{t}")
        print(f"[upload] tasks/{t} -> {REPO_ID}:{t}/", file=sys.stderr)
        api.upload_folder(repo_id=REPO_ID, repo_type="dataset",
                          folder_path=str(src), path_in_repo=t,
                          commit_message=f"{t}: sync from operator checkout",
                          ignore_patterns=["__pycache__/*", ".DS_Store"])
    print(f"[upload] done: {len(tasks)} task(s)")


def cmd_fetch(tasks: list[str], with_test: bool) -> None:
    from huggingface_hub import snapshot_download
    patterns = [f"{t}/**" for t in tasks]
    ignore = None if with_test else [f"{t}/test.json" for t in tasks]
    print(f"[fetch] {REPO_ID} -> tasks/ ({patterns}"
          f"{', WITH held-out test.json' if with_test else ', test.json skipped'})",
          file=sys.stderr)
    snapshot_download(repo_id=REPO_ID, repo_type="dataset", token=_token(),
                      local_dir=str(ROOT / "workspace_setup" / "tasks"),
                      allow_patterns=patterns, ignore_patterns=ignore)
    # the arbiter of "consistent with the codebase": the integrity pins
    import subprocess
    subprocess.run([sys.executable, str(ROOT / "bench.py"), "verify"], cwd=ROOT)


def cmd_list() -> None:
    api = _api()
    files = api.list_repo_files(REPO_ID, repo_type="dataset")
    by_task: dict[str, int] = {}
    for f in files:
        parts = Path(f).parts
        if len(parts) >= 2:
            by_task[parts[0]] = by_task.get(parts[0], 0) + 1
    for t, n in sorted(by_task.items()):
        print(f"  {t}/: {n} files")
    print(f"{len(files)} files total")


def main() -> None:
    ap = argparse.ArgumentParser(description="Sync task assets with the HF dataset.")
    ap.add_argument("cmd", choices=["upload", "fetch", "list"])
    ap.add_argument("--task", action="append", default=[],
                    help="task asset dir(s); default: every registered task")
    ap.add_argument("--with-test", action="store_true",
                    help="fetch: ALSO pull the held-out test.json (scoring "
                         "operator only)")
    args = ap.parse_args()
    if args.cmd == "list":
        cmd_list()
        return
    tasks = args.task or registered_tasks()
    if args.cmd == "upload":
        cmd_upload(tasks)
    else:
        cmd_fetch(tasks, with_test=args.with_test)


if __name__ == "__main__":
    main()
