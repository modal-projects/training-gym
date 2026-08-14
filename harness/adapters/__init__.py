"""Env adapters — the ONE seam an agentic task plugs into.

An agentic task registers `adapter: harness/adapters/<env>.py` in its
tasks/<T>/task.yaml. The adapter is OPERATOR-owned, frozen, and pinned
(integrity TASK_DATA_KEYS); harness/rollout.py drives it. The contract is two
module-level functions — no classes, no registry:

    load_split(split_path) -> list[dict]
        Scenario rows, each with an "id" (str). Usually just json.load.

    run_episode(agent, row, cfg) -> dict
        Run EXACTLY ONE episode of `row`'s scenario against the submission
        policy and return:
            {"reward": float in [0, 1],   # the env's own verifier score
             "steps": int,                # actions the env actually executed
             "done": bool,                # env reached a terminal state
             "transcript": list,          # audit trail (actions/observations)
             "info": dict}                # env-specific extras
        `agent` is the object submission/agent.py build() returned. The adapter
        decides how to drive it: text envs wrap env.step in an execute()
        callback and call agent.act(instruction, execute, driver=...); tau2
        points its native orchestrator at agent.base_url/agent.model. Either
        way the adapter is the only bridge between policy and scored env.
        `cfg` is the resolved task config plus {"trial": int, "seed": int}.

Envs with a NATIVE batch runner (tau2's run_domain owns trials, concurrency,
and the user simulator) may expose run_split(agent, rows, cfg) -> {qid:
{"rewards": [...], "steps": [...], "done": [...][, "episodes"/"error"]}}
INSTEAD of run_episode — rollout.rollout_rows drives whichever the adapter
provides. Trials (for run_episode adapters), pass^k, artifact writing, and the
leaderboard all belong to harness/rollout.py — an adapter never grades itself
beyond the env's own reward.
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path


def load_adapter(root: Path, rel_path: str):
    """Import an adapter module from ROOT by file path (NOT a package import:
    verifying/running a copied tree must load THAT tree's adapter, mirroring
    integrity.judge_prompt_sha's reasoning). Validates the contract surface."""
    path = Path(root) / rel_path
    if not path.exists():
        raise SystemExit(f"[adapters] no adapter at {path}")
    spec = importlib.util.spec_from_file_location(
        f"_lab_adapter_{hashlib.sha256(str(path).encode()).hexdigest()[:12]}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not callable(getattr(mod, "load_split", None)):
        raise SystemExit(f"[adapters] {rel_path} lacks required function load_split()")
    if not (callable(getattr(mod, "run_episode", None))
            or callable(getattr(mod, "run_split", None))):
        raise SystemExit(f"[adapters] {rel_path} needs run_episode() (per-episode "
                         "envs) or run_split() (native batch runners)")
    return mod
