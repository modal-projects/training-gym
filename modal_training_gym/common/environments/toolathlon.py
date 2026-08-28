"""Live, decoupled Toolathlon environment built on the env base abstractions.

Toolathlon (https://toolathlon.xyz) is a long-horizon, tool-using agent
benchmark. This module turns it into a reusable RL environment with three
cooperating pieces, layered on the abstractions in :mod:`.base`:

- **Config** — :class:`ToolathlonEnvConfig` holds every path/port/image name so
  nothing is hardcoded at module scope.
- **Environment + pool** — :class:`ToolathlonEnvironment` (one live episode: a
  Modal sandbox running the MCP tool gateway, graded by Toolathlon's own
  evaluator) and :class:`ToolathlonEnvPool` (warm sandboxes + the directory
  snapshot library that restores the world to a given trajectory step).
- **Trajectory data** — :class:`ToolathlonTrajectoryDataset` loads expert
  trajectories (remapping the original workspace path onto ours) and the
  :func:`build_prefix_messages` / :func:`prune_prefix` helpers reconstruct the
  step-``K`` prompt prefix.

Scope ("Tier A"): process-only MCP servers whose entire state lives in the agent
workspace (``filesystem``, ``excel``, ``pdf-tools``, ``word``, ``pptx``,
``memory``, ``howtocook``, ``arxiv_local``, ``terminal``), so it can be captured
with a single directory snapshot. Tasks backed by external services (k8s,
WooCommerce/MySQL, Canvas/Postgres, email) are out of scope.

Nothing here is tutorial-specific: a caller supplies the task split, the
tokenizer/budget for pruning, and (optionally) the system prompt.
"""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Callable, Literal

from modal_training_gym.common.dataset import DatasetConfig, DatasetRow
from modal_training_gym.common.environments.base import (
    DirectorySnapshotLibrary,
    EvalVerdict,
    Observation,
    SandboxEnvironment,
    SandboxEnvironmentPool,
    StepResult,
    ToolCall,
    render_tool_catalog as render_tool_catalog,
    tool_schemas_to_openai as tool_schemas_to_openai,
)

# MCP servers whose state is confined to the agent workspace (snapshot-safe "Tier A").
TIER_A_MCPS = frozenset(
    {
        "filesystem",
        "excel",
        "pdf-tools",
        "word",
        "pptx",
        "memory",
        "howtocook",
        "arxiv_local",
        "terminal",
    }
)

# Terminal "done" action names the agent can emit to end an episode.
DONE_TOOLS = frozenset({"claim_done", "local-claim_done"})

# Agent-side aux tools (handled by ``dispatch_tool``, NOT exposed by the MCP
# gateway), so ``list_tools`` can't discover them — inject them into the catalog
# by hand so the model knows it can call them.
LOCAL_TOOL_SCHEMAS: dict[str, dict] = {
    "local-claim_done": {
        "description": "Signal the task is complete. Call only once the success condition is actually satisfied.",
        "parameters": {"type": "object", "properties": {}},
    },
    "local-python-execute": {
        "description": "Run a short Python script in the task workspace and return its stdout/stderr.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python source to execute."},
                "filename": {
                    "type": "string",
                    "description": "Optional script filename.",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (max 120).",
                },
            },
            "required": ["code"],
        },
    },
}


# ── Config ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ToolathlonEnvConfig:
    """All paths/ports/image names for one decoupled Toolathlon environment.

    The defaults reproduce the reference setup. ``snapshot_root`` is the overlay
    mount + directory-snapshot root, pinned to a fixed path (via the patched run
    config's ``direct_to_dumps`` + ``dump_path``) so the workspace path is
    deterministic across the snapshot build and the rollout/eval pool.
    """

    image: str = "lockon0927/toolathlon-task-image:1016beta"
    commit: str = "main"  # pin a SHA for reproducibility in a real run
    snapshot_root: str = "/task"
    workspace_subdir: str = "workspace"
    # The expert trajectories were generated with the agent workspace at this absolute path; our MCP
    # filesystem server is rooted at ``workspace_path``, so callers remap this prefix across a loaded
    # trajectory (golden call args, observations, task text) so replay + student calls resolve.
    orig_workspace: str = "/workspace/dumps/workspace"
    eval_config: str = (
        "scripts/formal_run_modal.json"  # patched run config (direct_to_dumps -> root)
    )
    venv: str = "/opt/toolathlon/.venv"  # uv-synced venv with Toolathlon's deps
    gateway_port: int = 8765
    snapshot_catalog: str = (
        "toolathlon-tierA-snapshots"  # modal.Dict: (task, K) -> snapshot
    )
    sandbox_app: str = "toolathlon-tierA-env"

    @property
    def workspace_path(self) -> str:
        """Where the MCP servers + evaluator operate (= ``snapshot_root/workspace``)."""
        return f"{self.snapshot_root}/{self.workspace_subdir}"

    @property
    def bundle_path(self) -> str:
        # Inside the overlay -> travels in the directory snapshot.
        return f"{self.workspace_path}/.toolathlon_bundle.json"

    @property
    def traj_log_path(self) -> str:
        # Synthesized dump_line consumed by container_eval; in the snapshot root so it travels too.
        return f"{self.snapshot_root}/traj_log.json"

    @property
    def pybin(self) -> str:
        return f"{self.venv}/bin/python"

    @property
    def gateway_sse_url(self) -> str:
        return f"http://127.0.0.1:{self.gateway_port}/sse"

    @property
    def gateway_health_url(self) -> str:
        return f"http://127.0.0.1:{self.gateway_port}/health"


DEFAULT_CONFIG = ToolathlonEnvConfig()


# ── In-sandbox snippets ─────────────────────────────────────────────────────
# Run inside the sandbox with the synced venv's interpreter (``config.pybin``).

# Call one tool through the gateway's MCP SSE transport using the official `mcp` client and print a
# parseable result line. The decoupled gateway is a standard MCP SSE server, so this is the canonical
# client path — shared by golden replay (snapshot build) and the live env pool.
_MCP_CALL_SNIPPET = (
    "import sys, json, asyncio\n"
    "from mcp.client.sse import sse_client\n"
    "from mcp import ClientSession\n"
    "async def _main():\n"
    "    url, name, args = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])\n"
    "    async with sse_client(url) as (r, w):\n"
    "        async with ClientSession(r, w) as s:\n"
    "            await s.initialize()\n"
    "            res = await s.call_tool(name, args)\n"
    "            out = {\n"
    "                'isError': bool(getattr(res, 'isError', False)),\n"
    "                'content': [getattr(c, 'text', str(c)) for c in (res.content or [])],\n"
    "            }\n"
    "            print('__RESULT__' + json.dumps(out))\n"
    "asyncio.run(_main())\n"
)

# Run the `local-python-execute` aux tool. Unlike the MCP server tools, the Toolathlon "local" tools
# are agent-side FunctionTools (not gateway-exposed), so we run them directly here, mirroring upstream:
# write the code under {ws}/.python_tmp and `uv run` it in the workspace. argv: ws, filename, b64(code), timeout.
_LOCAL_PY_SNIPPET = (
    "import os, sys, json, base64, subprocess\n"
    "ws, fn, b64, timeout = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])\n"
    "tmp = os.path.join(ws, '.python_tmp'); os.makedirs(tmp, exist_ok=True)\n"
    "open(os.path.join(tmp, fn), 'w', encoding='utf-8').write(base64.b64decode(b64).decode())\n"
    "try:\n"
    "    r = subprocess.run(f'uv run --directory {ws} ./.python_tmp/{fn}', shell=True, "
    "capture_output=True, text=True, timeout=timeout)\n"
    "    parts = []\n"
    "    if r.stdout: parts += ['=== STDOUT ===', r.stdout.rstrip()]\n"
    "    if r.stderr: parts += ['=== STDERR ===', r.stderr.rstrip()]\n"
    "    parts += ['=== EXECUTION INFO ===', f'Return code: {r.returncode}']\n"
    "    print('__RESULT__' + json.dumps({'isError': False, 'content': ['\\n'.join(parts)]}))\n"
    "except subprocess.TimeoutExpired:\n"
    "    print('__RESULT__' + json.dumps({'isError': False, "
    "'content': [f'=== EXECUTION TIMEOUT === after {timeout}s']}))\n"
)

# List the gateway's tool catalog over the MCP SSE transport. argv: gateway url.
_LIST_TOOLS_SNIPPET = (
    "import sys, json, asyncio\n"
    "from mcp.client.sse import sse_client\n"
    "from mcp import ClientSession\n"
    "async def _main():\n"
    "    url = sys.argv[1]\n"
    "    async with sse_client(url) as (r, w):\n"
    "        async with ClientSession(r, w) as s:\n"
    "            await s.initialize()\n"
    "            res = await s.list_tools()\n"
    "            tools = [{'name': t.name, 'description': t.description or '', "
    "'parameters': t.inputSchema or {}} for t in res.tools]\n"
    "            print('__TOOLS__' + json.dumps(tools))\n"
    "asyncio.run(_main())\n"
)


def _trajlog_snippet(config: ToolathlonEnvConfig) -> str:
    """Synthesize the dump_line (traj_log.json) that ``container_eval`` consumes.

    We don't run Toolathlon's host agent loop (the agent drives tools directly), so we reconstruct the
    same TaskConfig preprocess built and write {config, status=SUCCESS}. status=SUCCESS forces the
    workspace-vs-groundtruth check to run -> that pass/fail IS the terminal verdict. argv: task_dir, out_path.
    """
    return (
        "import sys, json, os\n"
        "from utils.task_runner.runner import TaskRunner\n"
        "from utils.data_structures.task_config import TaskConfig\n"
        "from utils.roles.task_agent import TaskStatus\n"
        "task_dir, out_path = sys.argv[1], sys.argv[2]\n"
        f"ec = json.load(open('{config.eval_config}'))\n"
        "agent_config = TaskRunner.load_configs(ec)[1]\n"
        "tc = TaskConfig.build(task_dir, agent_config.model.short_name, ec['global_task_config'], "
        "single_turn_mode=True, cn_mode=False)\n"
        "tc.log_file = os.path.join(tc.task_root, 'traj_log.json')\n"
        "tc.agent_workspace = os.path.join(tc.task_root, 'workspace')\n"
        "json.dump({'config': tc.to_dict(), 'status': TaskStatus.SUCCESS.value}, open(out_path, 'w'))\n"
        "print('__TRAJLOG_OK__')\n"
    )


def _patch_config_snippet(config: ToolathlonEnvConfig) -> str:
    # Patch the run config so task_root is exactly snapshot_root (direct_to_dumps + dump_path) — makes
    # the workspace path deterministic (no model-name/turn components) so builder + pool agree.
    return (
        "import json; d=json.load(open('scripts/formal_run_v0.json')); "
        "g=d.setdefault('global_task_config', {}); "
        f"g['dump_path']='{config.snapshot_root}'; g['direct_to_dumps']=True; "
        f"json.dump(d, open('{config.eval_config}','w'), indent=2)"
    )


# ── Tool dispatch ─────────────────────────────────────────────────────────


def _absolutize_excel_paths(
    config: ToolathlonEnvConfig, name: str, arguments: dict
) -> dict:
    """Rewrite relative spreadsheet paths to absolute for excel-* tools.

    The excel MCP server runs in stdio mode here, which requires *absolute* file paths; the expert
    trajectories pass workspace-relative names (the original run used excel in SSE mode).
    """
    if not name.startswith("excel-"):
        return arguments
    exts = (".xlsx", ".xls", ".xlsm", ".csv")

    def _fix(v):
        if isinstance(v, str) and not v.startswith("/") and v.lower().endswith(exts):
            return f"{config.workspace_path}/{v}"
        return v

    return {k: _fix(v) for k, v in arguments.items()}


def dispatch_tool(
    sandbox: Any, config: ToolathlonEnvConfig, action: ToolCall
) -> Observation:
    """Run one tool call inside ``sandbox`` and return an :class:`Observation`.

    ``local-*`` tools are the agent's aux tools (not gateway-exposed): ``claim_done`` is a no-op done
    signal, ``python-execute`` runs code in the workspace. Everything else goes through the MCP gateway
    over localhost.
    """
    name = action.name
    arguments = _absolutize_excel_paths(config, name, action.arguments or {})
    if name == "local-claim_done":
        return Observation(text="task marked done", is_error=False)
    if name == "local-python-execute":
        fn = str(arguments.get("filename") or "snippet.py")
        if not fn.endswith(".py"):
            fn += ".py"
        timeout = min(int(arguments.get("timeout", 30) or 30), 120)
        b64 = base64.b64encode(str(arguments.get("code", "")).encode()).decode()
        proc = sandbox.exec(
            config.pybin,
            "-c",
            _LOCAL_PY_SNIPPET,
            config.workspace_path,
            fn,
            b64,
            str(timeout),
        )
    else:
        proc = sandbox.exec(
            config.pybin,
            "-c",
            _MCP_CALL_SNIPPET,
            config.gateway_sse_url,
            name,
            json.dumps(arguments),
        )
    proc.wait()
    for line in proc.stdout.read().splitlines():
        if line.startswith("__RESULT__"):
            out = json.loads(line[len("__RESULT__") :])
            return Observation(
                text="\n".join(out.get("content", [])),
                is_error=bool(out.get("isError")),
            )
    return Observation(text=proc.stderr.read()[:500], is_error=True)


# ── Gateway lifecycle (inside the sandbox) ──────────────────────────────────


def _await_gateway(
    sandbox: Any, config: ToolathlonEnvConfig, attempts: int = 60
) -> bool:
    """Block until the in-sandbox gateway answers /health (servers spawned + ready)."""
    import time

    for _ in range(attempts):
        proc = sandbox.exec("curl", "-fsS", config.gateway_health_url)
        proc.wait()
        if proc.returncode == 0:
            return True
        time.sleep(1)
    return False


def _start_gateway(sandbox: Any, config: ToolathlonEnvConfig) -> None:
    """Start the MCP gateway process (spawns the stdio MCP servers rooted at the workspace)."""
    sandbox.exec(
        config.pybin,
        "-m",
        "scripts.decoupled.container_tool_gateway",
        "--bundle_file",
        config.bundle_path,
        "--host",
        "0.0.0.0",
        "--port",
        str(config.gateway_port),
    )  # background process inside the sandbox


def _seed_workspace(sandbox: Any, config: ToolathlonEnvConfig, task_dir: str) -> None:
    """Preprocess (seed workspace) + synthesize the eval dump_line + start the gateway.

    The gateway spawns the stdio MCP servers rooted at the workspace, so it must be started *after*
    the workspace exists. Both gateway and any in-sandbox client talk over localhost.
    """
    # Preprocess seeds the workspace and writes the bundle (into the workspace, so it travels in the
    # directory snapshot). direct_to_dumps in eval_config pins task_root == snapshot_root.
    proc = sandbox.exec(
        config.pybin,
        "-m",
        "scripts.decoupled.container_preprocess",
        "--eval_config",
        config.eval_config,
        "--task_dir",
        task_dir,
        "--bundle_file",
        config.bundle_path,
    )
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(
            f"preprocess failed for {task_dir}: {proc.stderr.read()[:800]}"
        )
    # Synthesize traj_log.json so container_eval can score the workspace (we drive tools ourselves).
    proc = sandbox.exec(
        config.pybin, "-c", _trajlog_snippet(config), task_dir, config.traj_log_path
    )
    proc.wait()
    if "__TRAJLOG_OK__" not in proc.stdout.read():
        raise RuntimeError(
            f"traj_log synthesis failed for {task_dir}: {proc.stderr.read()[:800]}"
        )
    _start_gateway(sandbox, config)
    if not _await_gateway(sandbox, config):
        raise RuntimeError("gateway did not become ready")


# ── Base image (built lazily, cached per config) ────────────────────────────

_IMAGE_CACHE: dict[ToolathlonEnvConfig, Any] = {}


def build_env_image(config: ToolathlonEnvConfig = DEFAULT_CONFIG) -> Any:
    """The CPU image: Toolathlon task image + decoupled scripts + uv-locked Python deps.

    The base image ships node+npm (MCP servers run via npx) and `uv`, but *no* Python packages — the
    decoupled gateway/preprocess/eval expect a `uv sync`'d venv. The lockfile has no torch/GPU deps,
    so this stays lightweight. Cached per config to avoid redundant builds.
    """
    import modal

    if config not in _IMAGE_CACHE:
        _IMAGE_CACHE[config] = (
            modal.Image.from_registry(config.image)
            .apt_install("git", "curl")
            .run_commands(
                f"git clone --depth 1 -b {config.commit} "
                "https://github.com/hkust-nlp/Toolathlon /opt/toolathlon",
                # Materialize Toolathlon's locked deps into the venv (gateway needs aiohttp-sse/pyyaml/
                # mcp; eval needs openai-agents; MCP servers are pre-fetched).
                "cd /opt/toolathlon && /root/.local/bin/uv sync --frozen",
                # Decoupled scripts import configs.global_configs / token_key_session — the repo ships
                # only *_example.py; copy them (placeholder keys are fine for Tier A, no external APIs).
                "cp /opt/toolathlon/configs/global_configs_example.py /opt/toolathlon/configs/global_configs.py",
                "cp /opt/toolathlon/configs/token_key_session_example.py /opt/toolathlon/configs/token_key_session.py",
                f'cd /opt/toolathlon && {config.pybin} -c "{_patch_config_snippet(config)}"',
            )
            # PYTHONPATH so `scripts`/`utils` import; VIRTUAL_ENV pins the venv; PATH puts venv + uv first.
            .env(
                {
                    "PYTHONPATH": "/opt/toolathlon",
                    "VIRTUAL_ENV": config.venv,
                    "PATH": f"{config.venv}/bin:/root/.local/bin:/usr/local/sbin:"
                    "/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
                }
            )
            .workdir("/opt/toolathlon")
        )
    return _IMAGE_CACHE[config]


# ── Environment + pool ──────────────────────────────────────────────────────


class ToolathlonEnvironment(SandboxEnvironment):
    """One live Toolathlon episode backed by a Modal sandbox running the MCP gateway."""

    def __init__(
        self, sandbox: Any, config: ToolathlonEnvConfig, task_name: str
    ) -> None:
        super().__init__(sandbox)
        self.config = config
        self.task_name = task_name

    def step(self, action: ToolCall) -> StepResult:
        """Execute one tool call against the live MCP server; ``done`` on a terminal action."""
        obs = dispatch_tool(self.sandbox, self.config, action)
        return StepResult(observation=obs, done=action.name in DONE_TOOLS)

    def list_tools(self) -> dict[str, dict]:
        """The agent's tool catalog: live gateway tools + the agent-side aux tools.

        Returns ``{name: {description, parameters}}`` — the shape
        :func:`tool_schemas_to_openai` consumes — so a live rollout can hand the
        model the exact catalog without needing an expert trajectory.
        """
        proc = self.sandbox.exec(
            self.config.pybin, "-c", _LIST_TOOLS_SNIPPET, self.config.gateway_sse_url
        )
        proc.wait()
        schemas: dict[str, dict] = {}
        for line in proc.stdout.read().splitlines():
            if line.startswith("__TOOLS__"):
                for tool in json.loads(line[len("__TOOLS__") :]):
                    schemas[tool["name"]] = {
                        "description": tool["description"],
                        "parameters": tool["parameters"],
                    }
                break
        schemas.update(LOCAL_TOOL_SCHEMAS)
        return schemas

    def evaluate(self) -> EvalVerdict:
        """Score the restored workspace via Toolathlon's ``container_eval`` (exit 0 == pass).

        container_eval reads the synthesized traj_log (status=SUCCESS) and runs the task's
        workspace-vs-groundtruth check. Upstream ends with ``raise SystemExit(0 if pass else 1)``, so
        the exit code IS the verdict. It prints "Evaluation finished." just before exiting; a non-zero
        exit WITHOUT that marker means the evaluator itself crashed (bad bundle / harness error), not a
        legitimate task failure — flagged via ``harness_error`` so it isn't scored as a task failure.
        """
        proc = self.sandbox.exec(
            self.config.pybin,
            "-m",
            "scripts.decoupled.container_eval",
            "--bundle_file",
            self.config.bundle_path,
        )
        proc.wait()
        passed = proc.returncode == 0
        out = proc.stdout.read()
        if passed:
            return EvalVerdict(passed=True)
        if "Evaluation finished." not in out:
            return EvalVerdict(
                passed=False,
                harness_error=True,
                detail=f"container_eval crashed (rc={proc.returncode}): {proc.stderr.read()[:500]}",
            )
        detail = " | ".join(
            ln.strip()
            for ln in out.splitlines()
            if ln.startswith(("Details:", "Failure:"))
        )
        return EvalVerdict(passed=False, detail=detail[:500])


class ToolathlonEnvPool(SandboxEnvironmentPool):
    """Per-acquire sandbox pool that restores ``(task, K)`` from a directory snapshot.

    Each :meth:`acquire` creates a fresh sandbox from the cached image, mounts the ``(task, K)``
    snapshot, and starts the gateway. We do *not* reuse sandboxes via ``unmount_image`` — unmounting an
    in-use snapshot mount fails on Modal's backend with an InternalError (same beta limitation as the
    mount-of-a-plain-dir bug) — so :meth:`release` terminates the sandbox; creation from the cached
    image (no rebuild) is cheap.
    """

    def __init__(self, config: ToolathlonEnvConfig = DEFAULT_CONFIG) -> None:
        super().__init__()
        self.config = config
        self.snapshots = DirectorySnapshotLibrary(
            config.snapshot_catalog, config.snapshot_root
        )

    @property
    def app_name(self) -> str:
        return self.config.sandbox_app

    def build_image(self) -> Any:
        return build_env_image(self.config)

    def acquire(self, task_name: str, k: int) -> ToolathlonEnvironment:
        sandbox = self.create_sandbox(timeout=60 * 30, cpu=2.0, memory=4096)
        # Mount the (task, K) directory snapshot (workspace + bundle + traj_log) and start the gateway
        # (a process, not captured by the snapshot).
        self.snapshots.mount(sandbox, task_name, k)
        _start_gateway(sandbox, self.config)
        _await_gateway(sandbox, self.config, attempts=40)
        return ToolathlonEnvironment(sandbox, self.config, task_name)


# Lazily-created module-level pool, shared across rollouts in a worker.
_ENV_POOL_CACHE: dict[ToolathlonEnvConfig, ToolathlonEnvPool] = {}


def get_env_pool(config: ToolathlonEnvConfig = DEFAULT_CONFIG) -> ToolathlonEnvPool:
    """Process-wide :class:`ToolathlonEnvPool` for ``config`` (created lazily)."""
    if config not in _ENV_POOL_CACHE:
        _ENV_POOL_CACHE[config] = ToolathlonEnvPool(config)
    return _ENV_POOL_CACHE[config]


# ── Snapshot library builder (offline, once) ────────────────────────────────


def build_task_snapshots(
    task_name: str,
    golden_calls: list[dict],
    config: ToolathlonEnvConfig = DEFAULT_CONFIG,
) -> int:
    """Snapshot the workspace at every step K of one task's golden trajectory.

    K=0 is the pristine post-preprocess state; K=n is after the n-th golden call. Replaying a golden
    call advances env state only — the prompt text still comes from the original trace.
    """
    import modal

    app = modal.App.lookup(config.sandbox_app, create_if_missing=True)
    sandbox = modal.Sandbox._experimental_create(
        "sleep", "infinity", image=build_env_image(config), app=app, timeout=60 * 60
    )
    snapshots = DirectorySnapshotLibrary(config.snapshot_catalog, config.snapshot_root)

    def _seed(sb: Any) -> None:
        _seed_workspace(sb, config, f"finalpool/{task_name}")

    def _advance(sb: Any, step: int) -> None:
        call = golden_calls[step]
        obs = dispatch_tool(
            sb, config, ToolCall(call["name"], call.get("arguments", {}))
        )
        if (
            obs.is_error
        ):  # surface a broken gateway/tool immediately, don't emit identical snapshots
            raise RuntimeError(
                f"golden replay '{call['name']}' errored: {obs.text[:500]}"
            )

    try:
        return snapshots.build_item(
            sandbox, task_name, len(golden_calls), _seed, _advance
        )
    finally:
        sandbox.terminate()


# Parallel builder: a lightweight CPU Modal Function that builds one task's snapshots (spawning its own
# env sandbox). Mapping it over the tasks builds them all concurrently.
def _builder_app():
    import modal

    return modal.App(f"{DEFAULT_CONFIG.sandbox_app}-builder")


def build_snapshot_library(
    task_to_golden: dict[str, list[dict]], config: ToolathlonEnvConfig = DEFAULT_CONFIG
) -> None:
    """Build the ``(task, K)`` snapshot catalog, one Modal container per task, in parallel.

    Resumable and self-healing: a task is skipped only if *every* snapshot K = 0..len(golden) is
    still cataloged (catalog entries expire after 7 days of inactivity — see
    ``DirectorySnapshotLibrary.missing_steps``, whose reads also refresh the surviving entries'
    TTLs). Tasks with any expired/missing step are rebuilt in full.
    """
    import modal

    snapshots = DirectorySnapshotLibrary(config.snapshot_catalog, config.snapshot_root)
    pending = []
    for task_name, golden_calls in task_to_golden.items():
        missing = snapshots.missing_steps(task_name, len(golden_calls))
        if not missing:
            print(
                f"  {task_name}: all {len(golden_calls) + 1} snapshots cataloged (TTLs refreshed), skipping"
            )
        else:
            print(
                f"  {task_name}: {len(missing)}/{len(golden_calls) + 1} snapshots missing, rebuilding"
            )
            pending.append((task_name, golden_calls))
    if not pending:
        return

    app = modal.App(f"{config.sandbox_app}-builder")

    @app.function(
        image=modal.Image.debian_slim(python_version="3.12")
        .pip_install("modal~=1.4.3")
        .add_local_python_source("modal_training_gym", copy=True),
        timeout=60 * 60,
        max_containers=16,
        serialized=True,
    )
    def _build_one(item: tuple) -> tuple:
        task_name, golden_calls = item
        return task_name, build_task_snapshots(task_name, golden_calls, config)

    print(f"  building {len(pending)} task(s) in parallel...")
    with app.run():
        for task_name, n in _build_one.map(pending):
            print(f"  {task_name}: cataloged {n} (task, K) snapshots")


# ── Trajectory dataset ──────────────────────────────────────────────────────


class ToolathlonTrajectoryDataset(DatasetConfig):
    """Expert Toolathlon trajectories as a one-row-per-task training dataset.

    Each row carries the compact golden trajectory in its ``label``; the rollout/eval pick a start step
    K at call time and rebuild the step-K prefix via :func:`build_prefix_messages` / :func:`prune_prefix`.
    The original workspace path in the raw trace is remapped onto ``config.workspace_path`` so replay
    and student tool calls resolve.

    ``train_tasks`` and ``eval_tasks`` are task-name allowlists for their
    respective instances. An empty allowlist includes every task.
    """

    hf_repo: str = "hkust-nlp/Toolathlon-Trajectories"
    source_file: str = "deepseek-v3.2-exp_1.jsonl"
    train_tasks: tuple[str, ...] = ()
    eval_tasks: tuple[str, ...] = ()
    obs_limit: int = 1500  # truncate each observation to N chars in context
    writes_eval_paths = False

    def __init__(
        self,
        split: Literal["all", "train", "eval"] = "train",
        config: ToolathlonEnvConfig = DEFAULT_CONFIG,
        *,
        hf_repo: str | None = None,
        source_file: str | None = None,
        train_tasks: tuple[str, ...] | None = None,
        eval_tasks: tuple[str, ...] | None = None,
        obs_limit: int | None = None,
    ) -> None:
        self.split = split
        # Keep this while framework path resolution still uses ``hf_split``.
        self.hf_split = split
        self.config = config
        if hf_repo is not None:
            self.hf_repo = hf_repo
        if source_file is not None:
            self.source_file = source_file
        if train_tasks is not None:
            self.train_tasks = train_tasks
        if eval_tasks is not None:
            self.eval_tasks = eval_tasks
        if obs_limit is not None:
            self.obs_limit = obs_limit
        source = self.source_file.removesuffix(".jsonl")
        self.id = f"toolathlon-{source}-{split}-{self.cache_key}"
        super().__init__()

    input_key = "messages"
    label_key = "label"
    output_format: Literal["jsonl"] = "jsonl"

    @property
    def cache_key(self) -> str:
        payload = json.dumps(
            {
                "eval_tasks": self.eval_tasks,
                "hf_repo": self.hf_repo,
                "obs_limit": self.obs_limit,
                "orig_workspace": self.config.orig_workspace,
                "source_file": self.source_file,
                "split": self.split,
                "train_tasks": self.train_tasks,
                "workspace_path": self.config.workspace_path,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:8]

    def _tasks_for_split(self) -> set[str]:
        if self.split == "eval":
            return set(self.eval_tasks)
        if self.split == "train":
            return set(self.train_tasks)
        return set(self.train_tasks) | set(self.eval_tasks)

    def _load_trajectories(self, task_names: set[str] | None = None) -> list[dict]:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id=self.hf_repo, filename=self.source_file, repo_type="dataset"
        )
        keep = self._tasks_for_split() if task_names is None else task_names
        trajectories = []
        with open(path) as f:
            for line in f:
                traj = json.loads(line)
                if keep and traj.get("task_name") not in keep:
                    continue
                # Remap the original workspace prefix -> our workspace_path on the raw serialized
                # trajectory (covers golden-call args, observations, and the task text in one pass,
                # including arguments stored as nested JSON strings).
                raw_msgs = (traj.get("messages") or "[]").replace(
                    self.config.orig_workspace, self.config.workspace_path
                )
                msgs = json.loads(raw_msgs)
                if len(msgs) < 3:
                    continue
                trajectories.append(
                    {
                        "task_name": traj["task_name"],
                        "messages": msgs,
                        "tool_calls_meta": json.loads(traj.get("tool_calls", "{}"))
                        if traj.get("tool_calls")
                        else {},
                    }
                )
        return trajectories

    def _golden_calls(self, msgs: list[dict]) -> list[dict]:
        """The ordered list of expert (golden) tool calls in a trajectory."""
        calls = []
        for m in msgs:
            if m.get("role") == "assistant" and m.get("tool_calls"):
                tc = m["tool_calls"]
                if tc and isinstance(tc[0], dict) and "function" in tc[0]:
                    fn = tc[0]["function"]
                    args = fn.get("arguments", "{}")
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}
                    calls.append({"name": fn["name"], "arguments": args})
        return calls

    def _observations(self, msgs: list[dict]) -> list[str]:
        """Ordered tool observations (one per executed call), truncated to ``obs_limit``."""
        return [
            str(m.get("content", ""))[: self.obs_limit]
            for m in msgs
            if m.get("role") == "tool"
        ]

    def _task_request(self, msgs: list[dict]) -> str:
        for m in msgs:
            if m.get("role") == "user" and str(m.get("content", "")).strip():
                return str(m["content"])
        return ""

    def _tool_schemas(self, trajectory: dict) -> dict:
        # The exact tool catalog the expert was given via the OpenAI tools= param (gateway tools/list).
        # name -> {description, parameters} so the prompt can show which tools exist + their args.
        schemas = {}
        for t in trajectory.get("tool_calls_meta", {}).get("tools", []):
            if isinstance(t, dict) and "function" in t:
                fn = t["function"]
                schemas[fn["name"]] = {
                    "description": fn.get("description", ""),
                    "parameters": fn.get("parameters", {}),
                }
        return schemas

    def _make_row(self, trajectory: dict) -> dict:
        msgs = trajectory["messages"]
        golden = self._golden_calls(msgs)
        task_user = self._task_request(msgs)
        label = json.dumps(
            {
                "task_name": trajectory["task_name"],
                "total_steps": len(golden),
                "task_request": task_user,
                "golden_calls": golden,
                "observations": self._observations(msgs),
                "tool_schemas": self._tool_schemas(trajectory),
            }
        )
        # Minimal messages so the framework can load/tokenize the row; the rollout rebuilds the real
        # step-K prefix from the label per the curriculum (sample.prompt is not used directly).
        messages = [
            {
                "role": "system",
                "content": "You are a tool-calling agent. Output a JSON tool call.",
            },
            {"role": "user", "content": task_user},
        ]
        return {"messages": messages, "label": label}

    def rows(self) -> Iterable[DatasetRow]:
        for trajectory in self._load_trajectories():
            # Skip degenerate trajectories with no tool calls.
            if self._golden_calls(trajectory["messages"]):
                yield self._make_row(trajectory)

    def golden_by_task(self) -> dict[str, list[dict]]:
        """Map every configured task to golden calls for snapshot construction."""
        task_names = set(self.train_tasks) | set(self.eval_tasks)
        return {
            t["task_name"]: self._golden_calls(t["messages"])
            for t in self._load_trajectories(task_names)
        }


# ── Prompt-prefix reconstruction ────────────────────────────────────────────

DEFAULT_SYSTEM_PROMPT = """\
You are a tool-using agent completing a task against real tools. Work one step at a time.

Rules:
- Make EXACTLY ONE tool call per turn. Emit only the tool call — no extra prose, narration, or markdown fences around it.
- Use only the tools provided to you, with their exact names. Do not invent tools, arguments, or file paths.
- Pass file paths exactly as given in the task; inspect files (list/read) before editing or overwriting them.
- After each tool result, check whether it succeeded before continuing; do not blindly repeat a failed call with the same arguments.
- Call `local-claim_done` only once the task's success condition is actually satisfied.

Emit every tool call in exactly this format — a single <function> block wrapped in <tool_call>, with one <parameter> block per argument. For example, a real call that writes a JSON file (content abbreviated with … here) looks like:

<tool_call>
<function=filesystem-write_file>
<parameter=path>
train-ticket-plan.json
</parameter>
<parameter=content>
{"thursday": {"train number": "G385", "departure station": "Beijing South", "arrival station": "Qufu East", …}}
</parameter>
</function>
</tool_call>

Use the actual tool name, parameters, and full (untruncated) values required by your task; the call above is only an illustration of the wire format."""


def default_system_prompt(tool_schemas: dict) -> str:
    """Behavioral rules only — the tool catalog travels via the chat template's
    ``tools=`` parameter (see :func:`tool_schemas_to_openai`), not prompt text."""
    return DEFAULT_SYSTEM_PROMPT


def build_prefix_messages(
    label: dict,
    k: int,
    *,
    obs_limit: int = 1500,
    system_prompt: Callable[[dict], str] = default_system_prompt,
) -> list[dict]:
    """Rebuild the step-K prompt prefix from the compact trajectory in the dataset ``label``.

    system + task request + interleaved [assistant(golden call_i as a native ``tool_calls``
    entry), tool(obs_i)] for i in 0..k-1. Golden calls are structured ``tool_calls`` (not
    JSON text) so the chat template renders them in the model's native tool-call wire
    format — the same format the model is post-trained to emit — and observations are
    ``role: "tool"`` turns. Render with ``apply_chat_template(msgs,
    tools=tool_schemas_to_openai(label["tool_schemas"]), ...)`` (or pass both straight to a
    chat-completions request). We keep only the structured tool calls (no expert prose) and
    truncate each observation to ``obs_limit`` — matching how the prompt was constructed.
    """
    golden = label.get("golden_calls", [])
    obs = label.get("observations", [])
    msgs = [
        {"role": "system", "content": system_prompt(label.get("tool_schemas", {}))},
        {"role": "user", "content": label.get("task_request", "")},
    ]
    for i in range(min(k, len(golden))):
        call = golden[i] or {}
        msgs.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": f"call_{i}",
                        "type": "function",
                        "function": {
                            "name": call.get("name", ""),
                            "arguments": call.get("arguments") or {},
                        },
                    }
                ],
            }
        )
        if i < len(obs):
            msgs.append(
                {
                    "role": "tool",
                    "tool_call_id": f"call_{i}",
                    "content": str(obs[i])[:obs_limit],
                }
            )
    return msgs


def prune_prefix(
    messages: list[dict],
    *,
    budget: int,
    count_tokens: Callable[[str], int] | None = None,
    keep_recent: int = 6,
    old_call_chars: int = 200,
) -> list[dict]:
    """Recency-decayed prune of a Toolathlon prefix to fit ``budget`` content tokens.

    The prefix bulk lives in two places, both in the OLD steps: bulky tool **observations**
    (read-heavy tasks) and the bulky **arguments of old tool calls** (write/code-heavy tasks). Since
    the snapshot restores world state, old history only needs to preserve the action *trace* + recent
    context. So we always keep the system message + the task request + the last ``keep_recent``
    messages verbatim, then reclaim budget from the OLDEST steps in priority order:

    1. drop old observations (stale, bulky, re-readable via a tool);
    2. truncate old tool-call arguments to ``old_call_chars`` (keep the action name + a head like the
       file path — the cheap trace — and shed the big embedded scripts/data);
    3. as a last resort, drop old tool calls entirely.

    ``count_tokens`` measures a string's token length (pass the model tokenizer for CJK-accurate
    sizing); it falls back to a ~3.5 chars/token heuristic. ``budget`` should already account for
    chat-template overhead added on top of message content.
    """
    if count_tokens is None:

        def count_tokens(s: str) -> int:  # noqa: E306
            return int(len(str(s)) / 3.5)

    def _msg_text(m: dict) -> str:
        text = str(m.get("content", "") or "")
        for tc in m.get("tool_calls") or []:
            text += json.dumps(tc.get("function", {}))
        return text

    sizes = [count_tokens(_msg_text(m)) for m in messages]
    if sum(sizes) <= budget:
        return messages

    # Protected head: leading system message(s) + the first user (task) message.
    head = 0
    while head < len(messages) and messages[head].get("role") == "system":
        head += 1
    if head < len(messages) and messages[head].get("role") == "user":
        head += 1
    tail_start = max(head, len(messages) - keep_recent)
    old = list(range(head, tail_start))

    def _is_obs(m):
        return m.get("role") == "tool" or (
            m.get("role") == "user"
            and str(m.get("content", "")).startswith("<observation>")
        )

    def _truncate_assistant(m: dict) -> dict:
        """Shrink an old tool call to its head: name + the first ``old_call_chars`` of args."""
        if m.get("tool_calls"):
            calls = []
            for tc in m["tool_calls"]:
                fn = dict(tc.get("function", {}))
                args = fn.get("arguments", {})
                blob = args if isinstance(args, str) else json.dumps(args)
                if len(blob) > old_call_chars:
                    # truncated args are no longer valid JSON; keep them as a string trace
                    fn["arguments"] = blob[:old_call_chars] + " …"
                calls.append({**tc, "function": fn})
            return {**m, "content": "", "tool_calls": calls}
        c = str(m.get("content", ""))
        return (
            {**m, "content": c[:old_call_chars] + " …"}
            if len(c) > old_call_chars
            else m
        )

    cur = sum(sizes)
    to_drop: set[int] = set()
    truncated: dict[int, dict] = {}

    # Tier 1: drop old observations (oldest first).
    for i in old:
        if cur <= budget:
            break
        if _is_obs(messages[i]):
            cur -= sizes[i]
            to_drop.add(i)
    # Tier 2: truncate old tool-call arguments (oldest first), preserving the name + head.
    for i in old:
        if cur <= budget:
            break
        if i in to_drop or messages[i].get("role") != "assistant":
            continue
        new = _truncate_assistant(messages[i])
        cur -= sizes[i] - count_tokens(_msg_text(new))
        truncated[i] = new
    # Tier 3 (last resort): drop old tool calls entirely (oldest first).
    for i in old:
        if cur <= budget:
            break
        if i in to_drop or messages[i].get("role") != "assistant":
            continue
        cur -= count_tokens(_msg_text(truncated[i])) if i in truncated else sizes[i]
        to_drop.add(i)

    out = []
    for i, msg in enumerate(messages):
        if i in to_drop:
            continue
        if i in truncated:
            msg = truncated[i]
        out.append(msg)
    return out
