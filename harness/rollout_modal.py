#!/usr/bin/env python3
"""Modal execution of the agentic rollout — env deps + student serving in-container.

harness/rollout.py is the measurement logic (rows -> adapter episodes ->
judge-shaped artifacts); THIS file only supplies the place it runs for envs
with real dependencies: one Modal image per task (env pinned at build, mirrors
harness/eval.py's IMAGES pattern) over ONE shared serving stack (vLLM 0.25 with
Qwen3.5 tool-call parsing — env tasks drive the student through native tool
calls, so a server-side parser is mandatory), plus a local entrypoint that
gates integrity, ships rows in, and persists artifacts locally via
rollout.persist() — byte-identical artifact layout to a local rollout.

The submission contract is untouched: the container calls
submission/agent.py build(base_url=<in-container endpoint>) of the tree it
was launched from, so post-eval still drives exactly the policy the
contestant shipped; only the serving substrate is operator infrastructure.

    modal run harness/rollout_modal.py::rollout --task alfworld --split dev \
        --model Qwen/Qwen3.5-9B --limit 2 --allow-dirty
    modal run harness/rollout_modal.py::alfworld_splits   # (re)generate dev/test rows

bench.py `rollout`/`score` dispatch here automatically when a --model is given
(mock/--base-url runs stay local; see bench.py cmd_rollout).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:  # operator side: .env -> OPENAI_API_KEY etc. for the secrets built below.
    import envfile
    envfile.load_env(ROOT)
except ImportError:  # in-container: no .env; secrets arrive via Modal
    pass

app = modal.App("lab-rollout")
HF_CACHE = modal.Volume.from_name("lab-hf-cache", create_if_missing=True)
OUT = modal.Volume.from_name("lab-out", create_if_missing=True)
HF_CACHE_DIR = "/hf-cache"

ALFWORLD_PIN = "alfworld==0.4.2"           # keep == tasks/alfworld/task.yaml env.pin
ALFWORLD_DATA = "/alfworld-data"


def _base_image():
    """ONE serving stack for every env task: vLLM 0.25, the recipe validated
    for Qwen3.5 tool calling by the tau2 leaderboard baselines
    (agentic_catriges/cartridges/infra/modal_tau2_leaderboard.py). Env tasks
    drive the student through NATIVE tool calls, which needs a server-side
    tool-call parser — vLLM's `--tool-call-parser qwen3_coder` is the known-good
    one here, so both images share it rather than each picking a stack."""
    return (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("git", "curl", "ffmpeg")   # ffmpeg: vLLM 0.25 imports torchcodec
        .pip_install("vllm==0.25.0",
                     "huggingface_hub[hf_transfer]>=0.34.0,<2.0",
                     "requests", "pyyaml")
        .env({
            "HF_HOME": HF_CACHE_DIR,
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "VLLM_USE_V1": "1",
            "VLLM_USE_FLASHINFER_SAMPLER": "0",  # no nvcc JIT at runtime (vLLM 0.25)
            "TOKENIZERS_PARALLELISM": "true",
            "LITELLM_LOG": "ERROR",
        })
    )


def _with_repo(img):
    """The operator-side code an episode needs, mounted under /repo (the
    remote fn inserts these on sys.path). toolbox/ is mounted selectively:
    only the driver + client packages, not the vendored trainers."""
    return (img
            .add_local_dir(str(ROOT / "harness"), "/repo/harness")
            .add_local_dir(str(ROOT / "submission"), "/repo/submission")
            .add_local_dir(str(ROOT / "toolbox" / "harness_tool"),
                           "/repo/toolbox/harness_tool")
            .add_local_dir(str(ROOT / "toolbox" / "api_clients"),
                           "/repo/toolbox/api_clients")
            .add_local_dir(str(ROOT / "bench"), "/repo/bench"))


ALFWORLD_IMAGE = _with_repo(
    _base_image()
    .pip_install(ALFWORLD_PIN)
    .env({"ALFWORLD_DATA": ALFWORLD_DATA})
    # bake the pinned game data at build (layer-cached; text games only)
    .run_commands("alfworld-download")
    # claude CLI: lets --backend cli-claude run frontier REFERENCE baselines
    # against the env in-container (auth = CLAUDE_CODE_OAUTH_TOKEN secret)
    .run_commands("curl -fsSL https://claude.ai/install.sh | bash"
                  " && ln -sf /root/.local/bin/claude /usr/local/bin/claude")
)

# Operator-side claude-CLI credential for cli-claude reference baselines —
# score-time container only; prepare_workspace strips it from agent workspaces.
CLAUDE_SECRET = modal.Secret.from_dict(
    {"CLAUDE_CODE_OAUTH_TOKEN": __import__("os").environ.get("CLAUDE_CODE_OAUTH_TOKEN", "")})

# tau2: the base stack + the pinned repo cloned INTO the image (tau2 needs its
# data/ dir via TAU2_DATA_DIR). One image per env pin: airline/retail/telecom
# share 1901a301; banking_knowledge ships at aa74303c (its data changed after
# 1901a301 — see tasks/tau2_banking/task.yaml).
TAU2_PIN = "1901a301961cbbe3fd11f3e84a2a376530c759e3"   # == task.yaml env.pin
TAU2_BANKING_PIN = "aa74303ce5ff89a675297a3930b825bf1096f3fa"
TAU2_ROOT = "/workspace/tau2-bench"


def _tau2_image(pin: str, sandbox: bool = False):
    """sandbox=True adds the banking_knowledge shell-sandbox binaries (srt +
    ripgrep) — its default 'alltools' retrieval variant refuses to start
    without them."""
    img = _base_image()
    if sandbox:
        # srt needs bubblewrap + socat on Linux (macOS uses its native sandbox)
        img = img.apt_install("ripgrep", "bubblewrap", "socat",
                              "nodejs", "npm").run_commands(
            "npm install -g @anthropic-ai/sandbox-runtime@0.0.23")
    return _with_repo(
        img
        .run_commands(
            f"git clone https://github.com/sierra-research/tau2-bench {TAU2_ROOT}"
            f" && cd {TAU2_ROOT} && git checkout {pin}"
            f" && pip install -e '{TAU2_ROOT}[knowledge]'")
        .env({"PYTHONPATH": f"{TAU2_ROOT}/src", "TAU2_DATA_DIR": f"{TAU2_ROOT}/data"})
    )


TAU2_IMAGE = _tau2_image(TAU2_PIN)
TAU2_BANKING_IMAGE = _tau2_image(TAU2_BANKING_PIN, sandbox=True)

def _user_sim_url() -> str:
    """Where the customer simulator lives. Only the TOKEN needs to be in a
    .env: the URL is just where we deployed harness/user_sim.py, so ask Modal
    (LEARNING_AGENT_USER_SIM_URL still overrides, for a non-Modal deployment)."""
    import os
    url = os.environ.get("LEARNING_AGENT_USER_SIM_URL", "").rstrip("/")
    if url:
        return url
    try:
        return modal.Function.from_name(
            "lab-user-sim", "web").get_web_url().rstrip("/") + "/v1"
    except Exception:  # noqa: BLE001  not deployed / no creds -> preflight says so
        return ""


# How the rollout container reaches the customer simulator: the operator's
# user-sim service (harness/user_sim.py), which pins the model and holds the
# provider key. OPENAI_API_KEY is forwarded only when this entrypoint runs on a
# machine that has one (direct-to-provider fallback); it is empty, and unused,
# for a run launched from an agent workspace.
OPENAI_SECRET = modal.Secret.from_dict({
    **{k: __import__("os").environ.get(k, "")
       for k in ("OPENAI_API_KEY", "LEARNING_AGENT_USER_SIM_TOKEN", "LEARNING_AGENT_SESSION")},
    "LEARNING_AGENT_USER_SIM_URL": _user_sim_url()})


def _remote_rollout(task: str, rows: list, model: str, tcfg: dict, tp: int,
                    backend: str = "", base_url: str = ""):
    """Runs INSIDE the task image: serve student -> build submission agent ->
    adapter episodes. Returns (per_question, episodes). backend='cli-claude'
    skips serving and drives the claude CLI instead (reference baselines);
    base_url skips serving and drives an already-served endpoint (reference
    baselines against e.g. the team's GLM-5.2 — use the CPU binding)."""
    sys.path[:0] = ["/repo/harness", "/repo/submission", "/repo/toolbox"]
    import rollout as RO           # /repo/harness/rollout.py
    from adapters import load_adapter
    from agent import build        # /repo/submission/agent.py — THE submission
    proc = None
    try:
        if backend == "cli-claude":
            agent = build(backend="cli-claude", model=model)
        elif base_url:
            agent = build(base_url=base_url, model=model)
        else:
            base, proc = _serve_vllm_qwen(model, tp=tp)
            agent = build(base_url=base, model=model)
        adapter = load_adapter(Path("/repo"), tcfg["adapter"])
        episodes: list[tuple] = []
        per_question = RO.rollout_rows(
            rows, adapter, agent, tcfg,
            save_episode=lambda qid, t, ep: episodes.append((qid, t, ep)))
    finally:
        if proc is not None:
            proc.terminate()
    return per_question, episodes


def _serve_vllm_qwen(model: str, tp: int = 1, port: int = 8000,
                     max_model_len: int = 262144, max_num_seqs: int = 8):
    """THE serving command for every env task: the leaderboard-protocol vllm
    0.25 line for Qwen3.5. `--enable-auto-tool-choice --tool-call-parser
    qwen3_coder` is what makes driver: tools work at all — without it the
    model's calls come back as prose and never reach the environment.
    Returns (base_url, process)."""
    import subprocess
    import time
    import urllib.request
    # No --api-key: the endpoint is loopback-only inside the container, and a
    # configured key 401s every client that doesn't happen to send that exact
    # token (the submission's own OAIClient defaults to "EMPTY").
    cmd = ["vllm", "serve", model, "--served-model-name", model,
           "--host", "127.0.0.1", "--port", str(port),
           "--enable-prefix-caching", "--enable-auto-tool-choice",
           "--tool-call-parser", "qwen3_coder", "--reasoning-parser", "qwen3",
           "--additional-config", '{"gdn_prefill_backend": "triton"}',
           "--language-model-only", "--seed", "42",
           "--gpu-memory-utilization", "0.92",
           "--max-num-seqs", str(max_num_seqs),
           "--max-num-batched-tokens", "32768",
           "--max-model-len", str(max_model_len),
           "--tensor-parallel-size", str(tp)]
    proc = subprocess.Popen(cmd)
    base = f"http://127.0.0.1:{port}/v1"
    for _ in range(360):                      # up to 30 min (vllm cold start)
        if proc.poll() is not None:
            raise RuntimeError(f"vllm server exited rc={proc.returncode}")
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3)
            return base, proc
        except Exception:  # noqa: BLE001
            time.sleep(5)
    raise RuntimeError("student endpoint never came up")


# One Modal function per task so each binds its env-baked image (eval.py pattern).
@app.function(image=ALFWORLD_IMAGE, gpu="H200", timeout=180 * 60,
              volumes={HF_CACHE_DIR: HF_CACHE, "/out": OUT},
              secrets=[modal.Secret.from_name("huggingface-secret"), CLAUDE_SECRET])
def rollout_alfworld(rows: list, model: str, tcfg: dict, tp: int = 1,
                     backend: str = "", base_url: str = ""):
    return _remote_rollout("alfworld", rows, model, tcfg, tp,
                           backend=backend, base_url=base_url)


# cli-claude / external-endpoint episodes never touch the GPU: a separate
# CPU-only binding of the same image so reference baselines don't hold an
# idle H200 for hours.
@app.function(image=ALFWORLD_IMAGE, timeout=180 * 60,
              secrets=[modal.Secret.from_name("huggingface-secret"), CLAUDE_SECRET])
def rollout_alfworld_cpu(rows: list, model: str, tcfg: dict, tp: int = 1,
                         backend: str = "", base_url: str = ""):
    return _remote_rollout("alfworld", rows, model, tcfg, tp,
                           backend=backend, base_url=base_url)


def _tau2_episodes(rows: list, model: str, tcfg: dict, tp: int, backend: str,
                   base_url: str = ""):
    """Runs INSIDE a tau2 image: serve the student (or point tau2 at an
    already-served endpoint when base_url is given), hand tau2's native
    orchestrator the endpoint, collect judge-shaped episodes. Shared by every
    tau2_* task — the domain comes from tcfg, the env pin from the image."""
    if backend:
        raise RuntimeError("tau2 drives the policy through litellm (its native "
                           "orchestrator) — an external frontier policy needs a "
                           "real ANTHROPIC_API_KEY, not the claude CLI")
    sys.path[:0] = ["/repo/harness", "/repo/submission", "/repo/toolbox"]
    import rollout as RO
    from adapters import load_adapter
    proc = None
    try:
        from agent import build
        if base_url:
            agent = build(base_url=base_url, model=model)
        else:
            base, proc = _serve_vllm_qwen(model, tp=tp)
            agent = build(base_url=base, model=model)
        adapter = load_adapter(Path("/repo"), tcfg["adapter"])
        episodes: list[tuple] = []
        per_question = RO.rollout_rows(
            rows, adapter, agent, tcfg,
            save_episode=lambda qid, t, ep: episodes.append((qid, t, ep)))
    finally:
        if proc is not None:
            proc.terminate()
    return per_question, episodes


# One function per IMAGE (not per task): airline/retail/telecom share the
# 1901a301 env; banking binds its own pin's image.
@app.function(image=TAU2_IMAGE, gpu="H200", timeout=300 * 60,
              volumes={HF_CACHE_DIR: HF_CACHE, "/out": OUT},
              secrets=[modal.Secret.from_name("huggingface-secret"), OPENAI_SECRET])
def rollout_tau2(rows: list, model: str, tcfg: dict, tp: int = 1,
                 backend: str = "", base_url: str = ""):
    return _tau2_episodes(rows, model, tcfg, tp, backend, base_url=base_url)


@app.function(image=TAU2_BANKING_IMAGE, gpu="H200", timeout=300 * 60,
              volumes={HF_CACHE_DIR: HF_CACHE, "/out": OUT},
              secrets=[modal.Secret.from_name("huggingface-secret"), OPENAI_SECRET])
def rollout_tau2_banking(rows: list, model: str, tcfg: dict, tp: int = 1,
                         backend: str = "", base_url: str = ""):
    return _tau2_episodes(rows, model, tcfg, tp, backend, base_url=base_url)


ROLLOUT_FN = {"alfworld": rollout_alfworld,
              "tau2_airline": rollout_tau2,
              "tau2_retail": rollout_tau2,
              "tau2_telecom": rollout_tau2,
              "tau2_banking": rollout_tau2_banking}
CPU_ROLLOUT_FN = {"alfworld": rollout_alfworld_cpu}


@app.function(image=ALFWORLD_IMAGE, timeout=15 * 60)
def _list_alfworld_games() -> dict:
    """Enumerate the baked game files per upstream split (relative paths)."""
    import os
    root = Path(os.environ["ALFWORLD_DATA"])
    out = {}
    for split in ("valid_seen", "valid_unseen"):
        base = root / "json_2.1.1" / split
        out[split] = sorted(str(p.relative_to(root))
                            for p in base.rglob("*.tw-pddl"))
    return out


@app.local_entrypoint()
def alfworld_splits(dev_n: int = 50, test_n: int = 50, seed: int = 0):
    """(Re)generate tasks/alfworld/{dev,test}.json: a seeded sample of
    valid_seen -> dev and valid_unseen -> test. Pack files are pinned, so
    rerunning this is benchmark drift — freeze deliberately afterward."""
    import random
    games = _list_alfworld_games.remote()
    for name, n, src in (("dev", dev_n, "valid_seen"),
                         ("test", test_n, "valid_unseen")):
        files = games[src]
        picked = sorted(random.Random(seed).sample(files, min(n, len(files))))
        rows = [{"id": f"alfworld_{name}_{i:04d}",
                 "game_file": f,
                 "task_type": Path(f).parts[2].split("-")[0]}
                for i, f in enumerate(picked)]
        out = ROOT / "tasks" / "alfworld" / f"{name}.json"
        out.write_text(json.dumps(rows, indent=1) + "\n")
        print(f"[alfworld-splits] {name}: {len(rows)}/{len(files)} games "
              f"from {src} (seed {seed}) -> {out}")


@app.local_entrypoint()
def rollout(task: str, split: str = "dev", model: str = "", tag: str = "",
            config: str = "", limit: int = 0, no_record: bool = False,
            allow_dirty: bool = False, tp: int = 1, backend: str = "",
            base_url: str = ""):
    """Integrity-gate locally, run episodes in the task's container, persist
    judge-shaped artifacts locally (rollout.persist — identical layout)."""
    import config as labcfg
    import rollout as RO
    if task not in ROLLOUT_FN:
        raise SystemExit(f"[rollout-modal] no Modal image wired for task {task!r} "
                         f"(have: {sorted(ROLLOUT_FN)}) — self-contained envs "
                         "run locally via harness/rollout.py")
    if not model:
        raise SystemExit("[rollout-modal] --model required (weights to serve, "
                         "or a claude CLI alias with --backend cli-claude)")
    if backend == "cli-claude" and not no_record and not limit:
        # a frontier CLI policy is a REFERENCE, never a scoreable student (rule 6)
        print("[rollout-modal] cli-claude is a reference baseline: implying --no-record")
        no_record = True
    fn = ROLLOUT_FN[task]
    if backend == "cli-claude" or base_url:
        fn = CPU_ROLLOUT_FN.get(task, fn)   # no GPU held when nothing is served
    cfg = labcfg.load_config(ROOT)
    g = cfg["global"]
    if limit and not no_record:
        print(f"[rollout] --limit {limit} is a smoke test: implying --no-record")
        no_record = True
    integrity_status = RO.integrity_gate(allow_dirty, g)

    tcfg = labcfg.resolve(ROOT, task, config or None)
    if tcfg.get("archetype") != "agentic":
        raise SystemExit(f"[rollout-modal] task {task!r} is not agentic")
    problems = labcfg.validate_task(tcfg)
    if problems:
        raise SystemExit(f"[rollout-modal] invalid task config: {'; '.join(problems)}")

    # PRE-FLIGHT the user simulator BEFORE spawning a GPU container: the
    # adapter's own check runs in-container, i.e. after ~10 min of vLLM cold
    # start. Fail in one second with the fix instead.
    import os
    user_llm = (tcfg.get("env") or {}).get("user_llm")
    if user_llm and not (_user_sim_url() or os.environ.get("OPENAI_API_KEY")):
        raise SystemExit(
            f"[rollout-modal] {task} needs the {user_llm!r} customer simulator, "
            "but the user-sim service could not be reached.\n"
            "  Operator:  modal deploy harness/user_sim.py\n"
            "  Everyone:  LEARNING_AGENT_USER_SIM_TOKEN=<token> in .env (the URL resolves "
            "from Modal; set LEARNING_AGENT_USER_SIM_URL only for a non-Modal deployment).\n"
            "  The simulator is the same model for dev and scoring by design.")
    # the container has no repo tree beyond /repo mounts: inline the sys primer
    tcfg["_sys_text"] = (ROOT / tcfg["sys"]).read_text()

    from adapters import load_adapter
    adapter = load_adapter(ROOT, tcfg["adapter"])   # rows only; env imports stay lazy
    rows = adapter.load_split(ROOT / tcfg[split])
    if limit:
        rows = rows[:limit]
    tag = tag or f"{task}_{model.rstrip('/').split('/')[-1]}_{split}"
    # user-sim budget attribution: an agent session brings its own id (seeded by
    # prepare_workspace); an operator run is billed to its tag, so agent spend
    # can never exhaust the budget a scoring run needs.
    tcfg["_session"] = os.environ.get("LEARNING_AGENT_SESSION") or f"operator:{tag}"
    env = tcfg.get("env") or {}
    print(f"[rollout-modal] task={task} tag={tag} split={split} model={model} "
          f"n={len(rows)} num_trials={env.get('num_trials', 1)} "
          f"max_steps={env.get('max_steps')} integrity={integrity_status}")

    if backend == "cli-claude" and len(rows) > 8:
        # CLI episodes are slow (fresh `claude --print` per turn) and stateless
        # per row: fan out over parallel CPU containers instead of hitting the
        # single-container timeout. Chunks run on the DEPLOYED lab-rollout app
        # (`modal deploy harness/rollout_modal.py` first): ephemeral `modal
        # run` apps get stopped under long fan-outs, killing in-flight work —
        # deployed functions survive independent of this client app.
        import os
        fn_name = next(name for name, f in CPU_ROLLOUT_FN.items() if f is fn)
        dep = modal.Function.from_name(
            "lab-rollout", f"rollout_{fn_name}_cpu",
            environment_name=os.environ.get("MODAL_ENVIRONMENT"))
        per_question, episodes = {}, []
        pending = [rows[i:i + 5] for i in range(0, len(rows), 5)]
        for attempt in (1, 2):
            calls = [(c, dep.spawn(c, model, tcfg, tp, backend)) for c in pending]
            failed = []
            for chunk, call in calls:
                try:
                    pq, eps = call.get(timeout=3 * 3600)
                except Exception as e:  # noqa: BLE001  infra death, not episode failure
                    print(f"[rollout-modal] chunk {chunk[0]['id']}..: {type(e).__name__}")
                    failed.append(chunk)
                    continue
                per_question.update(pq)
                episodes.extend(eps)
            if not failed:
                break
            print(f"[rollout-modal] attempt {attempt}: {len(failed)} chunk(s) "
                  f"died to infra errors" + ("; retrying" if attempt == 1 else
                                             " — their rows are omitted"))
            pending = failed
        # re-key to row order for readability
        per_question = {str(r["id"]): per_question[str(r["id"])]
                        for r in rows if str(r["id"]) in per_question}
    else:
        per_question, episodes = fn.remote(rows, model, tcfg, tp, backend,
                                           base_url)
    episodes = [tuple(e) for e in episodes]
    RO.persist(task, split, tag, tcfg, g, per_question, episodes,
               integrity_status, limit, no_record,
               {"model": model,
                "backend": backend or ("endpoint" if base_url
                                       else "modal:sglang")})
