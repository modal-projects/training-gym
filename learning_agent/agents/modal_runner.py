"""lab-agent — Learning Agent contestant sessions in Modal containers.

One Modal app; every agent session is one container under it (spawned per
session by agents/run_sandbox_modal.sh, which prepares the session's subdir on
the agent-workspace volume first — this module is only the container side).

Volume layout ($MODAL_AGENT_VOLUME, default lab-agent-workspace):

    <task>/<session>/workspace   the agent's repo copy, mounted at
                                 /vol/<task>/<session>/workspace — everything
                                 the agent does in it persists on the volume
    <task>/<session>/logs        session home: CLI-native session state
                                 (opencode storage / claude session jsonl),
                                 obs watcher log, filled task spec
    (the task and its corpus ride inside each session's workspace at task/ —
     no shared state between sessions)

The container carries the `lab-agent-modal-token` secret so the agent's own
Modal GPU jobs (gpu_launcher.py / its own apps) and the live observatory
watcher work from inside; both target $MODAL_ENVIRONMENT.
"""
from __future__ import annotations

import os
from pathlib import Path

import modal

_HERE = Path(__file__).resolve().parent      # agents/
_REPO = _HERE.parent

AGENT_VOLUME = os.environ.get("MODAL_AGENT_VOLUME", "lab-agent-workspace")
OBS_VOLUME = os.environ.get("MODAL_OBS_VOLUME", "lab-observatory")
ENVIRONMENT = os.environ.get("MODAL_ENVIRONMENT", "lab-dev")

# Everything a contestant CLI needs, pinned: the opencode version is the one
# the modal_glm52 scaffold was verified against; the codex version is the one
# the codex_glm52 scaffold was verified against (0.145.0 on the operator Mac).
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl", "ca-certificates", "nodejs", "npm")
    .run_commands("npm install -g opencode-ai@1.18.8 @openai/codex@0.145.0")
    # placement salt: bump to change the image hash when the scheduler keeps
    # placing agent containers on a host with a broken network path to a team
    # endpoint (2026-08-13: repeated request-body truncation to the GLM
    # endpoint from the same warm pool; fresh-image sandboxes were clean)
    .env({"LEARNING_AGENT_IMAGE_SALT": "2026-08-13-a"})
    # httpx: agents/lib/responses_shim.py (the codex Responses->Chat bridge)
    .pip_install("pyyaml", "modal", "httpx==0.28.1")
    # Seed copies (read-only to the agent): the watcher must not be rewritable
    # by the process it reports on.
    .add_local_dir(str(_REPO / "observatory"), remote_path="/seed/observatory")
    .add_local_file(str(_HERE / "lib" / "container_entry.sh"),
                    remote_path="/seed/container_entry.sh")
)

volume = modal.Volume.from_name(AGENT_VOLUME, create_if_missing=True)
app = modal.App("lab-agent")


@app.function(
    image=image,
    volumes={"/vol": volume},
    secrets=[modal.Secret.from_name("lab-agent-modal-token")],
    timeout=24 * 60 * 60,   # Modal's ceiling; run_sandbox_modal.sh clamps the
    cpu=4,                  # budget so run.sh's own grace kill fires first
    memory=8192,
)
def run_session(task: str, session: str, scaffold: str = "codex_kimi3",
                hours: float = 23.5, model: str = "", track: str = "easy") -> dict:
    import subprocess
    import threading

    base = Path("/vol") / task / session
    ws, logs = base / "workspace", base / "logs"
    if not ws.is_dir():
        raise RuntimeError(f"no prepared workspace at {ws} — launch via agents/run_sandbox_modal.sh")
    logs.mkdir(parents=True, exist_ok=True)

    # The agent's `modal run` jobs and the watcher's uploads target the team env.
    os.environ["MODAL_ENVIRONMENT"] = ENVIRONMENT
    os.environ["MODAL_OBS_VOLUME"] = OBS_VOLUME

    # Long run: commit the volume periodically so progress is visible to the
    # operator (and survives a container crash), not only at function exit.
    stop = threading.Event()

    def _committer():
        while not stop.wait(120):
            try:
                volume.commit()
            except Exception:
                pass

    threading.Thread(target=_committer, daemon=True).start()

    # Own process group + SIGTERM forwarding: `modal container stop` signals
    # only this Python process — without forwarding, the bash/codex tree
    # ignores it and the container survives as a zombie for hours (observed
    # 2026-08-13). TERM the group, escalate to KILL after a grace period.
    import signal

    proc = subprocess.Popen(["bash", "/seed/container_entry.sh", scaffold, task,
                             str(hours), model, track, str(ws), str(logs), "/seed"],
                            start_new_session=True)

    def _forward_term(signum, frame):
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        def _escalate():
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        threading.Thread(target=_escalate, daemon=True).start()

    signal.signal(signal.SIGTERM, _forward_term)
    signal.signal(signal.SIGINT, _forward_term)

    rc = proc.wait()
    stop.set()
    volume.commit()

    status = {}
    runs_dir = ws / "agents" / "_runs"
    if runs_dir.is_dir():
        for st in sorted(runs_dir.glob("*/solve_status.txt")):
            status[st.parent.name] = st.read_text().strip().replace("\n", " ")
    return {"exit": rc, "session": f"{task}/{session}", "runs": status}
