# gpu_tools — Modal GPUs for training, serving, and debugging

Two scripts and three docs:

- `gpu_launcher.py` — run one command on an H200 (image, volumes, secret
  pre-wired). Start here for one-off jobs; `--help` documents the flags.
- `sandbox.md` + `gpu_sandbox.py` — an interactive GPU box with SSH for
  debugging, profiling (nsys/ncu), and prototyping.
- `modal.md` — the platform: apps, images, `modal run` / `deploy` / `serve`,
  the CLI, where the docs live.
- `training.md` — writing your own training app: the shared volumes, secrets,
  retries with checkpoint auto-resume, multi-node.

Rules:

1. You are already authenticated; everything runs in the `lab-dev`
   environment.
2. H200 only, always — `H200` or `H200:N` (up to 8 per node, 141 GB VRAM
   each), never another GPU type. This keeps every run's GPU-hours
   comparable. CPU-only work: no GPU (`none` in `gpu_launcher.py`).
3. HOW you run is your decision: a GPU sandbox, `modal run`, or a deployed
   app — pick what fits the job. The shortcut for one-off jobs is
   `gpu_launcher.py` (volumes and secret pre-wired); write your own Modal
   app when you need more (servers, multi-node, retries).
4. Volumes: `lab-out` (mount at `/out` — checkpoints go to
   `/out/models/$LEARNING_AGENT_RUN_ID/<tag>/merged`; the volume is SHARED across
   runs, so stay inside your run's namespace — other runs' artifacts are
   not yours to use or overwrite) and `lab-hf-cache` (mount at `/hf-cache`,
   set `HF_HOME=/hf-cache` — the base model is already cached; do not
   clean it). Secret: `huggingface-secret`.
5. GPU accounting: every GPU job puts one line in `runs/GPU_LOG.jsonl` —
   `gpu_launcher.py` and `gpu_sandbox.py` write it automatically; if you
   run your own Modal app, append the line yourself:
   `{"ts": "<iso>", "seconds": <wall>, "gpu": "H200:2", "n_gpus": 2,
   "command": "<what ran>", "exit": 0}`. The dashboard sums this into the
   run's GPU-hours. Your usage is ALSO metered independently from the Modal
   control plane and shown next to your self-report — unlogged sandboxes
   are visible either way, so log honestly, not selectively. If you create
   sandboxes with your own Modal code, pass
   `tags={"learning_agent_run_id": os.environ["LEARNING_AGENT_RUN_ID"]}` so the metering
   attributes them to your run precisely.
6. Do not modify the operator's deployed apps (the GLM endpoint, the judge
   service, the user simulator). Your GPU work is yours: sandboxes and
   apps you create.
