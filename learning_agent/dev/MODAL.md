# Modal inventory & team conventions

Everything Modal-specific in one place. The generic runbook (how to launch a
shared run) lives in the root [README, "Team runs"](../README.md#team-runs-one-launch-everything-shared);
this file records *our* account's names and the wiring behind that runbook.

## Environments (workspace: modal-labs)

Environments are **workspace-scoped**: every member of the `modal-labs`
workspace can see and run against any non-restricted env, so a shared env is
"shared with everyone" by default (the `*-restricted` envs are the opt-in
exception). Manage per-env access from the Modal dashboard.

| env | status |
|---|---|
| `lab-dev` | **operative shared team env — the committed `.env` default** (created 2026-07-22). Non-restricted, so usable by everyone in the workspace. Has `huggingface-secret` (empty `HF_TOKEN` — fine while all base models are public) and the deployed viewer: **https://modal-labs-lab-dev--lab-observatory-web.modal.run** (walkthrough at `/how`). The `lab-observatory` volume was created by the deploy; `lab-out` + `lab-hf-cache` auto-create on the first `bench.py train`/`score` (that first train repopulates the cache, ~18 GB once). Add `wandb-secret` before any RL run. |
| `leon-dev` | previous team env — has `huggingface-secret` + `wandb-secret`, the populated `lab-hf-cache` and `lab-out` (2026-07-18 verifications), and its own viewer: https://modal-labs-leon-dev--lab-observatory-web.modal.run |
| `learning-agent` | earlier dedicated-env attempt (2026-07-21), never finished wiring — superseded by `lab-dev` |
| `junlin-dev` | Junlin's own env (has its own secrets); the observatory's hardcoded fallback default in `volume_io.py` |

## Environment convention

- One knob: `MODAL_ENVIRONMENT` in the repo-root `.env` (see `.env.example`;
  committed default is `lab-dev`).
- Who reads it:
  - `agents/run.sh` exports it from `.env` at launch, so the agent's
    `bench.py train` / `bench.py rl` jobs (which shell out to `modal run`)
    land in the team environment — the Modal CLI reads only the process env,
    never `.env`.
  - `observatory/volume_io.py` reads process env, then `.env`, then falls
    back to `junlin-dev`.
- For manual operator commands outside a run (`modal deploy`,
  `modal volume ...`), export it in your shell or rely on your profile.

## Apps (all defined in this repo)

| app | source | GPU / timeout | what it does |
|---|---|---|---|
| `lab-sft` | `pipeline/train.py` | 1×H200, 90 min | LoRA-SFT via axolotl, merge, write to `lab-out` |
| `lab-eval` | `harness/eval.py` | 1×H200, 120 min, per-task image (dspy/openclaw/fav2) | serve a checkpoint as the ReAct search agent for scoring |
| `lab-rl-merge` | `pipeline/rl.py` | 1×H200, 2 h | convert the RL run's Megatron checkpoint to HF, write to `lab-out` |
| `lab-observatory` | `observatory/app.py` | CPU | the team viewer (`modal deploy observatory/app.py`) |
| `lab-agent` | `agents/modal_runner.py` | CPU (4 cpu / 8 GB), 24 h timeout | containerized contestant sessions — one container per session, spawned by `agents/run_sandbox_modal.sh`; carries `lab-agent-modal-token` so in-container `bench.py train` + obs ingest work |
| `ep-glm-5-2-fp8` | (deployed separately) | GPU serving | SGLang server for `zai-org/GLM-5.2-FP8` (1M ctx) at https://modal-labs-lab-dev--ep-glm-5-2-fp8-server.us-east.modal.direct — the `agents/modal_glm52` contestant's backend. OpenAI-compatible `/v1`; its Anthropic `/v1/messages` emulation is partial (Claude Code cannot use it). **Public and unauthenticated** — accepted risk; anyone with the URL can spend GPU time. |

RL *training* itself runs through `modal_training_gym` (slime): 1 node ×
2×H200, tp=2, actor and rollout engine colocated (one H200 does not fit
both). Launched from the local `.venv-rl` (python 3.12) by
`bench.py rl` → `pipeline/rl.py`. Needs the Modal secrets
`huggingface-secret` and `wandb-secret` in the environment.

## Volumes

| volume | contents | notes |
|---|---|---|
| `lab-out` | trained weights at `/out/models/<tag>/merged` | `rl_e2e_verify` and `smoke_len16k` are 2026-07-18 infra tests, not benchmark checkpoints |
| `lab-hf-cache` | base-model snapshot (`Qwen/Qwen3.5-9B`) | populated once by the first train; trainers then force HF offline — this is load-bearing (online, transformers probes for an image processor the repo doesn't ship and dies), do not "clean up" |
| `lab-observatory` | `runs/<run_id>/{record,workspace,status}.json + raw/` (trace, audit, prompt, `workspace.tar.gz`) | written by `obs ingest/watch`; read-only for the viewer |
| `lab-agent-workspace` | containerized sessions: `<task>/<session>/{workspace,logs}` + one-time `_shared/{corpus/<task>, training-toolbox}` | **V2 volume** (required: per-session `modal volume cp -r`); created 2026-07-28. `logs/home` holds the CLI's own session state (opencode storage / claude session jsonl) |
| `huggingface-cache` | HF cache used by the RL merge step | modal_training_gym's convention, distinct from `lab-hf-cache` |
| (gym checkpoint volumes) | Megatron checkpoints per RL run | created by modal_training_gym; name passed through to the merge |
| `la-ula-*` | legacy naming, orphaned | dead history — ignore; delete whenever |

## Handy commands

```bash
modal deploy observatory/app.py                 # (re)deploy the team viewer
modal volume ls lab-out models/                 # what checkpoints exist
modal volume get lab-observatory runs/<id>/raw/workspace.tar.gz .   # pull a submission
```

## Secrets (lab-dev)

- `huggingface-secret` — empty `HF_TOKEN`, fine while base models are public.
- `lab-agent-modal-token` — a Modal API token (created 2026-07-28 from the
  operator's profile) that `lab-agent` session containers carry so the agent's
  own `modal run` GPU jobs and observatory uploads work from inside.
  **Readable by anyone with lab-dev access** (workspace-scoped env) — rotate
  via `modal token new` + re-`modal secret create` if that trust changes.

## Observed costs (from the 2026-07-18 verifications)

- SFT smoke: 1×H200 ≈ 10 min, plus a one-time ~18 GB base-model download
  into `lab-hf-cache`.
- RL end-to-end (2 rollout iterations): 2×H200 ≈ 15 min.
- Judge calls (Anthropic API): pennies per eval at dev-set sizes.

## Security posture (current, deliberate)

The deployed viewer is a **public URL with no auth** (v1 decision in
`observatory/DESIGN.md`) — anyone holding the URL can read every ingested
run, including workspace archives. Acceptable in stealth; revisit before
sharing the URL beyond the team.

## De-stealth checklist

1. `git rm -r dev/` (this folder).
2. Replace the `MODAL_ENVIRONMENT=junlin-dev` default in `.env.example`
   (and `observatory/volume_io.py`'s fallback) with a neutral placeholder.
3. Held-out `tasks/*/test.json` must already be out of the public repo by
   then (tracked blocker — required even earlier, before any medium/hard run).
4. Decide viewer hosting: keep private, or add auth before publishing a URL.
5. Sweep volumes: archive or drop infra-test checkpoints on `lab-out`.
