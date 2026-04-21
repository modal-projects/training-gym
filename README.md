# Training Gym

> [!IMPORTANT]
> Modal's multi-node cluster training product is in early preview and not
> generally accessible. Please [**contact us**](https://modal.com/slack)
> for access.

Distributed training on [Modal](https://modal.com) without hand-rolling a
launcher each time. Pick a training framework (slime, verl, Megatron,
MS-SWIFT, Lightning, HF Accelerate, or raw torchrun), plug in a model +
dataset config, and `modal run` it — training-gym handles the image, the
cluster topology, the Ray/NCCL bring-up, volume mounts, and checkpointing.

Packaged as `modal-training-gym` — pip-install once, then import
framework-specific launchers from your own scripts or notebooks. Every
tutorial is a runnable `.py` file **and** a matching `.ipynb` with the same
steps narrated cell-by-cell — the notebook is the place to read the
walkthrough; this README is the map.

## Install

In a notebook or script:

```python
! pip install -q git+https://github.com/modal-projects/training-gym.git@joy/initial-setup
```

Every generated tutorial notebook has this line as its first code cell.

## Quickstart

**1. Validate your Modal setup.** Before launching anything expensive, run a
2 × 8 × H100 NCCL all-reduce to confirm multi-node training works in your
workspace:

```bash
uv run modal run --detach tutorials/nccl_benchmark/nccl_benchmark.py::run_benchmark
```

**2. Run a tutorial.** Qwen3-4B GRPO on GSM8K using SLIME:

```bash
uv run modal run tutorials/slime_gsm8k/slime_gsm8k.py::app.download_model
uv run modal run tutorials/slime_gsm8k/slime_gsm8k.py::app.prepare_dataset
uv run modal run --detach tutorials/slime_gsm8k/slime_gsm8k.py::app.train
```

Or open the matching `.ipynb` in Jupyter / Modal Notebooks and run
cell-by-cell — each notebook is a self-contained walkthrough. See
[`tutorials/README.md`](tutorials/README.md) for the full catalog.

## Pick your framework

Each framework package exposes `build_<name>_app(modal=..., config=...)` —
a factory that returns a `modal.App` with `download_model`,
`prepare_dataset`, and `train` functions. Shared container objects
(`DatasetConfig`, `Model`, `WandbConfig`) plug into the framework config;
each framework translates them into its own CLI vocabulary.

| Framework | Good for | Abstraction | Example |
|---|---|---|---|
| `torchrun` | Any `torchrun`-compatible script; BYO training loop | Thin — cluster + launch only | [`starcoder_llama2_7b`](tutorials/starcoder_llama2_7b/) |
| `hf_accelerate` | Accelerate-based SFT, FSDP | Thin | [`starcoder_llama2_7b`](tutorials/starcoder_llama2_7b/) |
| `lightning` | PyTorch Lightning Fabric scripts | Thin | [`lightning_fabric_demo`](tutorials/lightning_fabric_demo/) |
| `ms_swift` | LoRA / full SFT via ModelScope SWIFT (HF or Megatron backend) | Opinionated | [`ms_swift_glm_4_7_gsm8k`](tutorials/ms_swift_glm_4_7_gsm8k/), [`ms_swift_custom_hf`](tutorials/ms_swift_custom_hf/) |
| `megatron` | Full-parameter training on Megatron-LM (TP / PP / EP) | Opinionated | [`megatron_glm_4_7_longmit128k`](tutorials/megatron_glm_4_7_longmit128k/) |
| `slime` | GRPO / RL post-training — Ray + Megatron + SGLang | Opinionated | [`slime_gsm8k`](tutorials/slime_gsm8k/), [`slime_haiku`](tutorials/slime_haiku/) |
| `verl` | GRPO / RL post-training — Ray + Megatron + vLLM | Opinionated | [`verl_qwen3_32b_gsm8k`](tutorials/verl_qwen3_32b_gsm8k/) |

"Thin" launchers give you a cluster and a `torchrun` — bring your own
training script. "Opinionated" launchers wrap a specific upstream framework
and expect you to configure it via that framework's CLI/YAML vocabulary.

Source in `modal_training_gym/frameworks/`. Runnable examples in
[`tutorials/README.md`](tutorials/README.md).

## Documentation

- [Multi-node training guide (Notion)](https://modal-com.notion.site/Multi-node-docs-1281e7f16949806f966adedfe8b2cb74?pvs=4)
- [Multi-GPU Training on Modal](https://modal.com/docs/guide/gpu#multi-gpu-training)
- [Using CUDA on Modal](https://modal.com/docs/guide/cuda)
- [GPU Metrics](https://modal.com/docs/guide/gpu-metrics)

## License

[MIT](LICENSE).

---

# Developer Guide

## Layout

```
modal_training_gym/        ← installable package
├── common/                ← cross-framework classes (datasets, models, wandb, Ray cluster helpers)
└── frameworks/            ← one package per training framework (see tutorials/ for the full list)

tutorials/                 ← runnable examples — one folder per tutorial
├── tutorial_generator/    ← source files; each produces a .py + .ipynb
└── generate_tutorial.py   ← AST-walks the sources, regenerates .py + .ipynb

dashboards/                ← Grafana-style dashboards for monitoring runs
skills/                    ← agent skills for navigating this repo
```

## Dev setup

```bash
# editable install + pinned dev deps (pre-commit, ipykernel if you want)
uv sync

# register this venv as a Jupyter kernel (one-time, for notebook work)
uv run python -m ipykernel install --user --name=modal-training-gym

# install the pre-commit hook locally
uv run pre-commit install
```

Project Python is pinned to 3.12 (see `.python-version` / `pyproject.toml`);
every `@app.function(serialized=True)` requires the local ↔ remote Python
versions to match, and the framework images we use (slime nightly, NeMo
25.11, verl `vllm011.latest`) all ship py312.

## Authoring a new tutorial

See [`tutorials/README.md`](tutorials/README.md#authoring-a-new-tutorial)
for the generator-source format and the per-tutorial `TUTORIAL_METADATA`
schema.

## Contributing a new framework

1. Create `modal_training_gym/frameworks/<name>/` with `config.py`,
   `launcher.py`, `__init__.py`.
2. `build_<name>_app(*, modal, config, name=None) -> modal.App` is the
   public entrypoint — same shape as the existing frameworks. See
   `modal_training_gym/frameworks/verl/launcher.py` for a complete
   reference.
3. Add a new `tutorials/tutorial_generator/<tutorial>.py` demonstrating it,
   and run the generator.
4. Container configs (`dataset`, `model`, `wandb`) are interpreted
   explicitly in each framework — don't try to share CLI vocabularies.
   The common config classes stay pure data.

## Agent guide

- [Running training jobs on Modal](skills/agent-modal-training.md)
