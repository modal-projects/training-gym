# Training Gym

> [!IMPORTANT]
> Our multi-node cluster training product is in early preview and not generally accessible. Please [**contact us**](https://modal.com/slack) for access.

Well-documented examples of running distributed training jobs on [Modal](https://modal.com).

## Layout

- [**`frameworks/`**](frameworks/) — examples that show how to use Modal with specific frameworks and codebases.
  - [`slime/`](frameworks/slime/), [`miles/`](frameworks/miles/), [`ms-swift/`](frameworks/ms-swift/), [`verl/`](frameworks/verl/)
  - [`megatron/`](frameworks/megatron/), [`nanoGPT/`](frameworks/nanoGPT/), [`lightning/`](frameworks/lightning/), [`ray/`](frameworks/ray/)
  - [`resnet50/`](frameworks/resnet50/), [`starcoder/`](frameworks/starcoder/), [`benchmark/`](frameworks/benchmark/)
- [**`tutorials/`**](tutorials/) — quick one-pager examples that also run on Modal Notebooks (with a "Launch in Modal Notebooks" button). These have the repository mounted onto them so they can reference shared code (e.g. the slime launcher, HF-to-Megatron converters) without duplicating it.
- [**`common/`**](common/) — common shared utilities used across examples (SSH setup, health checks).
- [**`skills/`**](skills/) — agent skills for navigating this repository and running training on Modal.
- [**`dashboards/`**](dashboards/) — generic dashboards for monitoring training runs.

## Documentation

The multi-node training guide is currently available on Notion: [modal-com.notion.site/Multi-node-docs](https://modal-com.notion.site/Multi-node-docs-1281e7f16949806f966adedfe8b2cb74?pvs=4).

Other relevant documentation:

- [Multi-GPU Training](https://modal.com/docs/guide/gpu#multi-gpu-training)
- [Using CUDA on Modal](https://modal.com/docs/guide/cuda)
- [GPU Metrics](https://modal.com/docs/guide/gpu-metrics)
- [Agent guide for running training jobs on Modal](skills/agent-modal-training.md)

## License

The [MIT license](LICENSE).
