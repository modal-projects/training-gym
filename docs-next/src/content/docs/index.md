---
title: Training Gym
description: Reusable building blocks and runnable examples for distributed training on Modal.
hero:
  tagline: Distributed training on Modal has never been easier. Pick a framework, plug in a model, and let it jam on your dataset.
  actions:
    - text: Get started
      link: /tutorials/intro/quickstart/
      icon: right-arrow
    - text: View on GitHub
      link: https://github.com/modal-projects/training-gym
      variant: secondary
      icon: external
---

<div class="tg-landing-cards">

<a class="tg-landing-card" href="/tutorials/">
<h3>Tutorials</h3>
<p>Runnable notebooks and scripts covering RL post-training, SFT, and infrastructure patterns.</p>
</a>

<a class="tg-landing-card" href="/reference/">
<h3>API Reference</h3>
<p>Configuration classes, model definitions, and framework launchers — quick lookup for every public API.</p>
</a>

<a class="tg-landing-card" href="/support/">
<h3>Support</h3>
<p>Contact points for questions about Training Gym and access to Modal multi-node training.</p>
</a>

</div>

## Frameworks

Each framework package exposes a factory that returns a `modal.App` with `download_model`, `prepare_dataset`, and `train` functions.

| Framework | Good for | Example |
|---|---|---|
| `slime` | GRPO / RL post-training — Ray + Megatron + SGLang | [`slime_gsm8k`](/tutorials/rl/slime_gsm8k/), [`slime_haiku`](/tutorials/rl/slime_haiku/) |
| `ms_swift` | ms-swift Megatron SFT (single- and multi-node) | [`ms_swift_glm_4_7_gsm8k`](/tutorials/sft/ms_swift_glm_4_7_gsm8k/), [`ms_swift_custom_hf`](/tutorials/sft/ms_swift_custom_hf/) |
| `harbor` | Sandbox-based RL with Harbor tasks/agents | [`harbor_code_golf`](/tutorials/rl/harbor_code_golf/) |

## Install

```bash
pip install git+https://github.com/modal-projects/training-gym.git
```

## Quick links

- [Multi-node training guide](https://modal-com.notion.site/Multi-node-docs-1281e7f16949806f966adedfe8b2cb74?pvs=4)
- [Multi-GPU Training on Modal](https://modal.com/docs/guide/gpu#multi-gpu-training)
- [Using CUDA on Modal](https://modal.com/docs/guide/cuda)
- [GPU Metrics](https://modal.com/docs/guide/gpu-metrics)
