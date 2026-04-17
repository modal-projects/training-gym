# Tutorials

Every tutorial below has a one-click **Launch in Modal Notebook** button.
The button opens the `.ipynb` in a fresh Modal Notebook — the first code
cell is a `! pip install git+https://github.com/modal-projects/training-gym.git@joy/initial-setup`
that installs `modal-training-gym` into the notebook kernel, so the rest of
the cells run as-is.

To edit a tutorial, modify its source under `tutorial_generator/` — **not**
the generated `.py` / `.ipynb`. Then run `uv run python tutorials/generate_tutorial.py`
to regenerate (or let the pre-commit hook do it on commit).

## Tutorials

<!-- Launch URLs use the branch-scoped deep-link:
     https://modal.com/notebooks/modal-labs/joy-dev?import=url&url=<url-encoded raw GitHub URL>
     These resolve once the .ipynb has been pushed to origin/joy/initial-setup. -->

| Tutorial | Framework | Cluster shape | What it trains | Launch |
|---|---|---|---|---|
| [`slime_gsm8k`](slime_gsm8k/slime_gsm8k.ipynb) | `slime` | 4 × 8×H200 | Qwen3-4B GRPO on GSM8K (colocated) | [![Open in Modal](https://img.shields.io/badge/Open_in-Modal-7FEE64?style=for-the-badge&labelColor=0b0d0f)](https://modal.com/notebooks/modal-labs/joy-dev?import=url&url=https%3A%2F%2Fraw.githubusercontent.com%2Fmodal-projects%2Ftraining-gym%2Fjoy%2Finitial-setup%2Ftutorials%2Fslime_gsm8k%2Fslime_gsm8k.ipynb) |
| [`ms_swift_glm_4_7_gsm8k`](ms_swift_glm_4_7_gsm8k/ms_swift_glm_4_7_gsm8k.ipynb) | `ms_swift` | 4 × 8×B200 | GLM-4.7 LoRA SFT on GSM8K (Megatron) | [![Open in Modal](https://img.shields.io/badge/Open_in-Modal-7FEE64?style=for-the-badge&labelColor=0b0d0f)](https://modal.com/notebooks/modal-labs/joy-dev?import=url&url=https%3A%2F%2Fraw.githubusercontent.com%2Fmodal-projects%2Ftraining-gym%2Fjoy%2Finitial-setup%2Ftutorials%2Fms_swift_glm_4_7_gsm8k%2Fms_swift_glm_4_7_gsm8k.ipynb) |
| [`megatron_glm_4_7_longmit128k`](megatron_glm_4_7_longmit128k/megatron_glm_4_7_longmit128k.ipynb) | `megatron` | 4 × 8×B200 | GLM-4.7 LoRA on LongMIT-128K (NeMo bridge) | [![Open in Modal](https://img.shields.io/badge/Open_in-Modal-7FEE64?style=for-the-badge&labelColor=0b0d0f)](https://modal.com/notebooks/modal-labs/joy-dev?import=url&url=https%3A%2F%2Fraw.githubusercontent.com%2Fmodal-projects%2Ftraining-gym%2Fjoy%2Finitial-setup%2Ftutorials%2Fmegatron_glm_4_7_longmit128k%2Fmegatron_glm_4_7_longmit128k.ipynb) |
| [`verl_qwen3_32b_gsm8k`](verl_qwen3_32b_gsm8k/verl_qwen3_32b_gsm8k.ipynb) | `verl` | 4 × 8×H100 | Qwen3-32B GRPO on GSM8K (Megatron + vLLM) | [![Open in Modal](https://img.shields.io/badge/Open_in-Modal-7FEE64?style=for-the-badge&labelColor=0b0d0f)](https://modal.com/notebooks/modal-labs/joy-dev?import=url&url=https%3A%2F%2Fraw.githubusercontent.com%2Fmodal-projects%2Ftraining-gym%2Fjoy%2Finitial-setup%2Ftutorials%2Fverl_qwen3_32b_gsm8k%2Fverl_qwen3_32b_gsm8k.ipynb) |
| [`starcoder_llama2_7b`](starcoder_llama2_7b/starcoder_llama2_7b.ipynb) | `torchrun` + `hf_accelerate` | 2 × 8×H100 | Llama-2-7B SFT on Go + Rust (FSDP) | [![Open in Modal](https://img.shields.io/badge/Open_in-Modal-7FEE64?style=for-the-badge&labelColor=0b0d0f)](https://modal.com/notebooks/modal-labs/joy-dev?import=url&url=https%3A%2F%2Fraw.githubusercontent.com%2Fmodal-projects%2Ftraining-gym%2Fjoy%2Finitial-setup%2Ftutorials%2Fstarcoder_llama2_7b%2Fstarcoder_llama2_7b.ipynb) |
| [`nanogpt_owt`](nanogpt_owt/nanogpt_owt.ipynb) | — (clones `karpathy/nanoGPT`) | 2 × 8×H100 | GPT-2 124M on OpenWebText | [![Open in Modal](https://img.shields.io/badge/Open_in-Modal-7FEE64?style=for-the-badge&labelColor=0b0d0f)](https://modal.com/notebooks/modal-labs/joy-dev?import=url&url=https%3A%2F%2Fraw.githubusercontent.com%2Fmodal-projects%2Ftraining-gym%2Fjoy%2Finitial-setup%2Ftutorials%2Fnanogpt_owt%2Fnanogpt_owt.ipynb) |
| [`lightning_fabric_demo`](lightning_fabric_demo/lightning_fabric_demo.ipynb) | `lightning` | 2 × 8×H100 | Transformer on WikiText2 (Fabric DDP) | [![Open in Modal](https://img.shields.io/badge/Open_in-Modal-7FEE64?style=for-the-badge&labelColor=0b0d0f)](https://modal.com/notebooks/modal-labs/joy-dev?import=url&url=https%3A%2F%2Fraw.githubusercontent.com%2Fmodal-projects%2Ftraining-gym%2Fjoy%2Finitial-setup%2Ftutorials%2Flightning_fabric_demo%2Flightning_fabric_demo.ipynb) |
| [`resnet50_imagenet`](resnet50_imagenet/resnet50_imagenet.ipynb) | — (clones `pytorch/vision`) | 4 × 8×H100 | ResNet50 on ImageNet (DDP) | [![Open in Modal](https://img.shields.io/badge/Open_in-Modal-7FEE64?style=for-the-badge&labelColor=0b0d0f)](https://modal.com/notebooks/modal-labs/joy-dev?import=url&url=https%3A%2F%2Fraw.githubusercontent.com%2Fmodal-projects%2Ftraining-gym%2Fjoy%2Finitial-setup%2Ftutorials%2Fresnet50_imagenet%2Fresnet50_imagenet.ipynb) |
| [`nccl_benchmark`](nccl_benchmark/nccl_benchmark.ipynb) | — (NCCL utility) | 2 × 8×H100 | All-reduce bandwidth benchmark (500000×2000 fp32) | [![Open in Modal](https://img.shields.io/badge/Open_in-Modal-7FEE64?style=for-the-badge&labelColor=0b0d0f)](https://modal.com/notebooks/modal-labs/joy-dev?import=url&url=https%3A%2F%2Fraw.githubusercontent.com%2Fmodal-projects%2Ftraining-gym%2Fjoy%2Finitial-setup%2Ftutorials%2Fnccl_benchmark%2Fnccl_benchmark.ipynb) |
| [`ray_slime_standalone`](ray_slime_standalone/ray_slime_standalone.ipynb) | — (raw `ModalRayCluster`) | 2 × 8×H100 | Ray-on-Modal pattern demo | [![Open in Modal](https://img.shields.io/badge/Open_in-Modal-7FEE64?style=for-the-badge&labelColor=0b0d0f)](https://modal.com/notebooks/modal-labs/joy-dev?import=url&url=https%3A%2F%2Fraw.githubusercontent.com%2Fmodal-projects%2Ftraining-gym%2Fjoy%2Finitial-setup%2Ftutorials%2Fray_slime_standalone%2Fray_slime_standalone.ipynb) |

## Running from the CLI instead

Every tutorial is also a plain `.py` file runnable via `modal run`. See
the top-level [`README.md`](../README.md) for the usage pattern.
