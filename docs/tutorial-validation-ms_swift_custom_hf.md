# Tutorial Validation Report

**Date**: 2026-04-22 14:52:11 UTC
**Modal environment**: `joy-dev`
**Targets scanned**: 15

## Summary

| Outcome | Count |
|---|---|
| pass | 0 |
| fail | 1 |
| blocked | 0 |
| skipped | 14 |

## Results

| Tutorial | Target | Type | Outcome | Detail | Wall Time |
|---|---|---|---|---|---|
| ms_swift_custom_hf | ms_swift_custom_hf | training | **fail** | [training_runtime] exited 1: ─────────────────────────────────────╮
│ ms-swift failed with code 1   | 175.5s |
| quickstart | quickstart | concept_only | **skipped** | filter_excluded | - |
| lightning_fabric_demo | lightning_fabric_demo | training | **skipped** | filter_excluded | - |
| nanogpt_owt | nanogpt_owt/train_single_node | training | **skipped** | filter_excluded | - |
| nanogpt_owt | nanogpt_owt/train_multi_node | training | **skipped** | filter_excluded | - |
| nccl_benchmark | nccl_benchmark | benchmark | **skipped** | filter_excluded | - |
| ray_slime_standalone | ray_slime_standalone | pattern_demo | **skipped** | filter_excluded | - |
| resnet50_imagenet | resnet50_imagenet | training | **skipped** | filter_excluded | - |
| slime_gsm8k | slime_gsm8k | training | **skipped** | filter_excluded | - |
| slime_haiku | slime_haiku | training | **skipped** | filter_excluded | - |
| verl_qwen3_32b_gsm8k | verl_qwen3_32b_gsm8k | training | **skipped** | filter_excluded | - |
| megatron_glm_4_7_longmit128k | megatron_glm_4_7_longmit128k | training | **skipped** | filter_excluded | - |
| ms_swift_glm_4_7_gsm8k | ms_swift_glm_4_7_gsm8k | training | **skipped** | filter_excluded | - |
| starcoder_llama2_7b | starcoder_llama2_7b/torchrun | training | **skipped** | filter_excluded | - |
| starcoder_llama2_7b | starcoder_llama2_7b/accelerate | training | **skipped** | filter_excluded | - |

## Detailed Results

### ms_swift_custom_hf

- **Outcome**: fail
- **Command**: `modal run --detach tutorials/sft/ms_swift_custom_hf/ms_swift_custom_hf.py::app.train`
- **Failure source**: training_runtime
- **Detail**: exited 1: ─────────────────────────────────────╮
│ ms-swift failed with code 1                                                  │
╰──────────────────────────────────────────────────────────────────────────────╯
- **First error**: `─────────────────────────────────────╮
│ ms-swift failed with code 1                                                  │
╰──────────────────────────────────────────────────────────────────────────────╯`
- **Wall time**: 175.5s
- **Stages**:
  - `modal run tutorials/sft/ms_swift_custom_hf/ms_swift_custom_hf.py::app.download_model` → exit=0 app=ap-yFokDVvXBMgHumap0XDfvj (62.6s)
  - `modal run tutorials/sft/ms_swift_custom_hf/ms_swift_custom_hf.py::app.prepare_dataset` → exit=0 app=ap-2Y35ACYqLiVJF6s8jgAtpl (19.9s)
  - `modal run --detach tutorials/sft/ms_swift_custom_hf/ms_swift_custom_hf.py::app.train` → exit=1 (93.0s)
    error: `─────────────────────────────────────╮
│ ms-swift failed with code 1                                                  │
╰──────────────────────────────────────────────────────────────────────────────╯`

### quickstart

- **Outcome**: skipped
- **Command**: ``
- **Detail**: filter_excluded
- **Wall time**: 0.0s

### lightning_fabric_demo

- **Outcome**: skipped
- **Command**: ``
- **Detail**: filter_excluded
- **Wall time**: 0.0s

### nanogpt_owt/train_single_node

- **Outcome**: skipped
- **Command**: ``
- **Detail**: filter_excluded
- **Wall time**: 0.0s

### nanogpt_owt/train_multi_node

- **Outcome**: skipped
- **Command**: ``
- **Detail**: filter_excluded
- **Wall time**: 0.0s

### nccl_benchmark

- **Outcome**: skipped
- **Command**: ``
- **Detail**: filter_excluded
- **Wall time**: 0.0s

### ray_slime_standalone

- **Outcome**: skipped
- **Command**: ``
- **Detail**: filter_excluded
- **Wall time**: 0.0s

### resnet50_imagenet

- **Outcome**: skipped
- **Command**: ``
- **Detail**: filter_excluded
- **Wall time**: 0.0s

### slime_gsm8k

- **Outcome**: skipped
- **Command**: ``
- **Detail**: filter_excluded
- **Wall time**: 0.0s

### slime_haiku

- **Outcome**: skipped
- **Command**: ``
- **Detail**: filter_excluded
- **Wall time**: 0.0s

### verl_qwen3_32b_gsm8k

- **Outcome**: skipped
- **Command**: ``
- **Detail**: filter_excluded
- **Wall time**: 0.0s

### megatron_glm_4_7_longmit128k

- **Outcome**: skipped
- **Command**: ``
- **Detail**: filter_excluded
- **Wall time**: 0.0s

### ms_swift_glm_4_7_gsm8k

- **Outcome**: skipped
- **Command**: ``
- **Detail**: filter_excluded
- **Wall time**: 0.0s

### starcoder_llama2_7b/torchrun

- **Outcome**: skipped
- **Command**: ``
- **Detail**: filter_excluded
- **Wall time**: 0.0s

### starcoder_llama2_7b/accelerate

- **Outcome**: skipped
- **Command**: ``
- **Detail**: filter_excluded
- **Wall time**: 0.0s

