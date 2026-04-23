# Tutorial Validation Report

**Date**: 2026-04-22 20:00:41 UTC
**Modal environment**: `joy-dev`
**Targets scanned**: 15

## Summary

| Outcome | Count |
|---|---|
| pass | 1 |
| fail | 14 |
| blocked | 0 |
| skipped | 0 |

## Results

| Tutorial | Target | Type | Outcome | Detail | Wall Time |
|---|---|---|---|---|---|
| quickstart | quickstart | concept_only | **pass** | source compiles and imports resolve | 0.1s |
| lightning_fabric_demo | lightning_fabric_demo | training | **fail** | [prerequisite_stage] prereq 'modal run tutorials/misc/lightning_fabric_demo/lightning_fabric_demo.py: | 11.3s |
| nanogpt_owt | nanogpt_owt/train_single_node | training | **fail** | [prerequisite_stage] prereq 'modal run tutorials/misc/nanogpt_owt/nanogpt_owt.py::prepare_data' exite | 0.6s |
| nanogpt_owt | nanogpt_owt/train_multi_node | training | **fail** | [prerequisite_stage] prereq 'modal run tutorials/misc/nanogpt_owt/nanogpt_owt.py::prepare_data' exite | 0.6s |
| nccl_benchmark | nccl_benchmark | benchmark | **fail** | [training_runtime] exited 1: ╰───────────────────────────────────────────────────────────────────── | 0.5s |
| ray_slime_standalone | ray_slime_standalone | pattern_demo | **fail** | [training_runtime] exited 1: ╰───────────────────────────────────────────────────────────────────── | 0.5s |
| resnet50_imagenet | resnet50_imagenet | training | **fail** | [prerequisite_stage] prereq 'modal run tutorials/misc/resnet50_imagenet/resnet50_imagenet.py::downloa | 0.5s |
| slime_gsm8k | slime_gsm8k | training | **fail** | [timeout] ╭─ Error ──────────────────────────────────────────────────────────────────────╮ | 246.0s |
| slime_haiku | slime_haiku | training | **fail** | [prerequisite_stage] prereq 'modal run tutorials/rl/slime_haiku/slime_haiku.py::app.prepare_dataset'  | 150.6s |
| verl_qwen3_32b_gsm8k | verl_qwen3_32b_gsm8k | training | **fail** | [timeout] ╭─ Error ──────────────────────────────────────────────────────────────────────╮ | 840.0s |
| megatron_glm_4_7_longmit128k | megatron_glm_4_7_longmit128k | training | **fail** | [image_build] detached launch timed out after 600s | 10240.2s |
| ms_swift_custom_hf | ms_swift_custom_hf | training | **fail** | [image_build] detached launch timed out after 600s | 620.0s |
| ms_swift_glm_4_7_gsm8k | ms_swift_glm_4_7_gsm8k | training | **fail** | [image_build] detached launch timed out after 600s | 770.7s |
| starcoder_llama2_7b | starcoder_llama2_7b/torchrun | training | **fail** | [prerequisite_stage] prereq 'modal run tutorials/sft/starcoder_llama2_7b/starcoder_llama2_7b.py::torc | 28.3s |
| starcoder_llama2_7b | starcoder_llama2_7b/accelerate | training | **fail** | [prerequisite_stage] prereq 'modal run tutorials/sft/starcoder_llama2_7b/starcoder_llama2_7b.py::acce | 11.1s |

## Detailed Results

### quickstart

- **Outcome**: pass
- **Command**: ``
- **Detail**: source compiles and imports resolve
- **First signal**: `import_ok`
- **Wall time**: 0.1s

### lightning_fabric_demo

- **Outcome**: fail
- **Command**: `modal run tutorials/misc/lightning_fabric_demo/lightning_fabric_demo.py::app.download_dataset`
- **Failure source**: prerequisite_stage
- **Detail**: prereq 'modal run tutorials/misc/lightning_fabric_demo/lightning_fabric_demo.py::app.download_dataset' exited 1: ╰──────────────────────────────────────────────────────────────────────────────╯
- **First error**: `╰──────────────────────────────────────────────────────────────────────────────╯`
- **Wall time**: 11.3s
- **Stages**:
  - `modal run tutorials/misc/lightning_fabric_demo/lightning_fabric_demo.py::app.download_dataset` → exit=1 app=ap-k8x4aQfHzE3ddWlCEXmN4C (11.3s)
    error: `╰──────────────────────────────────────────────────────────────────────────────╯`

### nanogpt_owt/train_single_node

- **Outcome**: fail
- **Command**: `modal run tutorials/misc/nanogpt_owt/nanogpt_owt.py::prepare_data`
- **Failure source**: prerequisite_stage
- **Detail**: prereq 'modal run tutorials/misc/nanogpt_owt/nanogpt_owt.py::prepare_data' exited 1: ModuleNotFoundError: No module named 'matplotlib'
- **First error**: `ModuleNotFoundError: No module named 'matplotlib'`
- **Wall time**: 0.6s
- **Stages**:
  - `modal run tutorials/misc/nanogpt_owt/nanogpt_owt.py::prepare_data` → exit=1 (0.6s)
    error: `ModuleNotFoundError: No module named 'matplotlib'`

### nanogpt_owt/train_multi_node

- **Outcome**: fail
- **Command**: `modal run tutorials/misc/nanogpt_owt/nanogpt_owt.py::prepare_data`
- **Failure source**: prerequisite_stage
- **Detail**: prereq 'modal run tutorials/misc/nanogpt_owt/nanogpt_owt.py::prepare_data' exited 1: ModuleNotFoundError: No module named 'matplotlib'
- **First error**: `ModuleNotFoundError: No module named 'matplotlib'`
- **Wall time**: 0.6s
- **Stages**:
  - `modal run tutorials/misc/nanogpt_owt/nanogpt_owt.py::prepare_data` → exit=1 (0.6s)
    error: `ModuleNotFoundError: No module named 'matplotlib'`

### nccl_benchmark

- **Outcome**: fail
- **Command**: `modal run tutorials/misc/nccl_benchmark/nccl_benchmark.py::run_benchmark`
- **Failure source**: training_runtime
- **Detail**: exited 1: ╰──────────────────────────────────────────────────────────────────────────────╯
- **First error**: `╰──────────────────────────────────────────────────────────────────────────────╯`
- **Wall time**: 0.5s
- **Stages**:
  - `modal run tutorials/misc/nccl_benchmark/nccl_benchmark.py::run_benchmark` → exit=1 (0.5s)
    error: `╰──────────────────────────────────────────────────────────────────────────────╯`

### ray_slime_standalone

- **Outcome**: fail
- **Command**: `modal run tutorials/misc/ray_slime_standalone/ray_slime_standalone.py::run_ray_job`
- **Failure source**: training_runtime
- **Detail**: exited 1: ╰──────────────────────────────────────────────────────────────────────────────╯
- **First error**: `╰──────────────────────────────────────────────────────────────────────────────╯`
- **Wall time**: 0.5s
- **Stages**:
  - `modal run tutorials/misc/ray_slime_standalone/ray_slime_standalone.py::run_ray_job` → exit=1 (0.5s)
    error: `╰──────────────────────────────────────────────────────────────────────────────╯`

### resnet50_imagenet

- **Outcome**: fail
- **Command**: `modal run tutorials/misc/resnet50_imagenet/resnet50_imagenet.py::download_imagenet`
- **Failure source**: prerequisite_stage
- **Detail**: prereq 'modal run tutorials/misc/resnet50_imagenet/resnet50_imagenet.py::download_imagenet' exited 1: ╰──────────────────────────────────────────────────────────────────────────────╯
- **First error**: `╰──────────────────────────────────────────────────────────────────────────────╯`
- **Wall time**: 0.5s
- **Stages**:
  - `modal run tutorials/misc/resnet50_imagenet/resnet50_imagenet.py::download_imagenet` → exit=1 (0.5s)
    error: `╰──────────────────────────────────────────────────────────────────────────────╯`

### slime_gsm8k

- **Outcome**: fail
- **Command**: `modal run --detach tutorials/rl/slime_gsm8k/slime_gsm8k.py::app.train`
- **Failure source**: timeout
- **Detail**: ╭─ Error ──────────────────────────────────────────────────────────────────────╮
- **App ID**: `ap-HuRzuu7UoheNxN2YFdGwNw`
- **First error**: `╭─ Error ──────────────────────────────────────────────────────────────────────╮`
- **Wall time**: 246.0s
- **Stages**:
  - `modal run tutorials/rl/slime_gsm8k/slime_gsm8k.py::app.download_model` → exit=0 app=ap-4ydrIo4u6qBEdVkwznSQtY (14.9s)
  - `modal run tutorials/rl/slime_gsm8k/slime_gsm8k.py::app.prepare_dataset` → exit=0 app=ap-0NJz8VU6kM7ahvprughWF7 (21.3s)
  - `modal run --detach tutorials/rl/slime_gsm8k/slime_gsm8k.py::app.train` → timeout app=ap-HuRzuu7UoheNxN2YFdGwNw [timeout] (209.7s)
    error: `╭─ Error ──────────────────────────────────────────────────────────────────────╮`

### slime_haiku

- **Outcome**: fail
- **Command**: `modal run tutorials/rl/slime_haiku/slime_haiku.py::app.prepare_dataset`
- **Failure source**: prerequisite_stage
- **Detail**: prereq 'modal run tutorials/rl/slime_haiku/slime_haiku.py::app.prepare_dataset' exited 1: ╰──────────────────────────────────────────────────────────────────────────────╯
- **First error**: `╰──────────────────────────────────────────────────────────────────────────────╯`
- **Wall time**: 150.6s
- **Stages**:
  - `modal run tutorials/rl/slime_haiku/slime_haiku.py::app.download_model` → exit=0 app=ap-nB2QTVq9RT23MpdxMpeeit (109.3s)
  - `modal run tutorials/rl/slime_haiku/slime_haiku.py::app.prepare_dataset` → exit=1 app=ap-SVWkLuiCuscudgc30w4l27 (41.4s)
    error: `╰──────────────────────────────────────────────────────────────────────────────╯`

### verl_qwen3_32b_gsm8k

- **Outcome**: fail
- **Command**: `modal run --detach tutorials/rl/verl_qwen3_32b_gsm8k/verl_qwen3_32b_gsm8k.py::app.train_multi_node -- trainer.total_training_steps=1`
- **Failure source**: timeout
- **Detail**: ╭─ Error ──────────────────────────────────────────────────────────────────────╮
- **App ID**: `ap-xnexgmASnvb28WJ09ggURa`
- **First error**: `╭─ Error ──────────────────────────────────────────────────────────────────────╮`
- **Wall time**: 840.0s
- **Stages**:
  - `modal run tutorials/rl/verl_qwen3_32b_gsm8k/verl_qwen3_32b_gsm8k.py::app.prep_dataset` → exit=0 app=ap-P2RRNsH3MHRv0lBAP8Shfm (77.9s)
  - `modal run tutorials/rl/verl_qwen3_32b_gsm8k/verl_qwen3_32b_gsm8k.py::app.download_model` → exit=0 app=ap-dFWv0CiUfND2qBdvYpmhhj (20.4s)
  - `modal run tutorials/rl/verl_qwen3_32b_gsm8k/verl_qwen3_32b_gsm8k.py::app.convert_hf_to_mcore` → exit=0 app=ap-6UYoEWAqUhWGiZirVp2aSC (272.3s)
  - `modal run tutorials/rl/verl_qwen3_32b_gsm8k/verl_qwen3_32b_gsm8k.py::app.upload_reward` → exit=0 app=ap-FCHtwhKXd5sp2yqquQhTqo (13.0s)
  - `modal run --detach tutorials/rl/verl_qwen3_32b_gsm8k/verl_qwen3_32b_gsm8k.py::app.train_multi_node -- trainer.total_training_steps=1` → timeout app=ap-xnexgmASnvb28WJ09ggURa [timeout] (456.4s)
    error: `╭─ Error ──────────────────────────────────────────────────────────────────────╮`

### megatron_glm_4_7_longmit128k

- **Outcome**: fail
- **Command**: `modal run --detach tutorials/sft/megatron_glm_4_7_longmit128k/megatron_glm_4_7_longmit128k.py::app.train_lora`
- **Failure source**: image_build
- **Detail**: detached launch timed out after 600s
- **First error**: `detached launch timed out after 600s`
- **Wall time**: 10240.2s
- **Stages**:
  - `modal run tutorials/sft/megatron_glm_4_7_longmit128k/megatron_glm_4_7_longmit128k.py::app.download_model` → exit=0 app=ap-S6LDNgVL7rieU3897eO0vH (1936.0s)
  - `modal run tutorials/sft/megatron_glm_4_7_longmit128k/megatron_glm_4_7_longmit128k.py::app.convert_to_megatron` → exit=0 app=ap-ci10AWPtn08b3o5hp1qvp2 (4453.6s)
  - `modal run tutorials/sft/megatron_glm_4_7_longmit128k/megatron_glm_4_7_longmit128k.py::app.prepare_dataset` → exit=0 app=ap-YD5VX7GDEWNFNSdSyt76L3 (3250.5s)
  - `modal run --detach tutorials/sft/megatron_glm_4_7_longmit128k/megatron_glm_4_7_longmit128k.py::app.train_lora` → timeout [image_build] (600.0s)
    error: `detached launch timed out after 600s`

### ms_swift_custom_hf

- **Outcome**: fail
- **Command**: `modal run --detach tutorials/sft/ms_swift_custom_hf/ms_swift_custom_hf.py::app.train`
- **Failure source**: image_build
- **Detail**: detached launch timed out after 600s
- **First error**: `detached launch timed out after 600s`
- **Wall time**: 620.0s
- **Stages**:
  - `modal run tutorials/sft/ms_swift_custom_hf/ms_swift_custom_hf.py::app.download_model` → exit=0 app=ap-cz6zOEne7tnzkorVUlGDjU (7.4s)
  - `modal run tutorials/sft/ms_swift_custom_hf/ms_swift_custom_hf.py::app.prepare_dataset` → exit=0 app=ap-accZlXWNdqjOstxZ5Lg80F (12.5s)
  - `modal run --detach tutorials/sft/ms_swift_custom_hf/ms_swift_custom_hf.py::app.train` → timeout [image_build] (600.1s)
    error: `detached launch timed out after 600s`

### ms_swift_glm_4_7_gsm8k

- **Outcome**: fail
- **Command**: `modal run --detach tutorials/sft/ms_swift_glm_4_7_gsm8k/ms_swift_glm_4_7_gsm8k.py::app.train`
- **Failure source**: image_build
- **Detail**: detached launch timed out after 600s
- **First error**: `detached launch timed out after 600s`
- **Wall time**: 770.7s
- **Stages**:
  - `modal run tutorials/sft/ms_swift_glm_4_7_gsm8k/ms_swift_glm_4_7_gsm8k.py::app.download_model` → exit=0 app=ap-pRg8Mx9aXWPn97uMfjh7i0 (159.0s)
  - `modal run tutorials/sft/ms_swift_glm_4_7_gsm8k/ms_swift_glm_4_7_gsm8k.py::app.prepare_dataset` → exit=0 app=ap-10GwFRPxIan5tAwF9MxrRt (11.6s)
  - `modal run --detach tutorials/sft/ms_swift_glm_4_7_gsm8k/ms_swift_glm_4_7_gsm8k.py::app.train` → timeout [image_build] (600.1s)
    error: `detached launch timed out after 600s`

### starcoder_llama2_7b/torchrun

- **Outcome**: fail
- **Command**: `modal run tutorials/sft/starcoder_llama2_7b/starcoder_llama2_7b.py::torchrun_app.download_dataset`
- **Failure source**: prerequisite_stage
- **Detail**: prereq 'modal run tutorials/sft/starcoder_llama2_7b/starcoder_llama2_7b.py::torchrun_app.download_dataset' exited 1: ╰──────────────────────────────────────────────────────────────────────────────╯
- **First error**: `╰──────────────────────────────────────────────────────────────────────────────╯`
- **Wall time**: 28.3s
- **Stages**:
  - `modal run tutorials/sft/starcoder_llama2_7b/starcoder_llama2_7b.py::torchrun_app.download_dataset` → exit=1 app=ap-jiLUcrjLrqw5caaG4OETH7 (28.3s)
    error: `╰──────────────────────────────────────────────────────────────────────────────╯`

### starcoder_llama2_7b/accelerate

- **Outcome**: fail
- **Command**: `modal run tutorials/sft/starcoder_llama2_7b/starcoder_llama2_7b.py::accelerate_app.download_dataset`
- **Failure source**: prerequisite_stage
- **Detail**: prereq 'modal run tutorials/sft/starcoder_llama2_7b/starcoder_llama2_7b.py::accelerate_app.download_dataset' exited 1: ╰──────────────────────────────────────────────────────────────────────────────╯
- **First error**: `╰──────────────────────────────────────────────────────────────────────────────╯`
- **Wall time**: 11.1s
- **Stages**:
  - `modal run tutorials/sft/starcoder_llama2_7b/starcoder_llama2_7b.py::accelerate_app.download_dataset` → exit=1 app=ap-mVTXu007wA6oVD6FqHxbEH (11.1s)
    error: `╰──────────────────────────────────────────────────────────────────────────────╯`

