---
title: API Reference
description: API reference for training-gym public classes.
---

Complete reference for the training-gym Python library.

## Core

| Class | Description |
|-------|-------------|
| [`ModelConfig`](/reference/core/modelconfig/) | Base class for model identity and weight-download logic. |
| [`HFModelConfiguration`](/reference/core/hfmodelconfiguration/) | ModelConfig for models hosted on HuggingFace. |
| [`ModelArchitecture`](/reference/core/modelarchitecture/) | Transformer architecture parameters for a specific model. |
| [`DatasetConfig`](/reference/core/datasetconfig/) | Dataset configuration shared across training frameworks. |
| [`HuggingFaceDataset`](/reference/core/huggingfacedataset/) | Dataset backed by a HuggingFace ``datasets`` repo. |
| [`HarborDataset`](/reference/core/harbordataset/) | Dataset configuration shared across training frameworks. |
| [`WandbConfig`](/reference/core/wandbconfig/) | Weights & Biases logging configuration shared across all frameworks. |
| [`ModalRayCluster`](/reference/core/modalraycluster/) | Base class for bootstrapping a Ray cluster inside Modal clustered functions. |
| [`TrainResult`](/reference/core/trainresult/) | One completed training run's checkpoint handle. |

## Evaluation

| Class | Description |
|-------|-------------|
| [`EvalConfig`](/reference/evaluation/evalconfig/) | Evaluate a deployed model on a dataset config. |
| [`EvalResult`](/reference/evaluation/evalresult/) | Saved results for one evaluation run across a dataset. |
| [`EvalRowResult`](/reference/evaluation/evalrowresult/) | One evaluated row: score, response text, and optional metadata. |

## Models

| Class | Description |
|-------|-------------|
| [`Qwen3-0.6B`](/reference/models/qwen3_0_6b/) | Qwen3-0.6B (0.6 billion parameters) from Alibaba. |
| [`Qwen3-1.7B`](/reference/models/qwen3_1_7b/) | Qwen3-1.7B (1.7 billion parameters) from Alibaba. |
| [`Qwen3-4B`](/reference/models/qwen3_4b/) | Qwen3-4B (4 billion parameters) from Alibaba. |
| [`Qwen3-8B`](/reference/models/qwen3_8b/) | Qwen3-8B (8 billion parameters) from Alibaba. |
| [`Qwen3-14B`](/reference/models/qwen3_14b/) | Qwen3-14B (14 billion parameters) from Alibaba. |
| [`Qwen3-30B-A3B`](/reference/models/qwen3_30b/) | Qwen3-30B-A3B (30B total, ~3B active) MoE model from Alibaba. |
| [`Qwen3-32B`](/reference/models/qwen3_32b/) | Qwen3-32B (32 billion parameters) from Alibaba. |

## Training

| Class | Description |
|-------|-------------|
| [`TrainConfig`](/reference/training/trainconfig/) | Compose dataset, model, and recipe into one training entrypoint. |
| [`MultiTurn`](/reference/training/multiturn/) | MultiTurn(generate_function: 'Callable', reward_function: 'Callable | None' = None, max_turns: 'int | None' = None, log_multi_turn: 'bool | None' = None, config_overrides: 'Mapping[str, Any]' = <factory>) |
| [`SlimeRecipe`](/reference/training/slimerecipe/) | Recipe dataclass for configuring slime GRPO training on Modal. |

## Deployment

| Class | Description |
|-------|-------------|
| [`DeploymentConfig`](/reference/deployment/deploymentconfig/) | Deploy a model behind a serving engine. |
| [`ModelDeployment`](/reference/deployment/modeldeployment/) | A deployed model endpoint. |
| [`SglangRecipe`](/reference/deployment/sglangrecipe/) | SGLang serving configuration. |
| [`VllmRecipe`](/reference/deployment/vllmrecipe/) | vLLM serving configuration. |
