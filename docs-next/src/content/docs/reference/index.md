---
title: API Reference
description: API reference for training-gym public classes.
---

# API Reference

Complete reference for the training-gym Python library.

## Core

| Class | Description |
|-------|-------------|
| [`ModelConfig`](/reference/core/modelconfig/) | Base class for model identity and weight-download logic. |
| [`HFModelConfiguration`](/reference/core/hfmodelconfiguration/) | ModelConfig for models hosted on HuggingFace. |
| [`ModelArchitecture`](/reference/core/modelarchitecture/) | Transformer architecture parameters for a specific model. |
| [`DatasetConfig`](/reference/core/datasetconfig/) | Dataset configuration shared across training frameworks. |
| [`HuggingFaceDataset`](/reference/core/huggingfacedataset/) | Dataset backed by a HuggingFace ``datasets`` repo. |
| [`WandbConfig`](/reference/core/wandbconfig/) | Weights & Biases logging configuration shared across all frameworks.. |
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
| [`Qwen3-4B`](/reference/models/qwen3_4b/) | Qwen3-4B (4 billion parameters) from Alibaba. |

## Training

| Class | Description |
|-------|-------------|
| [`TrainConfig`](/reference/training/trainconfig/) | Compose dataset, model, and recipe into one training entrypoint. |
| [`SlimeRecipe`](/reference/training/slimerecipe/) | Recipe dataclass for configuring slime GRPO training on Modal. |

## Deployment

| Class | Description |
|-------|-------------|
| [`DeploymentConfig`](/reference/deployment/deploymentconfig/) | Deploy a model behind a serving engine. |
| [`ModelDeployment`](/reference/deployment/modeldeployment/) | A deployed model endpoint. |
| [`SglangRecipe`](/reference/deployment/sglangrecipe/) | SGLang serving configuration. |
| [`VllmRecipe`](/reference/deployment/vllmrecipe/) | vLLM serving configuration. |
