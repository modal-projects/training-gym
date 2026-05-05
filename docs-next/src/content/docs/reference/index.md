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
| [`WandbConfig`](/reference/core/wandbconfig/) | Weights & Biases logging configuration shared across all frameworks.. |
| [`ModalRayCluster`](/reference/core/modalraycluster/) | Base class for bootstrapping a Ray cluster inside Modal clustered functions. |
| [`LlmJudge`](/reference/core/llmjudge/) | LLM-as-judge client for an OpenAI-compatible chat-completions endpoint. |
| [`TrainResult`](/reference/core/trainresult/) | One completed training run's checkpoint handle. |

## Models

| Class | Description |
|-------|-------------|
| [`qwen3-4b`](/reference/models/qwen3_0_6b/) | qwen3-4b (0.6 billion parameters) from Alibaba. |
| [`Qwen3-4B`](/reference/models/qwen3_4b/) | Qwen3-4B (4 billion parameters) from Alibaba. |
| [`Qwen3-32B`](/reference/models/qwen3_32b/) | Qwen3-32B (32 billion parameters) from Alibaba. |
| [`GLM-4.7`](/reference/models/glm_4_7/) | GLM-4.7 large MoE model from Zhipu AI. |
| [`Llama2-7B`](/reference/models/llama2_7b/) | Llama 2 7B from Meta. |
| [`Kimi-K2.5`](/reference/models/kimi_k2_5/) | Kimi K2.5 from Moonshot AI. |

## Frameworks

| Class | Description |
|-------|-------------|
| [`SlimeConfig`](/reference/frameworks/slimeconfig/) | slime GRPO training configuration. |
| [`ModalConfig (slime)`](/reference/frameworks/modalconfig/) | Modal infrastructure configuration for slime — image setup and dev overlays. |
