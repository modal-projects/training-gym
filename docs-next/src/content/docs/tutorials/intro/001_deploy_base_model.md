---
title: "Deploy the base model"
description: "Qwen3-4B haiku evaluation with verifiable rewards — serve, evaluate, train, compare"
---

This tutorial shows you how to deploy a base model on Modal to use for evaluation. Here, we will use Qwen3-4B as the base model, but
you can use any other model that is supported by the training-gym, or configure your own.

```python
from modal_training_gym import (
    DeploymentConfig,
    Qwen3_4B,
)
```

## Serve the base model

To deploy the base model, we can use the `DeploymentConfig` class. This allows us to deploy the model on Modal and get a handle to the deployed model.

```python
base_model = Qwen3_4B()
base_model_deployment = DeploymentConfig(
    model=base_model,
).serve()
print(f"Base model deployed to {base_model_deployment.url}")
```

The `chat_template_kwargs` parameter allows us to pass in a dictionary of kwargs to the chat template.
This is useful for passing in additional parameters to the chat template, such as the enable_thinking parameter.

---

## Related API Reference

- [`Qwen3_4B`](/reference/models/qwen3_4b/)
- [`DeploymentConfig`](/reference/deployment/deploymentconfig/)

**Source:** [`tutorials/intro/001_deploy_base_model/001_deploy_base_model.py`](https://github.com/modal-projects/training-gym/blob/main/tutorials/intro/001_deploy_base_model/001_deploy_base_model.py)
 | <a href="https://modal.com/notebooks/new/https://github.com/modal-projects/training-gym/blob/main/tutorials/intro/001_deploy_base_model/001_deploy_base_model.ipynb" target="_blank" rel="noopener noreferrer">Open in Modal Notebook</a>
