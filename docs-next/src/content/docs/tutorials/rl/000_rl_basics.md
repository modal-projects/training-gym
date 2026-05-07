---
title: "Qwen3-4B haiku evaluation with verifiable rewards — serve, evaluate, train, compare"
description: "Qwen3-4B haiku evaluation with verifiable rewards — serve, evaluate, train, compare"
---

Let's also define a Haiku dataset.
Here, we use the statworx/haiku dataset from HuggingFace.
Each row has a `keywords` topic and a reference `text` haiku.
We can use this dataset to train our model.

Seems straightforward enough, right? How do we run an eval on our base model with this dataset?
We can transform our scoring function above into an Eval Configuration.

First, to explain, an Eval Configuration is a class that owns the model-calling loop.
The task-specific part is a scoring function passed to `.evaluate(...)`, which must
return `EvalRowResult`.

---

## Related API Reference

- [`Qwen3_4B`](/reference/models/qwen3_4b/)
- [`DeploymentConfig`](/reference/deployment/deploymentconfig/)
- [`EvalConfig`](/reference/evaluation/evalconfig/)
- [`EvalRowResult`](/reference/evaluation/evalrowresult/)
- [`TrainConfig`](/reference/training/trainconfig/)
- [`SlimeRecipe`](/reference/training/slimerecipe/)
- [`WandbConfig`](/reference/core/wandbconfig/)
- [`TrainResult`](/reference/core/trainresult/)

**Source:** [`tutorials/rl/000_rl_basics/000_rl_basics.py`](https://github.com/modal-projects/training-gym/blob/main/tutorials/rl/000_rl_basics/000_rl_basics.py)
 | <a href="https://modal.com/notebooks/new/https://github.com/modal-projects/training-gym/blob/main/tutorials/rl/000_rl_basics/000_rl_basics.ipynb" target="_blank" rel="noopener noreferrer">Open in Modal Notebook</a>
