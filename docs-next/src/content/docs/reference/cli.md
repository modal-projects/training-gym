---
title: CLI Reference
description: Command-line interface for the Training Gym SDK.
---

The `training-gym` CLI is installed automatically with the package.

```bash
pip install git+https://github.com/modal-projects/training-gym.git@main
```

## `training-gym setup`

Deploy the observability dashboard to your Modal account.

```bash
training-gym setup
```

This is equivalent to calling `setup()` from Python:

```python
import modal_training_gym

modal_training_gym.setup()
```

The command builds a Modal image (Node.js + Svelte frontend + FastAPI backend),
deploys it as a persistent web app, and prints the dashboard URL.

**Prerequisites:** a Modal account with `modal token set` already configured.

## `training-gym --help`

```
usage: training-gym [-h] {setup} ...

positional arguments:
  {setup}
    setup     Deploy the training-gym dashboard to Modal

options:
  -h, --help  show this help message and exit
```
