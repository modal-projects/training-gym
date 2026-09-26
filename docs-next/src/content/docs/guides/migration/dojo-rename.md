---
order: 1
---

# Migrating from Training Gym to Modal Dojo

The Training Gym has been renamed to Modal Dojo. Everything you already have — checkpoints, datasets, Modal volumes, secrets, and the `training-gym` Modal environment — keeps working unchanged. Only the names you reference from your own code and shell need to be updated.

## Package and import

The PyPI distribution is now `modal-dojo` and the import namespace is `modal_dojo`:

```bash
# Before
uv add modal-training-gym

# After
uv remove modal-training-gym
uv add modal-dojo
```

```python
# Before
from modal_training_gym import TrainConfig

# After
from modal_dojo import TrainConfig
```

The public API is otherwise unchanged. The exception base classes are now `TrainingDojoError` and `TrainingDojoConfigError`.

## CLI

The `training-gym` command is now `modal-dojo`. Every subcommand keeps its name and flags:

```bash
# Before
training-gym setup
training-gym skills install

# After
modal-dojo setup
modal-dojo skills install
```

## Config file

Settings now live at `~/.modal-dojo.toml`. If that file does not exist, the CLI still reads `~/.training-gym.toml`, so nothing breaks until you move it:

```bash
mv ~/.training-gym.toml ~/.modal-dojo.toml
```

## Dashboard

The dashboard Modal app is now `dojo-dashboard`, so its URL changes. Re-run setup to deploy it and print the new URL, then update any saved links:

```bash
modal-dojo setup
```

The old `training-gym-dashboard` app is left in place; remove it with `modal app stop training-gym-dashboard` once you have switched over.

## Agent skills

`modal-dojo skills install` installs `modal-dojo-overview` in place of `training-gym-overview`. If the old skill is present, the command prints a hint; rerun it with `--force` to remove the old copy and its `.claude/skills` link.

## Docs

The documentation now lives at [dojo.modal.dev](https://dojo.modal.dev). Links to `gym.modal.dev` redirect to the same page on the new domain.
