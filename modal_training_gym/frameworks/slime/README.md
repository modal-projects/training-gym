# slime — Modal launcher for slime training

Thin Modal launcher that runs [slime](https://github.com/THUDM/slime) RL training on GPU clusters.

## Prerequisites

- Modal CLI installed and authenticated
- Set your Modal environment: `export MODAL_ENVIRONMENT=<your-env>`

## Running an experiment

All commands take the experiment name via `EXPERIMENT_CONFIG`. Run from the repo root.

### 1. List available experiments

```bash
modal run slime/modal_train.py::list_configs
```

### 2. Download model (one-time)

Downloads the experiment's HF checkpoint to the `huggingface-cache` volume.

```bash
EXPERIMENT_CONFIG=glm47_flash_dapo modal run slime/modal_train.py::download
```

### 3. Prepare dataset (one-time)

Downloads and preprocesses the training dataset to the `slime-data` volume.
Only required if the experiment defines a `prepare_data()` function (see [Adding an experiment](#adding-an-experiment)).

```bash
EXPERIMENT_CONFIG=glm47_flash_dapo modal run slime/modal_train.py::prepare_dataset
```

### 4. Convert checkpoint (one-time, raw mode only)

Converts the HF checkpoint to `torch_dist` format. Only required when `megatron_to_hf_mode = "raw"`.
Skip this step if using bridge mode.

```bash
EXPERIMENT_CONFIG=glm47_flash_dapo modal run slime/modal_train.py::convert_checkpoint
```

### 5. Run training

```bash
EXPERIMENT_CONFIG=glm47_flash_dapo modal run -d slime/modal_train.py::train
```

Use `-d` (detached) to keep training running after you close your terminal.

## Adding an experiment

### 1. Create the config file

Create `configs/<your_experiment>.py`. Each config file must expose two module-level instances:
- `modal` — a `ModalConfig` instance (GPU type, image patches)
- `slime` — a `SlimeConfig` subclass instance (all slime training arguments)

```python
from .base import ModalConfig, SlimeConfig, DATA_PATH

modal = ModalConfig(gpu="H200")


class _Slime(SlimeConfig):
    # Launcher instructions (not passed to slime CLI)
    slime_model_script = "scripts/models/qwen3-8B.sh"  # sources MODEL_ARGS
    async_mode = False

    # Model
    hf_checkpoint = "Qwen/Qwen3-8B"
    load = "Qwen/Qwen3-8B"
    megatron_to_hf_mode = "bridge"  # or "raw" (requires convert_checkpoint)

    # Infrastructure
    actor_num_nodes = 1
    actor_num_gpus_per_node = 8
    colocate = True

    # Data
    prompt_data = f"{DATA_PATH}/my_dataset/train.parquet"
    input_key = "problem"
    label_key = "answer"
    rm_type = "math"

    # ... all other slime args as snake_case attributes


slime = _Slime()
```

Every attribute on `_Slime` (except `environment`, `async_mode`, `slime_model_script`) is forwarded to
slime as a CLI argument: `field_name` → `--field-name`. See `configs/base.py` for full rules.

### 2. Add a `prepare_data()` method (if needed)

If your experiment needs to download or preprocess a dataset, override `prepare_data()` on `_Slime`.
It runs inside the Modal container with the `slime-data` volume mounted at `DATA_PATH`.

```python
class _Slime(SlimeConfig):
    ...
    def prepare_data(self) -> None:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id="org/my-dataset",
            repo_type="dataset",
            local_dir=f"{DATA_PATH}/my_dataset",
        )
```

If `prepare_data()` is not overridden, `prepare_dataset` will raise `NotImplementedError` — simply skip that step.

### 3. Run the workflow

`EXPERIMENT_CONFIG` is the config filename without `.py`:

```bash
EXPERIMENT_CONFIG=my_experiment modal run slime/modal_train.py::download
EXPERIMENT_CONFIG=my_experiment modal run slime/modal_train.py::prepare_dataset  # if prepare_data() defined
EXPERIMENT_CONFIG=my_experiment modal run slime/modal_train.py::convert_checkpoint  # if megatron_to_hf_mode = "raw"
EXPERIMENT_CONFIG=my_experiment modal run -d slime/modal_train.py::train
```

No registration step needed — the launcher discovers configs automatically from the `configs/` directory.

## YAML config fields

`eval_config`, `custom_config_path`, and `sglang_config` normally take file paths in slime.
In Python configs you can write them as inline dicts — the launcher materializes them to temp YAML files automatically:

```python
class _Slime(SlimeConfig):
    eval_config = {
        "eval": {
            "defaults": {"max_response_len": 16384},
            "datasets": [
                {"name": "aime", "path": "/data/aime.jsonl", "rm_type": "deepscaler"},
            ],
        }
    }
    custom_config_path = {
        "max_turns": 3,
        "rollout_interaction_env_path": "examples.my_env.rollout",
    }
```

## Dev overlay

To test local slime changes without rebuilding the image, set `local_slime` in your `ModalConfig`:

```python
modal = ModalConfig(
    gpu="H200",
    local_slime="/path/to/your/slime",
)
```

## Applying patches to the image

To inject local patch files into the image (e.g. to patch SGLang), use `patch_files` and `image_run_commands`:

```python
modal = ModalConfig(
    gpu="H200",
    patch_files=["patches/sglang_fix.patch"],
    image_run_commands=["cd /sgl-workspace/sglang && git apply /tmp/sglang_fix.patch"],
)
```

Each file in `patch_files` is added to the image at `/tmp/<filename>`.
