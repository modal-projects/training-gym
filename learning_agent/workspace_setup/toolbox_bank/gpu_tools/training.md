# Training apps on Modal

One Python file: bake your code into the image, mount the shared volumes, run
it. `toolbox/gpu_tools/gpu_launcher.py` already does this for one-off
commands; write
your own app when you need retries, servers, or multi-node.

## The pattern

```python
import modal

app = modal.App("my-training")

out = modal.Volume.from_name("lab-out")
hf = modal.Volume.from_name("lab-hf-cache")

image = (
    modal.Image.from_registry("nvidia/cuda:12.6.0-devel-ubuntu22.04",
                              add_python="3.12")
    .pip_install("torch", "transformers", "datasets", "accelerate")
    .env({"HF_HOME": "/hf-cache"})
    # copy=True whenever add_local_file/add_local_dir is followed by
    # .run_commands() — without it the build step can't see the file.
    .add_local_file("train.py", remote_path="/root/train.py", copy=True)
)

@app.function(
    image=image,
    gpu="H200",                     # H200 only — README rule 2
    timeout=60 * 60 * 12,
    volumes={"/out": out, "/hf-cache": hf},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    retries=modal.Retries(max_retries=3),
)
def train(config: str = "default.yaml"):
    import subprocess
    subprocess.run(["python", "/root/train.py", "--config", config],
                   check=True)

@app.local_entrypoint()
def main(config: str = "default.yaml"):
    train.remote(config=config)
```

```bash
modal run my_train.py --config large.yaml
```

Checkpoints go under `/out/models/$LEARNING_AGENT_RUN_ID/<tag>/`, final merged weights
at `/out/models/$LEARNING_AGENT_RUN_ID/<tag>/merged` — `/out` is shared across runs and
your run id is your namespace ($LEARNING_AGENT_RUN_ID is in your environment, and
`gpu_launcher.py` injects it into every sandbox). When the run finishes,
append the GPU accounting line to `runs/GPU_LOG.jsonl` (README rule 5).

## Fault tolerance

`retries=modal.Retries(max_retries=N)` plus a script that detects an
existing checkpoint and resumes from it gives fault-tolerant training: if
the container is preempted, Modal retries the call and your script picks up
from the last checkpoint on `/out` instead of starting over.

## Volumes

Writes to a mounted volume persist when the function calls
`volume.commit()` (sandbox-attached volumes also flush automatically on
termination). Inspect from outside:

```bash
modal volume ls  lab-out /models
modal volume get lab-out /models/<tag>/merged/config.json ./
modal volume put lab-hf-cache ./local_file /remote/path
```

## Multi-node

More than 8 GPUs means multiple nodes, gang-scheduled and connected with
RDMA:

```python
import modal.experimental

@app.function(
    gpu="H200:8",
    timeout=60 * 60 * 24,
    retries=modal.Retries(initial_delay=0.0, max_retries=10),
    experimental_options={"efa_enabled": True},
)
@modal.experimental.clustered(size=2, rdma=True)
def train(config: str):
    info = modal.experimental.get_cluster_info()
    # info.rank (0 = leader), info.container_ips (sorted by rank) —
    # hand these to torchrun as node_rank / master_addr.
```

1. All containers launch together or not at all; if any is preempted,
   Modal terminates the cluster and retries the whole input — same
   checkpoint-resume pattern as above.
2. Only rank 0's return value reaches the caller.
3. Set `NCCL_NVLS_ENABLE=0`, `CUDA_DEVICE_MAX_CONNECTIONS=1`, and
   `PYTHONUNBUFFERED=1` in the training function for reliable multi-node
   NCCL.
