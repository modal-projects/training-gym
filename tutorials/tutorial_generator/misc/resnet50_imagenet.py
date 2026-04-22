"""Tutorial source for `resnet50_imagenet` — parsed by generate_tutorial.py.

Standalone Modal tutorial that trains ResNet50 on ImageNet across multiple
nodes. Uses PyTorch's official `torchvision` reference classification recipe
(cloned at image-build time) under `torchrun`, driven by
`@modal.experimental.clustered`.
"""

TUTORIAL_METADATA = {
    'framework': '— (clones `pytorch/vision`)',
    'cluster_shape': '4 × 8×H100',
    'summary': 'ResNet50 on ImageNet (DDP)',
    'difficulty': 'Intermediate',
    'order': 80,
}


from tutorial_generator import code, markdown, notebook_only, py_only, shell


@markdown
def _intro():
    """
    # ResNet50 on ImageNet — multi-node DDP on Modal

    **What this tutorial is.** A standalone Modal app (no
    `modal_training_gym.frameworks.*` import) that wraps PyTorch's
    official reference classification recipe
    (`torchvision/references/classification/train.py`, cloned into the
    image at build time) and runs it under `torchrun` on a 4 × 8×H100
    cluster via `@modal.experimental.clustered`. Useful as a reference
    for "I want to run an upstream recipe as-is on a Modal cluster"
    without a framework wrapper.

    **What it runs.** ResNet50 on ImageNet-1K with the stock
    torchvision hyperparameters — SGD + cosine decay, TrivialAugment,
    label smoothing, MixUp + CutMix. Reference: 90 epochs ≈ 40 min on
    4 × 8×H100, ~71% top-1. More epochs gets you to 80%+. (The script
    below is set to `--epochs 1` for a smoke run; bump for real
    training.)

    **Pipeline.**

    1. `download_imagenet` — pulls `ILSVRC/imagenet-1k` from HF (~155GB)
       onto a `imagenet` volume. Needs `huggingface-secret`.
    2. `train_resnet50` — clustered multi-node DDP under `torchrun`.
       RDMA + EFA on, retries up to 10 times.
    3. `evaluate_resnet50` — `--test-only` against the val split.

    **What to watch.** W&B (connect via `wandb-secret`). Val top-1
    should climb toward 70% by epoch 30 on the stock recipe.
    """


@notebook_only
@shell("! pip install -q modal")
def _install():
    pass


@code
def _imports_and_image():
    import os
    import subprocess

    import modal
    import modal.experimental

    _CUDA_VERSION = "12.4.0"
    _CUDA_FLAVOR = "devel"
    _CUDA_OS = "ubuntu22.04"
    _CUDA_TAG = f"{_CUDA_VERSION}-{_CUDA_FLAVOR}-{_CUDA_OS}"

    N_NODES = 4
    N_PROC_PER_NODE = 8

    # Build the training image. We pin torch + torchvision, add wandb for
    # logging, and clone pytorch/vision (pinned) for its reference
    # classification recipe at /root/vision.
    image = (
        modal.Image.from_registry(f"nvidia/cuda:{_CUDA_TAG}", add_python="3.12")
        .apt_install("git", "curl", "libibverbs-dev", "libibverbs1", "libhwloc15", "libnl-route-3-200")
        .pip_install(
            "torch==2.6.0",
            "torchvision==0.21.0",
            "wandb==0.19.11",
            "tqdm==4.67.1",
        )
        .run_commands(
            "git clone --depth 1 https://github.com/pytorch/vision.git /root/vision",
        )
    )

    data_volume = modal.Volume.from_name("imagenet", create_if_missing=True)

    app = modal.App("resnet50-imagenet", image=image)


@markdown
def _download_section():
    """
    ## 1. Download ImageNet

    ImageNet is gated on HuggingFace (`ILSVRC/imagenet-1k`). Configure a
    `huggingface-secret` Modal secret containing `HF_TOKEN` before running
    this stage. The download is large (~155 GB) and takes a while.
    """


@code
def _download_fn():
    @app.function(
        volumes={"/data": data_volume},
        secrets=[modal.Secret.from_name("huggingface-secret")],
        timeout=12 * 60 * 60,
        cpu=(0.2, 16),
    )
    def download_imagenet():
        """Pull ImageNet-1K shards from HuggingFace into the data volume."""
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id="ILSVRC/imagenet-1k",
            repo_type="dataset",
            local_dir="/data/imagenet",
            token=os.environ["HF_TOKEN"],
        )
        print("ImageNet downloaded to /data/imagenet")
        data_volume.commit()


@markdown
def _train_section():
    """
    ## 2. Multi-node training

    Launches `torchvision/references/classification/train.py` under
    `torchrun` on each clustered node. RDMA + EFA enabled, DDP gradient
    all-reduces flow over the high-speed fabric. Reference hyperparameters
    — SGD + cosine decay, 90 epochs, label smoothing — come straight from
    the torchvision recipe.
    """


@code
def _train_fn():
    @app.function(
        gpu=f"H100:{N_PROC_PER_NODE}",
        volumes={"/data": data_volume},
        secrets=[modal.Secret.from_name("wandb-secret")],
        experimental_options={"efa_enabled": True},
        timeout=6 * 60 * 60,
        retries=modal.Retries(initial_delay=0.0, max_retries=10),
    )
    @modal.experimental.clustered(size=N_NODES, rdma=True)
    def train_resnet50():
        """Clustered multi-node ResNet50 training via torchrun."""
        from torch.distributed.run import parse_args, run

        cluster_info = modal.experimental.get_cluster_info()
        rank = cluster_info.rank
        master_ip = cluster_info.container_ips[0]
        container_id = os.environ["MODAL_TASK_ID"]

        print(f"[rank {rank}/{N_NODES}] container {container_id}, master {master_ip}")

        # Torchvision reference recipe: ResNet50 on ImageNet, 90 epochs.
        # Hyperparameters match the official recipe
        # (https://github.com/pytorch/vision/tree/main/references/classification).
        args = [
            f"--nnodes={N_NODES}",
            f"--nproc-per-node={N_PROC_PER_NODE}",
            f"--node-rank={rank}",
            f"--master-addr={master_ip}",
            "--master-port=29500",
            "/root/vision/references/classification/train.py",
            "--model", "resnet50",
            "--data-path", "/data/imagenet",
            "--epochs", "1",
            "--batch-size", "128",
            "--lr", "0.5",  # scales with global batch; reference uses 0.5 for 1024
            "--lr-scheduler", "cosineannealinglr",
            "--lr-warmup-epochs", "0",
            "--lr-warmup-method", "linear",
            "--auto-augment", "ta_wide",
            "--ra-sampler",
            "--label-smoothing", "0.1",
            "--mixup-alpha", "0.2",
            "--cutmix-alpha", "1.0",
            "--weight-decay", "2e-5",
            "--norm-weight-decay", "0.0",
            "--output-dir", "/data/checkpoints/resnet50",
        ]
        print(f"Running torchrun {' '.join(args)}")
        run(parse_args(args))


@markdown
def _eval_section():
    """
    ## 3. (Optional) Evaluate

    Quick top-1/top-5 check against the val set with a trained checkpoint.
    """


@code
def _eval_fn():
    @app.function(
        gpu="H100",
        volumes={"/data": data_volume},
        timeout=60 * 60,
    )
    def evaluate_resnet50():
        """Run the torchvision reference eval against the val split."""
        subprocess.run(
            [
                "python",
                "/root/vision/references/classification/train.py",
                "--test-only",
                "--model", "resnet50",
                "--data-path", "/data/imagenet",
                "--resume", "/data/checkpoints/resnet50/checkpoint.pth",
            ],
            check=True,
        )


@py_only
@markdown
def _run_cli():
    """
    ```bash
    uv run modal run tutorials/misc/resnet50_imagenet/resnet50_imagenet.py::download_imagenet
    uv run modal run --detach tutorials/misc/resnet50_imagenet/resnet50_imagenet.py::train_resnet50
    uv run modal run tutorials/misc/resnet50_imagenet/resnet50_imagenet.py::evaluate_resnet50
    ```
    """


@notebook_only
@markdown
def _run_interactive():
    """
    Interactive — open an ephemeral app and run one stage per cell:
    """


@notebook_only
@code
def _invoke_download_imagenet():
    with app.run():
        download_imagenet.remote()


@notebook_only
@code
def _invoke_train_resnet50():
    with app.run():
        train_resnet50.remote()


@notebook_only
@code
def _invoke_evaluate_resnet50():
    with app.run():
        evaluate_resnet50.remote()
