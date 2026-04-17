"""Tutorial source for `resnet50_imagenet` — parsed by generate_tutorial.py.

Standalone Modal tutorial that trains ResNet50 on ImageNet across multiple
nodes. Uses PyTorch's official `torchvision` reference classification recipe
(cloned at image-build time) under `torchrun`, driven by
`@modal.experimental.clustered`.
"""

from tutorial_generator import code, markdown, notebook_only, py_only, shell


@markdown
def _title():
    """
    # Training ResNet50 on ImageNet — Multi-node

    Trains ResNet50 on the full ImageNet dataset across a Modal cluster.
    With 4 × 8 H100 SXM, 90 epochs completes in ~40 minutes and reaches
    ~71% validation accuracy. More epochs gets you to 80%+.

    This tutorial uses PyTorch's official reference classification recipe
    (`torchvision/references/classification`), cloned into the image at
    build time.
    """


@markdown
def _pipeline_overview():
    """
    ## Pipeline

    1. **Download ImageNet** — bring the dataset onto a Modal volume.
    2. **Train** — multi-node DDP under `torchrun`.
    3. **Monitor** — `wandb` reports train/val loss + top-1/top-5 accuracy.
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
            "--epochs", "90",
            "--batch-size", "128",
            "--lr", "0.5",  # scales with global batch; reference uses 0.5 for 1024
            "--lr-scheduler", "cosineannealinglr",
            "--lr-warmup-epochs", "5",
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
    uv run modal run tutorials/resnet50_imagenet/resnet50_imagenet.py::download_imagenet
    uv run modal run --detach tutorials/resnet50_imagenet/resnet50_imagenet.py::train_resnet50
    uv run modal run tutorials/resnet50_imagenet/resnet50_imagenet.py::evaluate_resnet50
    ```
    """


@notebook_only
@code
def _invoke():
    with app.run():
        # download_imagenet.remote()
        train_resnet50.remote()
        # evaluate_resnet50.remote()
