"""Tutorial source for `nanogpt_owt` — parsed by generate_tutorial.py.

Single-file Modal tutorial for karpathy/nanoGPT on OpenWebText. Unlike the
framework-based tutorials, this one doesn't import from
`modal_training_gym.frameworks.*` — it's a direct Modal app that clones
karpathy/nanoGPT into the image at build time and runs the stock scripts
(`train.py`, `bench.py`, `sample.py`) under `torchrun`.

The supplementary `transformer_sizing.ipynb` and `scaling_laws.ipynb` cells
from the upstream repo are appended at the bottom under `## Analysis`.
"""

TUTORIAL_METADATA = {
    'framework': '— (clones `karpathy/nanoGPT`)',
    'cluster_shape': '2 × 8×H100',
    'summary': 'GPT-2 124M on OpenWebText',
    'order': 60,
}


from tutorial_generator import code, markdown, notebook_only, py_only, shell


@markdown
def _title():
    """
    # nanoGPT on Modal — GPT-2 reproduction on OpenWebText

    ![nanoGPT](https://raw.githubusercontent.com/karpathy/nanoGPT/master/assets/nanogpt.jpg)

    The simplest, fastest repository for training/finetuning medium-sized
    GPTs. This tutorial wraps [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
    as-is (cloned into the image at build time) and adds the Modal bits on
    top: cluster-aware multi-node launches, volumes for data + checkpoints,
    and the supporting `prepare_data` / `bench` / `sample` entrypoints.
    """


@markdown
def _readme_overview():
    """
    ## Overview

    The `train.py` file reproduces GPT-2 (124M) on OpenWebText, running on a
    single 8×A100 40GB node in about 4 days of training. The code itself is
    plain and readable: `train.py` is a ~300-line boilerplate training loop
    and `model.py` a ~300-line GPT model definition. Because the code is so
    simple, it's very easy to hack to your needs, train new models from
    scratch, or finetune pretrained checkpoints.

    Pipeline on Modal:

    1. `prepare_data` — tokenizes OpenWebText, writes `train.bin`/`val.bin` to a volume
    2. `train_single_node` — one 8×A100 container (the classic karpathy recipe)
    3. `train_multi_node` — 2+ nodes × 8×H100, clustered via `@modal.experimental.clustered`
    4. `bench_single_gpu` — single-GPU forward/backward profiling
    5. `run_inference` — sample from a trained checkpoint
    """


@markdown
def _readme_mfu():
    """
    ### MFU Performance (from the reference runs)

    | GPU Config | Time per iteration | MFU     |
    |------------|--------------------|---------|
    | 8×A100     | ~360 ms            | ~47.84% |
    | 8×H100     | ~231 ms            | ~25.4%  |
    | 8×H200     | ~225 ms            | ~25.0%  |
    | 2×8×H100   | ~121 ms            | ~24.2%  |
    | 4×8×H100   | ~63 ms             | ~23.3%  |

    Good cluster-scale performance gives ~linear speedup as the number of
    nodes increases.
    """


@notebook_only
@shell("! pip install -q modal matplotlib numpy pandas")
def _install():
    pass


@code
def _imports_and_image():
    import os
    import shutil
    import subprocess

    import modal
    import modal.experimental

    # Instructions for installing flash-attn and friends come from the Modal
    # CUDA guide:
    # https://modal.com/docs/guide/cuda#for-more-complex-setups-use-an-officially-supported-cuda-image
    _CUDA_VERSION = "12.6.0"
    _CUDA_FLAVOR = "devel"
    _CUDA_OS = "ubuntu22.04"
    _CUDA_TAG = f"{_CUDA_VERSION}-{_CUDA_FLAVOR}-{_CUDA_OS}"

    image = (
        modal.Image.from_registry(f"nvidia/cuda:{_CUDA_TAG}", add_python="3.12")
        .apt_install(
            "git",
            "libibverbs-dev",
            "libibverbs1",
            "libhwloc15",
            "libnl-route-3-200",
        )
        .pip_install(
            "torch==2.6.0",
            "transformers==4.51.3",
            "datasets==3.6.0",
            "tiktoken==0.9.0",
            "wandb==0.19.11",
            "tqdm==4.67.1",
        )
        # Clone karpathy/nanoGPT into /root/nanogpt. The rest of the tutorial
        # treats /root/nanogpt as the working directory for every script run.
        .run_commands(
            "git clone https://github.com/karpathy/nanoGPT /root/nanogpt",
        )
    )

    # Volumes for the tokenized data + model checkpoints.
    data_volume = modal.Volume.from_name("nanogpt-owt-data", create_if_missing=True)
    output_volume = modal.Volume.from_name("nanogpt-owt-output", create_if_missing=True)

    app = modal.App("nanogpt-owt", image=image)

    # How many nodes and GPUs. `train_multi_node` reads these at decoration time.
    N_NODES = 2
    N_PROC_PER_NODE = 8


@markdown
def _data_section():
    """
    ## 1. Data preparation

    Run `nanogpt/data/openwebtext/prepare.py` to tokenize OpenWebText into
    two big binary files (`train.bin`, `val.bin`). This is the same script
    from upstream karpathy/nanoGPT — we just run it inside a Modal container
    and copy the results to a `Volume` for reuse across training runs.

    The prep is CPU-bound; we give it a generous CPU budget to avoid a
    bottleneck, and the whole thing typically takes ~30 minutes.
    """


@code
def _prepare_data():
    @app.function(
        timeout=3600,
        cpu=(0.2, 16),
        volumes={"/vol": data_volume},
    )
    def prepare_data():
        """Tokenize OpenWebText and copy `train.bin`/`val.bin` to the data volume."""
        os.environ["TRUST_REMOTE_CODE"] = "true"
        os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "true"
        subprocess.run(
            ["python", "/root/nanogpt/data/openwebtext/prepare.py"], check=True
        )
        print("Copying train.bin and val.bin to the data volume…")
        shutil.copy("/root/nanogpt/data/openwebtext/train.bin", "/vol/train.bin")
        shutil.copy("/root/nanogpt/data/openwebtext/val.bin", "/vol/val.bin")
        data_volume.commit()


@markdown
def _single_node_section():
    """
    ## 2. Single-node training

    Karpathy's classic recipe: one 8×A100 container runs `torchrun` over
    `train.py` with the stock `config/train_gpt2.py`. This is the original
    reference configuration — useful as a baseline before scaling out.

    This will run for about 4 days using PyTorch DDP and reach a loss of
    ~2.85 (see the README's baselines table for how that compares to the
    OpenAI GPT-2 checkpoints).
    """


@code
def _train_single_node_fn():
    def _run_train_from_vol(args: list[str]):
        """Symlink the vol'd data into the expected path, then launch torchrun."""
        from torch.distributed.run import parse_args, run

        for name in ("train.bin", "val.bin"):
            src = f"/vol/{name}"
            dst = f"/root/nanogpt/data/openwebtext/{name}"
            if not os.path.lexists(dst):
                os.symlink(src, dst)
        print(f"Running torchrun {' '.join(args)}")
        run(parse_args(args))

    @app.function(
        gpu=f"A100:{N_PROC_PER_NODE}",
        secrets=[modal.Secret.from_name("wandb-secret")],
        volumes={"/vol": data_volume, "/root/out": output_volume},
        timeout=60 * 60 * 24,
    )
    def train_single_node():
        """Single-container training — 8×A100, classic nanoGPT config."""
        _run_train_from_vol(
            [
                f"--nproc-per-node={N_PROC_PER_NODE}",
                "/root/nanogpt/train.py",
                "config/train_gpt2.py",
                "--max_iters=1",
                "--eval_interval=1",
                "--log_interval=1",
                "--eval_iters=1",
            ]
        )


@markdown
def _multi_node_section():
    """
    ## 3. Multi-node training (clustered)

    Same `train.py`, now launched on 2+ nodes with Modal's
    `@modal.experimental.clustered` decorator. RDMA + EFA are enabled, so
    inter-node gradient all-reduces run over the high-speed fabric. Good
    cluster-scale performance gives ~linear speedup as you add nodes.
    """


@code
def _train_multi_node_fn():
    @app.function(
        gpu=f"H100:{N_PROC_PER_NODE}",
        secrets=[modal.Secret.from_name("wandb-secret")],
        experimental_options={"efa_enabled": True},
        volumes={"/vol": data_volume, "/root/out": output_volume},
        timeout=60 * 60 * 24,
    )
    @modal.experimental.clustered(N_NODES, rdma=True)
    def train_multi_node():
        """Clustered multi-node training. Run with `modal run …::train_multi_node`."""
        from torch.distributed.run import parse_args, run

        cluster_info = modal.experimental.get_cluster_info()
        rank = cluster_info.rank
        master_ip = cluster_info.container_ips[0]
        container_id = os.environ["MODAL_TASK_ID"]

        # Symlink the vol'd data into the expected path.
        for name in ("train.bin", "val.bin"):
            src = f"/vol/{name}"
            dst = f"/root/nanogpt/data/openwebtext/{name}"
            if not os.path.lexists(dst):
                os.symlink(src, dst)

        args = [
            f"--nnodes={N_NODES}",
            f"--nproc-per-node={N_PROC_PER_NODE}",
            f"--node-rank={rank}",
            f"--master-addr={master_ip}",
            "/root/nanogpt/train.py",
            "config/train_gpt2.py",
            "--max_iters=1",
            "--eval_interval=1",
            "--log_interval=1",
            "--eval_iters=1",
            "--wandb_log=True",
            "--wandb_project=nanogpt-owt",
            f"--wandb_run_name=multinode-{container_id}",
        ]
        print(f"[rank {rank}/{N_NODES}] torchrun {' '.join(args)}")
        run(parse_args(args))


@markdown
def _bench_section():
    """
    ## 4. Single-GPU benchmarking

    `bench.py` runs exactly what happens in the meat of `train.py`'s
    training loop but strips away checkpointing, logging, DDP setup, etc.
    Useful for quick profiling / MFU measurements.
    """


@code
def _bench_fn():
    @app.function(
        gpu="H100",
        volumes={"/vol": data_volume, "/root/out": output_volume},
    )
    def bench_single_gpu(profile: bool = False):
        """Benchmark forward+backward on a single GPU."""
        from torch.distributed.run import parse_args, run

        if profile:
            os.environ["NANOGPT_PROFILE"] = "1"

        for name in ("train.bin", "val.bin"):
            src = f"/vol/{name}"
            dst = f"/root/nanogpt/data/openwebtext/{name}"
            if not os.path.lexists(dst):
                os.symlink(src, dst)

        args = ["--nproc-per-node=1", "/root/nanogpt/bench.py"]
        print(f"Running torchrun {' '.join(args)}")
        run(parse_args(args))


@markdown
def _inference_section():
    """
    ## 5. Sampling from a trained checkpoint

    `sample.py` loads checkpoints from the `/root/out` volume and samples
    from them. You can also sample from pre-trained GPT-2 by passing
    `init_from='gpt2'` (or `gpt2-medium`/`gpt2-large`/`gpt2-xl`).
    """


@code
def _inference_fn():
    @app.function(
        gpu="H100",
        volumes={"/root/out": output_volume},
    )
    def run_inference(prompt: str = "What is the answer to life, the universe, and everything?"):
        """Sample from a trained checkpoint sitting in /root/out/ckpts."""
        args = [
            "--init_from=resume",
            "--out_dir=/root/out/ckpts",
            "--num_samples=10",
            "--max_new_tokens=100",
            f"--start={prompt}",
        ]
        subprocess.run(["python", "/root/nanogpt/sample.py", *args], check=True)


@markdown
def _run_section():
    """
    ## Running it
    """


@py_only
@markdown
def _run_cli():
    """
    From the CLI:

    ```bash
    # 1. tokenize OpenWebText (~30 min, CPU)
    uv run modal run tutorials/nanogpt_owt/nanogpt_owt.py::prepare_data

    # 2. single-node baseline (8×A100, multi-day)
    uv run modal run --detach tutorials/nanogpt_owt/nanogpt_owt.py::train_single_node

    # 3. multi-node (set N_NODES above, then run)
    uv run modal run --detach tutorials/nanogpt_owt/nanogpt_owt.py::train_multi_node

    # 4. benchmark a single GPU
    uv run modal run tutorials/nanogpt_owt/nanogpt_owt.py::bench_single_gpu

    # 5. sample from the trained checkpoint
    uv run modal run tutorials/nanogpt_owt/nanogpt_owt.py::run_inference
    ```
    """


@notebook_only
@markdown
def _run_interactive():
    """
    Interactive — run one stage per cell. `train_single_node` and
    `train_multi_node` are alternative launch paths:
    """


@notebook_only
@code
def _invoke_prepare_data():
    with app.run():
        prepare_data.remote()


@notebook_only
@code
def _invoke_train_single_node():
    with app.run():
        train_single_node.remote()


@notebook_only
@code
def _invoke_train_multi_node():
    with app.run():
        train_multi_node.remote()


@notebook_only
@code
def _invoke_bench_single_gpu():
    with app.run():
        bench_single_gpu.remote()


@notebook_only
@code
def _invoke_run_inference():
    with app.run():
        run_inference.remote()


@markdown
def _analysis_section():
    """
    ---

    # Analysis — theoretical sizing & scaling laws

    The rest of the notebook ports karpathy's two analysis notebooks
    (`transformer_sizing.ipynb` and `scaling_laws.ipynb`) into this file so
    everything lives in one place. These cells are pure CPU computation —
    no Modal compute required. They estimate parameter counts, FLOPs,
    memory footprints, and replicate Chinchilla-style scaling-law plots.
    """


@markdown
def _sizing_intro():
    """
    ## Transformer Theoretical Model

    Analysis of a Transformer: estimates the number of FLOPs, parameters,
    peak memory footprint, checkpoint size, etc.
    """


@code
def _sizing_imports():
    from collections import OrderedDict


@code
def _sizing_constants():
    # config_args = {
    #     'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M
    #     'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M
    #     'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M
    #     'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M
    # }[model_type]

    block_size = 1024
    vocab_size = 50257
    n_layer = 12
    n_head = 12
    n_embd = 768
    bias = False
    assert not bias, "this analysis assumes bias=False just for simplicity"


@code
def _sizing_params():
    def params():
        """Estimate number of parameters in the model."""
        out = OrderedDict()
        # token and position embeddings
        out["embedding/position"] = n_embd * block_size
        out["embedding/token"] = n_embd * vocab_size
        out["embedding"] = out["embedding/position"] + out["embedding/token"]
        # attention blocks
        out["attention/ln"] = n_embd
        out["attention/kqv"] = n_embd * 3 * n_embd
        out["attention/proj"] = n_embd**2
        out["attention"] = out["attention/ln"] + out["attention/kqv"] + out["attention/proj"]
        # MLP blocks
        ffw_size = 4 * n_embd
        out["mlp/ln"] = n_embd
        out["mlp/ffw"] = n_embd * ffw_size
        out["mlp/proj"] = ffw_size * n_embd
        out["mlp"] = out["mlp/ln"] + out["mlp/ffw"] + out["mlp/proj"]
        # full transformer
        out["block"] = out["attention"] + out["mlp"]
        out["transformer"] = n_layer * out["block"]
        out["ln_f"] = n_embd
        out["dense"] = 0  # weight-tied with embedding
        out["total"] = out["embedding"] + out["transformer"] + out["ln_f"] + out["dense"]
        return out

    p = params()
    params_total = p["total"]
    print(f"we see: {params_total}, expected: 124337664, match: {params_total == 124337664}")
    print(f"{'name':20s} {'params':10s} {'ratio (%)':10s}")
    for k, v in p.items():
        print(f"{k:20s} {v:10d} {v / params_total * 100:10.4f}")


@code
def _sizing_checkpoint_bytes():
    # fp32 params + 2 AdamW buffers per param
    params_bytes = params_total * 4
    params_and_buffers_bytes = params_bytes + 2 * params_bytes
    print(f"est checkpoint size: {params_and_buffers_bytes / 1e9:.2f} GB")
    measured_bytes = 1542470366  # wc -c ckpt.pt
    print(f"measured with wc -c ckpt.pt: {measured_bytes}")
    print(f"fluff ratio: {measured_bytes / params_and_buffers_bytes * 100:.2f}%")


@markdown
def _sizing_memory_ratio():
    """
    We can also estimate the ratio of GPU memory that will be taken up just
    by the weights and the AdamW optimizer buffers.
    """


@code
def _sizing_memory_calc():
    gpu_memory = 40e9  # 40 GB A100 GPU, roughly
    print(
        f"memory ratio taken up just for parameters: "
        f"{params_and_buffers_bytes / gpu_memory * 100:.2f}%"
    )


@markdown
def _sizing_flops_intro():
    """
    Not that much for this tiny model — most of the memory is activations
    (forward and backward). This changes dramatically for larger models.

    Now let's estimate FLOPs for a single forward pass.
    """


@code
def _sizing_flops():
    def flops():
        # Count weight FLOPs only — LN, Softmax, etc. are effectively irrelevant.
        # Real FLOPs (not MACs), hence 2* everywhere.
        out = OrderedDict()
        head_size = n_embd // n_head
        # attention
        out["attention/kqv"] = 2 * block_size * (n_embd * 3 * n_embd)
        out["attention/scores"] = 2 * block_size * block_size * n_embd
        out["attention/reduce"] = 2 * n_head * (block_size * block_size * head_size)
        out["attention/proj"] = 2 * block_size * (n_embd * n_embd)
        out["attention"] = sum(
            out["attention/" + k] for k in ["kqv", "scores", "reduce", "proj"]
        )
        # MLP
        ffw_size = 4 * n_embd
        out["mlp/ffw1"] = 2 * block_size * (n_embd * ffw_size)
        out["mlp/ffw2"] = 2 * block_size * (ffw_size * n_embd)
        out["mlp"] = out["mlp/ffw1"] + out["mlp/ffw2"]
        # block + full
        out["block"] = out["attention"] + out["mlp"]
        out["transformer"] = n_layer * out["block"]
        out["dense"] = 2 * block_size * (n_embd * vocab_size)
        out["forward_total"] = out["transformer"] + out["dense"]
        out["backward_total"] = 2 * out["forward_total"]  # bwd ≈ 2 × fwd
        out["total"] = out["forward_total"] + out["backward_total"]
        return out

    f = flops()
    flops_total = f["forward_total"]
    print(f"{'name':20s} {'flops':14s} {'ratio (%)':10s}")
    for k, v in f.items():
        print(f"{k:20s} {v:14d} {v / flops_total * 100:10.4f}")


@code
def _sizing_palm_formula():
    # PaLM paper's MFU formula estimate.
    def palm_flops():
        N = params()["total"] - params()["embedding/position"]
        L, H, Q, T = n_layer, n_head, n_embd // n_head, block_size
        mf_per_token = 6 * N + 12 * L * H * Q * T
        return mf_per_token * block_size

    print(
        f"palm_flops: {palm_flops():d}, flops: {flops()['total']:d}, "
        f"ratio: {palm_flops() / flops()['total']:.4f}"
    )


@markdown
def _sizing_mfu_text():
    """
    The PaLM formula lands close to the explicit estimate, giving some
    confidence in the math. A100 is cited at 312 TFLOPS bfloat16 on tensor
    cores, so our MFU on a batch of 100 sequences running in 755 ms looks like:
    """


@code
def _sizing_mfu_calc():
    batch_size = 20 * 5  # 5 is grad_accum → total batch size 100
    measured_time = 0.755  # seconds per iteration
    measured_throughput = batch_size / measured_time
    flops_achieved = f["total"] * measured_throughput
    a100_flops_promised = 312e12  # bfloat16 tensor cores
    print(
        f"fraction of A100 used: "
        f"{flops_achieved / a100_flops_promised * 100:.2f}%"
    )


@code
def _sizing_6nd():
    # 6ND approximation — total training compute in FLOPs.
    model_size = params()["total"]
    tokens_num = 300e9  # 300B tokens
    a100_flops = 312e12
    assumed_mfu = 0.3
    flops_throughput = a100_flops * 8 * assumed_mfu  # 8×A100 at 30% MFU
    flops_needed = 6 * model_size * tokens_num  # 6ND
    time_needed_s = flops_needed / flops_throughput
    print(f"time needed to train the model: {time_needed_s / 3600 / 24:.2f} days")


@markdown
def _scaling_laws_intro():
    """
    ## Scaling Laws

    Reproducing some scaling laws results from
    [Chinchilla](https://arxiv.org/pdf/2203.15556.pdf). Can't get the
    numbers to match exactly but useful as a rough guide.
    """


@code
def _scaling_imports():
    import json

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd


@markdown
def _scaling_params():
    """
    ### Params

    First, parameter calculations:
    """


@code
def _scaling_gpt_params():
    def gpt_params(seq_len, vocab_size, d_model, num_heads, num_layers):
        """Given GPT config, estimate total number of parameters."""
        ffw_size = 4 * d_model
        embeddings = d_model * vocab_size + d_model * seq_len
        attention = 3 * d_model**2 + 3 * d_model
        attproj = d_model**2 + d_model
        ffw = d_model * ffw_size + ffw_size
        ffwproj = ffw_size * d_model + d_model
        layernorms = 2 * 2 * d_model
        ln_f = 2 * d_model
        dense = d_model * vocab_size
        return (
            num_layers * (attention + attproj + ffw + ffwproj + layernorms)
            + ln_f
            + dense
        )

    gpt2 = dict(seq_len=1024, vocab_size=50257, d_model=768, num_heads=12, num_layers=12)
    print(f"gpt2 params: {gpt_params(**gpt2) / 1e6:.2f}M")


@markdown
def _scaling_chinchilla_intro():
    """
    OpenAI reports GPT-2 (small) as having 124M params — the above matches.
    Now Chinchilla parameters (they use relative positional embeddings):
    """


@code
def _scaling_chinchilla_params():
    def chinchilla_params(seq_len, vocab_size, d_model, num_heads, num_layers, ffw_size):
        """Parameters in the Chinchilla models."""
        embeddings = d_model * vocab_size
        attention = 3 * d_model**2 + 3 * d_model
        relative_pos = d_model**2 + 2 * d_model
        attproj = d_model**2 + d_model
        ffw = d_model * ffw_size + ffw_size
        ffwproj = ffw_size * d_model + d_model
        layernorms = 2 * 2 * d_model
        ln_f = 2 * d_model
        dense = d_model * vocab_size
        return (
            num_layers
            * (attention + relative_pos + attproj + ffw + ffwproj + layernorms)
            + ln_f
            + dense
        )


@markdown
def _scaling_loss_intro():
    """
    ### Chinchilla Approach 3: loss fit L(N, D)

    Chinchilla's Approach 3 fits a function L(N, D) to approximate the final
    loss given model size N and dataset size D:

    $$L(N, D) = \\frac{A}{N^\\alpha} + \\frac{B}{D^\\beta} + E$$

    with `E = 1.69`, `A = 406.4`, `B = 410.7`, `α = 0.34`, `β = 0.28`.
    """


@code
def _scaling_loss_plot():
    def L(N, D):
        E, A, B, alpha, beta = 1.69, 406.4, 410.7, 0.34, 0.28
        return A / (N**alpha) + B / (D**beta) + E

    ns = 10 ** np.arange(7, 11, step=2**-4)  # 10M to 100B
    ds = 10 ** np.arange(9, 12, step=2**-4)  # 1B to 1T tokens
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    loss2d = np.log10(np.array([[L(n, d) for d in ds] for n in ns]))
    plt.imshow(loss2d, extent=[9, 12, 7, 11], origin="lower", alpha=0.5)
    plt.contour(loss2d, levels=30, extent=[9, 12, 7, 11], origin="lower")
    plt.xlabel("log10(dataset size)")
    plt.ylabel("log10(model size)")
    plt.title("loss")
    plt.colorbar()
    plt.subplot(122)
    compute2d = np.log10(np.array([[6 * n * d for d in ds] for n in ns]))
    plt.imshow(compute2d, extent=[9, 12, 7, 11], origin="lower", alpha=0.5)
    plt.contour(compute2d, levels=30, extent=[9, 12, 7, 11], origin="lower")
    plt.xlabel("log10(dataset size)")
    plt.ylabel("log10(model size)")
    plt.title("log10 flops")
    plt.colorbar()


@markdown
def _scaling_optimum_intro():
    """
    Given any N and D we can estimate the loss AND the compute
    (`FLOPs(N, D) = 6·N·D`). So for a fixed compute budget C, we can solve:

    $$N^*, D^* = \\arg\\min_{6ND = C} L(N, D)$$

    i.e. how big should the model be, and for how many tokens?
    """


@code
def _scaling_optimum():
    def L(N, D):
        return 406.4 / (N**0.34) + 410.7 / (D**0.28) + 1.69

    c = 2.21e19  # target compute budget
    ns = 10 ** np.arange(7, 11, step=2**-4)
    ds = c / (6 * ns)  # keep 6ND = C
    losses = L(ns, ds)
    best = np.argmin(losses)
    print(f"best model size: {ns[best] / 1e6:.2f}M")
    print(f"best dataset size: {ds[best] / 1e9:.2f}B")

    plt.figure(figsize=(3, 3))
    plt.plot(ns, losses)
    plt.xscale("log")
    plt.axvline(ns[best], color="red")
    plt.xlabel("model size")
    plt.ylabel("loss")
