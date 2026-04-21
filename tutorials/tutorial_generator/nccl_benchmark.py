"""Tutorial source for `nccl_benchmark` — parsed by generate_tutorial.py.

Standalone Modal tutorial for the NCCL all-reduce bandwidth benchmark.
No framework wrapping — this measures the fabric, not a training library.
"""

TUTORIAL_METADATA = {
    'framework': '— (NCCL utility)',
    'cluster_shape': '2 × 8×H100',
    'summary': 'All-reduce bandwidth benchmark (500000×2000 fp32)',
    'order': 90,
}


from tutorial_generator import code, markdown, notebook_only, py_only, shell


@markdown
def _title():
    """
    # Multi-node NCCL Benchmark

    Simple NCCL bandwidth benchmark for Modal's multi-node training
    infrastructure. This benchmark measures the all-reduce performance
    between nodes using a large tensor (500000×2000 float32), in order to
    understand the communication performance of your multi-node setup.
    """


@markdown
def _metrics():
    """
    ## Performance Metrics

    - **`algbw` (Algorithm Bandwidth)**: effective bandwidth from the
      application's perspective.
    - **`busbw` (Bus Bandwidth)**: actual hardware bandwidth utilization.

    Example output:

    ```
    The average bandwidth of all_reduce with a 4.0GB payload (50 trials, 16 ranks):
      algbw: 239.742 GBps (1917.9 Gbps)
      busbw: 449.515 GBps (3596.1 Gbps)
    ```

    For more details on these metrics, see the
    [NVIDIA NCCL Tests documentation](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md#bandwidth).
    """


@notebook_only
@shell("! pip install -q modal")
def _install():
    pass


@code
def _imports_and_image():
    import os
    import textwrap

    import modal
    import modal.experimental

    _CUDA_VERSION = "12.4.0"
    _CUDA_FLAVOR = "devel"
    _CUDA_OS = "ubuntu22.04"
    _CUDA_TAG = f"{_CUDA_VERSION}-{_CUDA_FLAVOR}-{_CUDA_OS}"

    N_NODES = 2
    N_PROC_PER_NODE = 8

    image = (
        modal.Image.from_registry(f"nvidia/cuda:{_CUDA_TAG}", add_python="3.12")
        .apt_install(
            "libibverbs-dev",
            "libibverbs1",
            "libhwloc15",
            "libnl-route-3-200",
        )
        .uv_pip_install("torch==2.6.0", "numpy", "importlib-metadata")
    )

    app = modal.App("multinode-nccl-benchmark", image=image)


@markdown
def _bench_script_section():
    """
    ## Benchmark script

    The script below does the actual all-reduce timing — warmup iterations,
    then 50 trials of `dist.all_reduce` on a `500000×2000 fp32` tensor
    (4 GB). It ends up baked into the image via `.add_local_file`.
    """


@code
def _define_script():
    BENCH_SCRIPT = textwrap.dedent(r'''
        import os

        import torch
        import torch.distributed as dist

        # Defaults borrowed from the EleutherAI cookbook.
        WARMUP_ITERS, TRIALS = 5, 50

        # Emulate a 4 GB fp32 payload: 500000 × 2000 × 4 bytes.
        N = 500000
        M = 2000


        def sync_all():
            torch.cuda.synchronize()
            dist.barrier()


        def timed_allreduce(mat, start_event, end_event, warmup_iters, iters, local_rank):
            sync_all()
            for _ in range(warmup_iters):
                dist.all_reduce(mat)
            sync_all()

            start_event.record()
            for _ in range(iters):
                dist.all_reduce(mat)
            end_event.record()

            sync_all()
            duration = start_event.elapsed_time(end_event) / 1000
            avg_duration = duration / iters

            n = dist.get_world_size()
            size = M * N * 4  # fp32
            algbw = torch.tensor([size / avg_duration]).cuda(local_rank)

            # Mean across all ranks.
            dist.reduce(algbw, dst=0, op=dist.ReduceOp.SUM)
            algbw /= n

            return algbw.item()


        def run(local_rank):
            is_global_rank_0 = dist.get_rank() == 0

            mat = torch.rand(N, M, dtype=torch.float32).cuda(local_rank)

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            algbw = timed_allreduce(
                mat, start_event, end_event,
                warmup_iters=WARMUP_ITERS, iters=TRIALS, local_rank=local_rank,
            )

            # 2*(n-1)/n busbw correction for all-reduce — see
            # https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md#allreduce
            n = dist.get_world_size()
            busbw = algbw * (2 * (n - 1) / n)

            if is_global_rank_0:
                print(
                    f"The average bandwidth of all_reduce with a {M * N * 4 / 1e9}GB "
                    f"payload ({TRIALS} trials, {n} ranks):\n"
                    f" algbw: {algbw / 1e9:.3f} GBps ({algbw * 8 / 1e9:.1f} Gbps)\n"
                    f" busbw: {busbw / 1e9:.3f} GBps ({busbw * 8 / 1e9:.1f} Gbps)\n"
                )


        def init_processes(local_rank, fn, backend="nccl"):
            torch.cuda.set_device(local_rank)
            dist.init_process_group(backend, device_id=torch.device(f"cuda:{local_rank}"))
            if dist.get_rank() == 0:
                print("Starting benchmark...")

            fn(local_rank)

            sync_all()
            dist.destroy_process_group()


        if __name__ == "__main__":
            local_rank = int(os.environ["LOCAL_RANK"])
            init_processes(local_rank=local_rank, fn=run)
    ''').lstrip("\n")

    # Materialize the script on the image at a known path.
    _script_remote_path = "/root/nccl_bench.py"
    image = image.run_commands(
        f"mkdir -p /root && cat > {_script_remote_path} <<'PYEOF'\n{BENCH_SCRIPT}\nPYEOF"
    )


@markdown
def _run_bench_section():
    """
    ## Run the benchmark

    `@modal.experimental.clustered(N_NODES, rdma=True)` brings up two RDMA
    peers; each node spawns 8 GPU workers via torchrun. Rank 0 prints the
    algorithm / bus bandwidth.
    """


@code
def _run_benchmark_fn():
    @app.function(
        gpu=f"H100:{N_PROC_PER_NODE}",
        image=image,
        # EFA unlocks extra H100 capacity; needs libhwloc15 + libnl-route-3-200
        # in the image (we installed them above) and a CUDA runtime.
        experimental_options={"efa_enabled": True},
        timeout=60 * 60,
    )
    @modal.experimental.clustered(size=N_NODES, rdma=True)
    def run_benchmark():
        """All-reduce a 4 GB tensor across the cluster and report bandwidth."""
        from torch.distributed.run import parse_args, run

        cluster_info = modal.experimental.get_cluster_info()
        rank = cluster_info.rank
        master_ip = cluster_info.container_ips[0]
        container_id = os.environ["MODAL_TASK_ID"]
        print(f"hello from {container_id}, rank {rank} of {N_NODES}")
        if rank == 0:
            print(f"main container's address: {master_ip}")

        args = [
            f"--nnodes={N_NODES}",
            f"--nproc-per-node={N_PROC_PER_NODE}",
            f"--node-rank={rank}",
            f"--master-addr={master_ip}",
            "/root/nccl_bench.py",
        ]
        print(f"Running torchrun with args: {' '.join(args)}")
        run(parse_args(args))


@py_only
@markdown
def _run_cli():
    """
    Run it:

    ```bash
    uv run modal run tutorials/nccl_benchmark/nccl_benchmark.py::run_benchmark
    ```
    """


@notebook_only
@code
def _invoke():
    with app.run():
        run_benchmark.remote()
