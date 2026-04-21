"""Tutorial source for `ray_slime_standalone` — parsed by generate_tutorial.py.

Standalone Ray-cluster-on-Modal pattern. Uses `ModalRayCluster` from
`modal_training_gym.common.ray_cluster` directly — no framework wrapper —
so you can see the bare Ray bootstrap + `JobSubmissionClient` submission
flow. The SLIME framework package internally builds on this same helper.
"""

TUTORIAL_METADATA = {
    'framework': '— (raw `ModalRayCluster`)',
    'cluster_shape': '2 × 8×H100',
    'summary': 'Ray-on-Modal pattern demo',
    'difficulty': 'Intermediate',
    'order': 100,
}


from tutorial_generator import code, markdown, notebook_only, py_only, shell


@markdown
def _title():
    """
    # Ray + Modal: standalone multi-node pattern

    This tutorial shows how to stand up a Ray cluster across Modal
    containers and submit jobs to it via Ray's `JobSubmissionClient`. The
    SLIME framework package (`modal_training_gym.frameworks.slime`) builds
    on exactly this pattern — here we use the `ModalRayCluster` helper
    directly, so you can see what's going on without any framework-specific
    sugar.

    Useful when you want a Ray cluster on Modal but your training code
    isn't one of the frameworks we've wrapped.
    """


@markdown
def _design():
    """
    ## Design

    - **Bootstrap** — on every clustered container, call
      `ModalRayCluster().start(n_nodes=...)`. Rank 0 starts the Ray head
      and waits for all workers to join; other ranks start Ray workers
      that register with the head.
    - **Job submission** — on the head only, use the cluster's
      `JobSubmissionClient` to submit a Python entrypoint and stream its
      logs.
    - **Dashboard** — open the Ray dashboard via `modal.forward` so you
      can watch the job from your browser.
    """


@notebook_only
@shell("! pip install -q git+https://github.com/modal-projects/training-gym.git@joy/initial-setup")
def _install():
    pass


@code
def _imports_and_image():
    import modal

    from modal_training_gym.common.ray_cluster import ModalRayCluster

    N_NODES = 2

    # Any image with Ray installed works. We use the SLIME nightly image
    # for parity with the framework-backed SLIME tutorial.
    image = (
        modal.Image.from_registry("slimerl/slime:nightly-dev-20260329a")
        .entrypoint([])
        .add_local_python_source("modal_training_gym", copy=True)
    )

    app = modal.App("ray-standalone", image=image)


@markdown
def _ray_function_section():
    """
    ## Clustered Ray + job submission

    One `@modal.experimental.clustered`-decorated function does it all:
    bootstrap Ray on every rank, then (on head) submit a Ray job and tail
    its logs. Other ranks sit in `wait_forever()` so the cluster stays up
    until the head finishes.
    """


@code
def _train_fn():
    @app.function(
        gpu=f"H100:8",
        timeout=24 * 60 * 60,
        experimental_options={"efa_enabled": True},
        serialized=True,
    )
    @modal.experimental.clustered(size=N_NODES, rdma=True)
    async def run_ray_job(entrypoint: str = "python -c 'import ray; ray.init(); print(ray.cluster_resources())'"):
        """Spin up Ray across the cluster, submit `entrypoint`, and tail its logs.

        The default entrypoint just prints Ray's view of cluster resources —
        swap in any shell command (e.g. `python3 my_train.py --flags …`) to
        run a real job.
        """
        cluster = ModalRayCluster()
        cluster.start(n_nodes=N_NODES)

        if not cluster.is_head:
            await cluster.wait_forever()
            return

        async with cluster.forward_dashboard() as tunnel:
            print(f"Ray dashboard: {tunnel.url}")
            status = await cluster.submit_and_tail(entrypoint)
            print(f"Final status: {status}")


@py_only
@markdown
def _run_cli():
    """
    ```bash
    # Default entrypoint prints Ray's cluster resources.
    uv run modal run tutorials/ray_slime_standalone/ray_slime_standalone.py::run_ray_job

    # Or pass your own entrypoint.
    uv run modal run tutorials/ray_slime_standalone/ray_slime_standalone.py::run_ray_job \\
        --entrypoint 'python3 /path/to/my_train.py --flag=value'
    ```
    """


@notebook_only
@code
def _invoke():
    with app.run():
        run_ray_job.remote()


@markdown
def _customization():
    """
    ## Customization

    To run a real training workload:

    - Override `image` above with the framework's expected environment.
    - Add volumes for datasets / checkpoints / HF cache as needed.
    - Pass the training command as `entrypoint`; `JobSubmissionClient`
      runs it on the head and streams logs over the Ray protocol.

    If you find yourself needing framework-specific plumbing (data prep,
    checkpoint conversion, etc.), the `slime` / `verl` framework packages
    already wrap this same pattern plus their respective training CLIs.
    """
