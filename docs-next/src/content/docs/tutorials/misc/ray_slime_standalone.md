---
title: "Ray-on-Modal pattern demo (standalone, no framework wrapper)"
description: "Ray-on-Modal pattern demo"
---

**What this tutorial is.** A peek under the hood of the `slime`
and `verl` frameworks. Both rely on the same primitive —
`ModalRayCluster` from `modal_training_gym.common.ray_cluster` —
to stand up a Ray cluster across clustered Modal containers. This
tutorial uses that primitive directly, so you can see what the
framework launchers are doing without any framework-specific
sugar.

**When to use this pattern.** When you want a Ray cluster on Modal
but your training code isn't one of the wrapped frameworks. For
example: a custom RL loop that needs `ray.actor`-based workers, a
hyperparameter sweep via Ray Tune, or a bring-your-own-training-
library setup where we haven't written a launcher yet.

**The design (what happens in the one `run_ray_job` function
below).**

- **Bootstrap.** Every clustered container calls
  `ModalRayCluster().start(n_nodes=...)`. Rank 0 starts the Ray
  head and waits for all workers to join; other ranks start Ray
  workers that register with the head and then
  `await cluster.wait_forever()` to keep the cluster up until the
  head finishes.
- **Job submission.** On the head only, the cluster's
  `JobSubmissionClient` submits a Python entrypoint (any shell
  command — the default just prints `ray.cluster_resources()`) and
  streams its logs back.
- **Dashboard.** `cluster.forward_dashboard()` uses `modal.forward`
  to expose the Ray dashboard URL, so you can watch the job from a
  browser.

```python
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
```

## Clustered Ray + job submission

One `@modal.experimental.clustered`-decorated function does it all:
bootstrap Ray on every rank, then (on head) submit a Ray job and tail
its logs. Other ranks sit in `wait_forever()` so the cluster stays up
until the head finishes.

```python
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
```

## Customization

To run a real training workload:

- Override `image` above with the framework's expected environment.
- Add volumes for datasets / checkpoints / HF cache as needed.
- Pass the training command as `entrypoint`; `JobSubmissionClient`
  runs it on the head and streams logs over the Ray protocol.

If you find yourself needing framework-specific plumbing (data prep,
checkpoint conversion, etc.), the `slime` / `verl` framework packages
already wrap this same pattern plus their respective training CLIs.

---

## Related API Reference

- [`ModalRayCluster`](/reference/core/modalraycluster/)

**Source:** [`tutorials/misc/ray_slime_standalone/ray_slime_standalone.py`](https://github.com/modal-projects/training-gym/blob/joy/initial-setup/tutorials/misc/ray_slime_standalone/ray_slime_standalone.py)
 | <a href="https://modal.com/notebooks/new/https://github.com/modal-projects/training-gym/blob/joy/initial-setup/tutorials/misc/ray_slime_standalone/ray_slime_standalone.ipynb" target="_blank" rel="noopener noreferrer">Open in Modal Notebook</a>
