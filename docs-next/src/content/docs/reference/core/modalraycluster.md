---
title: ModalRayCluster
description: API reference for ModalRayCluster
---

```python
from modal_training_gym.common.ray_cluster import ModalRayCluster
```

Base class for bootstrapping a Ray cluster inside Modal clustered functions.

## Methods

### `discover_cluster(self, n_nodes: int) -> None`

Populate `rank`, `head_addr`, `node_ip`, and `n_nodes` from Modal.

### `forward_dashboard(self)`

Return a `modal.forward` context manager for the Ray dashboard.

### `head_extra_start_args(self) -> list[str]`

Override to append flags to `ray start --head` (e.g. resource hints).

### `start(self, n_nodes: int, *, init_retries: int = 30, worker_wait_retries: int = 60) -> None`

Convenience: `discover_cluster(n_nodes)` followed by `start_ray(...)`.

### `start_ray(self, *, init_retries: int = 30, worker_wait_retries: int = 60) -> None`

Start the Ray head or worker using previously-discovered cluster state.

### `submit_and_tail(self, entrypoint: str, *, runtime_env: dict | None = None) -> str`

Submit a Ray job, stream its logs to stdout, and return the final status.

### `wait_forever(self, poll_seconds: float = 10) -> None`

Keep a worker container alive until the head terminates the cluster.

### `worker_extra_start_args(self) -> list[str]`

Override to append flags to `ray start` on worker ranks.

**Source:** [`modal_training_gym/common/ray_cluster.py`](https://github.com/modal-projects/training-gym/blob/main/modal_training_gym/common/ray_cluster.py)
