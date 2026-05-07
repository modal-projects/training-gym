"""Shared helpers for running a Ray cluster across Modal containers.

- `start_ray_head` / `start_ray_worker`: low-level primitives for spawning the
  Ray head/worker processes on a single container.
- `ModalRayCluster`: a base class that stitches those primitives together with
  Modal cluster discovery, the Ray `JobSubmissionClient`, dashboard port
  forwarding, and a worker keep-alive loop. Framework launchers (slime,
  verl, …) subclass or compose this.
"""

import asyncio
import subprocess
import time

RAY_PORT = 6379
RAY_DASHBOARD_PORT = 8265


def start_ray_head(
    node_ip: str,
    n_nodes: int,
    *,
    init_retries: int = 30,
    worker_wait_retries: int = 60,
    extra_start_args: list[str] | None = None,
) -> None:
    """Start the Ray head and block until all workers have joined.

    Spawns `ray start --head` in the background, then polls `ray.init()` until
    the head is reachable and `ray.nodes()` until `n_nodes` are alive. Raises
    `RuntimeError` if either poll times out.
    """
    import ray

    cmd = [
        "ray",
        "start",
        "--head",
        f"--node-ip-address={node_ip}",
        "--dashboard-host=0.0.0.0",
    ]
    if extra_start_args:
        cmd.extend(extra_start_args)
    subprocess.Popen(cmd)

    for _ in range(init_retries):
        try:
            ray.init(address="auto")
            break
        except ConnectionError:
            time.sleep(1)
    else:
        raise RuntimeError("Failed to connect to the Ray head node")

    for _ in range(worker_wait_retries):
        alive_nodes = [n for n in ray.nodes() if n["Alive"]]
        print(f"Waiting for Ray workers: {len(alive_nodes)}/{n_nodes} alive")
        if len(alive_nodes) == n_nodes:
            print("All Ray nodes connected")
            return
        time.sleep(1)
    raise RuntimeError(f"Timed out waiting for all {n_nodes} Ray nodes to join")


def start_ray_worker(
    node_ip: str,
    head_addr: str,
    *,
    head_port: int = RAY_PORT,
    extra_start_args: list[str] | None = None,
) -> None:
    """Start a Ray worker connected to the head at `head_addr:head_port`."""
    cmd = [
        "ray",
        "start",
        f"--node-ip-address={node_ip}",
        "--address",
        f"{head_addr}:{head_port}",
    ]
    if extra_start_args:
        cmd.extend(extra_start_args)
    subprocess.Popen(cmd)


class ModalRayCluster:
    """Base class for bootstrapping a Ray cluster inside Modal clustered functions.

    Manages cluster discovery, Ray head/worker startup,
    ``JobSubmissionClient`` creation, dashboard forwarding, and a
    worker keep-alive loop. Framework launchers (slime, etc.)
    compose or subclass this.

    Attributes (populated after ``discover_cluster()``)
    ---------------------------------------------------
    n_nodes : int
        Total cluster node count.
    rank : int
        This container's rank (0 = head).
    head_addr : str
        IPv4 address of the head node.
    node_ip : str
        IPv4 address of this container.
    is_head : bool
        Property — ``True`` when ``rank == 0``.

    Methods
    -------
    discover_cluster(n_nodes)
        Populate rank, addresses, and node count from Modal cluster info.
    start_ray(init_retries=30, worker_wait_retries=60)
        Start Ray head or worker using discovered cluster state.
    start(n_nodes, init_retries=30, worker_wait_retries=60)
        Convenience: ``discover_cluster`` + ``start_ray``.
    forward_dashboard()
        Return a ``modal.forward`` context manager for the Ray dashboard.
    submit_and_tail(entrypoint, runtime_env=None)
        Async: submit a Ray job, stream logs, return final status.
    wait_forever(poll_seconds=10)
        Async: keep a worker container alive until termination.
    """

    def __init__(self) -> None:
        # Populated by discover_cluster(); see _discovered for readiness check.
        self.n_nodes: int = 0
        self.rank: int = 0
        self.head_addr: str = ""
        self.node_ip: str = ""
        self._client = None
        self._discovered: bool = False
        self._started: bool = False

    def head_extra_start_args(self) -> list[str]:
        """Override to append flags to `ray start --head` (e.g. resource hints)."""
        return []

    def worker_extra_start_args(self) -> list[str]:
        """Override to append flags to `ray start` on worker ranks."""
        return []

    @property
    def is_head(self) -> bool:
        if not self._discovered:
            raise RuntimeError(
                "ModalRayCluster.discover_cluster() has not been called yet"
            )
        return self.rank == 0

    @property
    def client(self):
        """Ray `JobSubmissionClient`. Only valid on the head node."""
        if not self.is_head:
            raise RuntimeError("JobSubmissionClient is only available on the head node")
        return self._client

    def discover_cluster(self, n_nodes: int) -> None:
        """Populate `rank`, `head_addr`, `node_ip`, and `n_nodes` from Modal.

        Does not start Ray. Call this first if you need to read cluster state
        (e.g. to set framework-specific env vars) before `start_ray()` inherits
        the environment into the Ray daemon. Idempotent.
        """
        if self._discovered:
            return

        if n_nodes == 1:
            # Modal may omit container IPv4s for size-1 clustered functions.
            rank, head_addr, node_ip = 0, "127.0.0.1", "127.0.0.1"
        else:
            import modal.experimental

            info = modal.experimental.get_cluster_info()
            ips = list(info.container_ipv4_ips or [])
            if len(ips) != n_nodes:
                raise RuntimeError(
                    f"Modal cluster size mismatch: expected {n_nodes} nodes, got {len(ips)}"
                )
            rank = info.rank
            head_addr = ips[0]
            node_ip = ips[rank]

        self.n_nodes = n_nodes
        self.rank = rank
        self.head_addr = head_addr
        self.node_ip = node_ip
        self._discovered = True

    def start_ray(
        self,
        *,
        init_retries: int = 30,
        worker_wait_retries: int = 60,
    ) -> None:
        """Start the Ray head or worker using previously-discovered cluster state.

        Requires `discover_cluster()` to have been called. On the head also
        creates a `JobSubmissionClient`. Idempotent.
        """
        if self._started:
            return
        if not self._discovered:
            raise RuntimeError("discover_cluster() must be called before start_ray()")

        if self.rank == 0:
            start_ray_head(
                self.node_ip,
                self.n_nodes,
                init_retries=init_retries,
                worker_wait_retries=worker_wait_retries,
                extra_start_args=self.head_extra_start_args(),
            )
            from ray.job_submission import JobSubmissionClient

            self._client = JobSubmissionClient(f"http://127.0.0.1:{RAY_DASHBOARD_PORT}")
        else:
            start_ray_worker(
                self.node_ip,
                self.head_addr,
                extra_start_args=self.worker_extra_start_args(),
            )
        self._started = True

    def start(
        self,
        n_nodes: int,
        *,
        init_retries: int = 30,
        worker_wait_retries: int = 60,
    ) -> None:
        """Convenience: `discover_cluster(n_nodes)` followed by `start_ray(...)`."""
        self.discover_cluster(n_nodes)
        self.start_ray(
            init_retries=init_retries,
            worker_wait_retries=worker_wait_retries,
        )

    def forward_dashboard(self):
        """Return a `modal.forward` context manager for the Ray dashboard.

        Usable with either `with` or `async with`. Only valid on the head node.
        """
        import modal

        if not self.is_head:
            raise RuntimeError("forward_dashboard is only valid on the head node")
        return modal.forward(RAY_DASHBOARD_PORT)

    async def submit_and_tail(
        self,
        entrypoint: str,
        *,
        runtime_env: dict | None = None,
    ) -> str:
        """Submit a Ray job, stream its logs to stdout, and return the final status."""
        if not self.is_head:
            raise RuntimeError("submit_and_tail is only valid on the head node")
        assert self._client is not None
        job_id = self._client.submit_job(
            entrypoint=entrypoint,
            runtime_env=runtime_env or {},
        )
        print(f"Submitted Ray job: {job_id}")
        for line in self._client.tail_job_logs(job_id):
            print(line, end="", flush=True)
        status = self._client.get_job_status(job_id).value
        print(f"\nFinal Ray job status: {status}")
        if status not in ("SUCCEEDED",):
            raise RuntimeError(f"Ray job {job_id} finished with status: {status}")
        return status

    async def wait_forever(self, poll_seconds: float = 10) -> None:
        """Keep a worker container alive until the head terminates the cluster."""
        while True:
            await asyncio.sleep(poll_seconds)
