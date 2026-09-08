"""Training Gym's staleness-bounded fully-async slime rollout.

Decouples ``max_concurrent_tasks`` from ``rollout_batch_size``: a background
asyncio worker keeps a fixed pool of in-flight trajectories across rollout
boundaries, so the next training step doesn't have to wait for the slowest
in-flight sample to finish.

Use with ``--rollout-function-path
modal_training_gym.frameworks.slime.bounded_async_rollout.generate_rollout_fully_async``.
Plug in per-sample logic via ``--custom-generate-function-path`` and
per-sample reward via ``--custom-rm-path`` — the worker calls slime's stock
:func:`generate_and_rm_group` which dispatches to those.

Concurrency starts from slime's per-sample serving cap and is divided by
``n_samples_per_prompt`` to produce a prompt-group budget.
``rollout_max_staleness`` bounds the combined in-flight and completed pool to
``rollout_max_staleness * rollout_batch_size`` groups, and any finished group
launched more than ``rollout_max_staleness`` rollouts ago is dropped rather
than trained on.

The worker is intentionally oblivious to slime's higher-level pause /
weight-update signalling (e.g. ``GenerateState.aborted``). Each in-flight
generation short-circuits on those signals on its own and surfaces
:data:`Sample.Status.ABORTED`. Aborted or failed groups are returned to
``data_buffer`` so transient failures do not consume prompts, at most
``rollout_max_group_requeues`` times per group before the group is dropped.
"""

from __future__ import annotations

import asyncio
import atexit
import logging
import queue
import threading
import time
from typing import Any

__all__ = [
    "AsyncRolloutWorker",
    "generate_rollout_fully_async",
]

logger = logging.getLogger("modal_training_gym.slime.bounded_async_rollout")


# Global worker, shared across rollout calls so the queue stays warm.
_global_worker: AsyncRolloutWorker | None = None
_worker_lock = threading.Lock()


def _get_global_worker(args, data_buffer) -> AsyncRolloutWorker:
    from slime.utils.http_utils import get_rollout_num_engines

    global _global_worker
    with _worker_lock:
        if _global_worker is None or not _global_worker.worker_thread.is_alive():
            logger.info("starting fully-async rollout worker")
            sample_capacity = args.sglang_server_concurrency * get_rollout_num_engines(
                args
            )
            group_capacity = max(
                1,
                sample_capacity // max(1, args.n_samples_per_prompt),
            )
            _global_worker = AsyncRolloutWorker(
                args,
                data_buffer,
                concurrency=group_capacity,
            )
            _global_worker.start()
        return _global_worker


def _stop_global_worker() -> None:
    global _global_worker
    with _worker_lock:
        if _global_worker is not None:
            _global_worker.stop()
            _global_worker = None


atexit.register(_stop_global_worker)


class AsyncRolloutWorker:
    """Background thread + asyncio loop that continuously consumes groups
    from ``data_buffer`` and runs :func:`generate_and_rm_group` on each."""

    def __init__(self, args, data_buffer, concurrency: int = 10):
        self.args = args
        self.data_buffer = data_buffer
        self.concurrency = concurrency
        self.running = True
        # Done callbacks run on the worker event-loop thread. Keep this queue
        # unbounded so a full queue can never block that loop; pool_limit below
        # provides backpressure before new work is submitted.
        self.output_queue: queue.Queue[tuple[int, list[Any]]] = queue.Queue()
        self.worker_thread: threading.Thread | None = None
        self._failure_error: BaseException | None = None
        from slime.rollout.sglang_rollout import GenerateState

        self.state = GenerateState(args)
        # Bound every generated-but-unconsumed group, whether still active, in
        # the handoff queue, or buffered by the collector. This is the direct
        # Little's-law control on policy lag: at one batch consumed per update,
        # a pool of N * rollout_batch_size is at most roughly N updates deep.
        self.completed_buffer: dict[int, list[Any]] = {}
        self.inflight_gids: set[int] = set()
        self.current_rollout_id = 0
        self.launch_rid: dict[int, int] = {}
        self.requeues: dict[int, int] = {}
        self.max_requeues = getattr(args, "rollout_max_group_requeues", 3)
        self.max_loop_failures = getattr(args, "rollout_max_loop_failures", 5)
        self.aborted_groups_dropped = 0
        max_staleness = getattr(args, "rollout_max_staleness", None)
        staleness_limit = (
            max_staleness * args.rollout_batch_size if max_staleness else concurrency
        )
        self.pool_limit = min(concurrency, staleness_limit)

    # -- public --------------------------------------------------------------

    def start(self) -> None:
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.worker_thread = threading.Thread(
                target=self._thread_main, name="fully-async-rollout", daemon=True
            )
            self.worker_thread.start()

    def stop(self) -> None:
        self.running = False
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5)

    def get_completed_groups(self) -> list[tuple[int, list[Any]]]:
        completed: list[tuple[int, list[Any]]] = []
        while True:
            try:
                completed.append(self.output_queue.get_nowait())
            except queue.Empty:
                break
        return completed

    def queue_size(self) -> int:
        return self.output_queue.qsize()

    def raise_if_failed(self) -> None:
        if self._failure_error is not None:
            raise RuntimeError(
                "fully-async rollout producer failed"
            ) from self._failure_error

    # -- internals -----------------------------------------------------------

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._loop())
        except BaseException as exc:
            logger.exception("fully-async rollout producer crashed")
            self._fail(exc)

    def _fail(self, error: BaseException) -> None:
        self._failure_error = error
        self.running = False

    async def _loop(self) -> None:
        from slime.rollout.sglang_rollout import generate_and_rm_group

        active_tasks: set[asyncio.Task] = set()
        max_concurrent = self.concurrency
        gid_counter = 0
        consecutive_failures = 0

        while self.running:
            try:
                # Reap done tasks
                if active_tasks:
                    done = {t for t in active_tasks if t.done()}
                    for t in done:
                        try:
                            t.result()  # results already handled in callback
                        except Exception as e:  # noqa: BLE001
                            logger.warning("fully-async task crashed: %r", e)
                    active_tasks -= done

                # Top up only while the whole generated-but-unconsumed pool is
                # inside the serving and staleness budgets.
                while (
                    len(active_tasks) < max_concurrent
                    and len(active_tasks)
                    + self.output_queue.qsize()
                    + len(self.completed_buffer)
                    < self.pool_limit
                    and self.running
                ):
                    groups = self.data_buffer.get_samples(1)
                    if not groups:
                        break
                    for group in groups:
                        gid = gid_counter
                        gid_counter += 1
                        self.inflight_gids.add(gid)
                        self.launch_rid[gid] = self.current_rollout_id
                        task = asyncio.create_task(
                            generate_and_rm_group(
                                self.args,
                                group,
                                sampling_params=self.state.sampling_params.copy(),
                                evaluation=False,
                            )
                        )
                        task.add_done_callback(self._make_done_cb(gid, group))
                        active_tasks.add(task)

                consecutive_failures = 0
                await asyncio.sleep(1)
            except Exception as e:  # noqa: BLE001
                # Transient producer errors are retried; persistent ones must
                # surface via _fail instead of hanging the consumer forever.
                consecutive_failures += 1
                if consecutive_failures > self.max_loop_failures:
                    raise
                logger.exception("fully-async loop iteration error: %s", e)
                await asyncio.sleep(1)

        if active_tasks:
            logger.info(
                "fully-async: waiting for %d in-flight tasks to drain",
                len(active_tasks),
            )
            try:
                await asyncio.wait(active_tasks, timeout=30)
            except Exception:  # noqa: BLE001
                pass

    def _requeue_group(self, group: list[Any]) -> None:
        from slime.utils.types import Sample

        key = _group_key(group)
        self.requeues[key] = self.requeues.get(key, 0) + 1
        if self.requeues[key] > self.max_requeues:
            del self.requeues[key]
            self.aborted_groups_dropped += 1
            logger.warning(
                "fully-async: group %d aborted %d times; dropping",
                key,
                self.max_requeues + 1,
            )
            return
        for sample in group:
            sample.status = Sample.Status.PENDING
        try:
            self.data_buffer.add_samples([group])
        except Exception:  # noqa: BLE001
            logger.exception("fully-async: failed to requeue group")

    def _make_done_cb(self, gid: int, original_group: list[Any]):
        def _cb(done_task: asyncio.Task) -> None:
            from slime.utils.types import Sample

            self.inflight_gids.discard(
                gid
            )  # no longer generating → unpins the staleness window
            launch_rid = self.launch_rid.pop(gid, None)
            try:
                result = done_task.result()
            except Exception:  # noqa: BLE001
                logger.exception("fully-async: process task raised")
                self._requeue_group(original_group)
                return
            if not isinstance(result, list):
                logger.warning(
                    "fully-async: generate_and_rm_group returned %r, expected list[Sample]; requeueing",
                    type(result).__name__,
                )
                self._requeue_group(original_group)
                return
            # Aborted group → requeue for redo under a fresh gid, don't ship to training. Reset EVERY
            # sibling to PENDING: on re-pull, generate_and_rm short-circuits COMPLETED samples (returns
            # them verbatim, no regeneration), which would ship the siblings' old-policy trajectories under
            # a fresh gid — stale data outside the staleness window. PENDING forces a full-group redo.
            if any(getattr(s, "status", None) == Sample.Status.ABORTED for s in result):
                self._requeue_group(result)
                return
            if launch_rid is not None:
                self.launch_rid[gid] = launch_rid
            self.requeues.pop(_group_key(result), None)
            self.output_queue.put((gid, result))

        return _cb


async def _generate_rollout_async(
    args, rollout_id: int, data_buffer
) -> list[list[Any]]:
    from slime.utils.misc import load_function

    assert args.rollout_global_dataset
    worker = _get_global_worker(args, data_buffer)
    worker.current_rollout_id = rollout_id
    max_staleness = getattr(args, "rollout_max_staleness", None)

    target = args.rollout_batch_size
    logger.info(
        "fully-async rollout %d: target=%d queue_warm=%d",
        rollout_id,
        target,
        worker.queue_size(),
    )

    # Dynamic sampling (DAPO): if a filter is configured, discard groups it rejects (e.g. zero reward-variance
    # → no GRPO gradient) and keep pulling until `target` groups PASS. Over-generation is free here — the
    # windowed-FIFO already runs the generation pool ahead of the trainer — so we just filter the pool rather
    # than launch extra rounds. raw_reward is logged over ALL examined groups (pre-filter) so the metric is the
    # unbiased generation signal, not the selected-only (upward-biased) subset.
    dyn_filter = None
    fpath = getattr(args, "dynamic_sampling_filter_path", None)
    if fpath:
        dyn_filter = load_function(fpath)
    # Starvation guard: if the policy can't produce `target` passing groups even after a large over-sample,
    # accept rejected groups rather than hang the trainer (and the low kept_frac will show in the metrics).
    over_sample_cap = target * 8

    # Windowed-FIFO consumption: sample the oldest-completed groups first from the generation pool; never
    # block on an in-flight straggler (staleness is bounded on the generation side — see the worker's
    # __init__). Leftover completed groups stay buffered across steps.
    buf = worker.completed_buffer
    started = time.time()
    last_log = started
    LOG_EVERY = 30.0

    collected: list[list[Any]] = []
    n_examined = (
        0  # groups passed through the filter this rollout (pre-filter denominator)
    )
    n_kept = 0  # groups the filter accepted
    n_stale = 0  # groups launched more than max_staleness rollouts ago
    reward_sum_all = (
        0.0  # sum of per-group mean reward over ALL examined groups (pre-filter)
    )
    while len(collected) < target:
        worker.raise_if_failed()
        for gid, group in worker.get_completed_groups():
            buf[gid] = group

        for gid in sorted(buf):  # oldest-completed first
            if len(collected) >= target:
                break
            group = buf.pop(gid)
            age = rollout_id - worker.launch_rid.pop(gid, rollout_id)
            if max_staleness is not None and age > max_staleness:
                n_stale += 1
                continue
            if dyn_filter is None:
                collected.append(group)
                continue
            n_examined += 1
            reward_sum_all += _group_mean_reward(group, args)
            if dyn_filter(args, group).keep:
                collected.append(group)
                n_kept += 1
            elif n_examined >= over_sample_cap:
                collected.append(
                    group
                )  # starvation fallback — take a rejected group rather than hang

        if len(collected) < target:
            await asyncio.sleep(
                0.05
            )  # pool not yet deep enough — wait for more completions

        now = time.time()
        if now - last_log > LOG_EVERY:
            logger.info(
                "fully-async rollout %d: collected %d/%d, examined=%d kept=%d, buffered=%d, in_flight=%d, elapsed=%.1fs",
                rollout_id,
                len(collected),
                target,
                n_examined,
                n_kept,
                len(buf),
                len(worker.inflight_gids),
                now - started,
            )
            last_log = now

    # Order by sample.index for determinism (slime convention).
    out = sorted(collected, key=_group_key)
    n_aborted, worker.aborted_groups_dropped = worker.aborted_groups_dropped, 0
    metrics = {
        "rollout/stale_groups_dropped": float(n_stale),
        "rollout/aborted_groups_dropped": float(n_aborted),
    }
    if dyn_filter is not None and n_examined:
        kept_frac = n_kept / n_examined
        metrics.update(
            {
                "dynamic_sampling/kept_frac": kept_frac,
                "dynamic_sampling/filtered_frac": 1.0 - kept_frac,
                "dynamic_sampling/groups_examined": float(n_examined),
                "dynamic_sampling/raw_reward_all": reward_sum_all / n_examined,
            }
        )
    _log_rollout_metrics(args, rollout_id, metrics)
    logger.info(
        "fully-async rollout %d: done in %.1fs, stale_dropped=%d, buffered_left=%d, in_flight=%d",
        rollout_id,
        time.time() - started,
        n_stale,
        len(buf),
        len(worker.inflight_gids),
    )
    return out


def _group_key(group: list[Any]) -> int:
    for s in group:
        idx = getattr(s, "index", None)
        if idx is not None:
            return int(idx)
    return 0


def _group_mean_reward(group: list[Any], args) -> float:
    rs = [
        s.get_reward_value(args)
        for s in group
        if getattr(s, "reward", None) is not None
    ]
    return sum(rs) / len(rs) if rs else 0.0


def _log_rollout_metrics(args, rollout_id: int, metrics: dict[str, float]) -> None:
    """Emit rollout telemetry. dynamic_sampling/raw_reward_all is the UNBIASED mean reward over every generated
    group (pre-filter) — the true learning signal; slime's rollout/raw_reward is over the kept (non-zero-std)
    subset and reads high once the filter is active."""
    try:
        from slime.ray.rollout import compute_rollout_step
        from slime.utils import logging_utils

        metrics["rollout/step"] = compute_rollout_step(args, rollout_id)
        logging_utils.log(args, metrics, step_key="rollout/step")
        logger.info("fully-async rollout %d: %s", rollout_id, metrics)
    except Exception:  # noqa: BLE001 — telemetry must never crash the rollout
        logger.exception("rollout metrics logging failed (non-fatal)")


def generate_rollout_fully_async(
    args, rollout_id, data_buffer, evaluation: bool = False
):
    """Slime ``--rollout-function-path`` entrypoint."""

    if evaluation:
        raise ValueError("fully-async rollout doesn't support evaluation mode")
    from slime.utils.async_utils import run

    return run(_generate_rollout_async(args, rollout_id, data_buffer))
