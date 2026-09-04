"""Write torch_dist checkpoint shards from threads instead of forked processes.

Megatron's ``FileSystemWriterAsync.write_preloaded_data_multiproc`` forks one
worker process per write bucket, on every rank, at every save. Inside a Modal
container that is already running Megatron ranks, SGLang engines, Ray, and
rollout thread pools, that ``os.fork()`` can fail with
``BlockingIOError: [Errno 11] Resource temporarily unavailable``, which kills
the save and the run with it (seen on Qwen3.6-27B, 2026-09-04).

Upstream Megatron replaced the forked workers with threads for exactly this
reason (NVIDIA/Megatron-LM#3633, "Add single-process checkpoint save to avoid
forked multiprocessing"). This patch back-ports that behavior to the pinned
image: the method keeps its name and signature, so ``get_save_function_and_args``
and the async caller are untouched, and the per-bucket worker
``write_preloaded_data`` is unchanged -- it already talks to its queues through
``put``/``get``/``task_done``, which ``queue.Queue`` provides just as
``multiprocessing`` queues do.

Executed at image-build time via ``python3 <this file> [path]``.
"""

import pathlib
import re
import sys

DEFAULT_PATH = (
    "/root/Megatron-LM/megatron/core/dist_checkpointing/strategies/filesystem_async.py"
)
MARKER = "write(sync,threads)"

METHOD_PATTERN = re.compile(
    r"^    @staticmethod\n"
    r"(?:    @_disable_gc\(\)\n)?"
    r"    def write_preloaded_data_multiproc\((?P<signature>.*?)\)(?P<returns> -> None)?:\n"
    r".*?"
    r"(?=^    @staticmethod\n)",
    re.MULTILINE | re.DOTALL,
)

THREADED_METHOD = '''    @staticmethod
    @_disable_gc()
    def write_preloaded_data_multiproc({signature}) -> None:
        """
        Performs saving data to storage with multiple threads.

        Patched by modal-training-gym: the upstream version forks one process per
        write bucket, which fails with EAGAIN in process-heavy containers. Threads
        preserve the parallel write and the queue protocol the worker expects.
        """
        logger = logging.getLogger(__name__)
        w_start = time()
        write_results_or_exc: Union[dict, Exception] = dict()
        local_results_queue: queue.Queue = queue.Queue()
        count_queue: queue.Queue = queue.Queue()
        thread_list: List[threading.Thread] = []
        for i, write_bucket in enumerate(write_buckets):
            try:
                count_queue.put(i)
                kwargs = {{
                    "local_proc_idx": i,
                    "write_bucket": write_bucket,
                    "results_queue": local_results_queue,
                    "count_queue": count_queue,
                    "use_fsync": True,
                }}
{msc_block}
                thread_list.append(
                    threading.Thread(
                        target=partial({worker_target}),
                        kwargs=kwargs,
                        name=f"ckpt-writer-{{i}}",
                    )
                )
            except Exception as e:
                err_msg = f"An error is caught while a worker {{i}} is created, error: {{e}}"
                logger.error(err_msg)
                write_results_or_exc = RuntimeError(err_msg)

        if not isinstance(write_results_or_exc, Exception):
            for t in thread_list:
                t.start()

            logger.debug("FileSystemWriterAsync: collecting worker results...")

            # Every worker takes its ticket back off ``count_queue`` and marks it
            # done, so ``join`` returns once all buckets are written.
            count_queue.join()
            for _ in range(len(write_buckets)):
                try:
                    local_proc_idx, local_results_or_exc = local_results_queue.get_nowait()
                except queue.Empty:
                    write_results_or_exc = RuntimeError(
                        "Unexpected empty `local_results_queue`"
                        f" (expected {{len(write_buckets)}} items)"
                    )
                    break
                if isinstance(local_results_or_exc, Exception):
                    err_msg = (
                        f"Local worker {{local_proc_idx}} encountered"
                        f" an error: {{local_results_or_exc}}"
                    )
                    logger.error(err_msg)
                    write_results_or_exc = local_results_or_exc
                    break
                assert isinstance(local_results_or_exc, list), type(local_results_or_exc)
                write_results_or_exc[local_proc_idx] = local_results_or_exc
            for t in thread_list:
                t.join()

            logger.debug("FileSystemWriterAsync: collected worker results successfully")

        global_results_queue.put(write_results_or_exc)

        w_end = time()
        logger.debug(f"{{w_end}}, rank: {{rank}}, write(sync,threads): {{w_end - w_start}}")

'''

MSC_BLOCK = """
                if use_msc:
                    signature = inspect.signature(FileSystemWriterAsync.write_preloaded_data)
                    if len(signature.parameters) > 6:
                        kwargs["use_msc"] = use_msc
"""


def patch_source(src: str) -> str | None:
    """Return the patched source, or ``None`` when there is nothing to do."""
    if MARKER in src or "def write_preloaded_data_multithread" in src:
        return None
    match = METHOD_PATTERN.search(src)
    if match is None:
        raise SystemExit(
            "patch_ckpt_writer_threads: write_preloaded_data_multiproc not found; "
            "the Megatron checkpoint writer changed shape, refusing to guess"
        )
    signature = " ".join(match.group("signature").split()).rstrip(",").rstrip()
    params = [p.split(":")[0].strip() for p in signature.split(",") if p.strip()]
    for required in ("rank", "write_buckets", "global_results_queue"):
        if required not in params:
            raise SystemExit(
                f"patch_ckpt_writer_threads: unexpected signature ({signature!r})"
            )
    worker = "FileSystemWriterAsync.write_preloaded_data"
    worker_target = (
        f"{worker}, transform_list" if "transform_list" in params else worker
    )
    method = THREADED_METHOD.format(
        signature=signature,
        msc_block=MSC_BLOCK.rstrip("\n") if "use_msc" in params else "",
        worker_target=worker_target,
    )
    patched = src[: match.start()] + method + src[match.end() :]
    if not re.search(r"^import threading$", patched, re.MULTILINE):
        patched = re.sub(
            r"^import queue$",
            "import queue\nimport threading",
            patched,
            count=1,
            flags=re.MULTILINE,
        )
    if not re.search(r"^import queue$", patched, re.MULTILINE):
        raise SystemExit(
            "patch_ckpt_writer_threads: expected `import queue` in filesystem_async.py"
        )
    compile(patched, "filesystem_async.py", "exec")
    return patched


def main(argv: list[str]) -> None:
    path = pathlib.Path(argv[1] if len(argv) > 1 else DEFAULT_PATH)
    patched = patch_source(path.read_text())
    if patched is None:
        print("filesystem_async.py already writes checkpoint shards from threads")
        return
    path.write_text(patched)
    print("Patched filesystem_async.py to write checkpoint shards from threads")


if __name__ == "__main__":
    main(sys.argv)
