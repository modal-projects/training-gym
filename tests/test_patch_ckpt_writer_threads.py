"""The Megatron checkpoint-writer patch: threads instead of forked processes."""

from __future__ import annotations

import ast
import importlib.util
import os
import queue
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "modal_training_gym/common/megatron_patches/patch_ckpt_writer_threads.py"
FIXTURE = ROOT / "tests/testdata/megatron/filesystem_async.py.input"


def _load_patch_module():
    spec = importlib.util.spec_from_file_location("patch_ckpt_writer_threads", PATCH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def patched_source() -> str:
    patched = _load_patch_module().patch_source(FIXTURE.read_text())
    assert patched is not None
    return patched


def test_patch_replaces_forked_workers_with_threads(patched_source: str) -> None:
    ast.parse(patched_source)
    assert 'mp.get_context("fork")' not in patched_source
    assert "ctx.Process(" not in patched_source
    assert "threading.Thread(" in patched_source
    assert re.search(r"^import threading$", patched_source, re.MULTILINE)
    # The caller in get_save_function_and_args is untouched, so the method must
    # keep its name and full signature.
    assert (
        "def write_preloaded_data_multiproc(transform_list: List[_StorageWriterTransforms], "
        "use_msc: bool, rank: int, write_buckets: List[WriteBucket], "
        "global_results_queue: mp.Queue) -> None:"
    ) in patched_source
    assert (
        "partial(self.write_preloaded_data_multiproc, transform_list, self.use_msc)"
        in (patched_source)
    )


def test_patch_is_idempotent(patched_source: str) -> None:
    assert _load_patch_module().patch_source(patched_source) is None


def test_patch_is_a_no_op_once_upstream_uses_threads() -> None:
    upstream = "import queue\n\n\nclass FileSystemWriterAsync:\n    def write_preloaded_data_multithread(self):\n        pass\n"
    assert _load_patch_module().patch_source(upstream) is None


def test_patch_refuses_an_unrecognized_writer() -> None:
    with pytest.raises(SystemExit, match="not found"):
        _load_patch_module().patch_source(
            "import queue\n\nclass FileSystemWriterAsync:\n    pass\n"
        )


def _exec_writer_class(patched_source: str, namespace: dict) -> type:
    """Run the patched method and the unchanged worker against stub dependencies."""
    start = patched_source.index(
        "    @staticmethod\n    @_disable_gc()\n    def write_preloaded_data_multiproc("
    )
    class_body = patched_source[start:]
    source = "class FileSystemWriterAsync:\n" + class_body
    exec(compile(source, "patched_filesystem_async", "exec"), namespace)
    return namespace["FileSystemWriterAsync"]


def _stub_namespace() -> dict:
    import inspect
    import logging
    import threading
    from functools import partial
    from time import time
    from typing import List, Union

    def _disable_gc():
        return lambda fn: fn

    def _write_item(stream, data, write_item, storage_key):
        stream.write(data)
        return ("written", write_item, storage_key)

    return {
        "_disable_gc": _disable_gc,
        "_write_item": _write_item,
        "_process_memory": lambda: 0,
        "inspect": inspect,
        "logging": logging,
        "threading": threading,
        "partial": partial,
        "time": time,
        "List": List,
        "Union": Union,
        "queue": queue,
        "os": os,
    }


def test_threaded_writer_honors_the_worker_queue_protocol(
    patched_source: str, tmp_path: Path
) -> None:
    writer = _exec_writer_class(patched_source, _stub_namespace())
    buckets = [
        (
            str(tmp_path / f"__0_{i}.distcp"),
            f"key{i}",
            ([(f"item{i}", f"payload{i}".encode())], []),
        )
        for i in range(3)
    ]
    results: queue.Queue = queue.Queue()

    writer.write_preloaded_data_multiproc([], False, 0, buckets, results)

    written = results.get_nowait()
    assert isinstance(written, dict)
    assert sorted(written) == [0, 1, 2]
    for i in range(3):
        assert written[i] == [("written", f"item{i}", f"key{i}")]
        assert (tmp_path / f"__0_{i}.distcp").read_bytes() == f"payload{i}".encode()
    assert results.empty()


def test_threaded_writer_propagates_a_worker_failure(
    patched_source: str, tmp_path: Path
) -> None:
    writer = _exec_writer_class(patched_source, _stub_namespace())
    buckets = [
        (str(tmp_path / "__0_0.distcp"), "ok", ([("item", b"x")], [])),
        (str(tmp_path / "missing-dir" / "__0_1.distcp"), "bad", ([("item", b"y")], [])),
    ]
    results: queue.Queue = queue.Queue()

    writer.write_preloaded_data_multiproc([], False, 0, buckets, results)

    outcome = results.get_nowait()
    assert isinstance(outcome, FileNotFoundError)
