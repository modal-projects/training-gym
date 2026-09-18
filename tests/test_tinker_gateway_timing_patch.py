"""Gateway ``forward_backward`` timing patch for Miles' ``TinkerService``.

``tests/testdata/miles/tinker_service.py.input`` is upstream
``miles/tinker/core/service.py``; the golden output is what the patcher writes
into the image. Regenerate with ``uv run pytest
tests/test_tinker_gateway_timing_patch.py --rewrite``. The behavioral tests
run the injected preamble against the real ``recording_lane`` so the lane,
phase name and per-call accounting match what the dashboard expects.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import time
from enum import Enum
from pathlib import Path
from types import ModuleType

import pytest

from modal_training_gym.common import reporting, timing_recorder
from modal_training_gym.common.step_timing import RoleTimingRecord

TESTDATA = Path(__file__).parent / "testdata" / "miles"
PATCHER = (
    Path(__file__).parents[1]
    / "modal_training_gym"
    / "frameworks"
    / "miles"
    / "modal_helpers"
    / "patches"
    / "patch_tinker_timing.py"
)


@pytest.fixture(scope="session")
def patcher() -> ModuleType:
    spec = importlib.util.spec_from_file_location("patch_tinker_timing", PATCHER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["patch_tinker_timing"] = module
    spec.loader.exec_module(module)
    return module


def test_patch_matches_golden(patcher, request) -> None:
    src = (TESTDATA / "tinker_service.py.input").read_text()
    patched = patcher.patch_source(src)
    golden = TESTDATA / "tinker_service.py.timing.output"
    if request.config.getoption("--rewrite"):
        golden.write_text(patched)
        return

    assert patched == golden.read_text(), (
        f"golden mismatch for {golden.name}; rerun with --rewrite to accept"
    )
    compile(patched, "service.py", "exec")
    # Only the backend await moves under the timer; the surrounding
    # ``try/except UserInputError`` and everything else is untouched.
    assert patched.count("with _tg_batch_phase(batch.op):") == 1
    assert (
        "with _tg_batch_phase(batch.op):\n"
        "                outputs = await forward(self._batch_counter, "
        "slot_datums, batch.loss_fn, batch.loss_fn_config)\n"
    ) in patched
    assert patched.count("outputs = await forward(") == 1
    assert "_tg_role('driver', None)" in patched
    assert "phase('forward_backward')" in patched


def test_patch_is_idempotent(patcher) -> None:
    once = patcher.patch_source((TESTDATA / "tinker_service.py.input").read_text())
    assert patcher.patch_source(once) == once


def test_patch_file_skips_missing_checkout(patcher, tmp_path, capsys) -> None:
    assert patcher.patch_file(tmp_path / "missing.py") is False
    assert "skipping" in capsys.readouterr().out


def test_patch_file_rewrites_once(patcher, tmp_path) -> None:
    target = tmp_path / "service.py"
    target.write_text((TESTDATA / "tinker_service.py.input").read_text())
    assert patcher.patch_file(target) is True
    assert patcher.patch_file(target) is False
    assert patcher.PREAMBLE_MARKER in target.read_text()


def test_patch_rejects_unrecognized_source(patcher) -> None:
    with pytest.raises(RuntimeError, match="_run_batch"):
        patcher.patch_source("import os\n\nclass TinkerService:\n    pass\n")


class _CommandOp(str, Enum):
    FORWARD_BACKWARD = "forward_backward"
    FORWARD = "forward"


def _load_batch_phase(patcher):
    """Execute the injected preamble in a namespace shaped like ``service.py``
    (``CommandOp`` already imported) and return ``_tg_batch_phase``."""
    namespace: dict[str, object] = {"CommandOp": _CommandOp}
    exec(patcher.PREAMBLE, namespace)
    assert namespace["_tg_role"] is timing_recorder.recording_lane
    return namespace["_tg_batch_phase"]


def _configure(monkeypatch, snapshots: list):
    monkeypatch.setattr(timing_recorder, "MIN_PUBLISH_INTERVAL_S", 0.0)
    monkeypatch.setenv("TRAINING_GYM_SUBSTEP_TIMING", "auto")
    monkeypatch.setenv("TRAINING_GYM_FRAMEWORK_STATUS_URL", "https://dashboard.test")
    monkeypatch.setenv("TRAINING_GYM_TRAINING_RUN_ID", "gateway-run")
    monkeypatch.setattr(timing_recorder, "_TIMING_MODE_CACHE", None)
    monkeypatch.setattr(
        reporting,
        "_enqueue_timing",
        lambda payload, *, final=False: snapshots.append((payload, final)),
    )
    timing_recorder._PRELOOP_RECORDERS.clear()


def test_forward_backward_is_timed_on_persistent_driver_lane(
    patcher, monkeypatch
) -> None:
    snapshots: list = []
    _configure(monkeypatch, snapshots)
    batch_phase = _load_batch_phase(patcher)

    for _ in range(3):
        with batch_phase(_CommandOp.FORWARD_BACKWARD):
            time.sleep(0.001)

    payload, final = snapshots[-1]
    assert final is False, "persistent lane must not close between batches"
    record = RoleTimingRecord.model_validate(payload)
    assert record.storage_key == "pre-loop__driver"
    assert record.training_run_id == "gateway-run"
    phase = payload["phases"]["forward_backward"]
    assert phase["count"] == 3
    assert len(phase["invocations"]) == 3
    assert phase["busy_duration_s"] > 0
    assert set(payload["phases"]) == {"forward_backward"}


def test_forward_only_is_not_timed(patcher, monkeypatch) -> None:
    snapshots: list = []
    _configure(monkeypatch, snapshots)
    batch_phase = _load_batch_phase(patcher)

    with batch_phase(_CommandOp.FORWARD):
        pass

    assert snapshots == []
    assert timing_recorder._PRELOOP_RECORDERS == {}


def test_concurrent_tenant_batches_aggregate_on_one_lane(patcher, monkeypatch) -> None:
    snapshots: list = []
    _configure(monkeypatch, snapshots)
    batch_phase = _load_batch_phase(patcher)

    async def tenant_batch() -> None:
        with batch_phase(_CommandOp.FORWARD_BACKWARD):
            await asyncio.sleep(0.001)

    async def serve() -> None:
        await asyncio.gather(*(tenant_batch() for _ in range(4)))

    asyncio.run(serve())

    assert len(timing_recorder._PRELOOP_RECORDERS) == 1
    phase = snapshots[-1][0]["phases"]["forward_backward"]
    assert phase["count"] == 4
    assert len(phase["invocations"]) == 4


def test_timing_off_bypasses_recorder(patcher, monkeypatch) -> None:
    snapshots: list = []
    _configure(monkeypatch, snapshots)
    monkeypatch.setenv("TRAINING_GYM_SUBSTEP_TIMING", "off")
    monkeypatch.setattr(timing_recorder, "_TIMING_MODE_CACHE", None)
    batch_phase = _load_batch_phase(patcher)

    with batch_phase(_CommandOp.FORWARD_BACKWARD):
        pass

    assert snapshots == []
