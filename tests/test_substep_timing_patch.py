"""Golden-file test for the substep-timing patcher.

Anchors are literal source lines, so the test worth having is one that runs them
over the sources they were written for: ``tests/testdata`` holds slime's
``train.py`` after the rollout-status patcher (the state the timing patcher sees)
and miles' two entrypoints as shipped in the pinned image. Regenerate the
expected output with ``uv run pytest tests/test_substep_timing_patch.py
--rewrite``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

TESTDATA = Path(__file__).parent / "testdata"
FRAMEWORKS = Path(__file__).parents[1] / "modal_training_gym" / "frameworks"


def patcher_path(framework: str) -> Path:
    return (
        FRAMEWORKS / framework / "modal_helpers" / "patches" / "patch_substep_timing.py"
    )


# framework, fixture, entrypoint, golden output, driver phases the loop records.
DRIVERS = [
    (
        "slime",
        "slime/train.py.status.output",
        "train.py",
        "slime/train.py.timing.output",
        {
            "evaluate_rollouts",
            "generate_rollouts",
            "offload_rollout",
            "train_models",
            "checkpoint_save",
            "offload_train",
            "weight_sync",
            "initial_weight_sync",
            "evaluate_rollouts_end",
        },
    ),
    (
        "miles",
        "miles/train.py.input",
        "train.py",
        "miles/train.py.timing.output",
        {
            "evaluate_rollouts",
            "generate_rollouts",
            "offload_rollout",
            "train_models",
            "checkpoint_save",
            "offload_train",
            "weight_sync",
            "initial_weight_sync",
            "evaluate_rollouts_end",
        },
    ),
    (
        "miles",
        "miles/train_async.py.input",
        "train_async.py",
        "miles/train_async.py.timing.output",
        {
            "wait_for_rollout",
            "wait_for_next_rollout",
            "train_models",
            "checkpoint_save",
            "weight_sync",
            "initial_weight_sync",
            "evaluate_rollouts_end",
        },
    ),
    (
        "slime",
        "slime/train_async.py.status.output",
        "train_async.py",
        "slime/train_async.py.timing.output",
        {
            "wait_for_rollout",
            "wait_for_next_rollout",
            "train_models",
            "checkpoint_save",
            "weight_sync",
            "initial_weight_sync",
            "evaluate_rollouts_end",
        },
    ),
]


@pytest.fixture(scope="session")
def patchers() -> dict[str, object]:
    """Load each framework's patch script by path: they run standalone in the
    image, and are deliberately not importable as part of the package."""
    loaded = {}
    for framework in ("slime", "miles"):
        name = f"patch_substep_timing_{framework}"
        spec = importlib.util.spec_from_file_location(name, patcher_path(framework))
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module  # its dataclass resolves annotations here
        spec.loader.exec_module(module)
        loaded[framework] = module
    return loaded


@pytest.fixture(scope="session")
def miles(patchers):
    return patchers["miles"]


def _patched(patcher, tmp_path, fixture: str, entrypoint: str) -> str:
    work = tmp_path / entrypoint
    work.write_text((TESTDATA / fixture).read_text())
    patcher._patch_file(work, patcher.ENTRYPOINTS[entrypoint])
    return work.read_text()


@pytest.mark.parametrize(
    "framework, fixture, entrypoint, golden, expected_phases", DRIVERS
)
def test_patch_matches_golden(
    patchers, tmp_path, request, framework, fixture, entrypoint, golden, expected_phases
):
    patched = _patched(patchers[framework], tmp_path, fixture, entrypoint)
    golden_path = TESTDATA / golden

    if request.config.getoption("--rewrite"):
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        golden_path.write_text(patched)
        return

    assert patched == golden_path.read_text(), (
        f"golden mismatch for {golden}; rerun with --rewrite to accept"
    )
    assert "with _tg_role('driver', rollout_id) as _tg_rec:" in patched
    for phase in expected_phases:
        assert f"with _tg_rec.phase('{phase}'):" in patched
    compile(patched, entrypoint, "exec")


def test_a_conditional_phase_is_timed_inside_its_branch(patchers, tmp_path):
    """A skipped save must record nothing, not a 0s bar on every rollout.

    Its condition spans three lines, so the closing ``):`` sits at the ``if``'s
    own indent and must not be read as the start of another clause.
    """
    patched = _patched(
        patchers["slime"], tmp_path, "slime/train.py.status.output", "train.py"
    )
    save = patched.split("if release_train or should_run_periodic_action(")[1]
    before_wrap = save.split("with _tg_rec.phase('checkpoint_save'):")[0]
    assert before_wrap.split("#")[0].rstrip().endswith("):")


def test_patching_twice_is_a_no_op(miles, tmp_path, capsys):
    work = tmp_path / "train.py"
    work.write_text((TESTDATA / "miles/train.py.input").read_text())
    miles._patch_file(work, miles.ENTRYPOINTS["train.py"])
    once = work.read_text()
    miles._patch_file(work, miles.ENTRYPOINTS["train.py"])
    assert work.read_text() == once
    assert "already patched" in capsys.readouterr().out


def test_a_moved_anchor_fails_the_build(miles, tmp_path):
    """Half-instrumented timing is worse than none: a lane would just be absent."""
    work = tmp_path / "train.py"
    source = (TESTDATA / "miles/train.py.input").read_text()
    work.write_text(
        source.replace("await offload_train()", "await offload_train(args)")
    )
    with pytest.raises(RuntimeError, match="expected 1 occurrence"):
        miles._patch_file(work, miles.ENTRYPOINTS["train.py"])


@pytest.mark.parametrize("framework", ["miles", "slime"])
def test_per_sample_generation_target_wraps_only_generation_branch(
    patchers, tmp_path, framework
):
    patcher = patchers[framework]
    target = next(
        target
        for target in patcher.PACKAGE_TARGETS
        if target.path.endswith("sglang_rollout.py")
    )
    blocks = dict(target.blocks)
    source = (
        "async def generate_and_rm(args, sample, sampling_params, evaluation=False):\n"
        "    async with state.semaphore:\n"
        "        with state.dp_rank_context() as _:\n"
        f"{blocks['sample_generation']}"
        f"{blocks['reward']}"
        "    return sample\n"
    )
    work = tmp_path / target.path
    work.parent.mkdir(parents=True)
    work.write_text(source)
    patcher.patch_package_file(tmp_path, target)
    patched = work.read_text()
    assert patched.count("with _tg_time_phase('sample_generation'):") == 1
    assert patched.count("with _tg_time_phase('reward'):") == 1
    assert patched.index("with _tg_time_phase('sample_generation'):") < patched.index(
        "with _tg_time_phase('reward'):"
    )
    compile(patched, str(work), "exec")


def test_missing_package_file_warns_and_continues(miles, tmp_path, capsys):
    miles.patch_package_file(tmp_path, miles.PACKAGE_TARGETS[0])
    assert "substep timing patch skipped" in capsys.readouterr().out
