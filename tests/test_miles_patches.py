"""Golden-file tests for the miles rollout-status and advantage patchers."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from modal_training_gym.frameworks.miles.modal_helpers.patches import (
    patch_advantage_distribution as advantage_patcher,
    patch_mooncake_import_tolerance as mooncake_patcher,
    patch_skip_final_weight_sync as final_sync_patcher,
    patch_rollout_status_reporting as rollout_patcher,
)

TESTDATA = Path(__file__).parent / "testdata" / "miles"

STATUS_PATCH_FILES = ("train.py", "train_async.py", "log_utils.py")


@pytest.fixture(scope="session")
def miles_inputs() -> dict[str, str]:
    inputs = [TESTDATA / f"{name}.input" for name in STATUS_PATCH_FILES]
    assert inputs
    return {path.name.removesuffix(".input"): path.read_text() for path in inputs}


def test_all_miles_goldens_compile():
    for golden in TESTDATA.glob("*.output"):
        compile(golden.read_text(), str(golden), "exec")


def _apply_patcher(name: str, work: Path) -> None:
    if name == "log_utils.py":
        advantage_patcher._patch_file(work)
    else:
        rollout_patcher._patch_file(work)


def test_missing_patch_targets_are_skipped(tmp_path, capsys):
    missing = tmp_path / "train.py"
    rollout_patcher._patch_file(missing)
    assert not missing.exists()
    assert "not found, skipping rollout-status patch" in capsys.readouterr().out

    advantage_patcher._patch_file(missing)
    assert not missing.exists()
    assert "not found, skipping advantage-distribution patch" in capsys.readouterr().out


def test_patch_matches_golden(miles_inputs, tmp_path, request):
    rewrite_goldens = request.config.getoption("--rewrite")
    for name, source in miles_inputs.items():
        golden_path = TESTDATA / (
            f"{name}.advantage.output"
            if name == "log_utils.py"
            else f"{name}.status.output"
        )
        work = tmp_path / name
        work.write_text(source)
        _apply_patcher(name, work)
        actual = work.read_text()

        if rewrite_goldens:
            golden_path.write_text(actual)
            continue

        assert golden_path.exists(), (
            f"Golden output file does not exist: {golden_path}. "
            "Regenerate and review the expected patch output with "
            "uv run pytest tests/test_miles_patches.py --rewrite."
        )
        expected = golden_path.read_text()
        assert actual == expected, (
            f"golden mismatch for {name}; rerun with --rewrite to accept"
        )


def test_mooncake_import_is_moved_inside_p2p_setup(tmp_path):
    work = tmp_path / "p2p_transfer_utils.py"
    work.write_text(
        "from mooncake.engine import TransferEngine\n\n"
        "def setup_transfer_engine():\n"
        "    transfer_engine = TransferEngine()\n"
        "    return transfer_engine\n"
    )

    mooncake_patcher._patch_file(work)
    patched = work.read_text()
    tree = ast.parse(patched)

    assert mooncake_patcher.MARKER in patched
    assert not any(
        isinstance(node, ast.ImportFrom) and node.module == "mooncake.engine"
        for node in tree.body
    )
    setup = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "setup_transfer_engine"
    )
    assert any(
        isinstance(node, ast.ImportFrom) and node.module == "mooncake.engine"
        for node in ast.walk(setup)
    )

    # Image builds can layer the same patch more than once.
    mooncake_patcher._patch_file(work)
    assert work.read_text() == patched


def test_final_weight_sync_is_skipped_only_on_the_last_rollout(tmp_path):
    """The guard must break before the post-save sync on the final rollout,
    leave every earlier rollout's sync intact, and keep eval's weights fresh."""
    work = tmp_path / "train.py"
    work.write_text((TESTDATA / "train.py.input").read_text())

    final_sync_patcher._patch_file(work)
    patched = work.read_text()
    assert final_sync_patcher.MARKER in patched

    tree = ast.parse(patched)  # the patched loop must still be valid Python

    # Find the guard: an `if` whose body is a bare `break`, sitting in the
    # rollout loop ahead of the sync.
    guards = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and any(isinstance(b, ast.Break) for b in node.body)
        and "num_rollout" in ast.unparse(node.test)
        and "should_run_periodic_action" in ast.unparse(node.test)
    ]
    assert len(guards) == 1, "expected exactly one final-rollout break guard"
    assert "rollout_id + 1 >= args.num_rollout" in ast.unparse(guards[0].test)

    # It has to precede the sync, or it would not prevent anything.
    assert patched.index(final_sync_patcher.MARKER) < patched.index(
        "await offload_train()"
    )

    # The real (non-final) syncs and the eval call must survive untouched.
    original = (TESTDATA / "train.py.input").read_text()
    for line in (
        "await offload_train()",
        "await rollout_manager.onload_weights.remote()",
        "await actor_model.update_weights(rollout_id=rollout_id)",
        "await rollout_manager.onload_kv.remote()",
        "await rollout_manager.eval.remote(rollout_id)",
    ):
        assert patched.count(line) == original.count(line), line


def test_final_weight_sync_patch_is_idempotent_and_fails_on_drift(tmp_path):
    work = tmp_path / "train.py"
    work.write_text((TESTDATA / "train.py.input").read_text())
    final_sync_patcher._patch_file(work)
    once = work.read_text()
    final_sync_patcher._patch_file(work)
    assert work.read_text() == once

    drifted = tmp_path / "drifted.py"
    drifted.write_text(
        (TESTDATA / "train.py.input")
        .read_text()
        .replace("await offload_train()", "await offload_train_v2()")
    )
    with pytest.raises(SystemExit):
        final_sync_patcher._patch_file(drifted)


def _run_patched_loop_tail(num_rollout: int, eval_interval: int | None) -> list[str]:
    """Execute the patched save->guard->sync tail of miles' train loop with
    mocks, once per rollout, and return the ordered calls it makes."""
    import asyncio
    import textwrap

    work_src = (TESTDATA / "train.py.input").read_text()
    tmp = TESTDATA.parent / "_scratch_train.py"
    try:
        tmp.write_text(work_src)
        final_sync_patcher._patch_file(tmp)
        src = tmp.read_text()
    finally:
        tmp.unlink(missing_ok=True)

    start = src.index("        external_save = args.save_trigger_sentinel")
    end = src.index(
        "        if should_run_periodic_action(rollout_id, args.eval_interval"
    )
    tail = textwrap.dedent(src[start:end]).replace("break", "return 'BREAK'")

    calls: list[str] = []

    class _Args:
        save_trigger_sentinel = None
        save_interval = 50
        offload_rollout = True

    args = _Args()
    args.num_rollout = num_rollout
    args.eval_interval = eval_interval

    class _Remote:
        def __init__(self, name):
            self.name = name

        async def remote(self):
            calls.append(self.name)

    class _RM:
        onload_weights = _Remote("onload_weights")
        onload_kv = _Remote("onload_kv")

    class _Actor:
        async def update_weights(self, rollout_id):
            calls.append(f"update_weights[{rollout_id}]")

    async def save(rollout_id, force_sync=False):
        calls.append(f"save[{rollout_id}]")

    async def offload_train():
        calls.append("offload_train")

    def should_run_periodic_action(rid, interval, per_epoch, num_rollout=None):
        if interval is None:
            return False
        if num_rollout is not None and rid == num_rollout - 1:
            return True
        return (rid + 1) % interval == 0

    class _Log:
        def info(self, *a):
            pass

    ns = dict(
        args=args,
        rollout_manager=_RM,
        actor_model=_Actor(),
        save=save,
        offload_train=offload_train,
        should_run_periodic_action=should_run_periodic_action,
        num_rollout_per_epoch=10**9,
        logger=_Log(),
        os=__import__("os"),
    )

    async def loop():
        for rollout_id in range(num_rollout):
            ns["rollout_id"] = rollout_id
            exec("async def _b():\n" + textwrap.indent(tail, "    "), ns)
            if await ns["_b"]() == "BREAK":
                calls.append(f"BREAK@{rollout_id}")
                break

    asyncio.run(loop())
    return calls


def test_patched_loop_skips_only_the_final_post_save_sync():
    calls = _run_patched_loop_tail(num_rollout=3, eval_interval=None)
    # Final rollout: save, then break -- no onload/update_weights after it.
    assert calls[-2:] == ["save[2]", "BREAK@2"]
    assert "update_weights[2]" not in calls
    # Earlier rollouts still sync exactly as upstream does.
    assert calls.count("onload_weights") == 2
    assert "update_weights[0]" in calls and "update_weights[1]" in calls


def test_patched_loop_keeps_final_sync_when_eval_is_due():
    calls = _run_patched_loop_tail(num_rollout=3, eval_interval=3)
    assert "update_weights[2]" in calls
    assert not any(c.startswith("BREAK") for c in calls)


def test_patched_loop_single_rollout_saves_then_breaks():
    assert _run_patched_loop_tail(num_rollout=1, eval_interval=None) == [
        "save[0]",
        "BREAK@0",
    ]
