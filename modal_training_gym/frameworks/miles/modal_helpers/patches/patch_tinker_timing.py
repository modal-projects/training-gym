"""Time every ``forward_backward`` the Tinker gateway runs on the trainer.

``serve_tinker.py`` has no rollout loop, so the substep-timing patcher over
``train.py`` never fires for a gateway. The dispatcher in
``miles/tinker/core/service.py`` funnels every tenant's ``forward_backward`` /
``forward`` batch through ``TinkerService._run_batch``; this patcher wraps the
awaited trainer call there in a persistent ``driver`` lane (``rollout_id=None``,
the dashboard's ``pre-loop`` key) so each invocation lands in the run's timing
records with its own interval. ``forward_only`` batches are left unmeasured.

Runs after ``miles_git_ref`` overlays (which would otherwise revert it) and is
a no-op when the checkout has no Tinker gateway.
"""

from __future__ import annotations

import ast
from pathlib import Path

PREAMBLE_MARKER = "PATCHED_TRAINING_GYM_TINKER_TIMING_PREAMBLE"
PHASE_MARKER = "PATCHED_TRAINING_GYM_TIMING_FORWARD_BACKWARD"
PHASE = "forward_backward"
LANE_ROLE = "driver"

ROOT = Path("/root/miles")
TARGET = Path("miles/tinker/core/service.py")

PREAMBLE = (
    f"# {PREAMBLE_MARKER}: bootstrap gateway forward_backward timing\n"
    "import sys as _tg_sys\n"
    "from contextlib import contextmanager as _tg_cm\n"
    "if '/root' not in _tg_sys.path:\n"
    "    _tg_sys.path.insert(0, '/root')\n"
    "try:\n"
    "    from modal_training_gym.common.timing_recorder import (\n"
    "        recording_lane as _tg_role,\n"
    "    )\n"
    "except ImportError:\n"
    "    print('WARNING: modal_training_gym not importable; gateway timing off')\n"
    "    _tg_role = None\n"
    "\n"
    "\n"
    "@_tg_cm\n"
    "def _tg_batch_phase(op):\n"
    "    if _tg_role is None or op != CommandOp.FORWARD_BACKWARD:\n"
    "        yield\n"
    "        return\n"
    f"    with _tg_role({LANE_ROLE!r}, None) as _tg_rec, _tg_rec.phase({PHASE!r}):\n"
    "        yield\n"
    "\n"
)


def _after_imports(tree: ast.Module) -> int:
    """Line after the module docstring and top-level imports."""
    end_line = 0
    for index, node in enumerate(tree.body):
        if (
            index == 0
            and isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            end_line = node.end_lineno or node.lineno
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            end_line = node.end_lineno or node.lineno
        else:
            break
    return end_line


def _forward_call(tree: ast.Module) -> ast.Assign:
    """The ``outputs = await forward(...)`` statement in ``TinkerService._run_batch``."""
    for cls in tree.body:
        if not (isinstance(cls, ast.ClassDef) and cls.name == "TinkerService"):
            continue
        for fn in cls.body:
            if not (isinstance(fn, ast.AsyncFunctionDef) and fn.name == "_run_batch"):
                continue
            matches = [
                node
                for node in ast.walk(fn)
                if isinstance(node, ast.Assign)
                and isinstance(node.value, ast.Await)
                and isinstance(node.value.value, ast.Call)
                and isinstance(node.value.value.func, ast.Name)
                and node.value.value.func.id == "forward"
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    "expected 1 `await forward(...)` in TinkerService._run_batch,"
                    f" found {len(matches)}"
                )
            return matches[0]
    raise RuntimeError("TinkerService._run_batch not found")


def patch_source(src: str) -> str:
    if PREAMBLE_MARKER in src:
        return src
    tree = ast.parse(src)
    call = _forward_call(tree)
    lines = src.splitlines(keepends=True)
    start, end = call.lineno - 1, call.end_lineno or call.lineno
    indent = lines[start][: len(lines[start]) - len(lines[start].lstrip(" "))]
    wrapped = [
        f"{indent}# {PHASE_MARKER}\n",
        f"{indent}with _tg_batch_phase(batch.op):\n",
        *(f"    {line}" if line.strip() else line for line in lines[start:end]),
    ]
    lines[start:end] = wrapped
    cut = _after_imports(tree)
    patched = "".join(lines[:cut]) + PREAMBLE + "".join(lines[cut:])
    compile(patched, str(TARGET), "exec")
    return patched


def patch_file(path: Path) -> bool:
    if not path.exists():
        print(f"{path} not found; no Tinker gateway in this checkout, skipping")
        return False
    src = path.read_text()
    patched = patch_source(src)
    if patched == src:
        print(f"{path} already patched for gateway timing")
        return False
    path.write_text(patched)
    print(f"Patched {path} for gateway forward_backward timing")
    return True


def main() -> None:
    if not ROOT.is_dir():
        return
    patch_file(ROOT / TARGET)


if __name__ == "__main__":
    main()
