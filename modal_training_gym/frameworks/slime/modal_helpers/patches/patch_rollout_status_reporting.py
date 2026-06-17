"""Patch slime train entrypoints to report rollout-engine startup status."""

from __future__ import annotations

import re
from pathlib import Path


PREAMBLE_MARKER = "PATCHED_TRAINING_GYM_PREAMBLE"
ROLLOUT_MARKER = "PATCHED_TRAINING_GYM_ROLLOUT_STATUS"
WEIGHT_SYNC_MARKER = "PATCHED_TRAINING_GYM_WEIGHT_SYNC_STATUS"
GENERATE_ROLLOUT_MARKER = "PATCHED_TRAINING_GYM_GENERATE_ROLLOUT_STATUS"

PREAMBLE = (
    f"# {PREAMBLE_MARKER}: bootstrap phase reporter (runs once per process)\n"
    "import sys as _tg_sys\n"
    "if '/root' not in _tg_sys.path:\n"
    "    _tg_sys.path.insert(0, '/root')\n"
    "try:\n"
    "    from modal_training_gym.frameworks.slime.phase_reporting import (\n"
    "        install_base64_log_eliding as _tg_install_base64_log_eliding,\n"
    "        report_generate_rollouts as _tg_report_generate_rollouts,\n"
    "        report_rollout_initializing as _tg_report_rollout_initializing,\n"
    "        report_weight_sync as _tg_report_weight_sync,\n"
    "    )\n"
    "    _tg_install_base64_log_eliding()\n"
    "except ImportError:\n"
    "    def _tg_install_base64_log_eliding(): pass\n"
    "    def _tg_report_generate_rollouts(args): pass\n"
    "    def _tg_report_rollout_initializing(args): pass\n"
    "    def _tg_report_weight_sync(args): pass\n"
    "\n"
)


def _patch_file(path: Path) -> None:
    if not path.exists():
        print(f"WARNING: {path} not found, skipping rollout-status patch")
        return

    src = path.read_text()
    needs_preamble = PREAMBLE_MARKER not in src
    needs_rollout = ROLLOUT_MARKER not in src
    needs_weight_sync = WEIGHT_SYNC_MARKER not in src
    needs_generate_rollout = GENERATE_ROLLOUT_MARKER not in src

    if not (
        needs_preamble or needs_rollout or needs_weight_sync or needs_generate_rollout
    ):
        print(f"{path.name} already patched for rollout status reporting")
        return

    if needs_preamble:
        src = PREAMBLE + src
    elif "_tg_report_generate_rollouts" not in src:
        src = src.replace(
            "    from modal_training_gym.frameworks.slime.phase_reporting import (\n",
            "    from modal_training_gym.frameworks.slime.phase_reporting import (\n"
            "        report_generate_rollouts as _tg_report_generate_rollouts,\n",
            1,
        )
        src = src.replace(
            "except ImportError:\n",
            "except ImportError:\n    def _tg_report_generate_rollouts(args): pass\n",
            1,
        )

    rollout_count = 0
    if needs_rollout:
        rollout_pattern = re.compile(
            r"^(?P<indent>[ \t]*)rollout_manager, num_rollout_per_epoch = create_rollout_manager\(args, pgs\[\"rollout\"\]\)",
            re.M,
        )

        def _rollout_replacement(match: re.Match[str]) -> str:
            indent = match.group("indent")
            return (
                f"{indent}# {ROLLOUT_MARKER}: rollout engine startup state\n"
                f"{indent}_tg_report_rollout_initializing(args)\n"
                f'{indent}rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])'
            )

        src, rollout_count = rollout_pattern.subn(_rollout_replacement, src, count=1)

    weight_sync_count = 0
    if needs_weight_sync or needs_generate_rollout:
        weight_sync_pattern = re.compile(
            r"^(?P<indent>[ \t]*)(?P<call>(?:await[ \t]+)?(?:[A-Za-z_][A-Za-z0-9_]*\.)?update_weights\(\))",
            re.M,
        )

        def _weight_sync_replacement(match: re.Match[str]) -> str:
            indent = match.group("indent")
            call = match.group("call")
            lines = []
            if needs_weight_sync:
                lines.extend(
                    [
                        f"{indent}# {WEIGHT_SYNC_MARKER}: weight sync state",
                        f"{indent}_tg_report_weight_sync(args)",
                    ]
                )
            lines.extend(
                [
                    f"{indent}{call}",
                    f"{indent}# {GENERATE_ROLLOUT_MARKER}: rollout generation state",
                    f"{indent}_tg_report_generate_rollouts(args)",
                ]
            )
            return "\n".join(lines)

        src, weight_sync_count = weight_sync_pattern.subn(
            _weight_sync_replacement, src, count=1
        )

    failed = []
    if needs_rollout and rollout_count != 1:
        failed.append("rollout init")
    if (needs_weight_sync or needs_generate_rollout) and weight_sync_count != 1:
        failed.append("weight sync")
    if failed:
        print(f"WARNING: Could not patch {path.name} for: {', '.join(failed)}")

    path.write_text(src)
    print(f"Patched {path.name} with rollout status reporting")


_patch_file(Path("/root/slime/train.py"))
_patch_file(Path("/root/slime/train_async.py"))
