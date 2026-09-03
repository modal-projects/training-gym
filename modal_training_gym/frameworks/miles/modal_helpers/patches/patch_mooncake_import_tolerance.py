"""Import mooncake's TransferEngine only when Miles uses P2P weight transfer.

On some Modal EFA hosts the runtime bind-mounts the host's libibverbs over
the system path, and its private ABI (IBVERBS_PRIVATE_*) does not match the
image's libmlx5 — mooncake's TransferEngine import raises. The import sits at
module level on miles' actor import chain, so it kills training runs that
never use it (colocated weight sync runs on Ray IPC; TransferEngine only
backs the p2p update_weight_from_distributed path).

Trying and catching that import is not safe: the failed dlopen can leave
glibc's TLS bookkeeping partially mutated, and a later CUDA/NCCL load aborts
in ``_dl_allocate_tls_init``. This patch therefore removes the module-level
import entirely and performs it only inside the P2P setup function. Colocated
runs never touch mooncake; an actual P2P run gets a clear error at point of use.

Executed at image-build time via ``python3 <this file>``.
"""

import pathlib

MARKER = "PATCHED_MOONCAKE_IMPORT_TOLERANCE"

TARGET = pathlib.Path(
    "/root/miles/miles/backends/megatron_utils/update_weight/"
    "update_weight_from_distributed/p2p_transfer_utils.py"
)

OLD_IMPORT = "from mooncake.engine import TransferEngine"
NEW_IMPORT = "# " + MARKER + ": imported lazily by the P2P setup below"

OLD_USE = "    transfer_engine = TransferEngine()"
NEW_USE = (
    "    try:\n"
    "        from mooncake.engine import TransferEngine\n"
    "    except (ImportError, OSError) as exc:\n"
    "        raise RuntimeError(\n"
    '            "p2p weight transfer requires mooncake\'s TransferEngine, "\n'
    '            "which failed to import on this host (verbs stack mismatch)."\n'
    "        ) from exc\n"
    "    transfer_engine = TransferEngine()"
)


def _patch_file(target: pathlib.Path) -> None:
    if not target.exists():
        raise SystemExit(
            f"{target} not found; miles layout changed — re-check the patch."
        )

    src = target.read_text()
    if MARKER in src:
        print("mooncake import tolerance patch already applied")
        return

    if OLD_IMPORT not in src or OLD_USE not in src:
        raise SystemExit(
            "mooncake import tolerance patch did not match; miles' "
            "p2p_transfer_utils.py has changed. Re-check the import and the "
            "TransferEngine() call site before shipping."
        )

    src = src.replace(OLD_IMPORT, NEW_IMPORT, 1)
    src = src.replace(OLD_USE, NEW_USE, 1)
    target.write_text(src)
    print("Patched mooncake TransferEngine import to load lazily for P2P only")


if __name__ == "__main__":
    _patch_file(TARGET)
