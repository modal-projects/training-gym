"""Kimi K2.5 model spec as a concrete HFModelConfiguration subclass.

`moonshotai/Kimi-K2.5` ships native INT4 weights; downstream training (and
Megatron-bridge ingestion) operates on BF16, so `download()` is an
override of the generic HF path that does both the snapshot download *and*
the INT4 → BF16 conversion via the repo-bundled
`modal_training_gym/tools/convert_kimi_int4_to_bf16.py` script.

The script is mounted on every framework image at ``TOOLS_REMOTE_PATH`` by
each launcher (see ``common.framework.mount_tools_dir``), so this override
resolves the same path regardless of framework.
"""

import os
import subprocess

from .base import HFModelConfiguration, ModelTrainingConfig

# Mirrors common.framework.TOOLS_REMOTE_PATH — inlined here to avoid a
# cross-module dependency on framework helpers at model-import time.
_TOOLS_REMOTE_PATH = "/opt/training-gym/tools"


class Kimi_K2_5(HFModelConfiguration):
    """Kimi K2.5 from Moonshot AI.

    Ships native INT4 weights; ``download()`` performs both the
    HF snapshot download and INT4-to-BF16 conversion. The converted
    weights are written to ``model_path``.

    Methods
    -------
    download()
        Downloads INT4 weights and converts to BF16 using the bundled
        conversion script.
    """

    model_name = "moonshotai/Kimi-K2.5"
    model_path = "/checkpoints/Kimi-K2.5-bf16"
    training = ModelTrainingConfig(gpu_type="H100", n_nodes=4)

    def download(self) -> None:
        from huggingface_hub import snapshot_download

        source_dir = snapshot_download(repo_id=self.model_name)
        print(f"Source: {source_dir}")
        print(f"\n=== INT4 → BF16: {self.model_path} ===")
        subprocess.run(
            [
                "python",
                os.path.join(_TOOLS_REMOTE_PATH, "convert_kimi_int4_to_bf16.py"),
                "--model-dir",
                source_dir,
                "--output-dir",
                str(self.model_path),
            ],
            check=True,
        )
        print(f"\n=== Done: {self.model_path} ===")
