"""Tutorial source for `004_harbor_codegolf` — parsed by generate_tutorial.py.

Each `@markdown`-decorated function contributes a markdown cell (its
docstring); each `@code`-decorated function contributes a code cell (its
body, dedented). Function names are arbitrary. Cell order = source order.
"""

TUTORIAL_METADATA = {
    'framework': '`harbor`',
    'cluster_shape': '2 × 8×H100',
    'summary': 'Qwen3-4B RL code-golf on MBPP with Harbor sandboxes',
    'difficulty': 'Advanced',
    'order': 20,
    'api_classes': [
        'HarborConfig', 'HarborFrameworkConfig',
        'DatasetConfig', 'Qwen3_4B', 'WandbConfig',
    ],
}


from tutorial_generator import code, markdown, notebook_only, py_only, shell


@markdown
def _intro():
    """
    # Qwen3-4B Code Golf on MBPP with Harbor on Modal

    **What Harbor is.** Harbor is a sandbox-based agent evaluation
    framework. During RL training, each rollout spins up an isolated
    sandbox, runs the agent inside it, and scores the result with a
    verifier script. `modal-training-gym`'s `harbor` launcher wires
    Harbor + Miles onto a Modal multi-node cluster.

    **What this tutorial does.** GRPO-tunes Qwen3-4B on
    [MBPP](https://huggingface.co/datasets/Muennighoff/mbpp) (Mostly
    Basic Python Problems) with a code-golf reward: the agent must write
    correct Python, and shorter correct solutions earn a bonus. Each
    rollout creates a Harbor sandbox, the `SingleShotCodeAgent` generates
    code via the model's API, and a verifier checks correctness + awards
    a size bonus for brevity.

    **What you'll need.**
    - Access to Modal's multi-node training preview (2 × 8×H100).
    - A `huggingface-secret` Modal secret with `HF_TOKEN`.
    - A `wandb-secret` Modal secret with `WANDB_API_KEY`.
    - Patience: multi-hour run — use `modal run --detach`.
    """


@notebook_only
@shell("%uv pip install -q git+https://github.com/modal-projects/training-gym.git@main")
def _install():
    pass


@code
def _imports():
    import json
    import re
    import textwrap
    from pathlib import Path
    from typing import Any

    from modal_training_gym.common.dataset import DatasetConfig
    from modal_training_gym.common.models import Qwen3_4B
    from modal_training_gym.common.wandb import WandbConfig
    from modal_training_gym.frameworks.harbor import (
        HarborConfig,
        HarborFrameworkConfig,
    )


@markdown
def _explain_dataset():
    """
    ## Define the dataset

    The MBPP dataset ships as a JSON file of ~1,000 programming tasks.
    Each task has a prompt, a reference solution, and test assertions.
    This dataset class downloads MBPP, then converts each task into a
    Harbor task directory with:

    - `instruction.md` — the prompt (asking the agent to solve and write
      short code)
    - `task.toml` — Harbor metadata (timeouts, resource limits)
    - `tests/verify.py` — a verifier that runs the test assertions and
      computes a code-golf bonus reward
    - `environment/Dockerfile` — the sandbox image
    - `solution/solve.sh` — the reference solution (for debugging, not
      used during training)

    The code-golf reward is `pass_rate * (1 + 0.2 * size_bonus)` where
    `size_bonus = max(0, reference_bytes / candidate_bytes - 1)`. A
    correct solution that's half the reference length earns a 20% bonus.
    """


@code
def _define_dataset():
    DATA_PATH = Path("/data")
    MBPP_ROOT = DATA_PATH / "mbpp_harbor"
    TASKS_DIR = MBPP_ROOT / "tasks"

    LENGTH_BONUS_WEIGHT = 0.2


    def _extract_function_name(code: str) -> str:
        m = re.search(r"def\s+(\w+)\s*\(", code)
        return m.group(1) if m else "solution"


    def _build_instruction(text: str, function_name: str) -> str:
        return textwrap.dedent(f"""\
            You are solving a Python code-golf programming task.

            **Task:** {text}

            You must define a function named `{function_name}`.
            Write your solution as valid Python code to `/workspace/solution.py`.
            Shorter correct solutions earn a higher reward.
        """)


    def _build_task_toml(task_id: int) -> str:
        return textwrap.dedent(f"""\
            [task]
            version = "0.1"
            difficulty = "medium"

            [task.metadata]
            author = "MBPP"
            task_id = {task_id}
            category = "coding"
            tags = ["mbpp", "python", "code-golf"]

            [timeouts]
            agent = 180
            verifier = 180

            [environment]
            cpus = 1
            memory_mb = 2048
            storage_mb = 2048
            allow_internet = false
        """)


    def _build_verify_py(
        test_list: list[str],
        function_name: str,
        task_id: int,
        reference_bytes: int,
    ) -> str:
        tests_literal = json.dumps(test_list)
        return textwrap.dedent(f"""\
            import json, sys, traceback
            from pathlib import Path

            TESTS = {tests_literal}
            FUNCTION_NAME = {function_name!r}
            TASK_ID = {task_id}
            REFERENCE_BYTES = {reference_bytes}
            LENGTH_BONUS_WEIGHT = {LENGTH_BONUS_WEIGHT}

            log_dir = Path("/logs/verifier")
            log_dir.mkdir(parents=True, exist_ok=True)

            solution_path = Path("/workspace/solution.py")
            if not solution_path.exists():
                (log_dir / "reward.json").write_text(
                    json.dumps({{"reward": 0.0, "pass_rate": 0.0, "error": "solution.py not found"}})
                )
                sys.exit(0)

            code = solution_path.read_text()
            candidate_bytes = len(code.encode("utf-8"))

            namespace = {{}}
            try:
                exec(compile(code, "solution.py", "exec"), namespace)
            except Exception as exc:
                (log_dir / "reward.json").write_text(
                    json.dumps({{"reward": 0.0, "pass_rate": 0.0, "error": f"compile: {{exc}}"}})
                )
                sys.exit(0)

            passed = 0
            failures = []
            for i, test in enumerate(TESTS):
                try:
                    exec(test, namespace)
                    passed += 1
                except Exception as exc:
                    failures.append({{"test_index": i, "test": test, "error": str(exc)}})

            pass_rate = passed / len(TESTS) if TESTS else 0.0

            size_bonus = 0.0
            if pass_rate > 0 and candidate_bytes > 0 and REFERENCE_BYTES > 0:
                ratio = min(2.0, REFERENCE_BYTES / candidate_bytes)
                size_bonus = max(0.0, ratio - 1.0)

            reward = pass_rate * (1.0 + LENGTH_BONUS_WEIGHT * size_bonus)

            result = {{
                "reward": round(reward, 4),
                "pass_rate": round(pass_rate, 4),
                "passed": passed,
                "total": len(TESTS),
                "candidate_bytes": candidate_bytes,
                "reference_bytes": REFERENCE_BYTES,
                "size_bonus": round(size_bonus, 4),
            }}
            (log_dir / "reward.json").write_text(json.dumps(result))
            if failures:
                (log_dir / "details.json").write_text(json.dumps(failures, indent=2))
        """)


    def _build_test_sh(task_dir_name: str) -> str:
        return textwrap.dedent(f"""\
            #!/usr/bin/env bash
            set -euo pipefail
            mkdir -p /logs/verifier
            python3 /workspace/../tests/verify.py 2>&1 || true
            if [ ! -f /logs/verifier/reward.json ]; then
                echo '{{"reward": 0.0, "pass_rate": 0.0, "error": "verifier crashed"}}' > /logs/verifier/reward.json
            fi
        """)


    class MBPPCodeGolfDataset(DatasetConfig):
        """Downloads MBPP and creates Harbor task directories for code-golf training."""

        def __init__(self, train_size: int = 900, limit: int | None = None):
            super().__init__()
            self.train_size = train_size
            self.limit = limit

        def prepare(self) -> None:
            from huggingface_hub import hf_hub_download

            local_path = hf_hub_download(
                repo_id="Muennighoff/mbpp",
                filename="data/sanitized-mbpp.json",
                repo_type="dataset",
            )

            with open(local_path) as fh:
                records = json.load(fh)

            if self.limit:
                records = records[: self.limit]

            TASKS_DIR.mkdir(parents=True, exist_ok=True)

            for row in records:
                task_id = int(row["task_id"])
                text = str(row.get("prompt") or row.get("text", ""))
                code = str(row["code"])
                test_list = list(row.get("test_list", []))
                function_name = _extract_function_name(code)
                reference_bytes = len(code.encode("utf-8"))

                task_dir = TASKS_DIR / f"mbpp_{task_id:04d}"
                task_dir.mkdir(exist_ok=True)

                (task_dir / "instruction.md").write_text(
                    _build_instruction(text, function_name)
                )
                (task_dir / "task.toml").write_text(_build_task_toml(task_id))

                env_dir = task_dir / "environment"
                env_dir.mkdir(exist_ok=True)
                (env_dir / "Dockerfile").write_text(
                    "FROM python:3.11-slim\nWORKDIR /workspace\n"
                )

                tests_dir = task_dir / "tests"
                tests_dir.mkdir(exist_ok=True)
                (tests_dir / "verify.py").write_text(
                    _build_verify_py(test_list, function_name, task_id, reference_bytes)
                )
                (tests_dir / "test.sh").write_text(_build_test_sh(task_dir.name))

                solution_dir = task_dir / "solution"
                solution_dir.mkdir(exist_ok=True)
                (solution_dir / "solve.sh").write_text(
                    f"#!/usr/bin/env bash\ncat > /workspace/solution.py << 'PYEOF'\n{code}\nPYEOF\n"
                )

            print(f"Prepared {len(records)} MBPP tasks at {TASKS_DIR}")

            manifest = {
                "dataset": "Muennighoff/mbpp",
                "tasks": len(records),
                "task_root": str(TASKS_DIR),
            }
            (MBPP_ROOT / "manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n"
            )


@markdown
def _explain_config():
    """
    ## Define the experiment

    Harbor training uses the same Miles RL engine as slime but adds
    sandbox-based evaluation. The key config choices:

    **Agent**
    - `agent_import_path` points to `SingleShotCodeAgent`, which makes
      one chat-completion call and writes the extracted code to the
      sandbox. It records a structured trajectory (instruction + response
      with timing and token usage) for the observability dashboard.
    - `agent_kwargs={"temperature": 0.7, "max_tokens": 1024}` — sampling
      parameters forwarded to the agent.

    **Sandboxes**
    - `sandbox_timeout_secs=180` — each Harbor sandbox gets 3 minutes.
    - `task_root` / `instruction_path` — where to find task directories
      on the data volume.

    **Parallelism and cluster topology**
    - `Qwen3_4B` declares a `HarborPreset` with `n_nodes=2`,
      `tensor_model_parallel_size=2`, and `sequence_parallel=True`. These
      are applied automatically — no need to set them in `HarborFrameworkConfig`.
    - Model architecture flags (`--num-layers`, `--hidden-size`, etc.)
      are pulled automatically from `Qwen3_4B().architecture`.
    """


@code
def _define_config():
    AGENT_IMPORT_PATH = (
        "modal_training_gym.frameworks.harbor.agents.single_shot_code"
        ":SingleShotCodeAgent"
    )

    dataset = MBPPCodeGolfDataset(train_size=900)

    framework_config = HarborFrameworkConfig(
        agent_import_path=AGENT_IMPORT_PATH,
        agent_model_name="model",
        agent_kwargs={"temperature": 0.7, "max_tokens": 1024},
        task_root=str(TASKS_DIR),
        instruction_path="instruction.md",
        sandbox_timeout_secs=180,
        sandbox_idle_timeout_secs=60,
    )

    harbor = HarborConfig(
        model=Qwen3_4B(),
        dataset=dataset,
        wandb=WandbConfig(
            project="training-gym",
            group="harbor-code-golf",
            exp_name="qwen3-4b-mbpp",
        ),
        framework_config=framework_config,
    )


@markdown
def _explain_sandbox_scaling():
    """
    ## Understanding sandbox scaling

    Harbor plugs into Miles as a custom rollout generator. During each
    training step, Miles loads `num_rollout` prompts and requests
    `n_samples_per_prompt` completions per prompt. Each sample spawns
    exactly **one Modal sandbox** via the Harbor agent function. So the
    total number of sandboxes created per training step is:

    ```
    total_sandboxes = num_rollout × n_samples_per_prompt
    ```

    With the defaults (`num_rollout=200`, `n_samples_per_prompt=8`), each
    step creates **1,600 sandboxes**. Samples are grouped into
    `rollout_batch_size`-sized chunks during generation.

    The key scaling knobs in `HarborFrameworkConfig`:

    | Knob | Default | Effect |
    |------|---------|--------|
    | `num_rollout` | 200 | Prompts per training step |
    | `n_samples_per_prompt` | 8 | Samples per prompt — multiplies sandbox count |
    | `rollout_batch_size` | 64 | How samples are grouped during generation |
    | `n_nodes` | 1 | Cluster size — more nodes = more rollout throughput |
    | `colocate` | True | Whether actor + rollout share GPUs |
    | `sandbox_timeout_secs` | 1800 | Per-sandbox max runtime |
    | `sandbox_idle_timeout_secs` | 300 | Idle before sandbox cleanup |

    Sandboxes are spawned **sequentially** within the Miles rollout phase —
    there is no sandbox-level concurrency knob during training. To increase
    throughput, scale up `n_nodes` (more GPU capacity for faster model
    inference) or reduce `sandbox_timeout_secs` to fail slow sandboxes
    faster.

    The `eval_harbor` function (used after training) *does* have a
    `max_concurrency` parameter that parallelizes sandbox creation via an
    asyncio semaphore.
    """


@notebook_only
@code
def _print_sandbox_count():
    total = framework_config.num_rollout * framework_config.n_samples_per_prompt
    batches = -(-total // framework_config.rollout_batch_size)
    print(f"Sandboxes per training step: {framework_config.num_rollout} prompts × {framework_config.n_samples_per_prompt} samples = {total}")
    print(f"Rollout batches: {batches} (batch size {framework_config.rollout_batch_size})")
    print(f"Sandbox timeout: {framework_config.sandbox_timeout_secs}s")
    print(f"Cluster: {framework_config.n_nodes} node(s), colocate={framework_config.colocate}")


@markdown
def _build_section():
    """
    ## Build and run

    `build_app()` returns a Modal app with `download_model`,
    `prepare_dataset`, `train_multi_node`, and `eval_harbor`.
    """


@code
def _build_app():
    app = harbor.build_app(name="harbor-code-golf")


@py_only
@markdown
def _run_cli():
    """
    From the CLI:

    ```bash
    # 1. Prepare the dataset (once): download MBPP, create Harbor tasks
    uv run modal run tutorials/rl/004_harbor_codegolf/004_harbor_codegolf.py::app.prepare_dataset

    # 2. Download model weights (once)
    uv run modal run tutorials/rl/004_harbor_codegolf/004_harbor_codegolf.py::app.download_model

    # 3. Train
    uv run modal run --detach tutorials/rl/004_harbor_codegolf/004_harbor_codegolf.py::app.train_multi_node
    ```
    """


@notebook_only
@markdown
def _run_interactive():
    """
    Interactive — one stage per cell:
    """


@notebook_only
@code
def _invoke_prepare():
    import modal

    with modal.enable_output():
        with app.run():
            app.prepare_dataset.remote()


@notebook_only
@code
def _invoke_download():
    with modal.enable_output():
        with app.run():
            app.download_model.remote()


@notebook_only
@code
def _invoke_train():
    with modal.enable_output():
        with app.run():
            app.train_multi_node.remote()
