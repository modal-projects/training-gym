# pyright: reportUndefinedVariable=false

TUTORIAL_METADATA = {
    "framework": "`slime`",
    "cluster_shape": "4 × 8×H100",
    "summary": "Python code-golf with Modal Sandboxes",
    "difficulty": "Intermediate",
    "order": 20,
    "api_classes": [
        "HarborDataset",
        "Endpoint",
        "Qwen3_5_9B",
        "SlimeRecipe",
        "TrainConfig",
        "extract_code",
    ],
}


from tutorial_generator import code, markdown, notebook_only, py_only, shell


@markdown
def _intro():
    """
    # Code RL with Modal Sandboxes

    This tutorial trains Qwen3.5-9B on MBPP as Harbor Python code-golf tasks.
    Reward is Harbor verifier correctness plus a length bonus for shorter
    passing programs. Convert MBPP into Harbor task directories, score
    candidates in Modal sandboxes with each task's verifier, then GRPO-train
    and serve the checkpoint for eval.
    """


@py_only
@markdown
def _run_instructions():
    """
    Run with:
    ```
    uv run tutorials/rl/001_code_rl/001_code_rl.py
    ```
    """


@notebook_only
@shell(
    "import importlib.util\n"
    "\n"
    "if importlib.util.find_spec('modal_training_gym') is None:\n"
    "    %uv pip install -q git+https://github.com/modal-projects/training-gym.git@main"
)
def _install():
    pass


@code
def _imports():
    from modal_training_gym import (
        Endpoint,
        HarborDataset,
        Qwen3_5_9B,
        SlimeRecipe,
        TrainConfig,
        extract_code,
        list_checkpoints,
    )

    base_model = Qwen3_5_9B()


@markdown
def _conversion_intro():
    """
    ## Convert MBPP into Harbor tasks

    Each task directory is a Harbor unit. `instruction.md` is the prompt.
    `tests/verify.py` execs the candidate and runs assert tests.
    `label.json` holds `task_id`, `function_name`, and `reference_bytes`
    for the length bonus.
    """


@code
def _conversion_helpers():
    import ast
    import json
    import re
    import textwrap
    from dataclasses import dataclass
    from pathlib import Path

    from huggingface_hub import hf_hub_download

    MBPP_REPO_ID = "Muennighoff/mbpp"
    MBPP_JSONL_PATH = "data/mbpp.jsonl"
    _FUNCTION_RE = re.compile(r"^\s*def\s+([A-Za-z_]\w*)\s*\(", re.MULTILINE)

    @dataclass
    class MbppRecord:
        task_id: int
        text: str
        code: str
        test_setup_code: str
        test_list: list[str]
        challenge_test_list: list[str]
        function_name: str

    def normalize_code(text: str) -> str:
        return text.replace("\r\n", "\n").replace("\r", "\n")

    def extract_tested_function_name(code: str, tests: list[str]) -> str:
        definitions = set(_FUNCTION_RE.findall(normalize_code(code)))
        calls = {
            node.func.id
            for test in tests
            for node in ast.walk(ast.parse(test))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        candidates = definitions & calls
        if len(candidates) != 1:
            raise ValueError(
                f"Expected one tested function, found {sorted(candidates)}."
            )
        return next(iter(candidates))

    def _load_mbpp_records(
        repo_id: str = MBPP_REPO_ID, dataset_file: str = MBPP_JSONL_PATH
    ) -> list[MbppRecord]:
        mbpp_path = hf_hub_download(
            repo_id=repo_id, filename=dataset_file, repo_type="dataset"
        )
        rows = [
            json.loads(line)
            for line in Path(mbpp_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        records: list[MbppRecord] = []
        for row in rows:
            code = normalize_code(row["code"])
            test_list = list(row.get("test_list", []))
            challenge_test_list = list(row.get("challenge_test_list", []))
            records.append(
                MbppRecord(
                    task_id=int(row["task_id"]),
                    text=row["text"].strip(),
                    code=code,
                    test_setup_code=normalize_code(row.get("test_setup_code", "")),
                    test_list=test_list,
                    challenge_test_list=challenge_test_list,
                    function_name=extract_tested_function_name(
                        code, test_list + challenge_test_list
                    ),
                )
            )
        records.sort(key=lambda item: item.task_id)
        return records

    def _build_instruction(record: MbppRecord) -> str:
        tests = "\n".join(record.test_list)
        return (
            "You are solving a Python code-golf programming task.\n\n"
            "Task:\n"
            f"{record.text}\n\n"
            f"You must define a function named `{record.function_name}` and\n"
            "write only valid Python code to `/workspace/solution.py`.\n\n"
            "Your solution must pass these tests:\n"
            f"```python\n{tests}\n```\n"
        )

    def _build_verify_py(record: MbppRecord) -> str:
        all_tests = record.test_list + record.challenge_test_list
        return textwrap.dedent(
            f"""\
            import json
            import traceback
            from pathlib import Path

            TASK_ID = {record.task_id}
            FUNCTION_NAME = {record.function_name!r}
            TEST_SETUP_CODE = {record.test_setup_code!r}
            TEST_LIST = {json.dumps(all_tests)}

            SOLUTION_PATH = Path("/workspace/solution.py")
            REWARD_JSON = Path("/logs/verifier/reward.json")
            DETAILS_JSON = Path("/logs/verifier/details.json")


            def _write_outputs(reward: dict, details: dict) -> None:
                REWARD_JSON.parent.mkdir(parents=True, exist_ok=True)
                REWARD_JSON.write_text(json.dumps(reward), encoding="utf-8")
                DETAILS_JSON.write_text(json.dumps(details, indent=2), encoding="utf-8")


            def _load_solution() -> tuple[dict, str]:
                source = SOLUTION_PATH.read_text(encoding="utf-8")
                namespace: dict = {{}}
                exec(compile(source, str(SOLUTION_PATH), "exec"), namespace, namespace)
                return namespace, source


            def _run_tests(namespace: dict) -> tuple[int, int, list[dict]]:
                runtime = dict(namespace)
                if TEST_SETUP_CODE.strip():
                    exec(TEST_SETUP_CODE, runtime, runtime)

                passed = 0
                failures: list[dict] = []
                for test_expr in TEST_LIST:
                    try:
                        exec(test_expr, runtime, runtime)
                        passed += 1
                    except Exception as exc:
                        failures.append({{"test": test_expr, "error": repr(exc)}})
                return passed, len(TEST_LIST), failures


            def main() -> int:
                details = {{"task_id": TASK_ID, "function_name": FUNCTION_NAME}}
                try:
                    namespace, source = _load_solution()
                    passed, total, failures = _run_tests(namespace)
                    pass_rate = float(passed) / float(total) if total else 0.0
                    reward = 1.0 if total > 0 and passed == total else 0.0
                    details.update(
                        {{
                            "source_bytes": len(source.encode("utf-8")),
                            "passed": passed,
                            "total": total,
                            "failures": failures,
                        }}
                    )
                    _write_outputs(
                        {{
                            "reward": reward,
                            "pass_rate": pass_rate,
                            "passed": passed,
                            "total": total,
                        }},
                        details,
                    )
                    return 0
                except Exception as exc:
                    details["exception"] = repr(exc)
                    details["traceback"] = traceback.format_exc()
                    _write_outputs(
                        {{
                            "reward": 0.0,
                            "pass_rate": 0.0,
                            "passed": 0,
                            "total": len(TEST_LIST),
                        }},
                        details,
                    )
                    return 0


            if __name__ == "__main__":
                raise SystemExit(main())
            """
        )

    def _write_harbor_task(task_dir: Path, record: MbppRecord) -> None:
        (task_dir / "solution").mkdir(parents=True, exist_ok=True)
        (task_dir / "tests").mkdir(parents=True, exist_ok=True)

        (task_dir / "instruction.md").write_text(
            _build_instruction(record), encoding="utf-8"
        )
        (task_dir / "label.json").write_text(
            json.dumps(
                {
                    "task_id": record.task_id,
                    "function_name": record.function_name,
                    "reference_bytes": len(record.code.encode("utf-8")),
                }
            ),
            encoding="utf-8",
        )
        (task_dir / "solution" / "solution.py").write_text(
            record.code, encoding="utf-8"
        )
        (task_dir / "tests" / "verify.py").write_text(
            _build_verify_py(record), encoding="utf-8"
        )

    def write_harbor_tasks(tasks_root: Path, limit: int) -> None:
        records = _load_mbpp_records()[:limit]
        tasks_root.mkdir(parents=True, exist_ok=True)
        for record in records:
            _write_harbor_task(tasks_root / f"mbpp_{record.task_id:04d}", record)


@markdown
def _dataset_intro():
    """
    ## Load MBPP through HarborDataset

    `MbppHarborDataset` writes 72 Harbor task dirs, then
    `HarborDataset.prepare()` builds the slime parquet (64 train, 8 eval).

    `load()` writes a laptop cache and reads it on a fresh
    `HarborDataset` so this instance keeps `path` unset for remote train.
    """


@code
def _dataset():
    class MbppHarborDataset(HarborDataset):
        limit = 72
        train_size = 64
        eval_size = 8
        always_prepare = True
        shuffle_tasks = False
        label_metadata_path = "label.json"
        system_prompt = (
            "You write short Python that passes the tests. "
            "Define the required function. "
            "Markdown fences are optional."
        )

        def _materialize_tasks(self, tasks_root: Path) -> None:
            write_harbor_tasks(tasks_root, self.limit)

        def prepare(self, path, eval_paths=None):
            tasks_root = Path(path).parent / "tasks"
            self._materialize_tasks(tasks_root)
            self.path = str(tasks_root)
            super().prepare(path, eval_paths)

        def load(self, split="all"):
            cache_root = (
                Path.home() / ".cache" / "training-gym" / "mbpp-harbor" / "tasks"
            )
            self._materialize_tasks(cache_root)
            return HarborDataset(
                path=str(cache_root),
                train_size=self.train_size,
                eval_size=self.eval_size,
                label_metadata_path=self.label_metadata_path,
                system_prompt=self.system_prompt,
                shuffle_tasks=self.shuffle_tasks,
            ).load(split)

    dataset = MbppHarborDataset()


@notebook_only
@markdown
def _dataset_preview():
    """
    Each row has the Harbor instruction plus a label with task identity,
    `harbor_task_path`, and `reference_bytes`.
    """


@notebook_only
@code
def _dataset_preview_code():
    df = dataset.to_pandas()
    print(len(df))
    df.head(5)


@markdown
def _score_intro():
    """
    ## Score with the Harbor verifier

    `score_harbor_solution` reads `tests/verify.py` on this worker, uploads
    it with the candidate into a Python 3.11 Modal sandbox, and reads
    `/logs/verifier/reward.json`.

    Reward is 0 when `pass_rate` is 0. Passing programs get
    `pass_rate * (1 + 0.2 * size_bonus)`. Shorter-than-reference code
    earns the bonus.
    """


@code
def _score_harbor_solution():
    def _compose_harbor_reward(
        pass_rate: float,
        candidate_bytes: int,
        reference_bytes: int,
        length_weight: float = 0.2,
    ) -> float:
        if pass_rate <= 0:
            return 0.0
        size_bonus = max(0.0, min(2.0, reference_bytes / max(candidate_bytes, 1)) - 1.0)
        return pass_rate * (1.0 + length_weight * size_bonus)

    def score_harbor_solution(
        code: str,
        task_dir,
        reference_bytes=None,
    ) -> tuple[float, dict]:
        import json

        import modal

        metadata: dict = {}
        sandbox = None
        try:
            verify_src = (Path(task_dir) / "tests" / "verify.py").read_text(
                encoding="utf-8"
            )
            candidate_bytes = len(code.encode("utf-8"))
            ref_bytes = int(reference_bytes or 1)
            app = modal.App.lookup("training-gym-sandbox-rm", create_if_missing=True)
            image = modal.Image.debian_slim(python_version="3.11")
            sandbox = modal.Sandbox._experimental_create(
                "sleep",
                "infinity",
                app=app,
                image=image,
                timeout=120,
                cpu=1.0,
                memory=1024,
            )
            metadata["sandbox_id"] = sandbox.object_id
            sandbox.filesystem.write_text(code, "/workspace/solution.py")
            sandbox.filesystem.write_text(verify_src, "/workspace/verify.py")
            sandbox.exec("python", "/workspace/verify.py").wait()
            result = json.loads(
                sandbox.filesystem.read_text("/logs/verifier/reward.json")
            )
            pass_rate = float(result.get("pass_rate", result.get("reward", 0.0)))
            reward = _compose_harbor_reward(pass_rate, candidate_bytes, ref_bytes)
            metadata.update(
                {
                    "pass_rate": pass_rate,
                    "passed": result.get("passed"),
                    "total": result.get("total"),
                    "candidate_bytes": candidate_bytes,
                    "reference_bytes": ref_bytes,
                    "reward": reward,
                }
            )
            return reward, metadata
        except Exception as exc:
            metadata["error"] = repr(exc)
            return 0.0, metadata
        finally:
            if sandbox:
                try:
                    sandbox.terminate()
                except Exception:
                    pass


@markdown
def _gold_intro():
    """
    ## Prove the gold solution scores

    Load one converted task and run its reference `solution/solution.py`
    through the Harbor sandbox.
    """


@code
def _gold_proof():
    import os

    example = dataset.load()[0]
    label = example["label"]
    task_dir = Path(label["harbor_task_path"])
    gold = (task_dir / "solution" / "solution.py").read_text(encoding="utf-8")
    reward, meta = score_harbor_solution(gold, task_dir, label.get("reference_bytes"))
    print(
        f"sandbox_id={meta.get('sandbox_id')} "
        f"pass_rate={meta.get('pass_rate')} reward={reward}"
    )
    if os.environ.get("CODE_RL_SMOKE"):
        if meta.get("pass_rate"):
            raise SystemExit(0)
        raise SystemExit(f"gold Harbor score failed: {meta}")


@markdown
def _eval_intro():
    """
    ## Eval helper

    `run_eval` chats the served endpoint, extracts code, and scores with
    `score_harbor_solution`. A full base-model GPU eval is skipped here.
    The trained checkpoint is served and scored after train.
    """


@code
def _run_eval():
    def run_eval(endpoint, *, max_concurrency: int = 1) -> float:
        from concurrent.futures import ThreadPoolExecutor

        def _score_one(example):
            prompt = example["instruction"]
            messages = [
                {"role": "system", "content": dataset.system_prompt},
                {"role": "user", "content": prompt},
            ]
            message = endpoint.chat(
                messages,
                extra_parameters={
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            response = message.get("content") or ""
            code = extract_code(response, model=base_model)
            label = example["label"]
            reward, _meta = score_harbor_solution(
                code, label["harbor_task_path"], label.get("reference_bytes")
            )
            return float(reward)

        rows = dataset.load(split="eval")
        with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
            rewards = list(executor.map(_score_one, rows))
        return sum(rewards) / len(rewards) if rewards else float("nan")


@markdown
def _train_intro():
    """
    ## Train with SLIME

    `code_golf_rm` is the same Harbor scorer, wrapped for SLIME, on a
    4×8 H100 clustered RDMA layout with 10 rollouts, eval every 5 steps,
    and a checkpoint every 5. Qwen3.5-9B is the gym preset that fits this
    colocated GRPO shape (TP=2, sequence parallel, two GPUs per
    rollout engine) and still serves on a Modal endpoint.
    """


@code
def _train():
    async def code_golf_rm(args, sample, **kwargs) -> float:
        import asyncio
        import json

        raw = sample.label
        label = json.loads(raw) if isinstance(raw, str) else (raw or {})
        code = extract_code(sample.response, model=base_model)
        reward, meta = await asyncio.to_thread(
            score_harbor_solution,
            code,
            label["harbor_task_path"],
            label.get("reference_bytes"),
        )
        sample.metadata = {**(getattr(sample, "metadata", None) or {}), "harbor": meta}
        return float(reward)

    training_run = TrainConfig(
        model=base_model,
        dataset=dataset,
        recipe=SlimeRecipe(
            custom_rm_function=code_golf_rm,
            gpu_type="H100",
            actor_num_nodes=4,
            actor_num_gpus_per_node=8,
            colocate=True,
            tensor_model_parallel_size=2,
            sequence_parallel=True,
            rollout_num_gpus_per_engine=2,
            sglang_mem_fraction_static=0.25,
            num_rollout=10,
            rollout_batch_size=8,
            n_samples_per_prompt=8,
            rollout_max_response_len=2048,
            rollout_temperature=0.9,
            global_batch_size=16,
            eval_interval=5,
            eval_max_response_len=2048,
            n_samples_per_eval_prompt=8,
            max_tokens_per_gpu=4096,
            save_interval=5,
            apply_chat_template_kwargs='{"enable_thinking": false}',
        ),
    )
    print("Starting training...")
    train_result = training_run.train()
    print(f"Training run id: {train_result.training_run_id}")


@markdown
def _serve_trained_intro():
    """
    ## Serve and eval the trained checkpoint

    Slime writes Megatron checkpoints. Pass the latest one to
    `Endpoint.launch`; Megatron weights are converted to Hugging Face
    format during launch. `launch` returns as soon as a URL exists;
    `wait_until_ready` waits until the endpoint can serve.
    """


@code
def _serve_trained():
    megatron_checkpoint = list_checkpoints(train_result.training_run_id)[-1]
    trained_endpoint = Endpoint.launch(
        base_model,
        megatron_checkpoint,
        endpoint_name="qwen3-5-9b-code-golf",
        unauthenticated=True,
    )
    trained_endpoint.wait_until_ready()
    print(f"Trained model URL: {trained_endpoint.url}")

    trained_mean = run_eval(trained_endpoint)
    print(f"Trained mean reward: {trained_mean:.4f}")
