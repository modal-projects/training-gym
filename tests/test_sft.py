import pytest
from datasets import Dataset

from modal_training_gym import (
    DatasetConfig,
    GLM_4_7,
    HuggingFaceDataset,
    Qwen3_4B,
    Qwen3_4B_Recipe,
    SFTDataset,
    TrainConfig,
)
from modal_training_gym.train_recipes.slime_recipe import GLM_4_7_Recipe


def _recipe(**overrides):
    values = {
        "workload_type": "sft",
        "rollout_batch_size": 4,
        "global_batch_size": 4,
        "lr": 1e-5,
    }
    values.update(overrides)
    return Qwen3_4B_Recipe(**values)


def _flags(args: list[str]) -> dict[str, str | bool]:
    out: dict[str, str | bool] = {}
    for index, arg in enumerate(args):
        if arg.startswith("--"):
            out[arg] = (
                args[index + 1]
                if index + 1 < len(args) and not args[index + 1].startswith("--")
                else True
            )
    return out


def test_sft_workload_emits_train_only_flags() -> None:
    flags = _flags(_recipe().cli_args())

    assert "--workload-type" not in flags
    assert (
        flags["--rollout-function-path"] == "slime.rollout.sft_rollout.generate_rollout"
    )
    assert flags["--loss-type"] == "sft_loss"
    assert flags["--calculate-per-token-loss"] is True
    assert flags["--disable-compute-advantages-and-returns"] is True
    assert flags["--debug-train-only"] is True
    assert flags["--n-samples-per-prompt"] == "1"
    assert "--eval-interval" not in flags


def test_sft_workload_normalizes_rl_settings() -> None:
    recipe = _recipe(
        colocate=True,
        rollout_num_gpus=8,
        n_samples_per_prompt=8,
        eval_interval=10,
        use_fault_tolerance=True,
    )

    assert recipe.async_mode is True
    assert recipe.colocate is False
    assert recipe.rollout_num_gpus is None
    assert recipe.n_samples_per_prompt == 1
    assert recipe.eval_interval is None
    assert recipe.use_fault_tolerance is False


def test_sft_workload_requires_one_file_batch_per_step() -> None:
    with pytest.raises(ValueError, match="rollout_batch_size == global_batch_size"):
        _recipe(global_batch_size=8)


def test_sft_workload_rejects_escape_hatch_override() -> None:
    with pytest.raises(ValueError, match="remove them from extra_config: loss_type"):
        _recipe(extra_config={"loss_type": "policy_loss"})


def test_rl_workload_is_unchanged() -> None:
    recipe = Qwen3_4B_Recipe()

    assert recipe.workload_type == "rl"
    assert recipe.colocate is True
    assert recipe.n_samples_per_prompt == 8
    assert recipe.eval_interval == 10
    assert recipe.loss_type is None
    assert recipe.debug_train_only is False


def test_sft_preserves_qwen_model_recipe() -> None:
    recipe = _recipe()

    assert recipe.gpu_type == "H100"
    assert recipe.tensor_model_parallel_size == 1
    assert recipe.max_tokens_per_gpu == 8192
    assert recipe.lr == 1e-5
    assert recipe.gpu_allocation.actor_gpus == 8
    assert recipe.gpu_allocation.rollout_gpus == 0
    assert recipe.total_nodes == 1


def test_sft_preserves_large_model_topology() -> None:
    recipe = GLM_4_7_Recipe(workload_type="sft", global_batch_size=64)

    assert recipe.gpu_type == "H200"
    assert recipe.tensor_model_parallel_size == 8
    assert recipe.sequence_parallel is True
    assert recipe.actor_num_nodes == 8
    assert recipe.gpu_allocation.actor_gpus == 64
    assert recipe.gpu_allocation.rollout_gpus == 0
    assert recipe.total_nodes == 8


def test_sft_dataset_formats_prompt_completion_pair(monkeypatch) -> None:
    source = Dataset.from_dict({"question": ["2 + 2?"], "answer": ["4"]})
    dataset = SFTDataset(
        hf_repo="example/math",
        input_column="question",
        output_column="answer",
    )
    monkeypatch.setattr(dataset, "load", lambda *args, **kwargs: source)

    assert dataset._format_for_training(dataset.load())[0] == {
        "messages": [
            {"role": "user", "content": "2 + 2?"},
            {"role": "assistant", "content": "4"},
        ],
        "label": "",
    }


def test_sft_dataset_materializes_without_eval_split(monkeypatch, tmp_path) -> None:
    source = Dataset.from_dict({"question": ["2 + 2?"], "answer": ["4"]})
    dataset = SFTDataset(
        hf_repo="example/math",
        input_column="question",
        output_column="answer",
    )
    monkeypatch.setattr(dataset, "load", lambda *args, **kwargs: source)
    output = tmp_path / "train.parquet"

    dataset.prepare(str(output))
    dataset.validate_prepared(str(output))

    assert output.exists()
    assert dataset.writes_eval_paths is False


def test_sft_dataset_accepts_existing_conversations(monkeypatch) -> None:
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
    ]
    source = Dataset.from_dict({"conversation": [messages]})
    dataset = SFTDataset(hf_repo="example/chat", messages_column="conversation")
    monkeypatch.setattr(dataset, "load", lambda *args, **kwargs: source)

    assert dataset._format_for_training(dataset.load())[0] == {
        "messages": messages,
        "label": "",
    }


def test_sft_dataset_requires_an_assistant_turn(monkeypatch) -> None:
    source = Dataset.from_dict(
        {"conversation": [[{"role": "user", "content": "Hello"}]]}
    )
    dataset = SFTDataset(hf_repo="example/chat", messages_column="conversation")
    monkeypatch.setattr(dataset, "load", lambda *args, **kwargs: source)

    with pytest.raises(ValueError, match="assistant turn"):
        dataset._format_for_training(dataset.load())


@pytest.mark.parametrize(
    "override", [{"system_prompt": "Be helpful."}, {"prompt_template": "Q: {input}"}]
)
def test_sft_dataset_rejects_formatting_with_messages_column(override) -> None:
    with pytest.raises(ValueError, match="with messages_column"):
        SFTDataset(hf_repo="example/chat", messages_column="conversation", **override)


def test_sft_dataset_requires_input_and_output_columns_together() -> None:
    with pytest.raises(ValueError, match="input_column and output_column together"):
        SFTDataset(
            hf_repo="example/chat",
            messages_column="messages",
            input_column="prompt",
        )


def test_sft_rejects_prompt_only_dataset() -> None:
    dataset = HuggingFaceDataset(
        hf_repo="example/math",
        input_column="question",
        output_column="answer",
        apply_chat_template=False,
    )

    with pytest.raises(ValueError, match="supports_sft=True"):
        _recipe().validate_dataset(dataset)


def test_custom_dataset_can_opt_into_sft() -> None:
    dataset = DatasetConfig(
        input_key="messages",
        label_key="label",
        apply_chat_template=False,
        supports_sft=True,
    )

    _recipe().validate_dataset(dataset)


def test_train_config_preserves_model_recipe_and_workload_summary() -> None:
    config = TrainConfig(
        dataset=SFTDataset(
            hf_repo="example/math",
            input_column="question",
            output_column="answer",
        ),
        model=Qwen3_4B(),
        recipe=_recipe(),
    )

    resolved = config._prepare_recipe()
    assert isinstance(resolved, Qwen3_4B_Recipe)
    assert resolved.workload_type == "sft"
    assert config._build_config_summary("test-run")["objective"] == "sft"


def test_glm_sft_config_accepts_glm_model() -> None:
    config = TrainConfig(
        dataset=SFTDataset(hf_repo="example/chat", messages_column="messages"),
        model=GLM_4_7(),
        recipe=GLM_4_7_Recipe(workload_type="sft", global_batch_size=64),
    )

    assert config._prepare_recipe().workload_type == "sft"
