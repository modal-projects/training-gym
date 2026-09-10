import json

import pytest
from datasets import Dataset

from modal_training_gym import (
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
        "training_type": "sft",
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


def test_sft_training_type_emits_train_only_flags() -> None:
    flags = _flags(
        _recipe(
            num_steps=100, num_gpus_per_node=1, num_nodes=2, shuffle=True
        ).cli_args()
    )

    assert "--training-type" not in flags
    assert "--num-steps" not in flags
    assert "--num-gpus-per-node" not in flags
    assert "--num-nodes" not in flags
    assert "--shuffle" not in flags
    assert flags["--actor-num-gpus-per-node"] == "1"
    assert flags["--actor-num-nodes"] == "2"
    assert flags["--rollout-shuffle"] is True
    assert flags["--num-rollout"] == "100"
    assert (
        flags["--rollout-function-path"] == "slime.rollout.sft_rollout.generate_rollout"
    )
    assert flags["--loss-type"] == "sft_loss"
    assert flags["--calculate-per-token-loss"] is True
    assert flags["--disable-compute-advantages-and-returns"] is True
    assert flags["--debug-train-only"] is True
    assert flags["--n-samples-per-prompt"] == "1"
    assert "--eval-interval" not in flags


def test_sft_training_type_normalizes_rl_settings() -> None:
    recipe = _recipe(
        num_steps=100,
        num_rollout=2,
        num_gpus_per_node=1,
        num_nodes=2,
        shuffle=False,
        colocate=True,
        rollout_num_gpus=8,
        n_samples_per_prompt=8,
        eval_interval=10,
        use_fault_tolerance=True,
        use_critic=True,
        use_kl_loss=True,
        kl_coef=0.1,
    )

    assert recipe.num_rollout == recipe.num_steps == 100
    assert recipe.actor_num_gpus_per_node == 1
    assert recipe.actor_num_nodes == 2
    assert recipe.gpu_allocation.total_gpus == 2
    assert recipe.rollout_shuffle is False
    assert "--rollout-shuffle" not in _flags(recipe.cli_args())
    assert recipe.async_mode is True
    assert recipe.colocate is False
    assert recipe.rollout_num_gpus is None
    assert recipe.n_samples_per_prompt == 1
    assert recipe.eval_interval is None
    assert recipe.use_fault_tolerance is False
    assert recipe.gpu_allocation.critic_gpus == 0
    assert recipe.use_kl_loss is False
    assert recipe.kl_coef == 0


@pytest.mark.parametrize("overrides", [{}, {"rollout_batch_size": 2}])
def test_sft_training_type_derives_rollout_batch_size(overrides) -> None:
    recipe = _recipe(global_batch_size=32, **overrides)
    assert recipe.rollout_batch_size == recipe.global_batch_size == 32
    assert _flags(recipe.cli_args())["--rollout-batch-size"] == "32"


@pytest.mark.parametrize(
    "alias, field, current, value, expected",
    [
        ("num_gpus_per_node", "actor_num_gpus_per_node", 2, None, 2),
        ("num_gpus_per_node", "actor_num_gpus_per_node", 2, 1, 1),
        ("num_nodes", "actor_num_nodes", 3, None, 3),
        ("num_nodes", "actor_num_nodes", 3, 2, 2),
        ("shuffle", "rollout_shuffle", True, None, True),
        ("shuffle", "rollout_shuffle", False, None, False),
        ("shuffle", "rollout_shuffle", True, False, False),
        ("shuffle", "rollout_shuffle", False, True, True),
    ],
)
def test_sft_alias_overrides_or_preserves_field(alias, field, current, value, expected):
    recipe = _recipe(**{field: current, alias: value})

    assert getattr(recipe, field) == expected


@pytest.mark.parametrize(
    "key, value",
    [
        ("loss_type", "policy_loss"),
        ("rollout_batch_size", 2),
        ("num_rollout", 2),
        ("actor_num_gpus_per_node", 2),
        ("actor_num_nodes", 2),
        ("rollout_shuffle", False),
    ],
)
def test_sft_training_type_rejects_escape_hatch_override(key, value) -> None:
    with pytest.raises(ValueError, match=f"remove them from extra_config: {key}"):
        _recipe(extra_config={key: value})


def test_rl_training_type_is_unchanged() -> None:
    recipe = Qwen3_4B_Recipe(num_rollout=7)

    assert recipe.training_type == "rl"
    assert recipe.num_rollout == 7
    assert recipe.colocate is True
    assert recipe.n_samples_per_prompt == 8
    assert recipe.eval_interval is None
    assert recipe.loss_type is None
    assert recipe.debug_train_only is False


def test_sft_preserves_qwen_model_recipe() -> None:
    baseline = Qwen3_4B_Recipe(lr=1e-5)
    recipe = _recipe()

    for name in (
        "gpu_type",
        "tensor_model_parallel_size",
        "max_tokens_per_gpu",
        "lr",
        "rollout_shuffle",
    ):
        assert getattr(recipe, name) == getattr(baseline, name)
    assert recipe.gpu_allocation.actor_gpus == baseline.gpu_allocation.actor_gpus
    assert recipe.gpu_allocation.rollout_gpus == 0
    assert recipe.total_nodes == baseline.actor_num_nodes


def test_sft_preserves_large_model_topology() -> None:
    baseline = GLM_4_7_Recipe()
    config = TrainConfig(
        dataset=SFTDataset(hf_repo="example/chat", messages_column="messages"),
        model=GLM_4_7(),
        recipe=GLM_4_7_Recipe(training_type="sft", global_batch_size=64),
    )
    recipe = config._prepare_recipe()

    assert isinstance(recipe, GLM_4_7_Recipe)
    assert recipe.training_type == "sft"
    for name in (
        "gpu_type",
        "tensor_model_parallel_size",
        "sequence_parallel",
        "actor_num_nodes",
    ):
        assert getattr(recipe, name) == getattr(baseline, name)
    assert recipe.gpu_allocation.actor_gpus == baseline.gpu_allocation.actor_gpus
    assert recipe.gpu_allocation.rollout_gpus == 0
    assert recipe.total_nodes == baseline.actor_num_nodes


def test_sft_dataset_formats_prompt_completion_pair() -> None:
    source = Dataset.from_dict({"question": ["2 + 2?"], "answer": ["4"]})
    dataset = SFTDataset(
        hf_repo="example/math",
        input_column="question",
        output_column="answer",
    )

    assert dataset._format_for_training(source)[0] == {
        "question": [
            {"role": "user", "content": "2 + 2?"},
            {"role": "assistant", "content": "4"},
        ],
        "answer": "",
    }


@pytest.mark.parametrize(
    "columns",
    [{"input_column": "text", "output_column": "text"}, {"messages_column": "label"}],
)
def test_sft_dataset_rejects_matching_input_and_output_columns(columns) -> None:
    with pytest.raises(ValueError, match="input and output columns must be distinct"):
        SFTDataset(hf_repo="example/chat", **columns)


def test_sft_dataset_materializes_raw_conversations(monkeypatch, tmp_path) -> None:
    source = Dataset.from_dict({"question": ["2 + 2?"], "answer": ["4"]})
    dataset = SFTDataset(
        hf_repo="example/math",
        input_column="question",
        output_column="answer",
    )
    monkeypatch.setattr("datasets.load_dataset", lambda *args, **kwargs: source)
    output = tmp_path / "train.jsonl"

    dataset.write(str(output))
    dataset.validate_written(str(output))

    assert json.loads(output.read_text()) == {
        "question": [
            {"role": "user", "content": "2 + 2?"},
            {"role": "assistant", "content": "4"},
        ],
        "answer": "",
    }
    assert dataset.input_format == "raw"
    assert dataset.apply_chat_template() is False
    assert _recipe()._dataset_to_fields(dataset)["apply_chat_template"] is False


@pytest.mark.parametrize("input_format", ["text", "raw"])
def test_sft_prepared_data_does_not_share_rl_cache(input_format) -> None:
    source = dict(hf_repo="example/chat", input_column="prompt", output_column="answer")
    rl_path = Qwen3_4B_Recipe()._resolve_data_paths(
        HuggingFaceDataset(**source, input_format=input_format)
    )
    sft_path = _recipe()._resolve_data_paths(SFTDataset(**source))

    assert rl_path != sft_path
    assert sft_path == _recipe()._resolve_data_paths(SFTDataset(**source))


@pytest.mark.parametrize(
    "override",
    [
        {"messages_column": "conversation"},
        {"messages_column": "", "input_column": "messages", "output_column": "label"},
    ],
)
def test_sft_cache_key_includes_formatting(override):
    source = {"hf_repo": "example/chat", "messages_column": "messages"}
    assert (
        SFTDataset(**source).cache_key()
        != SFTDataset(**(source | override)).cache_key()
    )


def test_sft_always_download_disables_cache():
    dataset = SFTDataset(
        hf_repo="example/chat", messages_column="messages", always_download=True
    )
    assert dataset.cache_key() is None
    assert _recipe()._resolve_data_paths(dataset) != _recipe()._resolve_data_paths(
        dataset
    )


@pytest.mark.parametrize("messages_column", ["messages", "conversation"])
@pytest.mark.parametrize(
    "assistant",
    [
        {"role": "assistant", "content": "Hi"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"},
                }
            ],
        },
    ],
)
def test_sft_dataset_accepts_existing_conversations(messages_column, assistant) -> None:
    messages = [{"role": "user", "content": "Hello"}, assistant]
    source = Dataset.from_dict({messages_column: [messages]})
    dataset = SFTDataset(hf_repo="example/chat", messages_column=messages_column)

    assert dataset._format_for_training(source)[0] == {
        messages_column: source[0][messages_column],
        "label": "",
    }


def test_sft_dataset_requires_an_assistant_turn() -> None:
    source = Dataset.from_dict(
        {"conversation": [[{"role": "user", "content": "Hello"}]]}
    )
    dataset = SFTDataset(hf_repo="example/chat", messages_column="conversation")

    with pytest.raises(ValueError, match="assistant turn"):
        dataset._format_for_training(source)


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


def test_raw_dataset_can_opt_into_sft() -> None:
    dataset = HuggingFaceDataset(
        hf_repo="example/chat",
        input_column="messages",
        output_column="label",
        input_format="raw",
    )

    fields = _recipe()._dataset_to_fields(dataset)
    assert fields["input_key"] == "messages"
    assert fields["apply_chat_template"] is False


def test_train_config_preserves_model_recipe_and_training_type_summary() -> None:
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
    assert resolved.training_type == "sft"
    assert config._build_config_summary("test-run")["training_type"] == "sft"


@pytest.mark.parametrize("answer", [None, "", "   "])
def test_sft_rejects_empty_targets(answer):
    dataset = SFTDataset(hf_repo="test", input_column="q", output_column="a")
    with pytest.raises(ValueError, match="non-empty text"):
        dataset._format_for_training(
            Dataset.from_dict({"q": ["Question"], "a": [answer]})
        )


def test_sft_rejects_missing_columns():
    dataset = SFTDataset(hf_repo="test", messages_column="messages")
    with pytest.raises(ValueError, match="Missing SFT columns: messages"):
        dataset._format_for_training(Dataset.from_dict({"other": ["text"]}))


@pytest.mark.parametrize(
    "config",
    [
        {"training_type": "sft"},
        {},
    ],
)
def test_training_type_survives_public_summary(config):
    from modal_training_gym.common.run_summary import build_run_summary

    summary = build_run_summary({"training_run_id": "test", "config": config})
    assert summary.model_dump()["training_type"] == ("sft" if config else "rl")
